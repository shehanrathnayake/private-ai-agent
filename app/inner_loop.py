"""
inner_loop.py — Phase C: The Heartbeat

Runs as a daemon thread independently of the FastAPI request cycle.
Every INNER_LOOP_INTERVAL seconds it:
  1. Checks guards (circuit breaker, nothing-new, topic diversity)
  2. Calls ThoughtEngine.think() to generate a private thought
  3. Stores memory-worthy thoughts via memory_manager.add_vector()
  4. Enforces the thought count cap via memory_manager.prune_inner_thoughts()
  5. Every 10 cycles, triggers an introspection report

Design principles:
  - daemon=True  → thread dies cleanly when the server shuts down
  - Circuit breaker → 3 consecutive LLM failures pause the loop for 30 minutes
  - Topic diversity → skips storage if last 3 thoughts are semantically too similar
  - All guards are logged but never raise — the loop must never crash the server
"""

import threading
import time
from datetime import datetime
from typing import Optional

from app.config import (
    INNER_LOOP_ENABLED,
    INNER_LOOP_INTERVAL,
)


# How many consecutive failures before the circuit breaker trips
_CIRCUIT_BREAKER_THRESHOLD = 3
# How long (seconds) to pause after the circuit breaker trips
_CIRCUIT_BREAKER_PAUSE = 1800  # 30 minutes
# How many thought cycles between introspection reports
_INTROSPECTION_EVERY_N_CYCLES = 10
# Topic diversity: skip storage if cosine similarity to last N thoughts exceeds this
_DIVERSITY_THRESHOLD = 0.88
_DIVERSITY_LOOKBACK = 3


class InnerLoop(threading.Thread):
    """
    Background daemon thread that drives the inner monologue cycle.
    Instantiated once at server startup; never restarted.
    """

    def __init__(self, memory_manager):
        super().__init__(daemon=True, name="InnerLoop")
        self.mm = memory_manager

        # State visible to /debug loop
        self._lock = threading.Lock()           # Protects all state fields below
        self._running = False
        self._cycle_count = 0                   # Total cycles attempted
        self._thoughts_stored = 0               # Total thoughts stored
        self._last_run_at: Optional[datetime] = None
        self._last_thought_at: Optional[datetime] = None
        self._last_thought_text: str = ""
        self._consecutive_failures = 0
        self._circuit_open = False              # True = paused due to failures
        self._circuit_open_until: Optional[datetime] = None
        self._last_skip_reason: str = ""

        # Lazy-initialised inside the thread to avoid import-time side effects
        self._thought_engine = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        if not INNER_LOOP_ENABLED:
            print("[INNER_LOOP] Disabled via INNER_LOOP_ENABLED=false. Not starting.")
            return
        self._running = True
        super().start()
        print(f"[INNER_LOOP] Started. Interval={INNER_LOOP_INTERVAL}s, "
              f"CircuitBreaker={_CIRCUIT_BREAKER_THRESHOLD} failures → "
              f"{_CIRCUIT_BREAKER_PAUSE//60}min pause.")

    def stop(self):
        """Signals the loop to stop after the current sleep."""
        with self._lock:
            self._running = False
        print("[INNER_LOOP] Stop signal sent.")

    def get_status(self) -> str:
        """Returns a human-readable status string for /debug loop."""
        with self._lock:
            lines = [
                f"Running          : {self._running}",
                f"Cycle count      : {self._cycle_count}",
                f"Thoughts stored  : {self._thoughts_stored}",
                f"Last run at      : {self._last_run_at or 'Never'}",
                f"Last thought at  : {self._last_thought_at or 'Never'}",
                f"Circuit breaker  : {'OPEN (paused)' if self._circuit_open else 'Closed (healthy)'}",
            ]
            if self._circuit_open and self._circuit_open_until:
                lines.append(f"Circuit resumes  : {self._circuit_open_until}")
            if self._last_skip_reason:
                lines.append(f"Last skip reason : {self._last_skip_reason}")
            if self._last_thought_text:
                preview = self._last_thought_text[:80]
                lines.append(f"Last thought     : {preview}...")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Thread main loop
    # ------------------------------------------------------------------

    def run(self):
        """Entry point for the daemon thread."""
        # Lazy init — import here to avoid circular imports at module load time
        from app.thought_engine import ThoughtEngine
        self._thought_engine = ThoughtEngine(self.mm)

        print("[INNER_LOOP] Thread running.")

        while True:
            with self._lock:
                if not self._running:
                    break

            self._tick()

            # Sleep in small increments so stop() is responsive
            for _ in range(INNER_LOOP_INTERVAL):
                with self._lock:
                    if not self._running:
                        break
                time.sleep(1)

        print("[INNER_LOOP] Thread exited cleanly.")

    # ------------------------------------------------------------------
    # One thinking cycle
    # ------------------------------------------------------------------

    def _tick(self):
        """Executes one full thinking cycle with all guards."""
        now = datetime.now()

        with self._lock:
            self._cycle_count += 1
            self._last_run_at = now
            cycle_num = self._cycle_count

        print(f"[INNER_LOOP] Cycle #{cycle_num} starting at {now.strftime('%H:%M:%S')}")

        # Guard 1: Circuit breaker
        if self._is_circuit_open(now):
            return

        # Guard 2: Generate thought (ThoughtEngine handles "nothing new" internally)
        with self._lock:
            last_thought_ts = self._last_thought_at

        thought = self._thought_engine.think(last_thought_timestamp=last_thought_ts)

        if thought.skipped:
            with self._lock:
                self._last_skip_reason = thought.skip_reason
            print(f"[INNER_LOOP] Cycle #{cycle_num} skipped: {thought.skip_reason}")
            # A skip is not a failure — reset consecutive failure count
            with self._lock:
                self._consecutive_failures = 0
            self._journal_thought(now, cycle_num, thought, outcome="SKIPPED")
            return

        # Thought was generated — check for LLM error (empty thought = failure)
        if not thought.raw_thought:
            self._record_failure(cycle_num, "Empty thought returned (possible LLM error)")
            self._journal_thought(now, cycle_num, thought, outcome="FAILED")
            return

        # Success — reset circuit breaker
        with self._lock:
            self._consecutive_failures = 0
            if self._circuit_open:
                self._circuit_open = False
                self._circuit_open_until = None
                print("[INNER_LOOP] Circuit breaker reset — LLM responding normally.")

        print(f"[INNER_LOOP] Cycle #{cycle_num} thought generated "
              f"(salience={thought.salience:.2f}, act={thought.should_act}, "
              f"store={thought.memory_worthy})")
        print(f"[INNER_LOOP] THOUGHT: {thought.raw_thought[:100]}...")

        # Guard 3: Topic diversity check
        if thought.memory_worthy and self._is_too_similar_to_recent(thought.raw_thought):
            with self._lock:
                self._last_skip_reason = "Topic diversity guard: thought too similar to recent ones."
            print(f"[INNER_LOOP] Cycle #{cycle_num} storage skipped: diversity guard triggered.")
            # Still update last_thought_at so the guard timestamp advances
            with self._lock:
                self._last_thought_at = now
                self._last_thought_text = thought.raw_thought
            self._journal_thought(now, cycle_num, thought, outcome="DIVERSITY_BLOCKED")
            return

        # Store the thought
        if thought.memory_worthy:
            try:
                self.mm.add_vector(
                    thought.raw_thought,
                    session_id="__inner__",
                    mem_type="Inner Thought",
                    salience=thought.salience,
                )
                self.mm.prune_inner_thoughts()

                with self._lock:
                    self._thoughts_stored += 1
                    self._last_thought_at = now
                    self._last_thought_text = thought.raw_thought

                print(f"[INNER_LOOP] Cycle #{cycle_num} thought stored "
                      f"(total stored: {self._thoughts_stored})")
                self._journal_thought(now, cycle_num, thought, outcome="STORED")
            except Exception as e:
                self._record_failure(cycle_num, f"Storage error: {e}")
                self._journal_thought(now, cycle_num, thought, outcome="FAILED")
                return
        else:
            # Not memory-worthy — still update timestamp
            with self._lock:
                self._last_thought_at = now
                self._last_thought_text = thought.raw_thought
            print(f"[INNER_LOOP] Cycle #{cycle_num} thought not stored (memory_worthy=False).")
            self._journal_thought(now, cycle_num, thought, outcome="NOT_WORTHY")

        # Periodic introspection (every N cycles)
        if cycle_num % _INTROSPECTION_EVERY_N_CYCLES == 0:
            self._run_introspection()

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _is_circuit_open(self, now: datetime) -> bool:
        """Returns True if the circuit breaker is tripped and not yet reset."""
        with self._lock:
            if not self._circuit_open:
                return False
            if self._circuit_open_until and now >= self._circuit_open_until:
                # Auto-reset after pause duration
                self._circuit_open = False
                self._circuit_open_until = None
                self._consecutive_failures = 0
                print("[INNER_LOOP] Circuit breaker auto-reset after pause.")
                return False
            remaining = int((self._circuit_open_until - now).total_seconds() // 60)
            print(f"[INNER_LOOP] Circuit breaker OPEN. Resuming in ~{remaining} min.")
            return True

    def _is_too_similar_to_recent(self, new_thought: str) -> bool:
        """
        Checks if the new thought is semantically too similar to the last
        _DIVERSITY_LOOKBACK stored inner thoughts.
        Returns True if diversity guard should block storage.
        """
        try:
            import sqlite3
            import numpy as np
            import faiss
            from app.memory import METADATA_DB_PATH

            with sqlite3.connect(METADATA_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT content FROM vector_metadata
                    WHERE session_id = '__inner__' AND merged = 0
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (_DIVERSITY_LOOKBACK,))
                recent = [row[0] for row in cursor.fetchall()]

            if not recent:
                return False

            # Embed new thought
            model = self.mm._get_model()
            new_vec = model.encode([new_thought])
            new_vec = np.array(new_vec).astype('float32')
            faiss.normalize_L2(new_vec)

            # Compare against each recent thought
            for content in recent:
                old_vec = model.encode([content])
                old_vec = np.array(old_vec).astype('float32')
                faiss.normalize_L2(old_vec)
                sim = float(np.dot(new_vec, old_vec.T)[0][0])
                if sim >= _DIVERSITY_THRESHOLD:
                    print(f"[INNER_LOOP] Diversity guard: sim={sim:.3f} >= {_DIVERSITY_THRESHOLD} "
                          f"vs '{content[:50]}...'")
                    return True

            return False

        except Exception as e:
            # If diversity check fails, allow storage (fail open)
            print(f"[INNER_LOOP] Diversity check error (allowing storage): {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _journal_thought(self, now: datetime, cycle_num: int, thought, outcome: str):
        """
        Appends one entry to the daily thought journal in memory/thoughts/YYYY-MM-DD.md.
        This is the human-readable transparency layer — never read by the agent.
        Outcomes: STORED | SKIPPED | DIVERSITY_BLOCKED | NOT_WORTHY | FAILED
        """
        try:
            from app.bootstrap import THOUGHTS_DIR
            import os

            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            journal_path = os.path.join(THOUGHTS_DIR, f"{date_str}.md")

            # Outcome emoji for quick visual scanning
            icons = {
                "STORED":           "🧠",
                "SKIPPED":          "⏭️ ",
                "DIVERSITY_BLOCKED": "🔁",
                "NOT_WORTHY":       "💭",
                "FAILED":           "❌",
            }
            icon = icons.get(outcome, "?")

            lines = [f"\n## {icon} Cycle #{cycle_num} — {time_str} — {outcome}"]

            if thought.raw_thought:
                lines.append(f"**Thought:** {thought.raw_thought}")
                lines.append(f"**Salience:** {thought.salience:.2f} | "
                             f"**Act:** {thought.should_act} | "
                             f"**Action hint:** {thought.action_hint}")
            if thought.skipped and thought.skip_reason:
                lines.append(f"**Skip reason:** {thought.skip_reason}")

            lines.append("")  # trailing newline

            # Create file with header if it doesn't exist yet
            if not os.path.exists(journal_path):
                header = f"# Inner Monologue Journal — {date_str}\n"
                header += f"_Auto-generated by InnerLoop. Human read-only._\n"
                with open(journal_path, "w", encoding="utf-8") as f:
                    f.write(header)

            with open(journal_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))

        except Exception as e:
            # Journal write failure must never affect the loop
            print(f"[INNER_LOOP] Journal write error (non-fatal): {e}")

    def _record_failure(self, cycle_num: int, reason: str):
        """Increments failure counter and trips circuit breaker if threshold reached."""
        with self._lock:
            self._consecutive_failures += 1
            count = self._consecutive_failures
            self._last_skip_reason = reason

        print(f"[INNER_LOOP] Cycle #{cycle_num} FAILED ({count}/{_CIRCUIT_BREAKER_THRESHOLD}): {reason}")

        if count >= _CIRCUIT_BREAKER_THRESHOLD:
            from datetime import timedelta
            resume_at = datetime.now() + timedelta(seconds=_CIRCUIT_BREAKER_PAUSE)
            with self._lock:
                self._circuit_open = True
                self._circuit_open_until = resume_at
            print(f"[INNER_LOOP] ⚠️  Circuit breaker TRIPPED after {count} failures. "
                  f"Pausing until {resume_at.strftime('%H:%M:%S')}.")

    def _run_introspection(self):
        """Triggers an introspection report on a background cycle."""
        try:
            from app.introspection import Introspection
            # Use the most recent session from summaries
            import os
            from app.bootstrap import SUMMARIES_DIR
            files = [f for f in os.listdir(SUMMARIES_DIR) if f.endswith(".md")]
            if not files:
                return
            files.sort(
                key=lambda x: os.path.getmtime(os.path.join(SUMMARIES_DIR, x)),
                reverse=True
            )
            latest_session = files[0].replace(".md", "")
            introspection = Introspection(self.mm)
            report = introspection.generate_introspection_report(
                latest_session, flag_high_priority=True
            )
            if "[HIGH PRIORITY]" in report:
                print(f"[INNER_LOOP] ⚠️  Introspection flagged HIGH PRIORITY issues.")
            else:
                print(f"[INNER_LOOP] Introspection cycle complete — no critical issues.")
        except Exception as e:
            print(f"[INNER_LOOP] Introspection error (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Module-level singleton — imported by main.py and agent.py (/debug loop)
# ---------------------------------------------------------------------------

# Initialised to None here; set to a real InnerLoop instance in main.py
# after memory_manager is ready. This avoids circular imports.
inner_loop_instance: Optional[InnerLoop] = None
