import sqlite3
import os
import json
import numpy as np
import math
import faiss
from datetime import datetime
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from app.config import (
    OPENROUTER_API_KEY, VECTOR_DB_PATH, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    SIMILARITY_THRESHOLD, TOP_K_ASSOCIATIVE, IDENTITY_UPDATE_INTERVAL,
    DECAY_LAMBDA, REINFORCE_AMOUNT_ASSOCIATIVE, REINFORCE_AMOUNT_DETERMINISTIC,
    MAX_SALIENCE, MIN_SALIENCE, COMPRESSION_THRESHOLD, COMPRESSION_SALIENCE_BOOST,
    MAX_MERGES_PER_CYCLE, MIN_INDEX_SIZE_FOR_COMPRESSION
)

DB_PATH = "memory/agent_memory.db"
SUMMARIES_PATH = "memory/summaries"
VECTOR_INDEX_PATH = os.path.join(VECTOR_DB_PATH, "faiss.index")
METADATA_DB_PATH = os.path.join(VECTOR_DB_PATH, "metadata.db")
IDENTITY_FILE_PATH = "memory/identity.md"

class MemoryManager:
    def __init__(self):
        self._init_db()
        self.dimension = EMBEDDING_DIMENSION
        self.model = None # Lazy load on first use
        self._init_vector_db()
        
        if not os.path.exists(SUMMARIES_PATH):
            os.makedirs(SUMMARIES_PATH)

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
            conn.commit()

    def _init_vector_db(self):
        if not os.path.exists(VECTOR_DB_PATH):
            os.makedirs(VECTOR_DB_PATH)
            
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    vector_id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    type TEXT,
                    content TEXT,
                    salience REAL,
                    timestamp TEXT,
                    last_accessed_at TEXT,
                    merged INTEGER DEFAULT 0,
                    source_vector_ids TEXT,
                    scope TEXT DEFAULT 'session'
                )
            """)
            # Migration check
            cursor.execute("PRAGMA table_info(vector_metadata)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'last_accessed_at' not in columns:
                cursor.execute("ALTER TABLE vector_metadata ADD COLUMN last_accessed_at TEXT")
            if 'merged' not in columns:
                cursor.execute("ALTER TABLE vector_metadata ADD COLUMN merged INTEGER DEFAULT 0")
            if 'source_vector_ids' not in columns:
                cursor.execute("ALTER TABLE vector_metadata ADD COLUMN source_vector_ids TEXT")
            if 'scope' not in columns:
                cursor.execute("ALTER TABLE vector_metadata ADD COLUMN scope TEXT DEFAULT 'session'")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()
            
        # Initialize FAISS index
        try:
            if os.path.exists(VECTOR_INDEX_PATH):
                loaded_index = faiss.read_index(VECTOR_INDEX_PATH)
                # Check if dimension matches
                if loaded_index.d != self.dimension:
                    print(f"[PHASE3] Dimension mismatch. Resetting index.")
                    self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                    with sqlite3.connect(METADATA_DB_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM vector_metadata")
                        conn.commit()
                else:
                    if not isinstance(loaded_index, faiss.IndexIDMap):
                        print("[PHASE3] Converting IndexFlat to IndexIDMap for immutable ID support.")
                        # Create a new IndexIDMap and add existing vectors with their current sequential IDs
                        # This bridges the gap for existing contiguous indices.
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                        if loaded_index.ntotal > 0:
                            # Re-add all vectors from the loaded flat index
                            vectors = faiss.rev_swig_ptr(loaded_index.get_xb(), loaded_index.ntotal * loaded_index.d).reshape(loaded_index.ntotal, loaded_index.d)
                            ids = np.arange(loaded_index.ntotal).astype('int64')
                            self.index.add_with_ids(vectors, ids)
                    else:
                        self.index = loaded_index
            else:
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        except Exception as e:
            print(f"[PHASE3] Failed to load index: {e}. Starting fresh.")
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def _get_system_metadata(self, key: str, default: str = None) -> str:
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM system_metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default

    def _set_system_metadata(self, key: str, value: str):
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO system_metadata (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    def _get_model(self):
        if self.model is None:
            print(f"[PHASE3] Loading local embedding model: {EMBEDDING_MODEL} (this may take a minute)...")
            # Save model to memory volume to avoid re-downloads
            model_path = os.path.join("memory", "models")
            if not os.path.exists(model_path): os.makedirs(model_path)
            self.model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=model_path)
        return self.model

    def _get_embedding(self, text: str) -> List[float]:
        try:
            # Use local SentenceTransformer
            model = self._get_model()
            embedding = model.encode(text).tolist()
            return embedding
        except Exception as e:
            print(f"[PHASE3] Local embedding error: {e}")
            return [0.0] * self.dimension

    def add_vector(self, text: str, session_id: str, mem_type: str, salience: float = 0.5):
        if not text.strip(): return
        
        # Lightweight Deduplication: Check last 5 vectors of same type
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM vector_metadata WHERE type = ? ORDER BY vector_id DESC LIMIT 5",
                (mem_type,)
            )
            recent_contents = [row[0] for row in cursor.fetchall()]
            if text in recent_contents:
                # print(f"[MEMORY] Skipping duplicate vector insertion for type: {mem_type}")
                return

        embedding = self._get_embedding(text)
        if all(v == 0.0 for v in embedding):
            return
            
        try:
            vector = np.array([embedding]).astype('float32')
            faiss.normalize_L2(vector)
            
            with sqlite3.connect(METADATA_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO vector_metadata (session_id, type, content, salience, timestamp, scope) VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, mem_type, text, salience, datetime.now().isoformat(), 'session')
                )
                vector_id = cursor.lastrowid
                conn.commit()

            # Now add to FAISS with the permanent SQLite ID
            self.index.add_with_ids(vector, np.array([vector_id], dtype='int64'))
            faiss.write_index(self.index, VECTOR_INDEX_PATH)
            
            print(f"[PHASE3] Vector {vector_id} added: Type={mem_type}, Salience={salience}")
        except Exception as e:
            print(f"[PHASE3] Error adding vector: {e}")

    def query_associative(self, query_text: str, top_k: int = TOP_K_ASSOCIATIVE) -> List[Dict]:
        if not query_text.strip(): return []
        
        embedding = self._get_embedding(query_text)
        if all(v == 0.0 for v in embedding):
            return []
            
        try:
            vector = np.array([embedding]).astype('float32')
            faiss.normalize_L2(vector)
            
            distances, indices = self.index.search(vector, top_k)
            
            results = []
            with sqlite3.connect(METADATA_DB_PATH) as conn:
                cursor = conn.cursor()
                for i, idx in enumerate(indices[0]):
                    if idx == -1: continue
                    # normalized L2 distance to similarity
                    similarity = 1 - (distances[0][i] / 2)
                    if similarity >= SIMILARITY_THRESHOLD:
                        # Fix 4: Filter out merged=1 to prevent recall before/during rebuild
                        cursor.execute("SELECT type, content, salience FROM vector_metadata WHERE vector_id = ? AND merged = 0", (int(idx),))
                        row = cursor.fetchone()
                        if row:
                            # Fix 3: Update last_accessed_at for retrieved vectors
                            now_str = datetime.now().isoformat()
                            cursor.execute("UPDATE vector_metadata SET last_accessed_at = ? WHERE vector_id = ?", (now_str, int(idx)))
                            
                            results.append({
                                "vector_id": int(idx),
                                "type": row[0],
                                "content": row[1],
                                "salience": row[2],
                                "similarity": float(similarity)
                            })
                conn.commit()
            
            # Sort by salience then similarity
            results.sort(key=lambda x: (x['salience'], x['similarity']), reverse=True)
            return results
        except Exception as e:
            print(f"[PHASE3] Query error: {e}")
            return []

    def add_message(self, session_id: str, role: str, content: str):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def get_message_count(self, session_id: str) -> int:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
            return cursor.fetchone()[0]

    def get_summary(self, session_id: str) -> str:
        summary_file = os.path.join(SUMMARIES_PATH, f"{session_id}.md")
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def get_messages_since_last_summary(self, session_id: str, limit: int) -> List[Dict[str, str]]:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def write_session_summary(self, session_id: str, markdown: str):
        summary_file = os.path.join(SUMMARIES_PATH, f"{session_id}.md")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        # After writing, index sections for Phase 3
        sections = self.parse_summary_sections(markdown)
        for sec_type, content in sections.items():
            if content:
                # Salience logic: Open Threads get higher base salience
                # Also, check if this content already exists to avoid redundant vectors
                # (Simple check: last 5 vectors for this type)
                salience = 0.9 if sec_type == "Open Threads" else 0.5
                self.add_vector(content, session_id, sec_type, salience)
        
        # Increment summary update count for Phase 5 identity triggering
        count = int(self._get_system_metadata("summary_update_count", "0"))
        self._set_system_metadata("summary_update_count", str(count + 1))

    def get_summary_update_count(self) -> int:
        return int(self._get_system_metadata("summary_update_count", "0"))

    def get_knowledge(self) -> str:
        knowledge_file = "memory/knowledge.md"
        if os.path.exists(knowledge_file):
            with open(knowledge_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def parse_summary_sections(self, summary_text: str) -> Dict[str, str]:
        sections = {"Known Facts": "", "Preferences": "", "Open Threads": ""}
        current_section = None
        for line in summary_text.split("\n"):
            if "## Known Facts" in line: current_section = "Known Facts"
            elif "## Preferences" in line: current_section = "Preferences"
            elif "## Open Threads" in line: current_section = "Open Threads"
            elif "## Last Updated" in line: current_section = None
            elif current_section and line.strip():
                if not (line.strip().startswith("- <") and line.strip().endswith(">")):
                    sections[current_section] += line + "\n"
        return {k: v.strip() for k, v in sections.items()}

    def get_relevant_memory(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """Returns sections of the current session summary that are keyword-relevant and their vector IDs."""
        summary_text = self.get_summary(session_id)
        if not summary_text: return {"sections": {}, "vector_ids": []}
        sections = self.parse_summary_sections(summary_text)
        results = {}
        ui_lower = user_input.lower()
        # Identity/Origin Keywords
        id_keywords = ["name", "who am i", "remember me", "identity", "creator", "develop", "built", "origin", "who are you"]
        
        recalled_types = []
        if any(kw in ui_lower for kw in id_keywords) and sections["Known Facts"]:
            results["Known Facts"] = sections["Known Facts"]
            recalled_types.append("Known Facts")
        if any(kw in ui_lower for kw in ["preference", "like", "dislike", "favorite", "style", "prefer"]) and sections["Preferences"]:
            results["Preferences"] = sections["Preferences"]
            recalled_types.append("Preferences")
        if any(kw in ui_lower for kw in ["continue", "what about", "status", "next", "todo", "progress", "unfinished"]) and sections["Open Threads"]:
            results["Open Threads"] = sections["Open Threads"]
            recalled_types.append("Open Threads")
            
        # Fix 3: Deterministic recall only uses scope='session'
        vector_ids = self.get_vector_ids_for_session(session_id, recalled_types, scope='session')
        return {"sections": results, "vector_ids": vector_ids}

    def get_vector_ids_for_session(self, session_id: str, types: List[str], scope: str = None) -> List[int]:
        """Returns all vector IDs for the given types in a specific session."""
        if not types: return []
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            placeholders = ', '.join(['?'] * len(types))
            
            clause = "session_id = ? AND type IN ({})"
            params = [session_id] + types
            
            if scope:
                clause += " AND scope = ?"
                params.append(scope)
                
            query = f"SELECT vector_id FROM vector_metadata WHERE {clause.format(placeholders)}"
            cursor.execute(query, params)
            return [row[0] for row in cursor.fetchall()]

    def get_associative_memory(self, user_input: str, skip_content: List[str] = None, return_raw: bool = False) -> Any:
        results = self.query_associative(user_input)
        if not results:
            print("[PHASE3] Associative search: No results above threshold.")
            return [] if return_raw else ""
            
        skip_content = skip_content or []
        count = 0
        output_results = []
        for res in results:
            # Avoid direct duplicates from Phase 2
            is_duplicate = any(res['content'] in skip for skip in skip_content)
            if not is_duplicate:
                output_results.append(res)
                count += 1
        
        if return_raw:
            return output_results

        if count == 0:
            return ""
            
        print(f"[PHASE3] Associative search: {count} matches.")
        output = ["[PHASE3] ASSOCIATIVE MEMORIES:"]
        for res in output_results:
            output.append(f"[{res['type']}] (Salience: {res['salience']}): {res['content']}")
        return "\n".join(output)

    def get_identity(self, user_input: str, return_raw: bool = False) -> Any:
        if not os.path.exists(IDENTITY_FILE_PATH): 
            return {"content": "", "similarity": 0.0} if return_raw else ""
        
        with open(IDENTITY_FILE_PATH, "r", encoding="utf-8") as f:
            identity = f.read()
        
        # Identity-aware recall: only if semantic relevance is high
        embedding_ui = self._get_embedding(user_input)
        embedding_id = self._get_embedding(identity[:2000]) # Sample first 2k chars
        
        if all(v == 0.0 for v in embedding_ui) or all(v == 0.0 for v in embedding_id):
            return {"content": "", "similarity": 0.0, "vector_ids": []} if return_raw else ""
            
        vec_ui = np.array([embedding_ui]).astype('float32')
        vec_id = np.array([embedding_id]).astype('float32')
        faiss.normalize_L2(vec_ui)
        faiss.normalize_L2(vec_id)
        
        sim = float(np.dot(vec_ui, vec_id.T)[0][0])
        
        if return_raw:
            return {"content": identity, "similarity": sim, "vector_ids": []} # Identity file has no direct vectors yet

        if sim >= SIMILARITY_THRESHOLD:
            print(f"[PHASE3] Identity injection triggered (Sim: {sim:.2f})")
            return f"IDENTITY (Self-Model):\n{identity}"
        
        print(f"[PHASE3] Identity skipped (Sim: {sim:.2f})")
        return ""

    def update_identity(self):
        """Merges patterns from multiple summaries into the identity.md."""
        print("[PHASE3] Updating Identity self-model...")
        all_summaries = []
        try:
            for filename in os.listdir(SUMMARIES_PATH):
                if filename.endswith(".md"):
                    with open(os.path.join(SUMMARIES_PATH, filename), "r", encoding="utf-8") as f:
                        all_summaries.append(f.read())
        except Exception as e:
            print(f"[PHASE3] Error reading summaries for identity: {e}")
            return
        
        if not all_summaries: return

        # We'll use the LLM to merge these patterns
        from app.openrouter import run_openrouter
        merge_prompt = f"""
        Analyze the following session summaries and update the global identity/self-model of the user.
        Identify consistent behavioral patterns, recurring preferences, and core project themes.
        
        SUMMARIES TO MERGE:
        {chr(10).join(all_summaries)}
        
        MANDATORY OUTPUT STRUCTURE:
        # Identity: User Self-Model
        ## Behavioral Patterns
        - <recurring behaviors>
        ## Long-Term Interests
        - <stable interests>
        ## Unified Preferences
        - <cross-session preferences>
        """
        new_identity = run_openrouter(merge_prompt)
        if not new_identity.startswith("Error"):
            with open(IDENTITY_FILE_PATH, "w", encoding="utf-8") as f:
                f.write(new_identity)
            print("[PHASE3] Identity updated successfully.")
        else:
            print(f"[PHASE3] Identity update failed: {new_identity}")

    def detect_drift(self, session_id: str):
        """Logged-only drift detection for consistency monitoring."""
        print(f"[PHASE3.5] Scanning for drift in session {session_id}...")
        
        summary = self.get_summary(session_id)
        if not summary: return
        
        sections = self.parse_summary_sections(summary)
        
        # 1. Contradiction Check (Simple keyword search against identity)
        if os.path.exists(IDENTITY_FILE_PATH):
            with open(IDENTITY_FILE_PATH, "r", encoding="utf-8") as f:
                identity = f.read().lower()
            for section, content in sections.items():
                if "no" in content.lower() and "yes" in identity:
                    # Very primitive example of detection
                    pass
        
        # 2. Long-lived Open Threads
        if sections["Open Threads"]:
            thread_count = sections["Open Threads"].count("-")
            if thread_count > 5:
                print(f"[PHASE3.5] WARNING: High thread count ({thread_count}). Possible drift in resolution logic.")
        
        # 3. Repeated Content Detection (Check metadata DB)
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, COUNT(*) as c FROM vector_metadata GROUP BY content HAVING c > 1")
            repeats = cursor.fetchall()
            for r in repeats:
                print(f"[PHASE3.5] WARNING: Redundant memory detected ({r[1]} instances): {r[0][:50]}...")

    def decay_salience(self):
        """Implements time-based salience decay for associative memory using delta time."""
        print("[PHASE5] Starting salience decay process...")
        now = datetime.now()
        
        # Get last decay run
        last_run_str = self._get_system_metadata("last_decay_run")
        last_run = datetime.fromisoformat(last_run_str) if last_run_str else None
        
        # Decay Multipliers per Requirement
        multipliers = {
            "Open Threads": 0.5,
            "Preferences": 1.0,
            "Known Facts": 1.2,
            "Tool History": 1.5
        }
        
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            # Fetch vectors that are not protected (salience < 0.95)
            cursor.execute("SELECT vector_id, type, salience, timestamp, last_accessed_at FROM vector_metadata WHERE salience < 0.95")
            records = cursor.fetchall()
            
            decay_count = 0
            for vid, mtype, salience, ts_str, last_acc_str in records:
                if not ts_str: continue
                
                try:
                    # Issue 5: Skip decay for vectors reinforced since last decay run
                    if last_acc_str and last_run:
                        last_acc = datetime.fromisoformat(last_acc_str)
                        if last_acc > last_run:
                            print(f"[PHASE5] Skipping decay for recently accessed vector {vid}")
                            continue

                    created_at = datetime.fromisoformat(ts_str)
                    last_accessed = datetime.fromisoformat(last_acc_str) if last_acc_str else None
                    
                    # Issue 4: Decay reference time = max(created_at, last_accessed_at, last_decay_run)
                    reference_time = max(created_at, last_run) if last_run else created_at
                    if last_accessed:
                        reference_time = max(reference_time, last_accessed)
                        
                    delta_days = (now - reference_time).total_seconds() / 86400.0
                    
                    if delta_days <= 0: continue
                    
                    multiplier = multipliers.get(mtype, 1.0)
                    effective_lambda = DECAY_LAMBDA * multiplier
                    
                    # Formula: decayed_salience = current_salience * exp(-λ * delta_days)
                    new_salience = salience * math.exp(-effective_lambda * delta_days)
                    
                    # Floor check
                    if new_salience < 0.05:
                        new_salience = 0.05
                    
                    new_salience = round(new_salience, 4)
                    
                    if new_salience != salience:
                        cursor.execute("UPDATE vector_metadata SET salience = ? WHERE vector_id = ?", (new_salience, vid))
                        print(f"[PHASE5] Decayed vector {vid} ({mtype}): {salience} -> {new_salience} (delta: {delta_days:.4f} days)")
                        decay_count += 1
                        
                except (ValueError, TypeError):
                    continue
            
            conn.commit()
            
        # Update last run timestamp
        self._set_system_metadata("last_decay_run", now.isoformat())
        if decay_count > 0:
            print(f"[PHASE5] Salience decay complete. Updated {decay_count} records.")
        else:
            print("[PHASE5] Salience decay complete. No changes needed.")

    def reinforce_memory(self, vector_id: int, amount: float, source: str = "generic"):
        """Gradually increases the salience of a memory, countering decay."""
        try:
            with sqlite3.connect(METADATA_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT salience FROM vector_metadata WHERE vector_id = ?", (vector_id,))
                row = cursor.fetchone()
                if not row: return

                current_salience = row[0]
                new_salience = min(current_salience + amount, MAX_SALIENCE)
                new_salience = round(new_salience, 4)

                # Issue 4: Update last_accessed_at
                now_str = datetime.now().isoformat()
                cursor.execute("UPDATE vector_metadata SET salience = ?, last_accessed_at = ? WHERE vector_id = ?", (new_salience, now_str, vector_id))
                conn.commit()
                print(f"[PHASE5] Reinforced vector {vector_id} ({source}) (+{amount} -> {new_salience})")
        except Exception as e:
            print(f"[PHASE5] Reinforcement error: {e}")

    def rebuild_index(self):
        """Rebuilds the FAISS index while preserving immutable vector_ids."""
        print("[PHASE5.3] Rebuilding FAISS index (preserving permanent IDs)...")
        # Use a fresh IndexIDMap to allow gaps in IDs
        new_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            # Fix 1: Order by ID but do NOT reassign them
            cursor.execute("SELECT content, vector_id FROM vector_metadata WHERE merged = 0 ORDER BY vector_id ASC")
            records = cursor.fetchall()
            
            if not records:
                self.index = new_index
                faiss.write_index(self.index, VECTOR_INDEX_PATH)
                return

            # Batch process embeddings
            texts = [r[0] for r in records]
            model = self._get_model()
            embeddings = model.encode(texts)
            
            vectors = np.array(embeddings).astype('float32')
            faiss.normalize_L2(vectors)
            
            # Use original IDs from SQLite
            ids = np.array([r[1] for r in records]).astype('int64')
            new_index.add_with_ids(vectors, ids)
            
            # Delete old merged rows
            cursor.execute("DELETE FROM vector_metadata WHERE merged = 1")
            conn.commit()
            
        self.index = new_index
        faiss.write_index(self.index, VECTOR_INDEX_PATH)
        print(f"[PHASE5.3] Index rebuilt with {self.index.ntotal} vectors. All IDs preserved.")

    def compress_and_merge_memory(self):
        """Identifies and merges redundant or similar memories with structural hardening."""
        # Fix 1: Compression Re-entrancy Guard
        if self._get_system_metadata("compression_in_progress") == "1":
            print("[PHASE5.3] Guard: Compression already in progress. Aborting.")
            return

        try:
            self._set_system_metadata("compression_in_progress", "1")
            print("[PHASE5.3] Acquired compression lock.")

            # Guard: Abort compression if index size < threshold
            if self.index.ntotal < MIN_INDEX_SIZE_FOR_COMPRESSION:
                print(f"[PHASE5.3] Index size ({self.index.ntotal}) below threshold ({MIN_INDEX_SIZE_FOR_COMPRESSION}).")
                return

            print("[PHASE5.3] Starting hardened memory compression cycle...")
            
            with sqlite3.connect(METADATA_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT vector_id, content, salience, type FROM vector_metadata 
                    WHERE salience >= 0.3 AND salience < 0.95 
                    AND merged = 0 
                    AND scope = 'session'
                    AND type != 'Merged Memory'
                    AND type != 'Identity'
                    AND NOT (type = 'Open Threads' AND salience >= 0.8)
                """)
                candidates = cursor.fetchall()
                
            if len(candidates) < 2:
                print("[PHASE5.3] Insufficient candidates for compression.")
                return

            processed_ids = set()
            merged_count = 0
            new_merged_ids = []
            
            from app.openrouter import run_openrouter

            for vid, content, salience, mtype in candidates:
                if vid in processed_ids: continue
                if merged_count >= MAX_MERGES_PER_CYCLE: break
                
                results = self.query_associative(content, top_k=5)
                
                sim_neighbors = []
                for r in results:
                    n_vid = r['vector_id']
                    if n_vid == vid or n_vid in processed_ids: continue
                    if r['similarity'] < COMPRESSION_THRESHOLD: continue
                    if r['type'] != mtype: continue 
                    
                    with sqlite3.connect(METADATA_DB_PATH) as conn:
                        c2 = conn.cursor()
                        c2.execute("SELECT salience FROM vector_metadata WHERE vector_id = ? AND merged = 0 AND scope = 'session'", (n_vid,))
                        n_row = c2.fetchone()
                        if not n_row: continue
                        n_sal = n_row[0]
                        if mtype == 'Open Threads' and n_sal >= 0.8: continue
                        if n_sal < 0.3: continue 
                        
                    sim_neighbors.append(r)
                
                if len(sim_neighbors) < 1: continue 
                
                group = [(vid, content, salience)] + [(n['vector_id'], n['content'], n['salience']) for n in sim_neighbors]
                group_ids = [item[0] for item in group]
                group_texts = [item[1] for item in group]
                max_salience = max(item[2] for item in group)
                
                print(f"[PHASE5.3][MERGE] type={mtype} ids={group_ids} start_salience={salience}")
                
                merge_prompt = f"""
                The following memories are semantically redundant. 
                Merge them into a single, concise, and high-density memory statement.
                Preserve all unique facts, preferences, or technical details.
                
                MEMORIES TO MERGE:
                {chr(10).join([f"- {t}" for t in group_texts])}
                
                Merged Memory (Single Paragraph):
                """
                merged_text = run_openrouter(merge_prompt).strip()
                
                if merged_text and "Error" not in merged_text:
                    with sqlite3.connect(METADATA_DB_PATH) as conn:
                        cursor = conn.cursor()
                        placeholders = ', '.join(['?'] * len(group_ids))
                        cursor.execute(f"UPDATE vector_metadata SET merged = 1 WHERE vector_id IN ({placeholders})", group_ids)
                        
                        new_salience = min(max_salience + COMPRESSION_SALIENCE_BOOST, MAX_SALIENCE)
                        now_str = datetime.now().isoformat()
                        
                        # Fix 2 (Option A): Insert with merged = 1 until rebuild is complete
                        cursor.execute(
                            "INSERT INTO vector_metadata (session_id, type, content, salience, timestamp, last_accessed_at, merged, source_vector_ids, scope) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            ("global_merge", "Merged Memory", merged_text, new_salience, now_str, now_str, 1, json.dumps(group_ids), 'global')
                        )
                        new_id = cursor.lastrowid
                        new_merged_ids.append((new_id, merged_text))
                        conn.commit()
                    
                    processed_ids.update(group_ids)
                    merged_count += 1
                    print(f"[PHASE5.3][MERGE] SUCCESS -> new_id={new_id} preview='{merged_text[:60]}...'")

            if merged_count > 0:
                print(f"[PHASE5.3] Groups merged. Rebuilding index to purge old vectors...")
                self.rebuild_index()
                
                # Fix 2: Flip merged = 0 and add to FAISS after rebuild is complete
                with sqlite3.connect(METADATA_DB_PATH) as conn:
                    cursor = conn.cursor()
                    for new_id, text in new_merged_ids:
                        cursor.execute("UPDATE vector_metadata SET merged = 0 WHERE vector_id = ?", (new_id,))
                        # Embed and add to FAISS manually
                        emb = self._get_embedding(text)
                        vec = np.array([emb]).astype('float32')
                        faiss.normalize_L2(vec)
                        self.index.add_with_ids(vec, np.array([new_id], dtype='int64'))
                    conn.commit()
                faiss.write_index(self.index, VECTOR_INDEX_PATH)
                print(f"[PHASE5.3] Multi-merge complete. {merged_count} new memories active.")
            else:
                print("[PHASE5.3] Cycle finished with no merges.")

        except Exception as e:
            print(f"[PHASE5.3] Compression error: {e}")
        finally:
            self._set_system_metadata("compression_in_progress", "0")
            print("[PHASE5.3] Released compression lock.")

# Global instance
memory_manager = MemoryManager()
