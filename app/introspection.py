import os
import re
import datetime
from typing import Dict, List, Tuple
from app.memory import MemoryManager, IDENTITY_FILE_PATH, SUMMARIES_PATH

from app.bootstrap import REPORTS_DIR as REPORTS_PATH

class Introspection:
    def __init__(self, memory_manager: MemoryManager):
        self.mm = memory_manager

    def _get_sorted_session_files(self) -> List[str]:
        """Returns a list of session summary filenames sorted by modification time (descending)."""
        if not os.path.exists(SUMMARIES_PATH):
            return []
        files = [f for f in os.listdir(SUMMARIES_PATH) if f.endswith(".md")]
        # Sort by modification time, newest first
        files.sort(key=lambda x: os.path.getmtime(os.path.join(SUMMARIES_PATH, x)), reverse=True)
        return files

    def detect_semantic_contradictions(self, session_id: str) -> Dict[str, List[str]]:
        """Detects contradictions between Identity/Known Facts/Preferences."""
        contradictions = {
            "Identity_KnownFacts": [],
            "Preferences": [],
            "OpenThreads": []
        }
        
        # 1. Load Data
        summary_text = self.mm.get_summary(session_id)
        if not summary_text:
            return contradictions
            
        sections = self.mm.parse_summary_sections(summary_text)
        
        identity_text = ""
        if os.path.exists(IDENTITY_FILE_PATH):
            with open(IDENTITY_FILE_PATH, "r", encoding="utf-8") as f:
                identity_text = f.read().lower()
                
        known_facts = sections.get("Known Facts", "").lower()
        preferences = sections.get("Preferences", "").lower()
        open_threads = sections.get("Open Threads", "")
        
        # 2. Identity vs Known Facts (Heuristic: Negation overlaps)
        key_states = [
            (r"\bis\b", r"\bis not\b|\bisn't\b"),
            (r"\bcan\b", r"\bcannot\b|\bcan't\b"),
            (r"\bhas\b", r"\bdoes not have\b|\bdoesn't have\b|\bhas no\b"),
            (r"\benabled\b", r"\bdisabled\b"),
            (r"\bactive\b", r"\binactive\b")
        ]
        
        fact_lines = [f.strip("- ").strip() for f in known_facts.split("\n") if f.strip("- ")]
        for fact in fact_lines:
            for positive, negative in key_states:
                words = set(re.findall(r"\b\w{4,}\b", fact)) 
                for word in words:
                    if word in identity_text:
                        if (re.search(negative, fact) and re.search(positive, identity_text)) or \
                           (re.search(positive, fact) and re.search(negative, identity_text)):
                            contradictions["Identity_KnownFacts"].append(f"Conflict on '{word}': Fact='{fact}' vs Identity")
            
        # 3. Preferences Inconsistencies (Within Session)
        likes_patterns = [r"prefers?\b", r"likes?\b", r"loves?\b", r"enjoys?\b"]
        dislikes_patterns = [r"dislikes?\b", r"hates?\b", r"avoids?\b", r"not like\b"]
        
        def extract_prefs(text, patterns):
            results = []
            for pat in patterns:
                matches = re.findall(pat + r"\s+([\w\s]{2,50})", text, re.I)
                results.extend([m.strip() for m in matches])
            return list(set(results))

        curr_likes = extract_prefs(preferences, likes_patterns)
        curr_dislikes = extract_prefs(preferences, dislikes_patterns)
        
        for like in curr_likes:
            for dislike in curr_dislikes:
                if like in dislike or dislike in like:
                    contradictions["Preferences"].append(f"In-Session Conflict: Likes '{like}' but Dislikes '{dislike}'")

        # 4. Cross-Session Preferences Contradiction
        all_files = self._get_sorted_session_files()
        try:
            curr_idx = all_files.index(f"{session_id}.md")
            past_files = all_files[curr_idx+1 : curr_idx+6]
        except ValueError:
            past_files = all_files[:5]

        for p_file in past_files:
            p_path = os.path.join(SUMMARIES_PATH, p_file)
            with open(p_path, "r", encoding="utf-8") as f:
                p_sections = self.mm.parse_summary_sections(f.read())
            p_pref_text = p_sections.get("Preferences", "").lower()
            p_likes = extract_prefs(p_pref_text, likes_patterns)
            p_dislikes = extract_prefs(p_pref_text, dislikes_patterns)
            
            # Current Likes vs Past Dislikes
            for like in curr_likes:
                for p_dis in p_dislikes:
                    if like in p_dis or p_dis in like:
                        contradictions["Preferences"].append(f"Cross-Session Conflict ({p_file}): Now likes '{like}' but previously disliked it.")
            
            # Current Dislikes vs Past Likes
            for dislike in curr_dislikes:
                for p_like in p_likes:
                    if dislike in p_like or p_like in dislike:
                        contradictions["Preferences"].append(f"Cross-Session Conflict ({p_file}): Now dislikes '{dislike}' but previously liked it.")

        # 5. Open Threads - Stalled Threads detection
        thread_lines = [line.strip("- ").strip() for line in open_threads.split('\n') if line.strip() and not line.strip().startswith('##')]
        if len(thread_lines) > 5:
             contradictions["OpenThreads"].append(f"High thread count ({len(thread_lines)})")

        stalled = {}
        for p_file in past_files:
            p_path = os.path.join(SUMMARIES_PATH, p_file)
            with open(p_path, "r", encoding="utf-8") as f:
                p_sections = self.mm.parse_summary_sections(f.read())
            p_threads = [line.strip("- ").strip() for line in p_sections.get("Open Threads", "").split('\n') if line.strip() and not line.strip().startswith('##')]
            for t in thread_lines:
                if t in p_threads:
                    stalled[t] = stalled.get(t, 0) + 1
        
        for t, count in stalled.items():
            if count >= 3:
                contradictions["OpenThreads"].append(f"Stalled Task: '{t[:40]}...' persists across {count} sessions.")
        
        return contradictions

    def detect_behavioral_drift(self, session_id: str, lookback: int = 5) -> Dict[str, str]:
        """Tracks trends/shifts by comparing current session with previous ones."""
        drift_report = {}
        
        current_summary = self.mm.get_summary(session_id)
        if not current_summary:
            return {}
            
        current_sections = self.mm.parse_summary_sections(current_summary)
        current_prefs = set(current_sections.get("Preferences", "").lower().split('\n'))
        
        # Get past sessions
        all_files = self._get_sorted_session_files()
        # Find current session index
        try:
            curr_idx = all_files.index(f"{session_id}.md")
            past_files = all_files[curr_idx+1 : curr_idx+1+lookback]
        except ValueError:
            # Session might be new and not written to file yet, or file name mismatch
            # If not found, use top N
            past_files = all_files[:lookback]

        if not past_files:
            return {"Status": "No history for drift analysis."}

        for p_file in past_files:
            p_path = os.path.join(SUMMARIES_PATH, p_file)
            with open(p_path, "r", encoding="utf-8") as f:
                p_text = f.read()
            
            p_sections = self.mm.parse_summary_sections(p_text)
            p_prefs = set(p_sections.get("Preferences", "").lower().split('\n'))
            
            # Check for dropped preferences
            dropped = p_prefs - current_prefs
            # Check for new preferences
            new = current_prefs - p_prefs
            
            # Simple heuristic: If overlap is low (< 20%), flag major drift
            if p_prefs and current_prefs:
                overlap = len(p_prefs.intersection(current_prefs))
                score = overlap / len(p_prefs)
                if score < 0.2:
                    drift_report[p_file] = f"Major preference shift (Overlap: {score:.2f}). Dropped: {len(dropped)}, New: {len(new)}"
            elif p_prefs and not current_prefs:
                 drift_report[p_file] = "All past preferences dropped in current session."

        return drift_report

    def generate_introspection_report(self, session_id: str, flag_high_priority: bool = True) -> str:
        """Combines contradictions and drift into a readable report."""
        contradictions = self.detect_semantic_contradictions(session_id)
        drift = self.detect_behavioral_drift(session_id)
        
        report = []
        report.append("### PHASE 7: INTROSPECTION REPORT ###\n")
        
        report.append("#### Contradictions Detected ####")
        has_issues = False
        
        # Identity-KnownFacts
        if contradictions["Identity_KnownFacts"]:
            has_issues = True
            prefix = "[HIGH PRIORITY] " if flag_high_priority else ""
            report.append(f"{prefix}Identity_KnownFacts: {'; '.join(contradictions['Identity_KnownFacts'])}")
        else:
            report.append("Identity_KnownFacts: None detected")
            
        # Preferences
        if contradictions["Preferences"]:
            has_issues = True
            report.append(f"Preferences: {'; '.join(contradictions['Preferences'])}")
        else:
            report.append("Preferences: None detected")
            
        # OpenThreads
        if contradictions["OpenThreads"]:
            has_issues = True
            report.append(f"OpenThreads: {'; '.join(contradictions['OpenThreads'])}")
        else:
            report.append("OpenThreads: Normal")
            
        report.append("\n#### Behavioral Drift ####")
        if drift:
            for source, msg in drift.items():
                report.append(f"- Compared to {source}: {msg}")
        else:
            report.append("- No significant drift detected.")
            
        report.append("\n#####################################")
        
        report_text = "\n".join(report)
        
        # Save to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{session_id}_{timestamp}.md"
        report_path = os.path.join(REPORTS_PATH, report_filename)
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"[INTROSPECTION] Report saved to: {report_path}")
        except Exception as e:
            print(f"[INTROSPECTION] Failed to save report: {e}")
            
        return report_text
