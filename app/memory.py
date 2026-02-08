import sqlite3
import os
import json
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from app.config import (
    OPENROUTER_API_KEY, VECTOR_DB_PATH, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    SIMILARITY_THRESHOLD, TOP_K_ASSOCIATIVE, IDENTITY_UPDATE_INTERVAL
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
                    timestamp TEXT
                )
            """)
            conn.commit()
            
        # Initialize FAISS index
        try:
            if os.path.exists(VECTOR_INDEX_PATH):
                loaded_index = faiss.read_index(VECTOR_INDEX_PATH)
                # Check if dimension matches (important if we switched models)
                if loaded_index.d != self.dimension:
                    print(f"[PHASE3] Dimension mismatch (Index: {loaded_index.d}, Model: {self.dimension}). Resetting index.")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    # Also clear metadata DB if we reset the index
                    with sqlite3.connect(METADATA_DB_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM vector_metadata")
                        conn.commit()
                else:
                    self.index = loaded_index
                
                # Verify sync between SQLite and FAISS
                with sqlite3.connect(METADATA_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM vector_metadata")
                    db_count = cursor.fetchone()[0]
                    if db_count != self.index.ntotal:
                        print(f"[PHASE3] WARNING: Vector count mismatch (DB: {db_count}, FAISS: {self.index.ntotal}). Re-initializing.")
                        self.index = faiss.IndexFlatL2(self.dimension)
                        cursor.execute("DELETE FROM vector_metadata")
                        conn.commit()
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        except Exception as e:
            print(f"[PHASE3] Failed to load index: {e}. Starting fresh.")
            self.index = faiss.IndexFlatL2(self.dimension)

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
            
            vector_id = self.index.ntotal
            self.index.add(vector)
            faiss.write_index(self.index, VECTOR_INDEX_PATH)
            
            with sqlite3.connect(METADATA_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO vector_metadata (vector_id, session_id, type, content, salience, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (vector_id, session_id, mem_type, text, salience, datetime.now().isoformat())
                )
                conn.commit()
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
                        cursor.execute("SELECT type, content, salience FROM vector_metadata WHERE vector_id = ?", (int(idx),))
                        row = cursor.fetchone()
                        if row:
                            results.append({
                                "type": row[0],
                                "content": row[1],
                                "salience": row[2],
                                "similarity": float(similarity)
                            })
            
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

    def get_relevant_memory(self, session_id: str, user_input: str) -> Dict[str, str]:
        """Returns sections of the current session summary that are keyword-relevant."""
        summary_text = self.get_summary(session_id)
        if not summary_text: return {}
        sections = self.parse_summary_sections(summary_text)
        results = {}
        ui_lower = user_input.lower()
        if any(kw in ui_lower for kw in ["name", "who am i", "remember me", "identity"]) and sections["Known Facts"]:
            results["Known Facts"] = sections["Known Facts"]
        if any(kw in ui_lower for kw in ["preference", "like", "dislike", "favorite", "style", "prefer"]) and sections["Preferences"]:
            results["Preferences"] = sections["Preferences"]
        if any(kw in ui_lower for kw in ["continue", "what about", "status", "next", "todo", "progress", "unfinished"]) and sections["Open Threads"]:
            results["Open Threads"] = sections["Open Threads"]
        return results

    def get_associative_memory(self, user_input: str, skip_content: List[str] = None, return_raw: bool = False) -> any:
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

    def get_identity(self, user_input: str, return_raw: bool = False) -> any:
        if not os.path.exists(IDENTITY_FILE_PATH): 
            return {"content": "", "similarity": 0.0} if return_raw else ""
        
        with open(IDENTITY_FILE_PATH, "r", encoding="utf-8") as f:
            identity = f.read()
        
        # Identity-aware recall: only if semantic relevance is high
        embedding_ui = self._get_embedding(user_input)
        embedding_id = self._get_embedding(identity[:2000]) # Sample first 2k chars
        
        if all(v == 0.0 for v in embedding_ui) or all(v == 0.0 for v in embedding_id):
            return {"content": "", "similarity": 0.0} if return_raw else ""
            
        vec_ui = np.array([embedding_ui]).astype('float32')
        vec_id = np.array([embedding_id]).astype('float32')
        faiss.normalize_L2(vec_ui)
        faiss.normalize_L2(vec_id)
        
        sim = float(np.dot(vec_ui, vec_id.T)[0][0])
        
        if return_raw:
            return {"content": identity, "similarity": sim}

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

# Global instance
memory_manager = MemoryManager()
