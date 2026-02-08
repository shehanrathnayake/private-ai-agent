import sqlite3
import os
import json
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI
from app.config import (
    OPENROUTER_API_KEY, VECTOR_DB_PATH, EMBEDDING_MODEL, 
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
        self._init_vector_db()
        if not os.path.exists(SUMMARIES_PATH):
            os.makedirs(SUMMARIES_PATH)
        
        # Initialize OpenAI client for embeddings
        self.client = OpenAI(api_key=OPENROUTER_API_KEY) # Using same key for convenience

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
        # text-embedding-3-small dimension is 1536
        self.dimension = 1536
        if os.path.exists(VECTOR_INDEX_PATH):
            self.index = faiss.read_index(VECTOR_INDEX_PATH)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def _get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[PHASE3] Embedding error: {e}")
            return [0.0] * self.dimension

    def add_vector(self, text: str, session_id: str, mem_type: str, salience: float = 0.5):
        embedding = self._get_embedding(text)
        if all(v == 0.0 for v in embedding):
            return
            
        vector = np.array([embedding]).astype('float32')
        faiss.normalize_L2(vector) # Use cosine similarity via normalized L2
        
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
        print(f"[PHASE3] Vector added: Type={mem_type}, Salience={salience}")

    def query_associative(self, query_text: str, top_k: int = TOP_K_ASSOCIATIVE) -> List[Dict]:
        embedding = self._get_embedding(query_text)
        if all(v == 0.0 for v in embedding):
            return []
            
        vector = np.array([embedding]).astype('float32')
        faiss.normalize_L2(vector)
        
        distances, indices = self.index.search(vector, top_k)
        
        results = []
        with sqlite3.connect(METADATA_DB_PATH) as conn:
            cursor = conn.cursor()
            for i, idx in enumerate(indices[0]):
                if idx == -1: continue
                # In normalized L2, distance = 2 * (1 - cosine_similarity)
                similarity = 1 - (distances[0][i] / 2)
                if similarity >= SIMILARITY_THRESHOLD:
                    cursor.execute("SELECT type, content, salience FROM vector_metadata WHERE vector_id = ?", (int(idx),))
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            "type": row[0],
                            "content": row[1],
                            "salience": row[2],
                            "similarity": similarity
                        })
        return results

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
                salience = 0.8 if sec_type == "Open Threads" else 0.5
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

    def get_relevant_memory(self, session_id: str, user_input: str) -> str:
        summary_text = self.get_summary(session_id)
        if not summary_text: return ""
        sections = self.parse_summary_sections(summary_text)
        relevant_parts = []
        ui_lower = user_input.lower()
        if any(kw in ui_lower for kw in ["name", "who am i", "remember me", "identity"]) and sections["Known Facts"]:
            relevant_parts.append(f"RECALLED KNOWN FACTS:\n{sections['Known Facts']}")
        if any(kw in ui_lower for kw in ["preference", "like", "dislike", "favorite", "style", "prefer"]) and sections["Preferences"]:
            relevant_parts.append(f"RECALLED PREFERENCES:\n{sections['Preferences']}")
        if any(kw in ui_lower for kw in ["continue", "what about", "status", "next", "todo", "progress", "unfinished"]) and sections["Open Threads"]:
            relevant_parts.append(f"RECALLED OPEN THREADS:\n{sections['Open Threads']}")
        return "\n\n".join(relevant_parts)

    def get_associative_memory(self, user_input: str) -> str:
        results = self.query_associative(user_input)
        if not results: return ""
        output = ["[PHASE3] ASSOCIATIVE MEMORIES:"]
        for res in results:
            output.append(f"[{res['type']}] (Salience: {res['salience']}): {res['content']}")
        return "\n".join(output)

    def get_identity(self, user_input: str) -> str:
        if not os.path.exists(IDENTITY_FILE_PATH): return ""
        with open(IDENTITY_FILE_PATH, "r") as f:
            identity = f.read()
        
        # Identity-aware recall: only if semantic relevance is high
        embedding_ui = self._get_embedding(user_input)
        embedding_id = self._get_embedding(identity[:2000]) # Sample first 2k chars
        
        vec_ui = np.array([embedding_ui]).astype('float32')
        vec_id = np.array([embedding_id]).astype('float32')
        faiss.normalize_L2(vec_ui)
        faiss.normalize_L2(vec_id)
        
        sim = np.dot(vec_ui, vec_id.T)[0][0]
        if sim >= SIMILARITY_THRESHOLD:
            print(f"[PHASE3] Identity injection triggered (Sim: {sim:.2f})")
            return f"IDENTITY (Self-Model):\n{identity}"
        return ""

    def update_identity(self):
        """Merges patterns from multiple summaries into the identity.md."""
        print("[PHASE3] Updating Identity self-model...")
        all_summaries = []
        for filename in os.listdir(SUMMARIES_PATH):
            if filename.endswith(".md"):
                with open(os.path.join(SUMMARIES_PATH, filename), "r") as f:
                    all_summaries.append(f.read())
        
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
            with open(IDENTITY_FILE_PATH, "w") as f:
                f.write(new_identity)
            print("[PHASE3] Identity updated successfully.")

# Global instance
memory_manager = MemoryManager()
