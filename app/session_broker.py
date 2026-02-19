import os
import sqlite3
import datetime
import numpy as np
import faiss
from typing import Optional
from app.memory import memory_manager, METADATA_DB_PATH, DB_PATH
from app.config import SESSION_AUTO_IDLE_TIMEOUT, SESSION_TOPIC_SHIFT_THRESHOLD

class SessionBroker:
    """
    Automatically manages session IDs based on time gaps and topic shifts.
    Ensures the user doesn't have to manually provide a session ID.
    """

    def resolve_session_id(self, user_input: str, provided_id: Optional[str] = None) -> str:
        """
        Determines which session ID to use. 
        If provided_id is 'auto' or None, it uses internal logic.
        """
        if provided_id and provided_id != "auto":
            self._set_current_session(provided_id)
            return provided_id

        current_id = self._get_current_session()
        
        # 1. New Interaction (No current session)
        if not current_id:
            new_id = self._generate_session_id()
            self._set_current_session(new_id)
            print(f"[SESSION] Initializing first session: {new_id}")
            return new_id

        # 2. Temporal Check (Time Gap)
        last_ts = self._get_last_message_timestamp(current_id)
        if last_ts:
            delta = (datetime.datetime.now() - last_ts).total_seconds()
            if delta > SESSION_AUTO_IDLE_TIMEOUT:
                new_id = self._generate_session_id()
                self._set_current_session(new_id)
                print(f"[SESSION] Temporal break detected ({int(delta//3600)}h). New session: {new_id}")
                return new_id

        # 3. Contextual Check (Topic Shift)
        # We compare user input to the *summary* of the current session.
        summary = memory_manager.get_summary(current_id)
        if summary:
            # Use embeddings to check similarity
            sim = self._calculate_similarity(user_input, summary)
            if sim < SESSION_TOPIC_SHIFT_THRESHOLD:
                # If similarity is low, we might be starting a new topic.
                # However, only split if it's REALLY low or the session is already "full"
                # For now, let's just use the threshold.
                new_id = self._generate_session_id()
                self._set_current_session(new_id)
                print(f"[SESSION] Topic shift detected (Sim: {sim:.2f}). New session: {new_id}")
                return new_id

        return current_id

    def _generate_session_id(self) -> str:
        return datetime.datetime.now().strftime("session_%Y%m%d_%H%M")

    def _get_current_session(self) -> Optional[str]:
        return memory_manager._get_system_metadata("active_session_id")

    def _set_current_session(self, session_id: str):
        memory_manager._set_system_metadata("active_session_id", session_id)

    def _get_last_message_timestamp(self, session_id: str) -> Optional[datetime.datetime]:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (session_id,)
                )
                row = cursor.fetchone()
                if row:
                    # SQLite timestamp format: 2026-02-19 16:34:57
                    return datetime.datetime.fromisoformat(row[0].replace(" ", "T"))
        except Exception as e:
            print(f"[SESSION] Error fetching last timestamp: {e}")
        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Helper to compare semantic similarity between two texts."""
        try:
            emb1 = memory_manager._get_embedding(text1)
            emb2 = memory_manager._get_embedding(text2[:2000]) # Limit summary length
            
            vec1 = np.array([emb1]).astype('float32')
            vec2 = np.array([emb2]).astype('float32')
            faiss.normalize_L2(vec1)
            faiss.normalize_L2(vec2)
            
            return float(np.dot(vec1, vec2.T)[0][0])
        except Exception as e:
            print(f"[SESSION] Similarity calculation error: {e}")
            return 1.0 # Fail safe: don't split sessions on error

# Singleton instance
session_broker = SessionBroker()
