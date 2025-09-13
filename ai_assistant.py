from datetime import datetime
from typing import Dict

from db import get_db
from models import Intent
from nlu import NLU, handle_intent, log_trace, TraceStatus


class ParkingAssistant:
    """
    Router between small-talk (front-end LLM) and domain NLU (backend, DB-driven).
    Persists queries, messages, and traces in MongoDB, keyed by session_id.
    """

    def __init__(self) -> None:
        self.db = get_db()
        self.sessions: Dict[str, Dict] = {}
        self.nlu = NLU()

    def process_message(self, session_id: str, user_input: str) -> dict:
        now = datetime.utcnow()
        # Log raw query
        self.db["queries"].insert_one({
            "session_id": session_id,
            "text": user_input,
            "timestamp": now,
        })

        # Classify
        intent, confidence, entities = self.nlu.classify(user_input)

        # Delegate to LLM only for low-confidence unknown; handle greetings/gratitude ourselves
        llm_handoff = (intent == Intent.unknown and confidence < 0.6)

        if llm_handoff:
            # The frontend will call Puter GPT-5 nano; we store a placeholder message record
            self.db["messages"].insert_one({
                "session_id": session_id,
                "user_input": user_input,
                "intent": intent.value,
                "ai_response": None,
                "confidence": float(confidence),
                "timestamp": now,
            })
            log_trace(self.db, session_id, "llm_handoff", TraceStatus.decision, {"intent": intent.value}, confidence)
            return {
                "response": None,
                "intent": intent.value,
                "confidence": float(confidence),
                "session_id": session_id,
                "message_count": self.db["messages"].count_documents({"session_id": session_id}),
                "llm_handoff": True,
                "entities": entities,
            }

        # Handle via domain algorithm
        response = handle_intent(self.db, session_id, intent, entities, confidence)
        self.db["messages"].insert_one({
            "session_id": session_id,
            "user_input": user_input,
            "intent": intent.value,
            "ai_response": response,
            "confidence": float(confidence),
            "timestamp": now,
        })
        return {
            "response": response,
            "intent": intent.value,
            "confidence": float(confidence),
            "session_id": session_id,
            "message_count": self.db["messages"].count_documents({"session_id": session_id}),
            "llm_handoff": False,
            "entities": entities,
        }