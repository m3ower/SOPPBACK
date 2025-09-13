from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

ddgfdg
class Intent(Enum):
    LOST_TICKET = "lost_ticket"
    PAYMENT = "payment"
    FIND_CAR = "find_car"
    EXTEND_TIME = "extend_time"
    EMERGENCY = "emergency"
    BARRIER_CONTROL = "barrier_control"
    GREETING = "greeting"
    GRATITUDE = "gratitude"
    GENERAL_HELP = "general_help"
    UNCLEAR = "unclear"


@dataclass
class ConversationMessage:
    timestamp: datetime
    user_input: str
    intent: Intent
    ai_response: str
    confidence: float


class ConversationContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[ConversationMessage] = []
        self.current_intent: Optional[Intent] = None
        self.user_state = {}  # Store user-specific info like parking location, payment status, etc.
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

    def add_message(self, user_input: str, intent: Intent, ai_response: str, confidence: float):
        message = ConversationMessage(
            timestamp=datetime.now(),
            user_input=user_input,
            intent=intent,
            ai_response=ai_response,
            confidence=confidence
        )
        self.messages.append(message)
        self.current_intent = intent
        self.last_activity = datetime.now()

    def get_recent_context(self, num_messages: int = 3) -> List[ConversationMessage]:
        return self.messages[-num_messages:]


class ParkingAssistant:
    def __init__(self):
        self.sessions: Dict[str, ConversationContext] = {}

        # Intent classification keywords
        self.intent_keywords = {
            Intent.LOST_TICKET: ["lost", "missing", "ticket", "can't find", "forgot"],
            Intent.PAYMENT: ["pay", "payment", "cost", "charge", "bill", "money", "card"],
            Intent.FIND_CAR: ["find", "locate", "where", "car", "vehicle", "parked"],
            Intent.EXTEND_TIME: ["extend", "more time", "longer", "additional"],
            Intent.EMERGENCY: ["emergency", "urgent", "help", "stuck", "trapped"],
            Intent.BARRIER_CONTROL: ["barrier", "gate", "exit", "open", "close"],
            Intent.GREETING: ["hello", "hi", "hey", "good morning", "good afternoon"],
            Intent.GRATITUDE: ["thank", "thanks", "appreciate"],
            Intent.GENERAL_HELP: ["help", "assist", "support", "problem", "issue"]
        }

    def get_or_create_session(self, session_id: str) -> ConversationContext:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(session_id)
        return self.sessions[session_id]

    def classify_intent(self, text: str, context: ConversationContext) -> tuple[Intent, float]:
        text_lower = text.lower()
        intent_scores = {}

        # Score each intent based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            intent_scores[intent] = score

        # Consider context from previous messages
        if context.current_intent and len(context.messages) > 0:
            last_intent = context.current_intent
            # Boost score for continuing conversation
            if last_intent in intent_scores:
                intent_scores[last_intent] += 0.5

        # Find best match
        if not intent_scores or max(intent_scores.values()) == 0:
            return Intent.UNCLEAR, 0.3

        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent] / 3.0, 1.0)  # Normalize confidence

        return best_intent, confidence

    def generate_contextual_response(self, intent: Intent, text: str, context: ConversationContext) -> str:
        recent_messages = context.get_recent_context(2)

        if intent == Intent.LOST_TICKET:
            if any(msg.intent == Intent.LOST_TICKET for msg in recent_messages):
                return "I'm still working on your lost ticket issue. Let me generate a replacement code: LT-2024-" + context.session_id[
                                                                                                                     -4:].upper() + ". Show this to the attendant."
            else:
                return "I understand you've lost your parking ticket. Don't worry, I can help you with that. Do you remember approximately when you arrived?"

        elif intent == Intent.PAYMENT:
            if any(msg.intent == Intent.PAYMENT for msg in recent_messages):
                return "For payment continuation: The total is €4.50. Would you like to pay by card, mobile payment, or cash?"
            else:
                return "I'll help you process your parking payment. Let me calculate your total... That's €4.50 for your parking session. How would you like to pay?"

        elif intent == Intent.FIND_CAR:
            if any(msg.intent == Intent.FIND_CAR for msg in recent_messages):
                return "I'm still helping you locate your car. Try going to Level B2, Section C. Look for the blue signs. Can you see any familiar landmarks around your car?"
            else:
                context.user_state['looking_for_car'] = True
                return "I'll help you find your car. Do you remember which level or section you parked in? Any nearby landmarks or the color of the area signs?"

        elif intent == Intent.EXTEND_TIME:
            return "I can extend your parking time. How much additional time do you need? I can add 30 minutes, 1 hour, or 2 hours."

        elif intent == Intent.EMERGENCY:
            return "This sounds urgent. I'm immediately connecting you to emergency support. Help is on the way. Please stay calm and stay on the line."

        elif intent == Intent.BARRIER_CONTROL:
            return "I'm opening the exit barrier for you now. Please proceed slowly and have a safe journey!"

        elif intent == Intent.GREETING:
            if len(context.messages) == 0:
                return "Hello! I'm your smart parking assistant. I can help you with payments, finding your car, lost tickets, or any other parking needs. How can I assist you today?"
            else:
                return "Hello again! How else can I help you with your parking today?"

        elif intent == Intent.GRATITUDE:
            return "You're very welcome! I'm here whenever you need parking assistance. Have a great day!"

        elif intent == Intent.GENERAL_HELP:
            if "issue" in text.lower() or "problem" in text.lower():
                return "I'm here to help with any parking issues. Are you having trouble with payment, finding your car, a lost ticket, or something else?"
            else:
                return "I'm your parking assistant. I can help you with: finding your car, processing payments, replacing lost tickets, extending time, or opening barriers. What do you need?"

        else:  # Intent.UNCLEAR
            if len(context.messages) > 0:
                return "I want to make sure I understand correctly. Could you tell me more about what you need help with? I can assist with parking payments, finding your car, lost tickets, or other parking services."
            else:
                return "I'm here to help with your parking needs. Could you please tell me what you'd like assistance with? I can help with payments, finding your car, lost tickets, and more."

    def process_message(self, session_id: str, user_input: str) -> dict:
        # Get or create session
        context = self.get_or_create_session(session_id)

        # Classify intent
        intent, confidence = self.classify_intent(user_input, context)

        # Generate contextual response
        ai_response = self.generate_contextual_response(intent, user_input, context)

        # Add to conversation history
        context.add_message(user_input, intent, ai_response, confidence)

        return {
            "response": ai_response,
            "intent": intent.value,
            "confidence": confidence,
            "session_id": session_id,
            "message_count": len(context.messages)
        }