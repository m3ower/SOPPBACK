from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from ai_assistant import ParkingAssistant
import stripe

app = FastAPI()

# Initialize the AI assistant
ai_assistant = ParkingAssistant()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stripe configuration (keys embedded as requested)
STRIPE_PUBLISHABLE_KEY = "pk_test_51S6vz2JJF1Q4l4kQ7tE8niMN6mEtzAuhdjhTuLAnz3BZg5fSjFvtDsTUwXi0ddzjRDlReAEGQcsigst8OCr67sFX00SNSeyydv"
STRIPE_SECRET_KEY = "sk_test_51S6vz2JJF1Q4l4kQWKJpu9jt5CWhzXNbwWlYZC6kxbpwKh9HtGoZRzsA7BW5PO0mV4PTuCVz2ZvME12Fkxuisg6G00EZyqjY5z"
stripe.api_key = STRIPE_SECRET_KEY


class TranscriptRequest(BaseModel):
    text: str
    session_id: str = None
    timestamp: str = None


class CheckoutRequest(BaseModel):
    amount: int  # in cents
    currency: str = "eur"
    success_url: str
    cancel_url: str
    description: str | None = "Parking payment"


@app.post("/api/transcribe")
async def receive_transcript(request: TranscriptRequest):
    """Receive transcribed text from frontend and generate contextual response"""

    # Process with AI assistant (handles classification and context)
    result = ai_assistant.process_message(
        session_id=request.session_id or "default_session",
        user_input=request.text
    )

    # Enhanced console logging
    print(f"\n{'=' * 60}")
    print(f"CONVERSATIONAL AI - SESSION: {result['session_id']}")
    print(f"{'=' * 60}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Message #{result['message_count']}")
    print(f"User said: {request.text}")
    print(f"Detected intent: {result['intent']} (confidence: {result['confidence']:.2f})")
    print(f"LLM handoff: {result.get('llm_handoff', False)}")
    print(f"AI response: {result['response']}")
    print(f"{'=' * 60}\n")

    # Return response to frontend
    return {
        "success": True,
        "reply": result["response"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "session_id": result["session_id"],
        "message_count": result["message_count"],
        "processed_at": datetime.now().isoformat(),
        "llm_handoff": result.get("llm_handoff", False),
        "entities": result.get("entities", {}),
    }


@app.post("/api/pay/create-checkout-session")
async def create_checkout_session(payload: CheckoutRequest):
    """Create a Stripe Checkout Session and return the URL for redirection."""
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": payload.currency,
                    "unit_amount": int(payload.amount),
                    "product_data": {
                        "name": payload.description or "Parking payment",
                    },
                },
                "quantity": 1,
            }],
            success_url=payload.success_url,
            cancel_url=payload.cancel_url,
        )
        return {"checkout_url": session.url, "publishable_key": STRIPE_PUBLISHABLE_KEY}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history and traces for a session from DB"""
    db = ai_assistant.db
    messages = list(db["messages"].find({"session_id": session_id}).sort("timestamp", 1))
    traces = list(db["traces"].find({"session_id": session_id}).sort("timestamp", 1))
    history = [
        {
            "timestamp": m.get("timestamp").isoformat() if m.get("timestamp") else None,
            "user_input": m.get("user_input"),
            "intent": m.get("intent"),
            "ai_response": m.get("ai_response"),
            "confidence": m.get("confidence"),
        }
        for m in messages
    ]
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history,
        "traces": [
            {
                "timestamp": t.get("timestamp").isoformat() if t.get("timestamp") else None,
                "step": t.get("step"),
                "status": t.get("status"),
                "details": t.get("details"),
                "confidence": t.get("confidence"),
            }
            for t in traces
        ],
    }


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session"""
    if session_id in ai_assistant.sessions:
        del ai_assistant.sessions[session_id]
        return {"success": True, "message": f"Session {session_id} cleared"}
    else:
        return {"error": "Session not found"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(ai_assistant.sessions)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)