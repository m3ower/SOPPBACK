from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from ai_assistant import ParkingAssistant

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


class TranscriptRequest(BaseModel):
    text: str
    session_id: str = None
    timestamp: str = None


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
        "processed_at": datetime.now().isoformat()
    }


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    if session_id in ai_assistant.sessions:
        context = ai_assistant.sessions[session_id]
        history = []
        for msg in context.messages:
            history.append({
                "timestamp": msg.timestamp.isoformat(),
                "user_input": msg.user_input,
                "intent": msg.intent.value,
                "ai_response": msg.ai_response,
                "confidence": msg.confidence
            })
        return {
            "session_id": session_id,
            "message_count": len(history),
            "history": history,
            "created_at": context.created_at.isoformat(),
            "last_activity": context.last_activity.isoformat()
        }
    else:
        return {"error": "Session not found"}


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