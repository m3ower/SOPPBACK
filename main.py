from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from ai_assistant import ParkingAssistant
import stripe
from db import get_collections, ensure_indexes, ensure_validators
from bson import ObjectId
from typing import Optional

app = FastAPI()

# Initialize the AI assistant
ai_assistant = ParkingAssistant()

# Ensure DB schema and indexes
try:
    ensure_validators()
    ensure_indexes()
except Exception as _:
    pass

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


class StartSessionRequest(BaseModel):
    ticket_id: str
    plate: str
    vehicle_type: Optional[str] = None  # car, motorcycle, truck, rv
    gate_name: Optional[str] = None
    started_at: Optional[str] = None  # ISO string; default now if missing


class ExtendSessionRequest(BaseModel):
    ticket_id: str
    extra_minutes: int = 30


class PaymentSuccessRequest(BaseModel):
    ticket_id: Optional[str] = None
    session_db_id: Optional[str] = None


class QRPayload(BaseModel):
    # Accept both long and compact keys via aliases
    version: Optional[str] = Field(default=None, alias='v')
    ticketId: Optional[str] = Field(default=None, alias='t')
    plate: Optional[str] = Field(default=None, alias='p')
    vehicleType: Optional[str] = Field(default=None, alias='y')
    gateName: Optional[str] = Field(default=None, alias='g')
    issuedAt: Optional[str] = Field(default=None, alias='i')  # ISO time
    # Additional human-readable fields with hyphenated names
    entryTimeHuman: Optional[str] = Field(default=None, alias='entry-time')  # e.g., 13-09-2025
    carTypeHyphen: Optional[str] = Field(default=None, alias='car-type')     # e.g., c | car

    class Config:
        allow_population_by_field_name = True
    # Future fields can be added here; unknowns ignored on store


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


def _compute_amount_due_cents(started_at: datetime, now: Optional[datetime] = None, per_hour: float = 2.0, max_daily: Optional[float] = 20.0) -> int:
    now = now or datetime.utcnow()
    elapsed_sec = (now - started_at).total_seconds()
    if elapsed_sec <= 0:
        elapsed_sec = 1
    hours = max(1, int((elapsed_sec + 3599) // 3600))  # round up to next hour, min 1
    raw = hours * per_hour
    if max_daily is not None:
        raw = min(raw, max_daily)
    return int(round(raw * 100))


def _validate_qr(payload: QRPayload) -> dict:
    # Basic schema and content validation with support for clear template keys
    version = payload.version or '1'

    # Gather inputs possibly provided via hyphenated keys
    plate = (payload.plate or "").strip().upper()
    vtype_raw = (payload.vehicleType or payload.carTypeHyphen or "").strip().lower()
    issued_raw = payload.issuedAt or payload.entryTimeHuman

    # Check requireds and build missing list using requested names
    missing = []
    if not plate:
        missing.append('plate')
    if not vtype_raw:
        missing.append('car-type')
    if not issued_raw:
        missing.append('entry-time')
    if missing:
        raise HTTPException(status_code=400, detail=f"QR missing required fields: {', '.join(missing)}")

    if len(plate) < 3:
        raise HTTPException(status_code=400, detail="Invalid plate in QR")

    # Map short vehicle type codes
    code_map = {'c': 'car', 'm': 'motorcycle', 't': 'truck', 'r': 'rv'}
    vtype = code_map.get(vtype_raw, vtype_raw or 'car')
    if vtype not in {'car', 'motorcycle', 'truck', 'rv'}:
        raise HTTPException(status_code=400, detail="Invalid car-type; use c/m/t/r or car/motorcycle/truck/rv")

    # Parse issued time: try ISO first, then DD-MM-YYYY, then DD/MM/YYYY
    issued_at_iso: Optional[str] = None
    if issued_raw:
        try:
            # Accept exact ISO-8601
            _dt = datetime.fromisoformat(issued_raw.replace('Z', '+00:00'))
            issued_at_iso = _dt.isoformat()
        except Exception:
            # Fallbacks for human date
            for fmt in ("%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    _dt = datetime.strptime(issued_raw, fmt)
                    # Assume local naive as UTC for simplicity
                    issued_at_iso = _dt.isoformat()
                    break
                except Exception:
                    continue
    if not issued_at_iso:
        raise HTTPException(status_code=400, detail="Invalid entry-time format; expected ISO-8601 or DD-MM-YYYY")

    # Normalize
    return {
        "version": str(version),
        "ticketId": (str(payload.ticketId).strip() if payload.ticketId else None),
        "plate": plate,
        "vehicleType": vtype,
        "gateName": payload.gateName.strip() if payload.gateName else None,
        "issuedAt": issued_at_iso,
    }


@app.post("/api/session/start")
async def start_session(payload: StartSessionRequest):
    cols = get_collections()
    plate = payload.plate.strip().upper()
    vehicle = cols["vehicles"].find_one({"plate": plate})
    if not vehicle:
        vehicle_doc = {"plate": plate, "vehicle_type": payload.vehicle_type}
        res = cols["vehicles"].insert_one(vehicle_doc)
        vehicle_id = str(res.inserted_id)
    else:
        vehicle_id = str(vehicle.get("_id"))
        if payload.vehicle_type and vehicle.get("vehicle_type") != payload.vehicle_type:
            cols["vehicles"].update_one({"_id": vehicle["_id"]}, {"$set": {"vehicle_type": payload.vehicle_type}})

    # Find or create gate
    gate_id = None
    if payload.gate_name:
        gate = cols["gates"].find_one({"name": payload.gate_name})
        if not gate:
            gate_res = cols["gates"].insert_one({"name": payload.gate_name})
            gate_id = str(gate_res.inserted_id)
        else:
            gate_id = str(gate.get("_id"))

    # Close any stale open session for same plate (safety)
    cols["parking_sessions"].update_many({"vehicle_id": vehicle_id, "open": True}, {"$set": {"open": False, "ended_at": datetime.utcnow()}})
    # Mark any ongoing processes for same vehicle as completed
    try:
        cols["ongoing_processes"].update_many({"vehicle_id": vehicle_id, "status": "ongoing"}, {"$set": {"status": "completed", "ended_at": datetime.utcnow(), "updated_at": datetime.utcnow()}})
    except Exception:
        pass

    started_at = datetime.fromisoformat(payload.started_at) if payload.started_at else datetime.utcnow()
    sess_doc = {
        "vehicle_id": vehicle_id,
        "gate_id": gate_id,
        "ticket_id": payload.ticket_id,
        "started_at": started_at,
        "open": True,
        "paid": False,
    }
    ins = cols["parking_sessions"].insert_one(sess_doc)
    # Upsert ongoing process row
    try:
        now = datetime.utcnow()
        cols["ongoing_processes"].update_one(
            {"session_id": str(ins.inserted_id)},
            {"$set": {
                "session_id": str(ins.inserted_id),
                "ticket_id": payload.ticket_id,
                "vehicle_id": vehicle_id,
                "plate": plate,
                "gate_id": gate_id,
                "started_at": datetime.fromisoformat(payload.started_at) if payload.started_at else now,
                "last_scanned_at": now,
                "status": "ongoing",
                "created_at": now,
                "updated_at": now,
            }},
            upsert=True,
        )
    except Exception:
        pass
    return {"success": True, "session_db_id": str(ins.inserted_id)}


@app.post("/api/qr/ingest")
async def ingest_qr(payload: QRPayload):
    cols = get_collections()
    data = _validate_qr(payload)
    # Ensure vehicle
    plate = data["plate"]
    vehicle = cols["vehicles"].find_one({"plate": plate})
    if not vehicle:
        res = cols["vehicles"].insert_one({"plate": plate, "vehicle_type": data["vehicleType"]})
        vehicle_id = str(res.inserted_id)
    else:
        vehicle_id = str(vehicle["_id"])
        if data["vehicleType"] and vehicle.get("vehicle_type") != data["vehicleType"]:
            cols["vehicles"].update_one({"_id": vehicle["_id"]}, {"$set": {"vehicle_type": data["vehicleType"]}})
    # Ensure gate
    gate_id = None
    if data.get("gateName"):
        gate = cols["gates"].find_one({"name": data["gateName"]})
        if not gate:
            gate_res = cols["gates"].insert_one({"name": data["gateName"]})
            gate_id = str(gate_res.inserted_id)
        else:
            gate_id = str(gate["_id"])
    # Find existing open session by ticket or by vehicle
    sess = None
    if data.get("ticketId"):
        sess = cols["parking_sessions"].find_one({"ticket_id": data["ticketId"], "open": True})
    if not sess:
        sess = cols["parking_sessions"].find_one({"vehicle_id": vehicle_id, "open": True})
    now = datetime.utcnow()
    if not sess:
        # Start new session from QR
        started_at = now
        if data.get("issuedAt"):
            try:
                started_at = datetime.fromisoformat(data["issuedAt"])  # assume ISO string
            except Exception:
                started_at = now
        # Generate ticket id if missing
        ticket_id = data.get("ticketId") or f"TKT-{data['plate']}-{int(now.timestamp())}"
        doc = {
            "vehicle_id": vehicle_id,
            "gate_id": gate_id,
            "ticket_id": ticket_id,
            "started_at": started_at,
            "open": True,
            "paid": False,
            "qr_version": data["version"],
            "qr_raw": data,
            "last_scanned_at": now,
        }
        ins = cols["parking_sessions"].insert_one(doc)
        sess = cols["parking_sessions"].find_one({"_id": ins.inserted_id})
        # Clear any older ongoing entries for same vehicle
        try:
            cols["ongoing_processes"].update_many({"vehicle_id": vehicle_id, "session_id": {"$ne": str(ins.inserted_id)}, "status": "ongoing"}, {"$set": {"status": "completed", "ended_at": now, "updated_at": now}})
            cols["ongoing_processes"].delete_many({"vehicle_id": vehicle_id, "session_id": {"$ne": str(ins.inserted_id)}})
        except Exception:
            pass
        # Upsert ongoing process entry
        try:
            cols["ongoing_processes"].update_one(
                {"session_id": str(ins.inserted_id)},
                {"$set": {
                    "session_id": str(ins.inserted_id),
                    "ticket_id": ticket_id,
                    "vehicle_id": vehicle_id,
                    "plate": plate,
                    "gate_id": gate_id,
                    "started_at": started_at,
                    "last_scanned_at": now,
                    "qr_version": data["version"],
                    "qr_raw": data,
                    "status": "ongoing",
                    "created_at": now,
                    "updated_at": now,
                }},
                upsert=True,
            )
        except Exception as _:
            pass
    else:
        # Update existing session with latest QR info
        update_doc = {
            "gate_id": gate_id or sess.get("gate_id"),
            "ticket_id": data.get("ticketId") or sess.get("ticket_id"),
            "qr_version": data["version"],
            "qr_raw": data,
            "last_scanned_at": now,
        }
        cols["parking_sessions"].update_one({"_id": sess["_id"]}, {"$set": update_doc})
        sess = cols["parking_sessions"].find_one({"_id": sess["_id"]})
        # Sync ongoing process entry
        try:
            cols["ongoing_processes"].update_one(
                {"session_id": str(sess["_id"])},
                {"$set": {
                    "ticket_id": sess.get("ticket_id"),
                    "vehicle_id": sess.get("vehicle_id"),
                    "plate": plate,
                    "gate_id": sess.get("gate_id"),
                    "last_scanned_at": now,
                    "qr_version": data["version"],
                    "qr_raw": data,
                    "status": "ongoing",
                    "updated_at": now,
                }},
                upsert=True,
            )
        except Exception as _:
            pass

    # Compute amount due based on up-to-date session & pricing
    started_at = sess["started_at"]
    gate_id = sess.get("gate_id")
    veh = cols["vehicles"].find_one({"_id": ObjectId(sess["vehicle_id"])}) if sess.get("vehicle_id") else None
    vtype = (veh or {}).get("vehicle_type") or "car"
    pr = cols["prices"].find_one({"gate_id": gate_id}) if gate_id else None
    per_hour = 2.0
    max_daily = 20.0
    if pr:
        per_hour = float(pr.get("per_hour", per_hour))
        tiers = pr.get("per_hour_by_type") or {}
        if vtype in tiers:
            per_hour = float(tiers[vtype])
        if pr.get("max_daily") is not None:
            max_daily = float(pr.get("max_daily"))
    amount_cents = _compute_amount_due_cents(started_at, per_hour=per_hour, max_daily=max_daily)
    return {
        "success": True,
        "session_id": str(sess["_id"]),
        "ticket_id": sess.get("ticket_id"),
        "vehicle_type": vtype,
        "per_hour": per_hour,
        "amount_cents": amount_cents,
        "open": bool(sess.get("open", True)),
        "paid": bool(sess.get("paid", False)),
        "started_at": (sess.get("started_at").isoformat() if sess.get("started_at") else None),
    }


@app.post("/api/session/extend")
async def extend_session(payload: ExtendSessionRequest):
    cols = get_collections()
    sess = cols["parking_sessions"].find_one({"ticket_id": payload.ticket_id, "open": True})
    if not sess:
        raise HTTPException(status_code=404, detail="Open session not found for ticket")
    # Compute new planned end time
    now = datetime.utcnow()
    planned_end = sess.get("planned_end_at") or now
    if isinstance(planned_end, str):
        try:
            planned_end = datetime.fromisoformat(planned_end)
        except Exception:
            planned_end = now
    if planned_end < now:
        planned_end = now
    new_planned_end = planned_end + timedelta(minutes=int(max(1, payload.extra_minutes)))
    cols["parking_sessions"].update_one({"_id": sess["_id"]}, {"$set": {"planned_end_at": new_planned_end}})
    # Touch ongoing process update time
    try:
        cols["ongoing_processes"].update_one({"session_id": str(sess["_id"])}, {"$set": {"updated_at": datetime.utcnow()}})
    except Exception:
        pass
    return {"success": True, "planned_end_at": new_planned_end.isoformat()}


@app.get("/api/session/{ticket_id}/amount-due")
async def get_amount_due(ticket_id: str):
    cols = get_collections()
    sess = cols["parking_sessions"].find_one({"ticket_id": ticket_id, "open": True})
    if not sess:
        raise HTTPException(status_code=404, detail="Open session not found for ticket")
    started_at = sess["started_at"]
    gate_id = sess.get("gate_id")
    # Resolve vehicle type
    veh = cols["vehicles"].find_one({"_id": ObjectId(sess["vehicle_id"])}) if sess.get("vehicle_id") else None
    vtype = (veh or {}).get("vehicle_type") or "car"
    # Load pricing for gate
    pr = cols["prices"].find_one({"gate_id": gate_id}) if gate_id else None
    per_hour = 2.0
    max_daily = 20.0
    if pr:
        per_hour = float(pr.get("per_hour", per_hour))
        tiers = pr.get("per_hour_by_type") or {}
        if vtype in tiers:
            per_hour = float(tiers[vtype])
        if pr.get("max_daily") is not None:
            max_daily = float(pr.get("max_daily"))
    amount_cents = _compute_amount_due_cents(started_at, per_hour=per_hour, max_daily=max_daily)
    return {"amount_cents": amount_cents, "per_hour": per_hour, "vehicle_type": vtype}


@app.post("/api/session/paid")
async def mark_session_paid(payload: PaymentSuccessRequest):
    cols = get_collections()
    query = {}
    if payload.session_db_id:
        try:
            query["_id"] = ObjectId(payload.session_db_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid session id")
    elif payload.ticket_id:
        query["ticket_id"] = payload.ticket_id
        query["open"] = True
    else:
        raise HTTPException(status_code=400, detail="Provide session_db_id or ticket_id")

    sess = cols["parking_sessions"].find_one(query)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    # Compute final cost and close session
    started_at = sess["started_at"]
    # Use same pricing resolution as amount-due
    gate_id = sess.get("gate_id")
    veh = cols["vehicles"].find_one({"_id": ObjectId(sess["vehicle_id"])}) if sess.get("vehicle_id") else None
    vtype = (veh or {}).get("vehicle_type") or "car"
    pr = cols["prices"].find_one({"gate_id": gate_id}) if gate_id else None
    per_hour = 2.0
    max_daily = 20.0
    if pr:
        per_hour = float(pr.get("per_hour", per_hour))
        tiers = pr.get("per_hour_by_type") or {}
        if vtype in tiers:
            per_hour = float(tiers[vtype])
        if pr.get("max_daily") is not None:
            max_daily = float(pr.get("max_daily"))
    amount_cents = _compute_amount_due_cents(started_at, per_hour=per_hour, max_daily=max_daily)
    cols["parking_sessions"].update_one({"_id": sess["_id"]}, {"$set": {
        "paid": True,
        "open": False,
        "ended_at": datetime.utcnow(),
        "cost": amount_cents / 100.0,
    }})
    # Mark ongoing process as completed
    try:
        cols["ongoing_processes"].delete_one({"session_id": str(sess["_id"])})
    except Exception as _:
        pass
    return {"success": True, "final_amount_cents": amount_cents}


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


@app.get("/api/ongoing")
async def list_ongoing():
    """List ongoing processes (dev/debug)."""
    cols = get_collections()
    docs = list(cols["ongoing_processes"].find({}).sort("updated_at", -1))
    def _fmt(d):
        return {
            "session_id": str(d.get("session_id")),
            "ticket_id": d.get("ticket_id"),
            "plate": d.get("plate"),
            "status": d.get("status"),
            "started_at": d.get("started_at").isoformat() if d.get("started_at") else None,
            "ended_at": d.get("ended_at").isoformat() if d.get("ended_at") else None,
            "updated_at": d.get("updated_at").isoformat() if d.get("updated_at") else None,
            "gate_id": d.get("gate_id"),
        }
    return {"count": len(docs), "items": [_fmt(x) for x in docs]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)