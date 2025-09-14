from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
try:
    # Pydantic v2 config
    from pydantic import ConfigDict  # type: ignore
except Exception:
    ConfigDict = None  # type: ignore
from datetime import datetime, timedelta
from ai_assistant import ParkingAssistant
import stripe
from db import get_collections, ensure_indexes, ensure_validators, get_db
from bson import ObjectId
from typing import Optional
from typing import List

app = FastAPI()

# Initialize the AI assistant
ai_assistant = ParkingAssistant()
ai_assistant.db = get_db()

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

    # Pydantic v2 style config; fallback left for v1
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:
        class Config:  # type: ignore
            allow_population_by_field_name = True
    # Future fields can be added here; unknowns ignored on store
class ExitAttemptRequest(BaseModel):
    vrn: str
    gate_name: Optional[str] = None


class VRNSearchRequest(BaseModel):
    vrn: Optional[str] = None
    entry_time_from: Optional[str] = None
    entry_time_to: Optional[str] = None
    limit: int = 10


class VRNUpdateRequest(BaseModel):
    session_id: Optional[str] = None
    ticket_id: Optional[str] = None
    old_vrn: Optional[str] = None
    new_vrn: str


class BarrierControlRequest(BaseModel):
    action: str  # 'allow' | 'deny'
    gate_name: Optional[str] = None


class MessageLogRequest(BaseModel):
    session_id: str
    user_input: Optional[str] = None
    ai_response: Optional[str] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[dict] = None
    timestamp: Optional[str] = None  # ISO 8601


def _fuzzy_score(a: str, b: str) -> int:
    # Simple Levenshtein distance
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


@app.post("/api/exit/attempt")
async def exit_attempt(payload: ExitAttemptRequest):
    cols = get_collections()
    vrn = payload.vrn.strip().upper()
    # Find vehicle
    vehicle = cols["vehicles"].find_one({"plate": vrn})
    if not vehicle:
        return {"success": False, "status": "vrn_not_found"}
    vid = str(vehicle.get("_id"))
    # Prefer ongoing open, else open session
    ongoing = cols["ongoing_processes"].find_one({"vehicle_id": vid, "open": True})
    sess = None
    if not ongoing:
        sess = cols["parking_sessions"].find_one({"vehicle_id": vid, "open": True})
    src = ongoing or sess
    if not src:
        # No active session - allow crossing (some ticketless systems allow exit)
        return {"success": True, "status": "allowed", "reason": "no_active_session", "session": None}
    # Check payment
    if src.get("paid") or not src.get("open", True):
        return {"success": True, "status": "allowed", "session": {
            "session_id": src.get("session_id") or src.get("ticket_id"),
            "entry_time": (src.get("started_at").isoformat() if src.get("started_at") else None),
            "plate": vrn,
            "payment_amount": int((src.get("cost") or 0.0) * 100),
        }}
    # If unpaid, return payment required
    return {"success": True, "status": "payment_required", "session": {
        "session_id": src.get("session_id") or src.get("ticket_id"),
        "entry_time": (src.get("started_at").isoformat() if src.get("started_at") else None),
        "plate": vrn,
    }}


@app.post("/api/vrn/search")
async def vrn_search(payload: VRNSearchRequest):
    cols = get_collections()
    vrn = (payload.vrn or "").strip().upper()
    # Collect candidate open sessions (ongoing + sessions)
    candidates: List[dict] = []
    candidates.extend(list(cols["ongoing_processes"].find({"open": True})))
    candidates.extend(list(cols["parking_sessions"].find({"open": True})))
    # Filter by entry time if provided
    def within_time(doc: dict) -> bool:
        if not payload.entry_time_from and not payload.entry_time_to:
            return True
        try:
            st = doc.get("started_at")
            if not st:
                return False
            if isinstance(st, str):
                st = datetime.fromisoformat(st)
            if payload.entry_time_from:
                if st < datetime.fromisoformat(payload.entry_time_from):
                    return False
            if payload.entry_time_to:
                if st > datetime.fromisoformat(payload.entry_time_to):
                    return False
            return True
        except Exception:
            return False
    candidates = [c for c in candidates if within_time(c)]
    # Score by VRN similarity
    results = []
    for c in candidates:
        # resolve plate
        plate = None
        if c.get("plate"):
            plate = c["plate"].upper()
        else:
            veh = cols["vehicles"].find_one({"_id": ObjectId(c.get("vehicle_id"))}) if c.get("vehicle_id") else None
            if veh:
                plate = str(veh.get("plate", "")).upper()
        if not plate:
            continue
        score = _fuzzy_score(vrn, plate) if vrn else 0
        results.append({
            "session_id": c.get("session_id") or c.get("ticket_id"),
            "plate": plate,
            "entry_time": (c.get("started_at").isoformat() if c.get("started_at") else None),
            "open": bool(c.get("open", True)),
            "paid": bool(c.get("paid", False)),
            "score": score,
        })
    results.sort(key=lambda r: r["score"])  # lower distance is better
    return {"success": True, "results": results[: max(1, int(payload.limit))]}


@app.post("/api/vrn/update")
async def vrn_update(payload: VRNUpdateRequest):
    cols = get_collections()
    if not payload.new_vrn:
        raise HTTPException(status_code=400, detail="new_vrn required")
    new_plate = payload.new_vrn.strip().upper()
    # Ensure vehicle exists for new VRN
    vehicle = cols["vehicles"].find_one({"plate": new_plate})
    if not vehicle:
        vehicle_id = str(cols["vehicles"].insert_one({"plate": new_plate}).inserted_id)
    else:
        vehicle_id = str(vehicle["_id"])
    # Build query to locate session/ongoing
    q = {}
    if payload.session_id:
        q["session_id"] = payload.session_id
    if payload.ticket_id:
        q["ticket_id"] = payload.ticket_id
    if not q:
        raise HTTPException(status_code=400, detail="Provide session_id or ticket_id")
    updated = 0
    for colname in ["ongoing_processes", "parking_sessions"]:
        r = cols[colname].update_many(q, {"$set": {"plate": new_plate, "vehicle_id": vehicle_id}})
        updated += r.modified_count
    return {"success": True, "updated": int(updated)}


@app.post("/api/barrier/control")
async def barrier_control(payload: BarrierControlRequest):
    # In a real system this would signal hardware; here we return success
    if payload.action not in {"allow", "deny"}:
        raise HTTPException(status_code=400, detail="Invalid action")
    return {"success": True, "action": payload.action, "gate": payload.gate_name}


@app.get("/api/anpr/live")
async def anpr_live(vrn: Optional[str] = None):
    # Simulate ANPR camera feed status
    return {"success": True, "vrn": (vrn or None), "camera": "EXIT_CAM_1", "timestamp": datetime.utcnow().isoformat()}


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

    # Create or update an ongoing process record (do not finalize into parking_sessions yet)
    now = datetime.utcnow()
    started_at = datetime.fromisoformat(payload.started_at) if payload.started_at else now
    # Session id: prefer provided ticket id; else synthesize one
    session_id = (payload.ticket_id or f"SESS-{plate}-{int(now.timestamp())}")
    doc = {
        "session_id": session_id,
        "plate": plate,
        "vehicle_id": vehicle_id,
        "gate_id": gate_id,
        "ticket_id": payload.ticket_id,
        "started_at": started_at,
        "open": True,
        "paid": False,
        "status": "open",
        "last_scanned_at": now,
        "source": "session_start",
    }
    query = {"ticket_id": payload.ticket_id} if payload.ticket_id else {"vehicle_id": vehicle_id, "open": True}
    existing = cols["ongoing_processes"].find_one(query)
    if existing:
        cols["ongoing_processes"].update_one({"_id": existing["_id"]}, {"$set": doc})
        oid = existing["_id"]
    else:
        ins = cols["ongoing_processes"].insert_one(doc)
        oid = ins.inserted_id
    return {"success": True, "ongoing_id": str(oid)}


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
    # Prefer ongoing process first
    ongoing = None
    if data.get("ticketId"):
        ongoing = cols["ongoing_processes"].find_one({"ticket_id": data["ticketId"], "open": True})
    if not ongoing:
        ongoing = cols["ongoing_processes"].find_one({"vehicle_id": vehicle_id, "open": True})
    # Back-compat: open session if no ongoing exists
    sess = None
    if not ongoing and data.get("ticketId"):
        sess = cols["parking_sessions"].find_one({"ticket_id": data["ticketId"], "open": True})
    if not ongoing and not sess:
        sess = cols["parking_sessions"].find_one({"vehicle_id": vehicle_id, "open": True})
    now = datetime.utcnow()
    if not ongoing and not sess:
        # Start new ongoing from QR
        started_at = now
        if data.get("issuedAt"):
            try:
                started_at = datetime.fromisoformat(data["issuedAt"])  # assume ISO string
            except Exception:
                started_at = now
        # Generate ticket id if missing
        ticket_id = data.get("ticketId") or f"TKT-{data['plate']}-{int(now.timestamp())}"
        session_id = ticket_id
        doc = {
            "session_id": session_id,
            "plate": plate,
            "vehicle_id": vehicle_id,
            "gate_id": gate_id,
            "ticket_id": ticket_id,
            "started_at": started_at,
            "open": True,
            "paid": False,
            "status": "open",
            "qr_version": data["version"],
            "qr_raw": data,
            "last_scanned_at": now,
        }
        ins = cols["ongoing_processes"].insert_one(doc)
        ongoing = cols["ongoing_processes"].find_one({"_id": ins.inserted_id})
    else:
        # Update existing ongoing or session with latest QR info
        update_doc = {
            "gate_id": gate_id or (sess.get("gate_id") if sess else None),
            "ticket_id": data.get("ticketId") or (sess.get("ticket_id") if sess else None),
            "plate": plate,
            "status": "open",
            "qr_version": data["version"],
            "qr_raw": data,
            "last_scanned_at": now,
        }
        if ongoing:
            cols["ongoing_processes"].update_one({"_id": ongoing["_id"]}, {"$set": update_doc})
            ongoing = cols["ongoing_processes"].find_one({"_id": ongoing["_id"]})
        elif sess:
            cols["parking_sessions"].update_one({"_id": sess["_id"]}, {"$set": update_doc})
            sess = cols["parking_sessions"].find_one({"_id": sess["_id"]})

    # Compute amount due based on up-to-date session & pricing
    source_doc = ongoing or sess
    started_at = source_doc["started_at"]
    gate_id = source_doc.get("gate_id")
    veh = cols["vehicles"].find_one({"_id": ObjectId(source_doc["vehicle_id"])}) if source_doc.get("vehicle_id") else None
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
        "session_id": str(source_doc["_id"]),
        "ticket_id": source_doc.get("ticket_id"),
        "vehicle_type": vtype,
        "per_hour": per_hour,
        "amount_cents": amount_cents,
        "open": bool(source_doc.get("open", True)),
        "paid": bool(source_doc.get("paid", False)),
        "started_at": (source_doc.get("started_at").isoformat() if source_doc.get("started_at") else None),
    }


@app.post("/api/session/extend")
async def extend_session(payload: ExtendSessionRequest):
    cols = get_collections()
    # Prefer ongoing process if present
    ongoing = cols["ongoing_processes"].find_one({"ticket_id": payload.ticket_id, "open": True})
    sess = None
    if not ongoing:
        sess = cols["parking_sessions"].find_one({"ticket_id": payload.ticket_id, "open": True})
    if not ongoing and not sess:
        raise HTTPException(status_code=404, detail="Open session not found for ticket")
    # Compute new planned end time
    now = datetime.utcnow()
    src = ongoing or sess
    planned_end = src.get("planned_end_at") or now
    if isinstance(planned_end, str):
        try:
            planned_end = datetime.fromisoformat(planned_end)
        except Exception:
            planned_end = now
    if planned_end < now:
        planned_end = now
    new_planned_end = planned_end + timedelta(minutes=int(max(1, payload.extra_minutes)))
    if ongoing:
        cols["ongoing_processes"].update_one({"_id": ongoing["_id"]}, {"$set": {"planned_end_at": new_planned_end}})
    else:
        cols["parking_sessions"].update_one({"_id": sess["_id"]}, {"$set": {"planned_end_at": new_planned_end}})
    return {"success": True, "planned_end_at": new_planned_end.isoformat()}


@app.get("/api/session/{ticket_id}/amount-due")
async def get_amount_due(ticket_id: str):
    cols = get_collections()
    ongoing = cols["ongoing_processes"].find_one({"ticket_id": ticket_id, "open": True})
    sess = None
    if not ongoing:
        sess = cols["parking_sessions"].find_one({"ticket_id": ticket_id, "open": True})
    src = ongoing or sess
    if not src:
        raise HTTPException(status_code=404, detail="Open session not found for ticket")
    started_at = src["started_at"]
    gate_id = src.get("gate_id")
    # Resolve vehicle type
    veh = cols["vehicles"].find_one({"_id": ObjectId(src["vehicle_id"])}) if src.get("vehicle_id") else None
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
    # Prefer ongoing
    ongoing = cols["ongoing_processes"].find_one(query)
    sess = None
    if not ongoing:
        sess = cols["parking_sessions"].find_one(query)
    if not ongoing and not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    # Compute final cost and close session
    src = ongoing or sess
    started_at = src["started_at"]
    # Use same pricing resolution as amount-due
    gate_id = src.get("gate_id")
    veh = cols["vehicles"].find_one({"_id": ObjectId(src["vehicle_id"])}) if src.get("vehicle_id") else None
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
    if ongoing:
        # Move ongoing into parking_sessions as finalized record
        now = datetime.utcnow()
        move_doc = {k: v for k, v in ongoing.items() if k != "_id"}
        move_doc.update({
            "paid": True,
            "open": False,
            "ended_at": now,
            "cost": amount_cents / 100.0,
            "finalized_from": "ongoing_processes",
            "status": "paid",
        })
        ins = cols["parking_sessions"].insert_one(move_doc)
        cols["ongoing_processes"].delete_one({"_id": ongoing["_id"]})
        return {"success": True, "final_amount_cents": amount_cents, "session_db_id": str(ins.inserted_id)}
    else:
        cols["parking_sessions"].update_one({"_id": src["_id"]}, {"$set": {
            "paid": True,
            "open": False,
            "ended_at": datetime.utcnow(),
            "cost": amount_cents / 100.0,
            "status": "paid",
        }})
        return {"success": True, "final_amount_cents": amount_cents, "session_db_id": str(src["_id"]) }


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


@app.post("/api/messages/log")
async def log_message(payload: MessageLogRequest):
    """Persist a conversation message (user and/or assistant) into MongoDB.

    This is used by the frontend which runs the AI locally (Puter SDK) to still
    record transcripts server-side for history and analytics.
    """
    db = ai_assistant.db
    if not payload.session_id or not isinstance(payload.session_id, str):
        raise HTTPException(status_code=400, detail="session_id is required")
    # Parse timestamp or default to now (UTC)
    ts = datetime.utcnow()
    if payload.timestamp:
        try:
            # Support both Z and timezone-less formats
            iso = payload.timestamp.replace('Z', '+00:00')
            ts_parsed = datetime.fromisoformat(iso)
            # Store naive UTC
            ts = ts_parsed if ts_parsed.tzinfo is None else ts_parsed.replace(tzinfo=None)
        except Exception:
            pass

    # Coerce types and provide defaults to satisfy collection validator
    user_input = "" if payload.user_input is None else str(payload.user_input)
    ai_response = "" if payload.ai_response is None else str(payload.ai_response)
    intent = payload.intent if payload.intent is not None else "unknown"
    try:
        confidence = float(payload.confidence) if payload.confidence is not None else 0.0
    except Exception:
        confidence = 0.0

    doc = {
        "session_id": payload.session_id,
        "timestamp": ts,
        "user_input": user_input,
        "ai_response": ai_response,
        "intent": intent,
        "confidence": confidence,
        "metadata": payload.metadata or {},
    }
    res = db["messages"].insert_one(doc)
    return {"success": True, "id": str(res.inserted_id)}


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