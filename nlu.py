from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from pymongo.database import Database
from bson import ObjectId

from db import get_collections
from models import Intent, TraceStatus


# Require at least one digit to avoid matching words like 'PAYMENT'; allow common plate formats
PLATE_REGEX = re.compile(r"\b([A-Z]{1,3}\d{1,4}[A-Z]{0,2}|(?=[A-Z0-9]{5,8}\b)[A-Z0-9]*\d+[A-Z0-9]*)\b", re.I)
TIME_REGEX = re.compile(r"\b(\d{1,2})\s*(minutes?|mins?|hours?|hrs?)\b", re.I)


class NLU:
	"""
	Simple rule-based NLU for domain intents with confidence scores and entity extraction.
	"""

	DOMAIN_KEYWORDS = {
		Intent.lost_ticket: ["lost", "missing", "ticket"],
		Intent.payment: ["pay", "payment", "price", "charge", "bill", "cost"],
		Intent.find_car: ["find", "locate", "where", "car", "vehicle", "parked"],
		Intent.extend_time: ["extend", "more time", "extra", "add time"],
		Intent.emergency: ["emergency", "urgent", "stuck", "trapped"],
		Intent.barrier_control: ["barrier", "gate", "open", "close"],
		Intent.price_inquiry: ["price", "how much", "rate", "per hour"],
		Intent.start_session: ["start", "arrived", "enter"],
		Intent.end_session: ["end", "exit", "leave", "finish"],
		Intent.open_gate: ["open gate", "open the gate", "open barrier"],
		Intent.general_help: ["help", "assist", "support"],
		Intent.greeting: ["hello", "hi", "hey", "morning", "afternoon"],
		Intent.gratitude: ["thanks", "thank you", "appreciate"],
		# New intents
		Intent.validate_plate_and_pay: [
			"validate plate",
			"validate my plate",
			"confirm plate",
			"payment options",
			"pay with card",
			"mobile payment",
			"account billing",
		],
		Intent.check_payment_status: [
			"paid already",
			"already paid",
			"payment status",
			"check payment",
			"pre-paid",
			"prepaid",
		],
		Intent.smart_plate_detection: [
			"scan plate",
			"read plate",
			"license recognition",
			"plate recognition",
			"alpr",
		],
		Intent.classify_vehicle_and_price: [
			"vehicle type",
			"motorcycle",
			"bike",
			"truck",
			"rv",
			"pricing tier",
			"vehicle pricing",
		],
		Intent.handle_plate_reading_errors: [
			"unreadable plate",
			"can't read",
			"cannot read",
			"blurry",
			"obscured",
			"damaged plate",
		],
	}

	def classify(self, text: str) -> Tuple[Intent, float, Dict[str, Any]]:
		t = text.lower()
		scores: Dict[Intent, float] = {i: 0.0 for i in self.DOMAIN_KEYWORDS}

		for intent, kws in self.DOMAIN_KEYWORDS.items():
			for kw in kws:
				if kw in t:
					scores[intent] += 1.0

		# Entities boost
		entities: Dict[str, Any] = {}
		plate = self.extract_plate(text)
		if plate:
			entities["plate"] = plate
			scores[Intent.find_car] += 0.5
			scores[Intent.start_session] += 0.2
			scores[Intent.end_session] += 0.2
			scores[Intent.payment] += 0.2
			scores[Intent.validate_plate_and_pay] += 0.4

		duration = self.extract_duration(text)
		if duration:
			entities["duration_minutes"] = duration
			scores[Intent.extend_time] += 0.7

		# Vehicle type extraction (simple)
		vehicle_type = self.extract_vehicle_type(text)
		if vehicle_type:
			entities["vehicle_type"] = vehicle_type
			scores[Intent.classify_vehicle_and_price] += 0.8

		# Context flags for smart detection
		context_flags = self.extract_context_flags(text)
		if context_flags:
			entities["context"] = context_flags
			scores[Intent.smart_plate_detection] += 0.5

		# Payment method extraction (advisory)
		payment_method = self.extract_payment_method(text)
		if payment_method:
			entities["payment_method"] = payment_method
			scores[Intent.validate_plate_and_pay] += 0.3

		# Decide best intent
		best_intent = max(scores, key=scores.get)
		best_score = scores[best_intent]

		# Confidence normalization (0..1)
		confidence = min(best_score / 3.0, 1.0)

		# If no meaningful keywords, treat as unknown/small talk
		if best_score == 0:
			return Intent.unknown, 0.2, entities

		return best_intent, confidence, entities

	@staticmethod
	def extract_plate(text: str) -> Optional[str]:
		# Gather all candidates that contain at least one digit
		cands = []
		for m in PLATE_REGEX.finditer(text):
			s = m.group(1)
			if any(ch.isdigit() for ch in s):
				cands.append((m.start(), s.upper()))
		if not cands:
			return None
		# Prefer the plate closest to the keyword 'plate' or 'license'
		t = text.lower()
		pivot_positions = [t.find("plate"), t.find("license"), t.find("licence")]
		pivot_positions = [p for p in pivot_positions if p != -1]
		if pivot_positions:
			pivot = min(pivot_positions)
			best = min(cands, key=lambda x: abs(x[0] - pivot))
			return best[1]
		# Otherwise return the first candidate with digits
		return cands[0][1]

	@staticmethod
	def extract_duration(text: str) -> Optional[int]:
		m = TIME_REGEX.search(text)
		if not m:
			return None
		val = int(m.group(1))
		unit = m.group(2).lower()
		if unit.startswith("hour") or unit.startswith("hr"):
			return val * 60
		return val

	@staticmethod
	def extract_vehicle_type(text: str) -> Optional[str]:
		t = text.lower()
		if "motorcycle" in t or "bike" in t:
			return "motorcycle"
		if "truck" in t:
			return "truck"
		if "rv" in t:
			return "rv"
		# Avoid labeling generic 'car' to reduce false positives
		return None

	@staticmethod
	def extract_context_flags(text: str) -> Dict[str, bool]:
		t = text.lower()
		flags = {
			"at_gate": any(kw in t for kw in ["at gate", "approaching gate", "near gate", "gate camera", "barrier"]),
			"in_zone": any(kw in t for kw in ["scanning zone", "scan zone", "designated zone", "marked zone"]),
			"good_lighting": any(kw in t for kw in ["good lighting", "bright", "well lit"]) and not any(kw in t for kw in ["dark", "low light", "poor lighting"]),
		}
		return {k: v for k, v in flags.items() if v}

	@staticmethod
	def extract_payment_method(text: str) -> Optional[str]:
		t = text.lower()
		if "credit card" in t or "card" in t:
			return "credit_card"
		if "mobile" in t or "apple pay" in t or "google pay" in t or "wallet" in t:
			return "mobile_payment"
		if "account" in t or "bill my account" in t:
			return "account_billing"
		return None


def log_trace(db: Database, session_id: str, step: str, status: TraceStatus, details: Dict[str, Any], confidence: Optional[float] = None):
	db["traces"].insert_one({
		"session_id": session_id,
		"step": step,
		"status": status.value,
		"details": details,
		"confidence": confidence,
		"timestamp": datetime.utcnow(),
	})


def handle_intent(db: Database, session_id: str, intent: Intent, entities: Dict[str, Any], confidence: Optional[float] = None) -> str:
	cols = get_collections()
	now = datetime.utcnow()
	log_trace(db, session_id, "handle_intent", TraceStatus.started, {"intent": intent.value, "entities": entities})

	# Payment / end session: compute cost and close session (mark paid)
	if intent in (Intent.payment, Intent.end_session, Intent.validate_plate_and_pay):
		if intent == Intent.validate_plate_and_pay and (confidence or 0) < 0.5:
			log_trace(db, session_id, "validate_pay_low_confidence", TraceStatus.decision, {"confidence": confidence or 0})
			return "Please confirm: do you want to validate your plate and pay now?"
		plate = entities.get("plate")
		if not plate:
			log_trace(db, session_id, "payment_missing_plate", TraceStatus.error, {})
			# For validate_plate_and_pay, allow prompting with options after plate provided
			return "To process payment, please provide your license plate number."

		vehicle = cols["vehicles"].find_one({"plate": plate})
		if not vehicle:
			log_trace(db, session_id, "payment_vehicle_not_found", TraceStatus.error, {"plate": plate})
			return f"I couldn't find vehicle with plate {plate}. You can enter it manually or contact an attendant."

		ps = cols["parking_sessions"].find_one({"vehicle_id": str(vehicle.get("_id")), "open": True}, sort=[("started_at", -1)])
		if not ps:
			log_trace(db, session_id, "payment_session_not_found", TraceStatus.error, {"vehicle_id": str(vehicle.get("_id"))})
			return "I couldn't find an active parking session for your vehicle."

		# Determine gate price
		gate_id = ps.get("gate_id")
		price = cols["prices"].find_one({"gate_id": gate_id}) if gate_id else cols["prices"].find_one({})
		per_hour = (price or {}).get("per_hour", 3.0)
		# Tiered pricing by vehicle type
		vtype = vehicle.get("vehicle_type")
		if price and price.get("per_hour_by_type") and vtype and vtype in price["per_hour_by_type"]:
			per_hour = float(price["per_hour_by_type"][vtype])
		started_at = ps["started_at"]
		minutes = max(1, int((now - started_at).total_seconds() // 60))
		cost = round(per_hour * (minutes / 60.0), 2)
		if price and price.get("max_daily"):
			cost = min(cost, float(price["max_daily"]))

		# For validate_plate_and_pay, present payment methods first; only close if method chosen
		if intent == Intent.validate_plate_and_pay:
			method = entities.get("payment_method")
			if not method:
				log_trace(db, session_id, "payment_options_presented", TraceStatus.decision, {"amount_due": cost, "per_hour": per_hour})
				choices = "credit card, mobile payment, or account billing"
				return f"Plate {plate} validated. Amount due: €{cost:.2f}. Choose a payment method: {choices}."
			# Close and mark paid with selected method
			cols["parking_sessions"].update_one({"_id": ps["_id"]}, {"$set": {"open": False, "ended_at": now, "cost": cost, "paid": True}})
			log_trace(db, session_id, "payment_completed", TraceStatus.success, {"minutes": minutes, "cost": cost, "per_hour": per_hour, "paid": True, "method": method})
			return f"Plate {plate} validated. Paid €{cost:.2f} via {method.replace('_',' ')}."

		# Default payment/end_session: close and mark paid
		cols["parking_sessions"].update_one({"_id": ps["_id"]}, {"$set": {"open": False, "ended_at": now, "cost": cost, "paid": True}})
		log_trace(db, session_id, "payment_completed", TraceStatus.success, {"minutes": minutes, "cost": cost, "per_hour": per_hour, "paid": True})
		return f"Your parking session has been closed. Total time: {minutes} minutes. Amount due: €{cost:.2f}."

	# Start session
	if intent == Intent.start_session:
		plate = entities.get("plate")
		if not plate:
			log_trace(db, session_id, "start_missing_plate", TraceStatus.error, {})
			return "To start a session, please provide your license plate number."
		vehicle = cols["vehicles"].find_one({"plate": plate})
		if not vehicle:
			# Auto-register minimal vehicle
			user = cols["users"].find_one({})
			v_id = cols["vehicles"].insert_one({"plate": plate, "user_id": str(user["_id"]) if user else None}).inserted_id
			vehicle = cols["vehicles"].find_one({"_id": v_id})
			log_trace(db, session_id, "vehicle_autoregistered", TraceStatus.success, {"plate": plate})
		# choose a default gate
		gate = cols["gates"].find_one({})
		ps = {
			"vehicle_id": str(vehicle["_id"]),
			"gate_id": str(gate["_id"]) if gate else None,
			"started_at": now,
			"open": True,
			"cost": None,
		}
		cols["parking_sessions"].insert_one(ps)
		log_trace(db, session_id, "session_started", TraceStatus.success, {"plate": plate, "gate": (gate or {}).get("name")})
		return f"Started a new parking session for plate {plate}. Enjoy your stay!"

	# Find car
	if intent == Intent.find_car:
		plate = entities.get("plate")
		if not plate:
			return "To locate your car, please provide your license plate number."
		vehicle = cols["vehicles"].find_one({"plate": plate})
		if not vehicle:
			return f"I couldn't find vehicle with plate {plate}."
		ps = cols["parking_sessions"].find_one({"vehicle_id": str(vehicle.get("_id"))}, sort=[("started_at", -1)])
		if not ps:
			return "I couldn't find any recent sessions for your vehicle."
		gate = None
		gid = ps.get("gate_id")
		if gid:
			try:
				gate = cols["gates"].find_one({"_id": ObjectId(gid)})
			except Exception:
				gate = None
		gate_name = (gate or {}).get("name", "Main Garage")
		started = ps.get("started_at")
		log_trace(db, session_id, "find_car_result", TraceStatus.success, {"plate": plate, "gate": gate_name})
		return f"Your car was last seen at {gate_name}. Session started at {started.strftime('%H:%M')}."

	# Extend time (keep session open; advisory)
	if intent == Intent.extend_time:
		plate = entities.get("plate")
		minutes = entities.get("duration_minutes") or 60
		if not plate:
			return "To extend your time, please provide your license plate number."
		vehicle = cols["vehicles"].find_one({"plate": plate})
		if not vehicle:
			return f"I couldn't find vehicle with plate {plate}."
		ps = cols["parking_sessions"].find_one({"vehicle_id": str(vehicle.get("_id")), "open": True}, sort=[("started_at", -1)])
		if not ps:
			return "I couldn't find an active session to extend."
		# Log extension request (no actual timer in DB, just advisory)
		log_trace(db, session_id, "extend_time", TraceStatus.success, {"plate": plate, "added_minutes": minutes})
		return f"I've noted an extension of {minutes} minutes for plate {plate}. Charges continue to accrue while parked."

	# Lost ticket
	if intent == Intent.lost_ticket:
		log_trace(db, session_id, "lost_ticket", TraceStatus.decision, {})
		return "No worries. We can verify your stay using your license plate and entry time. Please provide your plate number."

	# Price inquiry
	if intent == Intent.price_inquiry:
		price = cols["prices"].find_one({})
		if not price:
			return "Standard rate is €3.00 per hour."
		resp = f"Rates: €{price['per_hour']:.2f}/hour"
		if price.get("max_daily"):
			resp += f", max €{price['max_daily']:.2f} per day"
		log_trace(db, session_id, "price_inquiry", TraceStatus.success, {"per_hour": price["per_hour"], "max_daily": price.get("max_daily")})
		return resp + "."

	# Gate control (simulated)
	if intent in (Intent.barrier_control, Intent.open_gate):
		log_trace(db, session_id, "open_gate", TraceStatus.success, {})
		return "I've signaled the barrier to open. Please proceed cautiously."

	# Check payment status before opening gate
	if intent == Intent.check_payment_status:
		if (confidence or 0) < 0.5:
			log_trace(db, session_id, "check_status_low_confidence", TraceStatus.decision, {"confidence": confidence or 0})
			return "Please confirm you want to check payment status (e.g., 'check payment status for plate ABC123')."
		plate = entities.get("plate")
		if not plate:
			log_trace(db, session_id, "check_status_missing_plate", TraceStatus.error, {})
			return "To check payment status, please provide your license plate number."
		vehicle = cols["vehicles"].find_one({"plate": plate})
		if not vehicle:
			log_trace(db, session_id, "check_status_vehicle_not_found", TraceStatus.error, {"plate": plate})
			return f"I couldn't find vehicle with plate {plate}."
		ps = cols["parking_sessions"].find_one({"vehicle_id": str(vehicle.get("_id")), "open": True}, sort=[("started_at", -1)])
		if ps and ps.get("paid"):
			log_trace(db, session_id, "payment_confirmed_open_gate", TraceStatus.success, {"plate": plate})
			return "Payment confirmed. Opening gate now."
		if ps and not ps.get("paid"):
			log_trace(db, session_id, "payment_required", TraceStatus.decision, {"plate": plate})
			return "Payment required before exit. Please complete payment at the kiosk or via mobile to open the gate."
		# No open session; check if a recently closed paid session exists (grace window 2h)
		recent = cols["parking_sessions"].find_one({"vehicle_id": str(vehicle.get("_id")), "open": False, "paid": True}, sort=[("ended_at", -1)])
		if recent and recent.get("ended_at") and (now - recent["ended_at"]).total_seconds() <= 2 * 3600:
			log_trace(db, session_id, "recent_paid_session_open_gate", TraceStatus.success, {"plate": plate})
			return "Recent payment detected. Opening gate."
		return "No valid payment found for the current session. Please proceed to payment."

	# Smart plate detection based on context flags
	if intent == Intent.smart_plate_detection:
		ctx = entities.get("context", {}) if isinstance(entities.get("context"), dict) else {}
		at_gate = ctx.get("at_gate", False)
		in_zone = ctx.get("in_zone", False)
		good_light = ctx.get("good_lighting", False)
		# Confidence threshold: require decent signal for this control intent
		if (confidence or 0) < 0.4:
			log_trace(db, session_id, "smart_detection_low_confidence", TraceStatus.decision, {"confidence": confidence or 0})
			return "I’m not fully sure you want a plate scan. Say 'scan plate at gate' to confirm."
		if not (at_gate and in_zone and good_light):
			log_trace(db, session_id, "smart_detection_not_appropriate", TraceStatus.decision, {"context": ctx})
			return "Not attempting plate read yet. Please position at the gate scanning zone with adequate lighting."
		log_trace(db, session_id, "smart_detection_begin", TraceStatus.started, {"context": ctx})
		return "Initiating plate scan. Please hold steady for a moment."

	# Vehicle classification and pricing tier
	if intent == Intent.classify_vehicle_and_price:
		if (confidence or 0) < 0.5:
			log_trace(db, session_id, "classify_low_confidence", TraceStatus.decision, {"confidence": confidence or 0})
			# Continue but ask clarification if type missing
			pass
		vtype = entities.get("vehicle_type")
		if not vtype:
			return "What type of vehicle is this? (motorcycle, truck, RV)"
		price = cols["prices"].find_one({})
		per_hour = (price or {}).get("per_hour", 3.0)
		if price and price.get("per_hour_by_type") and vtype in price["per_hour_by_type"]:
			per_hour = float(price["per_hour_by_type"][vtype])
		log_trace(db, session_id, "vehicle_classified_pricing", TraceStatus.success, {"vehicle_type": vtype, "per_hour": per_hour})
		return f"Detected {vtype}. Rate is €{per_hour:.2f} per hour."

	# Exception handling for unreadable plates
	if intent == Intent.handle_plate_reading_errors:
		if (confidence or 0) < 0.4:
			log_trace(db, session_id, "plate_error_low_confidence", TraceStatus.decision, {"confidence": confidence or 0})
			return "Did the camera fail to read the plate? If so, you can: enter plate manually, reposition, or call an attendant."
		log_trace(db, session_id, "plate_read_error", TraceStatus.decision, {})
		return "Plate unreadable. Options: 1) Manual plate entry, 2) Reposition vehicle for a clearer view, 3) Request attendant assistance."

	# Emergency
	if intent == Intent.emergency:
		log_trace(db, session_id, "emergency", TraceStatus.started, {})
		return "Emergency support notified. Please stay where you are; help is on the way."

	# Greetings / gratitude: concise, parking-only
	if intent == Intent.greeting:
		return "Hi! I can help with parking: payment, finding your car, time extension, or gate access."
	if intent == Intent.gratitude:
		return "You’re welcome. Need any other parking help?"

	# General help or unknown
	return "I can help with parking sessions, payments, finding your car, and gate access. How can I assist?"

