from __future__ import annotations

import os
from typing import Dict

from pymongo import MongoClient
from pymongo.server_api import ServerApi


# Connection details (can be overridden via env vars)
# Provided by user: mongodb+srv://admin:admin@giga.djwcvxc.mongodb.net/
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://admin:admin@giga.djwcvxc.mongodb.net/")
DB_NAME = os.getenv("MONGO_DB", "GigaHackDB")

_client: MongoClient | None = None


def get_client() -> MongoClient:
	global _client
	if _client is None:
		_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
		# Best-effort ping to validate connection; ignore failure so app still starts
		try:
			_client.admin.command("ping")
		except Exception:
			pass
	return _client


def get_db():
	return get_client()[DB_NAME]


def get_collections() -> Dict[str, any]:
    db = get_db()
    return {
        "vehicles": db["vehicles"],
        "gates": db["gates"],
        "prices": db["prices"],
        "parking_sessions": db["parking_sessions"],
        "ongoing_processes": db["ongoing_processes"],
        "messages": db["messages"],
        "traces": db["traces"],
        "barrier_logs": db["barrier_logs"],
        "anpr_readings": db["anpr_readings"],
    }
def ensure_indexes() -> None:
	cols = get_collections()
	# Vehicles: plate unique
	try:
		cols["vehicles"].create_index("plate", unique=True, name="uniq_plate")
	except Exception:
		pass
	# Gates: name unique
	try:
		cols["gates"].create_index("name", unique=True, name="uniq_gate_name")
	except Exception:
		pass
	# Ongoing processes: by ticket/open and vehicle/open
	try:
		cols["ongoing_processes"].create_index([("ticket_id", 1), ("open", 1)], name="ticket_open")
		cols["ongoing_processes"].create_index([("vehicle_id", 1), ("open", 1)], name="vehicle_open")
		cols["ongoing_processes"].create_index("last_scanned_at", name="last_scanned_at")
	except Exception:
		pass
	# Sessions: by ticket/open and vehicle/open
	try:
		cols["parking_sessions"].create_index([("ticket_id", 1), ("open", 1)], name="session_ticket_open")
		cols["parking_sessions"].create_index([("vehicle_id", 1), ("open", 1)], name="session_vehicle_open")
		cols["parking_sessions"].create_index("started_at", name="started_at")
	except Exception:
		pass
	# Prices: by gate
	try:
		cols["prices"].create_index("gate_id", name="gate_id")
	except Exception:
		pass
	# Messages & traces (for history endpoint)
	try:
		cols["messages"].create_index([("session_id", 1), ("timestamp", 1)], name="msg_session_ts")
		cols["traces"].create_index([("session_id", 1), ("timestamp", 1)], name="trace_session_ts")
	except Exception:
		pass
	# Barrier logs and ANPR readings
	try:
		cols["barrier_logs"].create_index([("gate_name", 1), ("timestamp", -1)], name="barrier_gate_ts")
		cols["anpr_readings"].create_index([("plate", 1), ("timestamp", -1)], name="anpr_plate_ts")
		cols["anpr_readings"].create_index([("gate_name", 1), ("timestamp", -1)], name="anpr_gate_ts")
	except Exception:
		pass


def ensure_validators() -> None:
	# Keep as no-op for Atlas unless explicit JSON schema desired.
	# Validators can be added here with runCommand if needed.
	return None

