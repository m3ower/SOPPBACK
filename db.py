"""
MongoDB connection helper for the Parking Assistant backend.

Uses environment variable `MONGODB_URI` if available, otherwise falls back
to the provided Atlas connection string. Provides a cached `get_db()` accessor
returning the `GigaHackDB` database and exposes typed collection helpers.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from pymongo import MongoClient, ASCENDING


DEFAULT_URI = (
	os.environ.get("MONGODB_URI")
	or "mongodb+srv://admin:admin@giga.djwcvxc.mongodb.net/?retryWrites=true&w=majority"
)
DB_NAME = os.environ.get("MONGODB_DB", "GigaHackDB")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
	client = MongoClient(
		DEFAULT_URI,
		serverSelectionTimeoutMS=2000,
		connectTimeoutMS=2000,
		socketTimeoutMS=2000,
		retryWrites=False,
	)
	# Trigger server selection early to fail fast if misconfigured
	try:
		client.admin.command("ping")
	except Exception:
		# Let it be lazy in non-critical paths; errors will surface on actual ops
		pass
	return client


def get_db():
	return get_client()[DB_NAME]


def get_collections() -> Dict[str, Any]:
	db = get_db()
	return {
		"users": db["users"],
		"vehicles": db["vehicles"],
		"gates": db["gates"],
		"prices": db["prices"],
		"parking_sessions": db["parking_sessions"],
		"ongoing_processes": db["ongoing_processes"],
		"messages": db["messages"],  # conversation messages
		"traces": db["traces"],      # algorithm traces/steps
		"queries": db["queries"],    # raw user queries
	}


def ensure_indexes() -> None:
	cols = get_collections()
	cols["users"].create_index([("email", ASCENDING)], unique=True, name="uniq_email")
	cols["vehicles"].create_index([("plate", ASCENDING)], unique=True, name="uniq_plate")
	cols["vehicles"].create_index([("user_id", ASCENDING)], name="idx_vehicle_user")
	cols["parking_sessions"].create_index([("vehicle_id", ASCENDING)], name="idx_session_vehicle")
	cols["parking_sessions"].create_index([("open", ASCENDING), ("started_at", ASCENDING)], name="idx_session_open_started")
	cols["parking_sessions"].create_index([("vehicle_id", ASCENDING), ("open", ASCENDING), ("paid", ASCENDING), ("ended_at", ASCENDING)], name="idx_session_paid_status")
	cols["parking_sessions"].create_index([("ticket_id", ASCENDING)], name="idx_session_ticket")
	cols["ongoing_processes"].create_index([("session_id", ASCENDING)], unique=True, name="uniq_ongoing_session")
	cols["ongoing_processes"].create_index([("ticket_id", ASCENDING)], name="idx_ongoing_ticket")
	cols["ongoing_processes"].create_index([("status", ASCENDING), ("started_at", ASCENDING)], name="idx_ongoing_status_started")
	cols["gates"].create_index([("name", ASCENDING)], unique=True, name="uniq_gate_name")
	cols["messages"].create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)], name="idx_msg_session_time")
	cols["traces"].create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)], name="idx_trace_session_time")
	cols["queries"].create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)], name="idx_query_session_time")


def ensure_validators() -> None:
	db = get_db()

	def _create_or_mod(name: str, schema: dict):
		if name in db.list_collection_names():
			try:
				db.command({
					"collMod": name,
					"validator": {"$jsonSchema": schema},
					"validationLevel": "moderate",
				})
			except Exception:
				# Some MongoDBs may not support collMod on Atlas free tier without existing validator; ignore
				pass
		else:
			db.create_collection(name, validator={"$jsonSchema": schema})

	# JSON Schemas
	users_schema = {
		"bsonType": "object",
		"required": ["name", "email"],
		"properties": {
			"name": {"bsonType": "string"},
			"email": {"bsonType": "string"},
			"phone": {"bsonType": ["string", "null"]},
		},
		"additionalProperties": True,
	}

	vehicles_schema = {
		"bsonType": "object",
		"required": ["plate"],
		"properties": {
			"user_id": {"bsonType": ["string", "null"]},
			"plate": {"bsonType": "string"},
			"color": {"bsonType": ["string", "null"]},
			"make": {"bsonType": ["string", "null"]},
			"model": {"bsonType": ["string", "null"]},
			"vehicle_type": {"bsonType": ["string", "null"]},
		},
		"additionalProperties": True,
	}

	gates_schema = {
		"bsonType": "object",
		"required": ["name"],
		"properties": {
			"name": {"bsonType": "string"},
			"location": {"bsonType": ["string", "null"]},
		},
		"additionalProperties": True,
	}

	prices_schema = {
		"bsonType": "object",
		"required": ["per_hour"],
		"properties": {
			"gate_id": {"bsonType": ["string", "null"]},
			"per_hour": {"bsonType": ["double", "int"]},
			"max_daily": {"bsonType": ["double", "int", "null"]},
			"per_hour_by_type": {"bsonType": ["object", "null"]},
		},
		"additionalProperties": True,
	}

	sessions_schema = {
		"bsonType": "object",
		"required": ["vehicle_id", "started_at", "open"],
		"properties": {
			"vehicle_id": {"bsonType": "string"},
			"gate_id": {"bsonType": ["string", "null"]},
			"started_at": {"bsonType": "date"},
			"ended_at": {"bsonType": ["date", "null"]},
			"planned_end_at": {"bsonType": ["date", "null"]},
			"open": {"bsonType": "bool"},
			"cost": {"bsonType": ["double", "int", "null"]},
			"paid": {"bsonType": ["bool", "null"]},
			"ticket_id": {"bsonType": ["string", "null"]},
			"qr_version": {"bsonType": ["string", "null"]},
			"qr_raw": {"bsonType": ["object", "null"]},
			"last_scanned_at": {"bsonType": ["date", "null"]},
		},
		"additionalProperties": True,
	}

	ongoing_schema = {
		"bsonType": "object",
		"required": ["session_id", "plate", "started_at", "status"],
		"properties": {
			"session_id": {"bsonType": "string"},
			"ticket_id": {"bsonType": ["string", "null"]},
			"vehicle_id": {"bsonType": ["string", "null"]},
			"plate": {"bsonType": "string"},
			"gate_id": {"bsonType": ["string", "null"]},
			"started_at": {"bsonType": "date"},
			"last_scanned_at": {"bsonType": ["date", "null"]},
			"qr_version": {"bsonType": ["string", "null"]},
			"qr_raw": {"bsonType": ["object", "null"]},
			"status": {"bsonType": "string"},  # ongoing | completed
			"created_at": {"bsonType": ["date", "null"]},
			"updated_at": {"bsonType": ["date", "null"]},
			"ended_at": {"bsonType": ["date", "null"]},
		},
		"additionalProperties": True,
	}

	messages_schema = {
		"bsonType": "object",
		"required": ["session_id", "user_input", "intent", "confidence", "timestamp"],
		"properties": {
			"session_id": {"bsonType": "string"},
			"user_input": {"bsonType": "string"},
			"intent": {"bsonType": "string"},
			"ai_response": {"bsonType": ["string", "null"]},
			"confidence": {"bsonType": ["double", "int"]},
			"timestamp": {"bsonType": "date"},
		},
		"additionalProperties": True,
	}

	traces_schema = {
		"bsonType": "object",
		"required": ["session_id", "step", "status", "timestamp"],
		"properties": {
			"session_id": {"bsonType": "string"},
			"step": {"bsonType": "string"},
			"status": {"bsonType": "string"},
			"details": {"bsonType": ["object", "array", "null"]},
			"confidence": {"bsonType": ["double", "int", "null"]},
			"timestamp": {"bsonType": "date"},
		},
		"additionalProperties": True,
	}

	queries_schema = {
		"bsonType": "object",
		"required": ["session_id", "text", "timestamp"],
		"properties": {
			"session_id": {"bsonType": "string"},
			"text": {"bsonType": "string"},
			"timestamp": {"bsonType": "date"},
		},
		"additionalProperties": True,
	}

	_create_or_mod("users", users_schema)
	_create_or_mod("vehicles", vehicles_schema)
	_create_or_mod("gates", gates_schema)
	_create_or_mod("prices", prices_schema)
	_create_or_mod("parking_sessions", sessions_schema)
	_create_or_mod("ongoing_processes", ongoing_schema)
	_create_or_mod("messages", messages_schema)
	_create_or_mod("traces", traces_schema)
	_create_or_mod("queries", queries_schema)


__all__ = [
	"get_client",
	"get_db",
	"get_collections",
	"ensure_indexes",
	"ensure_validators",
]

