"""
Pydantic models and enums for Parking Assistant domain.

These help serialize/validate documents exchanged via FastAPI and stored in MongoDB.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
	lost_ticket = "lost_ticket"
	payment = "payment"
	find_car = "find_car"
	extend_time = "extend_time"
	emergency = "emergency"
	barrier_control = "barrier_control"
	greeting = "greeting"
	gratitude = "gratitude"
	general_help = "general_help"
	price_inquiry = "price_inquiry"
	start_session = "start_session"
	end_session = "end_session"
	open_gate = "open_gate"
	unknown = "unknown"
	# New intents
	validate_plate_and_pay = "validate_plate_and_pay"
	check_payment_status = "check_payment_status"
	smart_plate_detection = "smart_plate_detection"
	classify_vehicle_and_price = "classify_vehicle_and_price"
	handle_plate_reading_errors = "handle_plate_reading_errors"


class TraceStatus(str, Enum):
	started = "started"
	decision = "decision"
	db_query = "db_query"
	success = "success"
	error = "error"


class MongoBase(BaseModel):
	class Config:
		arbitrary_types_allowed = True


class User(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	name: str
	email: str
	phone: Optional[str] = None


class Vehicle(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	user_id: Optional[str] = None
	plate: str
	color: Optional[str] = None
	make: Optional[str] = None
	model: Optional[str] = None
	vehicle_type: Optional[str] = None  # car, motorcycle, truck, rv


class Gate(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	name: str
	location: Optional[str] = None


class Price(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	gate_id: Optional[str] = None
	per_hour: float
	max_daily: Optional[float] = None
	# Optional tiered pricing by vehicle type
	per_hour_by_type: Optional[Dict[str, float]] = None


class ParkingSession(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	vehicle_id: str
	gate_id: Optional[str] = None
	started_at: datetime
	ended_at: Optional[datetime] = None
	planned_end_at: Optional[datetime] = None
	open: bool = True
	cost: Optional[float] = None
	paid: bool = False
	ticket_id: Optional[str] = None
	qr_version: Optional[str] = None
	qr_raw: Optional[dict] = None
	last_scanned_at: Optional[datetime] = None


class QueryLog(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	session_id: str
	text: str
	timestamp: datetime


class MessageLog(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	session_id: str
	user_input: str
	intent: str
	ai_response: Optional[str] = None
	confidence: float
	timestamp: datetime


class TraceLog(MongoBase):
	id: Optional[str] = Field(alias="_id", default=None)
	session_id: str
	step: str
	status: TraceStatus
	details: Dict[str, Any] = {}
	confidence: Optional[float] = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)


__all__ = [
	"Intent",
	"TraceStatus",
	"User",
	"Vehicle",
	"Gate",
	"Price",
	"ParkingSession",
	"QueryLog",
	"MessageLog",
	"TraceLog",
]

