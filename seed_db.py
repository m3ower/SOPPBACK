from __future__ import annotations

from datetime import datetime, timedelta
import random
import string

from db import get_db, ensure_indexes, ensure_validators


def seed():
	db = get_db()
	ensure_validators()
	ensure_indexes()

	# Clear existing (for dev/demo)
	for name in ["users", "vehicles", "gates", "prices", "parking_sessions", "messages", "traces", "queries"]:
		db[name].delete_many({})

	# Deterministic seed for reproducibility
	random.seed(42)

	# Users (generate 100 total; 4 base + 96 generated)
	base_users = [
		{"name": "Alice Smith", "email": "alice@example.com", "phone": "+123456789"},
		{"name": "Bob Johnson", "email": "bob@example.com", "phone": "+198765432"},
		{"name": "Carol Lee", "email": "carol@example.com", "phone": "+311111111"},
		{"name": "Dan Brown", "email": "dan@example.com", "phone": "+322222222"},
	]
	gen_users = []
	first_names = ["Eve","Frank","Grace","Heidi","Ivan","Judy","Karl","Liam","Mia","Nina","Omar","Pia","Quinn","Ruth","Seth","Tina","Uma","Vik","Wes","Xena","Yuri","Zoe"]
	last_names = ["Adams","Baker","Clark","Davis","Evans","Foster","Green","Hughes","Irwin","Jones","King","Lane","Moore","Ng","Owens","Price","Reed","Shaw","Turner","Usher","Voss","Wong","Xu","Young","Zimmer"]
	used_emails = set(u["email"] for u in base_users)
	for _ in range(96):
		fn = random.choice(first_names)
		ln = random.choice(last_names)
		email = f"{fn.lower()}.{ln.lower()}@example.com"
		# ensure unique
		while email in used_emails:
			email = f"{fn.lower()}.{ln.lower()}{random.randint(1,999)}@example.com"
		used_emails.add(email)
		gen_users.append({"name": f"{fn} {ln}", "email": email, "phone": f"+1{random.randint(2000000000, 9999999999)}"})

	users = base_users + gen_users
	user_ids = db["users"].insert_many(users).inserted_ids

	# Vehicles (preserve legacy plates + generate 196 new → 200 total)
	def rand_plate():
		# 3 letters + 3 digits or 2 letters + 4 digits (ensure digits present)
		pattern = random.choice(["LLLDDD","LLDDDD"])  # L=letter D=digit
		s = []
		for ch in pattern:
			if ch == "L":
				s.append(random.choice(string.ascii_uppercase))
			else:
				s.append(str(random.randint(0,9)))
		return "".join(s)

	vehicle_types = ["car","motorcycle","truck","rv"]
	colors = ["Blue","Red","Black","White","Grey","Silver","Green"]
	makes = ["Toyota","Honda","Ford","VW","BMW","Kia","Tesla"]
	models = ["Corolla","Civic","Focus","Golf","3 Series","Soul","Model 3","Camry","Accord","F-150","Tiguan"]

	vehicles = [
		{"user_id": str(user_ids[0]), "plate": "ABC123", "color": "Blue", "make": "Toyota", "model": "Corolla", "vehicle_type": "car"},
		{"user_id": str(user_ids[1]), "plate": "XYZ789", "color": "Red", "make": "Honda", "model": "Civic", "vehicle_type": "motorcycle"},
		{"user_id": str(user_ids[2]), "plate": "LMN456", "color": "Black", "make": "Ford", "model": "Focus", "vehicle_type": "truck"},
		{"user_id": str(user_ids[3]), "plate": "JKL321", "color": "White", "make": "VW", "model": "Golf", "vehicle_type": "car"},
	]
	used_plates = set(v["plate"] for v in vehicles)
	for i in range(196):
		plate = rand_plate()
		while plate in used_plates:
			plate = rand_plate()
		used_plates.add(plate)
		owner = str(random.choice(user_ids))
		vehicles.append({
			"user_id": owner,
			"plate": plate,
			"color": random.choice(colors),
			"make": random.choice(makes),
			"model": random.choice(models),
			"vehicle_type": random.choice(vehicle_types),
		})
	vehicle_ids = db["vehicles"].insert_many(vehicles).inserted_ids

	# Gates
	gates = [
		{"name": "Main Garage", "location": "123 Center St"},
		{"name": "West Lot", "location": "45 West Ave"},
		{"name": "North Tower", "location": "77 North Rd"},
	]
	gate_ids = db["gates"].insert_many(gates).inserted_ids

	# Prices
	prices = [
		{"gate_id": str(gate_ids[0]), "per_hour": 3.0, "max_daily": 12.0, "per_hour_by_type": {"motorcycle": 1.5, "truck": 5.0, "rv": 6.0, "car": 3.0}},
		{"gate_id": str(gate_ids[1]), "per_hour": 2.5, "max_daily": 10.0, "per_hour_by_type": {"motorcycle": 1.2, "truck": 4.0, "rv": 5.0, "car": 2.5}},
		{"gate_id": str(gate_ids[2]), "per_hour": 4.0, "max_daily": 15.0, "per_hour_by_type": {"motorcycle": 2.0, "truck": 6.0, "rv": 7.0, "car": 4.0}},
	]
	db["prices"].insert_many(prices)

	# Sessions: generate a rich mix
	now = datetime.utcnow()
	sessions = []
	# Keep a few baseline sessions (matching earlier behavior)
	sessions.extend([
		{"vehicle_id": str(vehicle_ids[0]), "gate_id": str(gate_ids[0]), "started_at": now - timedelta(hours=2, minutes=15), "open": True, "cost": None, "paid": False},
		{"vehicle_id": str(vehicle_ids[1]), "gate_id": str(gate_ids[1]), "started_at": now - timedelta(hours=1, minutes=30), "ended_at": now - timedelta(minutes=10), "open": False, "cost": 3.75, "paid": True},
		{"vehicle_id": str(vehicle_ids[2]), "gate_id": str(gate_ids[2]), "started_at": now - timedelta(minutes=40), "open": True, "cost": None, "paid": False},
		{"vehicle_id": str(vehicle_ids[3]), "gate_id": str(gate_ids[0]), "started_at": now - timedelta(hours=5), "ended_at": now - timedelta(hours=1), "open": False, "cost": 10.0, "paid": True},
	])

	# Build cost helper using tiered pricing
	price_map = {p["gate_id"]: p for p in prices}
	def compute_cost(gate_id: str, vehicle_type: str, minutes: int) -> float:
		base = 3.0
		per_hour = base
		pr = price_map.get(gate_id)
		if pr:
			per_hour = float(pr.get("per_hour", base))
			tiers = pr.get("per_hour_by_type") or {}
			if vehicle_type in tiers:
				per_hour = float(tiers[vehicle_type])
			max_daily = pr.get("max_daily")
		else:
			max_daily = None
		cost = round(per_hour * (minutes / 60.0), 2)
		if max_daily is not None:
			cost = min(cost, float(max_daily))
		return cost

	# Richer sessions: for vehicles after the first 4 (baseline),
	# create one closed session, maybe a second older closed, and maybe one open.
	vehicle_by_id = {str(vehicle_ids[i]): vehicles[i] for i in range(len(vehicle_ids))}
	for vid in vehicle_ids[4:]:
		vid_str = str(vid)
		vinfo = vehicle_by_id.get(vid_str, {})
		vtype = vinfo.get("vehicle_type", "car")

		# Always one closed session (duration up to 24h to hit max_daily sometimes)
		gate_oid = str(random.choice(gate_ids))
		mins = random.randint(15, 24*60)
		start_time = now - timedelta(minutes=mins + random.randint(5, 240))
		ended_at = start_time + timedelta(minutes=mins)
		cost = compute_cost(str(gate_oid), vtype, mins)
		is_paid = random.random() < 0.6
		sessions.append({
			"vehicle_id": vid_str,
			"gate_id": str(gate_oid),
			"started_at": start_time,
			"ended_at": ended_at,
			"open": False,
			"cost": cost,
			"paid": is_paid,
		})

		# Maybe a second older closed session
		if random.random() < 0.5:
			gate_oid2 = str(random.choice(gate_ids))
			mins2 = random.randint(30, 18*60)
			start_time2 = start_time - timedelta(minutes=mins2 + random.randint(60, 6*60))
			ended_at2 = start_time2 + timedelta(minutes=mins2)
			cost2 = compute_cost(str(gate_oid2), vtype, mins2)
			is_paid2 = random.random() < 0.7
			sessions.append({
				"vehicle_id": vid_str,
				"gate_id": str(gate_oid2),
				"started_at": start_time2,
				"ended_at": ended_at2,
				"open": False,
				"cost": cost2,
				"paid": is_paid2,
			})

		# Maybe one current open session
		if random.random() < 0.35:
			gate_oid3 = str(random.choice(gate_ids))
			mins3 = random.randint(5, 8*60)
			start_time3 = now - timedelta(minutes=mins3)
			# open sessions accrue; cost optional for display
			cost3 = None if random.random() < 0.8 else compute_cost(str(gate_oid3), vtype, mins3)
			is_paid3 = False  # typically unpaid
			sessions.append({
				"vehicle_id": vid_str,
				"gate_id": str(gate_oid3),
				"started_at": start_time3,
				"ended_at": None,
				"open": True,
				"cost": cost3,
				"paid": is_paid3,
			})

	db["parking_sessions"].insert_many(sessions)

	print("Seed complete.")


if __name__ == "__main__":
	seed()

