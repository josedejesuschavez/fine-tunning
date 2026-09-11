"""Genera un dataset sintetico de rate confirmations -> JSON estructurado."""

import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

BROKERS = [
    "TQL", "C.H. Robinson", "Landstar", "Coyote Logistics", "Echo Global",
    "RXO", "Uber Freight", "Arrive Logistics", "Worldwide Express", "Convoy",
    "Nolan Transportation", "England Logistics", "Allen Lund Company",
]

CITIES = [
    ("Midland", "TX"), ("Odessa", "TX"), ("Hobbs", "NM"), ("Roswell", "NM"),
    ("Carlsbad", "NM"), ("Houston", "TX"), ("Dallas", "TX"), ("San Antonio", "TX"),
    ("Oklahoma City", "OK"), ("Denver", "CO"), ("Phoenix", "AZ"), ("Amarillo", "TX"),
    ("Lubbock", "TX"), ("El Paso", "TX"), ("Tulsa", "OK"), ("Wichita", "KS"),
    ("Albuquerque", "NM"), ("Fort Worth", "TX"), ("Laredo", "TX"), ("Memphis", "TN"),
]

EQUIPMENT = ["Flatbed", "Dry Van", "Step Deck", "Reefer", "Hotshot"]
COMMODITIES = ["Pipe", "Drilling Equipment", "Frac Sand", "Valves", "General Freight",
               "Steel Coils", "Machinery", "Casing"]


def rand_date():
    d = date(2026, 1, 1) + timedelta(days=random.randint(0, 364))
    return d


def make_record():
    broker = random.choice(BROKERS)
    origin_city, origin_st = random.choice(CITIES)
    dest_city, dest_st = random.choice(CITIES)
    while (dest_city, dest_st) == (origin_city, origin_st):
        dest_city, dest_st = random.choice(CITIES)
    return {
        "broker": broker,
        "load_number": random.choice(["", "L", "BOL-", "#"]) + str(random.randint(100000, 9999999)),
        "origin": f"{origin_city}, {origin_st}",
        "destination": f"{dest_city}, {dest_st}",
        "pickup_date": rand_date().isoformat(),
        "rate": round(random.uniform(650, 4800), 2),
        "equipment": random.choice(EQUIPMENT),
        "commodity": random.choice(COMMODITIES),
    }


def fmt_date(iso, style):
    y, m, d = iso.split("-")
    if style == 0:
        return f"{m}/{d}/{y}"
    if style == 1:
        months = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]
        return f"{months[int(m) - 1]} {int(d)}, {y}"
    return f"{int(m)}-{int(d)}-{y[2:]}"


TEMPLATES = [
    # Formato tabular clasico
    lambda r, ds: (
        f"RATE CONFIRMATION\n"
        f"{r['broker']}\n"
        f"Load #: {r['load_number']}\n"
        f"Equipment: {r['equipment']}\n"
        f"Commodity: {r['commodity']}\n"
        f"PU: {r['origin']} on {fmt_date(r['pickup_date'], ds)}\n"
        f"DEL: {r['destination']}\n"
        f"Total Rate: ${r['rate']:,.2f}\n"
        f"Please sign and return."
    ),
    # Formato email informal
    lambda r, ds: (
        f"Hi, confirming the load below with {r['broker']}.\n\n"
        f"Reference {r['load_number']}. We need a {r['equipment'].lower()} picking up "
        f"{r['commodity'].lower()} in {r['origin']} on {fmt_date(r['pickup_date'], ds)}, "
        f"delivering to {r['destination']}.\n"
        f"Agreed rate is ${r['rate']:,.2f} all in.\n\n"
        f"Thanks"
    ),
    # Formato de campos etiquetados desordenado
    lambda r, ds: (
        f"CARRIER RATE AGREEMENT - {r['broker']}\n"
        f"ORDER NUMBER {r['load_number']}  |  TRAILER TYPE: {r['equipment'].upper()}\n"
        f"SHIPPER ....... {r['origin']}\n"
        f"PICKUP DATE ... {fmt_date(r['pickup_date'], ds)}\n"
        f"CONSIGNEE ..... {r['destination']}\n"
        f"COMMODITY ..... {r['commodity']}\n"
        f"LINEHAUL ...... ${r['rate']:,.2f}\n"
    ),
    # Formato denso de una sola linea
    lambda r, ds: (
        f"{r['broker']} rate con {r['load_number']}: {r['origin']} -> {r['destination']}, "
        f"pu {fmt_date(r['pickup_date'], ds)}, {r['equipment']}, {r['commodity']}, "
        f"${r['rate']:,.2f}."
    ),
]

INSTRUCTION = (
    "Extract the load details from the rate confirmation below and return them as JSON "
    "with the keys: broker, load_number, origin, destination, pickup_date (YYYY-MM-DD), rate (number)."
)


def make_example():
    r = make_record()
    tpl = random.choice(TEMPLATES)
    text = tpl(r, random.randint(0, 2))
    target = {
        "broker": r["broker"],
        "load_number": r["load_number"],
        "origin": r["origin"],
        "destination": r["destination"],
        "pickup_date": r["pickup_date"],
        "rate": r["rate"],
    }
    return {
        "instruction": INSTRUCTION,
        "input": text,
        "output": json.dumps(target, ensure_ascii=False),
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    examples = [make_example() for _ in range(600)]
    train, test = examples[:550], examples[550:]

    for name, rows in (("train", train), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} ejemplos")


if __name__ == "__main__":
    main()
