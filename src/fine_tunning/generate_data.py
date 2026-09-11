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

# Solo aparecen en test: miden si el modelo generaliza a valores que nunca vio.
NEW_BROKERS = ["J.B. Hunt", "Schneider", "Mode Transportation", "GlobalTranz"]
NEW_CITIES = [
    ("Shreveport", "LA"), ("Little Rock", "AR"), ("Las Cruces", "NM"),
    ("Corpus Christi", "TX"), ("Salt Lake City", "UT"), ("Kansas City", "MO"),
]

EQUIPMENT = ["Flatbed", "Dry Van", "Step Deck", "Reefer", "Hotshot"]
COMMODITIES = ["Pipe", "Drilling Equipment", "Frac Sand", "Valves", "General Freight",
               "Steel Coils", "Machinery", "Casing"]

# Para armar direcciones completas (instalacion, calle, ciudad, estado y codigo postal).
FACILITIES = ["Permian Pipe Yard", "Basin Valve Co.", "Lone Star Steel", "Mesa Distribution",
              "Red River Warehouse", "Summit Supply"]
STREETS = ["Industrial Blvd", "County Rd 1250", "Hwy 285", "Commerce St", "Frontage Rd", "Pipeline Dr"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def rand_date(rng):
    d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))
    return d


def rand_site(rng):
    return {
        "facility": rng.choice(FACILITIES),
        "street": f"{rng.randint(100, 9999)} {rng.choice(STREETS)}",
        "zip": rng.randint(70000, 89999),
    }


def make_record(rng, brokers, cities):
    broker = rng.choice(brokers)
    origin_city, origin_st = rng.choice(cities)
    dest_city, dest_st = rng.choice(cities)
    while (dest_city, dest_st) == (origin_city, origin_st):
        dest_city, dest_st = rng.choice(cities)
    return {
        "broker": broker,
        # Sin prefijo "#": en textos como "Load #123" el "#" se lee como "numero",
        # asi que incluirlo en el ID hacia ambigua la respuesta esperada.
        "load_number": rng.choice(["", "L", "BOL-"]) + str(rng.randint(100000, 9999999)),
        "origin": f"{origin_city}, {origin_st}",
        "destination": f"{dest_city}, {dest_st}",
        "pickup_date": rand_date(rng).isoformat(),
        "rate": round(rng.uniform(650, 4800), 2),
        "equipment": rng.choice(EQUIPMENT),
        "commodity": rng.choice(COMMODITIES),
        "pu_site": rand_site(rng),
        "del_site": rand_site(rng),
        "addr_style": rng.randint(0, 1),
    }


def fmt_date(iso, style):
    y, m, d = iso.split("-")
    if style == 0:
        return f"{m}/{d}/{y}"
    if style == 1:
        return f"{MONTHS[int(m) - 1]} {int(d)}, {y}"
    if style == 3:  # solo en la plantilla de test
        return f"{MONTHS[int(m) - 1][:3]} {int(d)}, {y}"
    return f"{int(m)}-{int(d)}-{y[2:]}"


def fmt_address(site, city_state, style):
    if style == 0:
        return f"{site['facility']}, {site['street']}, {city_state} {site['zip']}"
    return f"{site['facility']}\n  {site['street']}\n  {city_state} {site['zip']}"


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
    # Direcciones completas: el modelo debe reducirlas a "City, ST".
    lambda r, ds: (
        f"LOAD CONFIRMATION - {r['broker']}\n"
        f"Pro number: {r['load_number']}\n"
        f"Ship from: {fmt_address(r['pu_site'], r['origin'], r['addr_style'])}\n"
        f"Ship to: {fmt_address(r['del_site'], r['destination'], r['addr_style'])}\n"
        f"Ship date: {fmt_date(r['pickup_date'], ds)}\n"
        f"{r['equipment']} / {r['commodity']}\n"
        f"Carrier pay: ${r['rate']:,.2f}"
    ),
]

# Solo en test: etiquetas nuevas, destino ANTES que origen, "USD" y fecha abreviada.
HOLDOUT_TEMPLATES = [
    lambda r, ds: (
        f"LOAD TENDER - {r['broker']}\n"
        f"Tender ref: {r['load_number']}\n"
        f"Deliver to: {r['destination']}\n"
        f"Pick up at: {r['origin']}\n"
        f"Ready: {fmt_date(r['pickup_date'], 3)}\n"
        f"Trailer: {r['equipment']} / {r['commodity']}\n"
        f"Pay: USD {r['rate']:,.2f}"
    ),
]

# --- Casos dificiles (solo en test) ---------------------------------------
# Ruido realista: direcciones completas, varias fechas, montos y referencias,
# correcciones sobre valores anteriores y texto en espanol.

CARRIERS = ["Chavez Trucking LLC", "Rio Grande Hauling", "Desert Line Transport", "Mesa Freight Inc."]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def hard_extras(rng, r):
    """Datos extra que sirven de distractores; ninguno forma parte del JSON esperado."""
    pickup = date.fromisoformat(r["pickup_date"])
    return {
        "carrier": rng.choice(CARRIERS),
        "mc": rng.randint(100000, 999999),
        "po": rng.randint(10000, 99999),
        "quote": f"Q-{rng.randint(1000, 9999)}",
        "tender_date": (pickup - timedelta(days=rng.randint(1, 5))).isoformat(),
        "delivery_date": (pickup + timedelta(days=rng.randint(1, 4))).isoformat(),
        "old_pickup_date": (pickup - timedelta(days=rng.randint(1, 3))).isoformat(),
        "old_rate": round(r["rate"] - rng.uniform(50, 400), 2),
        "advance": rng.choice([300, 400, 500]),
        "detention": rng.choice([40, 50, 75]),
        "lumper": rng.choice([100, 150, 200]),
    }


def fmt_fecha(iso):
    y, m, d = iso.split("-")
    return f"{int(d)} de {MESES[int(m) - 1]} de {y}"


HARD_TEMPLATES = {
    # Rate con completa: el origen/destino salen de direcciones con codigo postal.
    "distractores": lambda r, ds, x: (
        f"CARRIER RATE CONFIRMATION\n"
        f"Quote ID: {x['quote']}    Tender date: {fmt_date(x['tender_date'], ds)}\n"
        f"Carrier: {x['carrier']} (MC# {x['mc']})    Broker: {r['broker']}\n"
        f"Load ID: {r['load_number']}    Customer PO #: {x['po']}\n\n"
        f"STOP 1 - PICKUP  {fmt_date(r['pickup_date'], ds)} 08:00-14:00\n"
        f"  {fmt_address(r['pu_site'], r['origin'], 0)}\n"
        f"STOP 2 - DELIVERY  {fmt_date(x['delivery_date'], ds)} by 16:00\n"
        f"  {fmt_address(r['del_site'], r['destination'], 0)}\n\n"
        f"Equipment: {r['equipment']} | Commodity: {r['commodity']} | Weight: 42,000 lbs\n"
        f"Rate: ${r['rate']:,.2f} (all-in, includes fuel)\n"
        f"Detention: ${x['detention']:.2f}/hr after 2 hrs free | "
        f"Lumper: reimbursable up to ${x['lumper']:.2f}\n"
        f"Carrier may request a quick-pay advance of up to ${x['advance']:.2f}.\n"
        f"Double brokering is prohibited. Carrier must call {r['broker']} dispatch before loading."
    ),
    # Hilo de correo con una correccion: valen la fecha y la tarifa NUEVAS.
    "correcciones": lambda r, ds, x: (
        f"Subject: RE: Load {r['load_number']} - update\n\n"
        f"Hi, quick correction on load {r['load_number']} with {r['broker']}: pickup was moved "
        f"from {fmt_date(x['old_pickup_date'], ds)} to {fmt_date(r['pickup_date'], ds)}, and the rate "
        f"was raised from ${x['old_rate']:,.2f} to ${r['rate']:,.2f} for the extra wait.\n"
        f"Lane is still {r['origin']} to {r['destination']}.\n\n"
        f"-----Original Message-----\n"
        f"Load {r['load_number']}: {r['origin']} -> {r['destination']}, "
        f"PU {fmt_date(x['old_pickup_date'], ds)}, ${x['old_rate']:,.2f}."
    ),
    # Rate con en espanol, con el mes escrito en espanol.
    "espanol": lambda r, ds, x: (
        f"CONFIRMACIÓN DE TARIFA\n"
        f"Broker: {r['broker']}\n"
        f"Transportista: {x['carrier']}\n"
        f"No. de carga: {r['load_number']}\n"
        f"Origen: {r['origin']}\n"
        f"Destino: {r['destination']}\n"
        f"Fecha de recolección: {fmt_fecha(r['pickup_date'])}\n"
        f"Equipo: {r['equipment']} | Mercancía: {r['commodity']}\n"
        f"Tarifa total: ${r['rate']:,.2f} USD\n"
        f"Favor de firmar y regresar."
    ),
}

INSTRUCTION = (
    "Extract the load details from the rate confirmation below and return them as JSON "
    "with the keys: broker, load_number, origin (City, ST), destination (City, ST), "
    "pickup_date (YYYY-MM-DD), rate (number)."
)


def make_example(rng=random, brokers=BROKERS, cities=CITIES, templates=TEMPLATES):
    r = make_record(rng, brokers, cities)
    tpl = rng.choice(templates)
    return to_example(r, tpl(r, rng.randint(0, 2)))


def make_hard_example(rng, template):
    r = make_record(rng, BROKERS, CITIES)
    ds = rng.randint(0, 2)
    return to_example(r, template(r, ds, hard_extras(rng, r)))


def to_example(r, text):
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


def build_datasets():
    examples = [make_example() for _ in range(600)]
    train = examples[:550]
    test = [dict(ex, group="in_dist") for ex in examples[550:]]

    # Grupos fuera de distribucion. Usan su propio generador para no alterar train.
    ood_rng = random.Random(7)
    test += [dict(make_example(ood_rng, brokers=NEW_BROKERS, cities=NEW_CITIES), group="nuevos_valores")
             for _ in range(25)]
    test += [dict(make_example(ood_rng, templates=HOLDOUT_TEMPLATES), group="nueva_plantilla")
             for _ in range(25)]

    # Casos dificiles, con otro generador para no alterar los grupos anteriores.
    hard_rng = random.Random(11)
    for group, template in HARD_TEMPLATES.items():
        test += [dict(make_hard_example(hard_rng, template), group=group) for _ in range(25)]
    return train, test


def main():
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    train, test = build_datasets()
    for name, rows in (("train", train), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} ejemplos")


if __name__ == "__main__":
    main()
