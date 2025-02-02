import json
from logic import *

# Load traffic data
with open("traffic_data.json", "r") as file:
    traffic_data = json.load(file)

# Get unique drivers
drivers = set(entry["Driver"] for entry in traffic_data)

# Process each driver separately
for driver in drivers:
    print(f"\nTraffic Violation Check for {driver}:")

    # Filter the data for this driver
    driver_data = [entry for entry in traffic_data if entry["Driver"] == driver]

    # Define logical symbols for each day's speed violation
    speed_symbols = {}
    violation_symbols = {}

    for entry in driver_data:
        day = entry["Day"]
        speed_symbols[day] = Symbol(f"Speed_Over_Limit_{driver}_Day{day}")
        violation_symbols[day] = Symbol(f"Violation_{driver}_Day{day}")

    # Logical rules for traffic violations (if speed is over limit, then violation)
    rules = And(*[
        Implication(speed_symbols[entry["Day"]], violation_symbols[entry["Day"]])
        for entry in driver_data
    ])

    # Assign known values from data (whether speeding occurred)
    facts = And(*[
        speed_symbols[entry["Day"]] if entry["Violation"] == "Yes" else Not(speed_symbols[entry["Day"]])
        for entry in driver_data
    ])

    # Check for violations using model_check()
    for entry in driver_data:
        day = entry["Day"]
        violation_result = model_check(And(rules, facts), violation_symbols[day])
        print(f"Day {day} Violation: {violation_result}")
