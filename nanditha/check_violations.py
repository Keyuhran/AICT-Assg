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

    # Filter data for the current driver
    driver_data = [entry for entry in traffic_data if entry["Driver"] == driver]

    # Define logical symbols for each day's violations
    speed_symbols = {}
    red_light_symbols = {}
    location_symbols = {}
    violation_symbols = {}

    for entry in driver_data:
        day = entry["Day"]
        speed_symbols[day] = Symbol(f"Speed_Over_Limit_{driver}_Day{day}")
        red_light_symbols[day] = Symbol(f"Jumped_Red_Light_{driver}_Day{day}")
        location_symbols[day] = Symbol(f"Multiple_Locations_{driver}_Day{day}")
        violation_symbols[day] = Symbol(f"Violation_{driver}_Day{day}")

    # Logical rules for traffic violations
    rules = And(*[
        Implication(speed_symbols[entry["Day"]], violation_symbols[entry["Day"]])
        for entry in driver_data
    ] + [
        Implication(red_light_symbols[entry["Day"]], violation_symbols[entry["Day"]])
        for entry in driver_data
    ] + [
        Implication(location_symbols[entry["Day"]], violation_symbols[entry["Day"]])
        for entry in driver_data
    ])

    # Assign known values from data
    facts = And(*[
        speed_symbols[entry["Day"]] if "Speeding" in entry["Violation"] else Not(speed_symbols[entry["Day"]])
        for entry in driver_data
    ] + [
        red_light_symbols[entry["Day"]] if "Jumped Red Light" in entry["Violation"] else Not(red_light_symbols[entry["Day"]])
        for entry in driver_data
    ] + [
        location_symbols[entry["Day"]] if "Multiple Locations" in entry["Violation"] else Not(location_symbols[entry["Day"]])
        for entry in driver_data
    ])

    # Check for violations using model_check()
    for entry in driver_data:
        day = entry["Day"]
        violation_result = model_check(And(rules, facts), violation_symbols[day])

        # Print the violation type if there is a violation
        if violation_result:
            print(f"Day {day} Violation: {entry['Violation']}")
        else:
            print(f"Day {day} Violation: No Violation")
