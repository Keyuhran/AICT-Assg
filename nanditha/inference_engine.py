import json
from sympy import symbols, And, Implies

class InferenceEngine:
    def __init__(self, data_file):
        """Loads traffic data from a file and defines logical rules."""
        self.data_file = data_file
        self.setup_rules()

    def setup_rules(self):
        """Defines basic traffic violation rules using simple logic."""
        self.speed = symbols("Speed")
        self.speed_limit = symbols("Speed_Limit")
        self.speed_violation = symbols("Speed_Violation")

        self.signal = symbols("Signal")
        self.at_intersection = symbols("At_Intersection")
        self.signal_violation = symbols("Signal_Violation")

        self.location_1 = symbols("Location_1")
        self.location_2 = symbols("Location_2")
        self.location_violation = symbols("Location_Violation")

        # Basic traffic rules
        self.rules = [
            Implies(self.speed > self.speed_limit, self.speed_violation),  # Speeding
            Implies(And(self.signal == "Red", self.at_intersection), self.signal_violation),  # Running red light
            Implies(And(self.location_1, self.location_2), self.location_violation)  # Being in two places at once
        ]

    def check_violations(self):
        """Reads traffic data and checks for violations based on predefined rules."""
        with open(self.data_file, "r") as file:
            traffic_data = json.load(file)

        violations = []
        
        for vehicle in traffic_data:
            vehicle_violations = []
            vehicle_id = vehicle["id"]
            speed = vehicle["speed"]
            speed_limit = vehicle["speed_limit"]
            signal = vehicle["signal"]
            locations = vehicle["locations"]

            # Speed Violation Check
            if speed > speed_limit:
                vehicle_violations.append(f"Speed Violation: {speed} km/h (Limit: {speed_limit} km/h)")

            # Signal Violation Check
            if signal == "Red" and "Intersection" in locations:
                vehicle_violations.append("Signal Violation: Ran a red light at Intersection")

            # Location Consistency Check
            if len(set(locations)) > 1:
                formatted_locations = " and ".join(locations)
                vehicle_violations.append(f"Location Violation: Detected at {formatted_locations} simultaneously")

            # Store results
            if vehicle_violations:
                violations.append({vehicle_id: vehicle_violations})

        return violations
