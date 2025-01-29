from inference_engine import InferenceEngine

def main():
    """Runs the traffic violation detection system."""
    print("Traffic Rule Violation Detection System\n")

    # Load traffic data and check for violations
    inference = InferenceEngine("traffic_data.json")
    violations = inference.check_violations()

    # Print results
    if violations:
        print("Traffic Violations Detected:")
        for vehicle in violations:
            for vehicle_id, vios in vehicle.items():
                print(f"\n{vehicle_id}:")
                for v in vios:
                    print(f"  - {v}")
    else:
        print("No violations detected.")

if __name__ == "__main__":
    main()
