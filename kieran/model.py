#kieran, QN3

from pomegranate import Node, DiscreteDistribution, ConditionalProbabilityTable, BayesianNetwork


# Weather node has no parents
Weather = Node(DiscreteDistribution({
    "sunny": 0.7,
    "rainy": 0.26,
    "foggy": 0.04
}), name="weather")

# Time of day node has no parents
Time_of_day = Node(DiscreteDistribution({
    "morning": 0.34,
    "afternoon": 0.33,
    "evening": 0.33
}), name="time_of_day")

# Historical congestion level node depends on Weather and Time_of_day
Historical_congestion_level = Node(ConditionalProbabilityTable([
    ["rainy", "morning", "high", 0.6],
    ["rainy", "morning", "medium", 0.3],
    ["rainy", "morning", "low", 0.1],
    ["rainy", "afternoon", "high", 0.8],
    ["rainy", "afternoon", "medium", 0.15],
    ["rainy", "afternoon", "low", 0.05],
    ["rainy", "evening", "high", 0.7],
    ["rainy", "evening", "medium", 0.2],
    ["rainy", "evening", "low", 0.1],
    ["sunny", "morning", "high", 0.4],
    ["sunny", "morning", "medium", 0.4],
    ["sunny", "morning", "low", 0.2],
    ["sunny", "afternoon", "high", 0.3],
    ["sunny", "afternoon", "medium", 0.5],
    ["sunny", "afternoon", "low", 0.2],
    ["sunny", "evening", "high", 0.35],
    ["sunny", "evening", "medium", 0.45],
    ["sunny", "evening", "low", 0.2],
    ["foggy", "morning", "high", 0.5],
    ["foggy", "morning", "medium", 0.4],
    ["foggy", "morning", "low", 0.1],
    ["foggy", "afternoon", "high", 0.6],
    ["foggy", "afternoon", "medium", 0.3],
    ["foggy", "afternoon", "low", 0.1],
    ["foggy", "evening", "high", 0.55],
    ["foggy", "evening", "medium", 0.35],
    ["foggy", "evening", "low", 0.1],
], [Weather.distribution, Time_of_day.distribution]), name="historical_congestion_level")

# Current congestion level node depends on Historical_congestion_level
Current_congestion_level = Node(ConditionalProbabilityTable([
    ["high", "high", 0.9],
    ["high", "medium", 0.1],
    ["high", "low", 0.0],
    ["medium", "high", 0.6],
    ["medium", "medium", 0.3],
    ["medium", "low", 0.1],
    ["low", "high", 0.3],
    ["low", "medium", 0.4],
    ["low", "low", 0.3],
], [Historical_congestion_level.distribution]), name="current_congestion_level")

# Create a Bayesian Network and add states
model = BayesianNetwork("Traffic Congestion Prediction")
model.add_states(Weather, Time_of_day, Historical_congestion_level, Current_congestion_level)
# Add edges connecting nodes
model.add_edge(Weather, Historical_congestion_level)
model.add_edge(Time_of_day, Historical_congestion_level)
model.add_edge(Historical_congestion_level, Current_congestion_level)

# Finalize model
model.bake()

# User input
weather_input = input("Enter the weather (sunny, rainy, foggy): ").strip().lower()
time_of_day_input = input("Enter the time of day (morning, afternoon, evening): ").strip().lower()

# Validate input
if weather_input not in ["sunny", "rainy", "foggy"] or time_of_day_input not in ["morning", "afternoon", "evening"]:
    print("Invalid input. Please enter valid weather and time of day.")
else:
    # Predict probabilities
    predictions = model.predict_proba({
        "weather": weather_input,
        "time_of_day": time_of_day_input
    })

    # Print results
    print("\nPredicted probabilities for Current Congestion Level:")
    for value, probability in predictions[-1].parameters[0].items():
        print(f"  {value}: {probability:.4f}")
