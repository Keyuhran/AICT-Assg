from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the Bayesian Network
model = BayesianNetwork([
    ("weather", "historical_congestion_level"),
    ("time_of_day", "historical_congestion_level"),
    ("historical_congestion_level", "current_congestion_level")
])

# Define CPDs
cpd_weather = TabularCPD(variable="weather", variable_card=3,
                         values=[[0.7], [0.26], [0.04]],
                         state_names={"weather": ["sunny", "rainy", "foggy"]})

cpd_time_of_day = TabularCPD(variable="time_of_day", variable_card=3,
                              values=[[0.34], [0.33], [0.33]],
                              state_names={"time_of_day": ["morning", "afternoon", "evening"]})

cpd_historical = TabularCPD(
    variable="historical_congestion_level",
    variable_card=3,
    values=[
        [0.1, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],  # Low
        [0.3, 0.15, 0.2, 0.4, 0.5, 0.45, 0.4, 0.3, 0.35], # Medium
        [0.6, 0.8, 0.7, 0.4, 0.3, 0.35, 0.5, 0.6, 0.55]   # High
    ],
    evidence=["weather", "time_of_day"],
    evidence_card=[3, 3],
    state_names={
        "historical_congestion_level": ["low", "medium", "high"],
        "weather": ["sunny", "rainy", "foggy"],
        "time_of_day": ["morning", "afternoon", "evening"]
    }
)

cpd_current = TabularCPD(
    variable="current_congestion_level",
    variable_card=3,
    values=[
        [0.0, 0.1, 0.3],  # Low
        [0.1, 0.3, 0.4],  # Medium
        [0.9, 0.6, 0.3]   # High
    ],
    evidence=["historical_congestion_level"],
    evidence_card=[3],
    state_names={
        "current_congestion_level": ["low", "medium", "high"],
        "historical_congestion_level": ["low", "medium", "high"]
    }
)
# Add CPDs to the model
model.add_cpds(cpd_weather, cpd_time_of_day, cpd_historical, cpd_current)

# Check the model structure
assert model.check_model()

# Perform inference
inference = VariableElimination(model)

# User input
weather_input = input("Enter the weather (sunny, rainy, foggy): ").strip().lower()
time_of_day_input = input("Enter the time of day (morning, afternoon, evening): ").strip().lower()

# Validate input
if weather_input not in ["sunny", "rainy", "foggy"] or time_of_day_input not in ["morning", "afternoon", "evening"]:
    print("Invalid input. Please enter valid weather and time of day.")
else:
    # Query the model
    query_result = inference.query(
        variables=["current_congestion_level"],
        evidence={"weather": weather_input, "time_of_day": time_of_day_input}
    )

# Print results
print("\nPredicted probabilities for Current Congestion Level:")
for state, prob in enumerate(query_result.values):
    state_name = query_result.state_names["current_congestion_level"][state]
    print(f"  {state_name}: {prob:.4f}")
