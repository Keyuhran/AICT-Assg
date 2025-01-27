from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the Bayesian Network
model = BayesianNetwork([
    ("weather", "historical_congestion_level"),
    ("time_of_day", "historical_congestion_level"),
    ("is_accident", "historical_congestion_level"),
    ("day_of_week", "historical_congestion_level"),
    ("historical_congestion_level", "current_congestion_level"),
    ("weather", "current_congestion_level"),
    ("time_of_day", "current_congestion_level"),
    ("is_accident", "current_congestion_level"),
    ("day_of_week", "current_congestion_level")
])

# Define CPDs
cpd_weather = TabularCPD(variable="weather", variable_card=3,
                         values=[[0.7], [0.26], [0.04]],
                         state_names={"weather": ["sunny", "rainy", "foggy"]})

cpd_time_of_day = TabularCPD(variable="time_of_day", variable_card=3,
                              values=[[0.34], [0.33], [0.33]],
                              state_names={"time_of_day": ["morning", "afternoon", "evening"]})

cpd_is_accident = TabularCPD(variable="is_accident", variable_card=2,
                              values=[[0.9], [0.1]],
                              state_names={"is_accident": ["no", "yes"]})

cpd_day_of_week = TabularCPD(variable="day_of_week", variable_card=7,
                              values=[[0.14], [0.14], [0.14], [0.14], [0.14], [0.15], [0.15]],
                              state_names={"day_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]})

# Historical congestion level
low_congestion = []
medium_congestion = []
high_congestion = []

for day in range(7):  # Days of the week
    for time in range(3):  # Times of the day
        for weather in range(3):  # Weather states
            for accident in range(2):  # Accident states
                if day in [5, 6]:  # Weekends
                    low_congestion.append(0.4 if accident == 0 else 0.2)
                    medium_congestion.append(0.5 if accident == 0 else 0.6)
                    high_congestion.append(0.1 if accident == 0 else 0.2)
                else:  # Weekdays
                    low_congestion.append(0.1 if accident == 0 else 0.05)
                    medium_congestion.append(0.7 if accident == 0 else 0.6)
                    high_congestion.append(0.2 if accident == 0 else 0.35)

cpd_historical = TabularCPD(
    variable="historical_congestion_level",
    variable_card=3,
    values=[low_congestion, medium_congestion, high_congestion],
    evidence=["weather", "time_of_day", "is_accident", "day_of_week"],
    evidence_card=[3, 3, 2, 7],
    state_names={
        "historical_congestion_level": ["low", "medium", "high"],
        "weather": ["sunny", "rainy", "foggy"],
        "time_of_day": ["morning", "afternoon", "evening"],
        "is_accident": ["no", "yes"],
        "day_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    }
)

# Current congestion level
current_values = []
for hist_level in range(3):  # Historical congestion levels
    for day in range(7):  # Days of the week
        for time in range(3):  # Times of the day
            for weather in range(3):  # Weather states
                for accident in range(2):  # Accident states
                    if accident == 1:  # Accidents increase congestion
                        if hist_level == 0:
                            current_values.append([0.1, 0.4, 0.5])
                        elif hist_level == 1:
                            current_values.append([0.15, 0.45, 0.4])
                        else:
                            current_values.append([0.25, 0.4, 0.35])
                    else:  # No accident
                        if day in [5, 6]:  # Weekends
                            if time == 0:  # Morning
                                if hist_level == 0:
                                    current_values.append([0.6, 0.3, 0.1])
                                elif hist_level == 1:
                                    current_values.append([0.45, 0.4, 0.15])
                                else:
                                    current_values.append([0.3, 0.4, 0.3])
                            else:  # Afternoon/Evening
                                if hist_level == 0:
                                    current_values.append([0.5, 0.4, 0.1])
                                elif hist_level == 1:
                                    current_values.append([0.35, 0.45, 0.2])
                                else:
                                    current_values.append([0.25, 0.4, 0.35])
                        else:  # Weekdays
                            if time == 0:  # Morning
                                if hist_level == 0:
                                    current_values.append([0.5, 0.4, 0.1])
                                elif hist_level == 1:
                                    current_values.append([0.4, 0.4, 0.2])
                                else:
                                    current_values.append([0.3, 0.4, 0.3])
                            elif time == 1:  # Afternoon
                                if hist_level == 0:
                                    current_values.append([0.4, 0.45, 0.15])
                                elif hist_level == 1:
                                    current_values.append([0.3, 0.5, 0.2])
                                else:
                                    current_values.append([0.2, 0.4, 0.4])
                            else:  # Evening
                                if hist_level == 0:
                                    current_values.append([0.35, 0.5, 0.15])
                                elif hist_level == 1:
                                    current_values.append([0.25, 0.5, 0.25])
                                else:
                                    current_values.append([0.2, 0.45, 0.35])

# Transpose to match expected shape
current_values = list(zip(*current_values))

cpd_current = TabularCPD(
    variable="current_congestion_level",
    variable_card=3,
    values=current_values,
    evidence=["historical_congestion_level", "weather", "time_of_day", "is_accident", "day_of_week"],
    evidence_card=[3, 3, 3, 2, 7],
    state_names={
        "current_congestion_level": ["low", "medium", "high"],
        "historical_congestion_level": ["low", "medium", "high"],
        "weather": ["sunny", "rainy", "foggy"],
        "time_of_day": ["morning", "afternoon", "evening"],
        "is_accident": ["no", "yes"],
        "day_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    }
)

# Add CPDs to the model
model.add_cpds(cpd_weather, cpd_time_of_day, cpd_is_accident, cpd_day_of_week, cpd_historical, cpd_current)

# Check the model structure
assert model.check_model()

# Perform inference
inference = VariableElimination(model)

# User input
weather_input = input("Enter the weather (sunny, rainy, foggy): ").strip().lower()
time_of_day_input = input("Enter the time of day (morning, afternoon, evening): ").strip().lower()
is_accident_input = input("Is there an accident on the road? (no, yes): ").strip().lower()
day_of_week_input = input("Enter the day of the week (monday, tuesday, wednesday, thursday, friday, saturday, sunday): ").strip().lower()

# Validate input
if (weather_input not in ["sunny", "rainy", "foggy"] or
    time_of_day_input not in ["morning", "afternoon", "evening"] or
    is_accident_input not in ["no", "yes"] or
    day_of_week_input not in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
    print("Invalid input. Please enter valid values.")
else:
    # Query the model
    query_result = inference.query(
        variables=["current_congestion_level"],
        evidence={
            "weather": weather_input,
            "time_of_day": time_of_day_input,
            "is_accident": is_accident_input,
            "day_of_week": day_of_week_input
        }
    )

    # Print results
    print("\nPredicted probabilities for Current Congestion Level:")
    for state, prob in enumerate(query_result.values):
        state_name = query_result.state_names["current_congestion_level"][state]
        print(f"  {state_name}: {prob:.4f}")
