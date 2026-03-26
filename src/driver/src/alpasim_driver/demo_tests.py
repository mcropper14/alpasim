from safety_validator import SafetyValidator

# Print scenario result cleanly
def print_result(title, description, ego_state, proposed_action, obstacles, result):
    print("Technical trace data:")
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(description)
    print()
    print("ego_state:", ego_state)
    print("proposed_action:", proposed_action)
    print("obstacles:", obstacles)
    print()
    print("is_safe:", result["is_safe"])
    print("validated_action:", result["validated_action"])
    print("violations:", result["violations"])
    print()
    print("decision_trace:")
    print(result["decision_trace"])
    print()
    print("trace_data:")
    print(result["trace_data"])
    print()


# Scenario 1:
# Alpamayo proposes a sensible action and the validator confirms it.
def scenario_action_is_correct():
    ego_state = {
        "x": 102.4,
        "y": 48.7,
        "speed": 8.0,
        "heading": 0.15,
    }
    proposed_action = {
        "target_speed": 7.5,
        "steering_angle": 0.03,
    }
    obstacles = [
        {"id": 1, "x": 130.0, "y": 55.0, "vx": 2.0, "vy": 0.2, "confidence": 0.93, "source": "lidar"},
        {"id": 2, "x": 120.0, "y": 42.0, "vx": 0.0, "vy": 0.0, "confidence": 0.88, "source": "radar"},
        {"id": 3, "x": 90.0, "y": 47.0, "vx": 0.0, "vy": 0.0, "confidence": 0.95, "source": "lidar"},
    ]
    description = (
        "Alpamayo proposes a moderate speed and small steering command. "
        "Obstacles are either outside the predicted path or far enough away that "
        "the proposed action is already physically safe. The validator should approve it."
    )
    return description, ego_state, proposed_action, obstacles


# Scenario 2:
# Alpamayo proposes a speed that is too high for an obstacle ahead.
# The validator should reduce the speed but not emergency brake.
def scenario_action_too_fast():
    ego_state = {
        "x": 215.0,
        "y": -37.5,
        "speed": 11.0,
        "heading": -0.08,
    }
    proposed_action = {
        "target_speed": 16.0,
        "steering_angle": 0.01,
    }
    obstacles = [
        {"id": 4, "x": 233.0, "y": -39.0, "vx": 3.5, "vy": 0.0, "confidence": 0.96, "source": "lidar"},
        {"id": 5, "x": 246.0, "y": -33.0, "vx": 0.0, "vy": 0.0, "confidence": 0.72, "source": "radar"},
        {"id": 6, "x": 221.0, "y": -50.0, "vx": 0.0, "vy": 0.0, "confidence": 0.90, "source": "lidar"},
    ]
    description = (
        "Alpamayo requests a speed-up, but there is a credible obstacle ahead in-path. "
        "The action is not catastrophically unsafe, but the commanded speed is too aggressive. "
        "The validator should reduce the target speed."
    )
    return description, ego_state, proposed_action, obstacles


# Scenario 3:
# Alpamayo proposes an unsafe action with a very close obstacle ahead.
# The validator should override with emergency braking.
def scenario_action_requires_emergency_brake():
    ego_state = {
        "x": -54.0,
        "y": 310.0,
        "speed": 13.5,
        "heading": 0.02,
    }
    proposed_action = {
        "target_speed": 14.0,
        "steering_angle": 0.0,
    }
    obstacles = [
        {"id": 7, "x": -45.5, "y": 310.3, "vx": 0.0, "vy": 0.0, "confidence": 0.99, "source": "lidar"},
        {"id": 8, "x": -40.0, "y": 315.5, "vx": 0.0, "vy": 0.0, "confidence": 0.65, "source": "radar"},
    ]
    description = (
        "Alpamayo proposes to continue at roughly current speed, but a high-confidence obstacle "
        "is dangerously close in the predicted path. The validator should reject the action and "
        "issue emergency braking."
    )
    return description, ego_state, proposed_action, obstacles


# Scenario 4:
# Alpamayo proposes a turn. Straight-path checking might miss the obstacle,
# but our steering-aware path should catch it.
def scenario_turning_conflict():
    ego_state = {
        "x": 400.0,
        "y": 122.0,
        "speed": 10.0,
        "heading": 0.35,
    }

    proposed_action = {
        "target_speed": 11.5,
        "steering_angle": 0.32,
    }

    obstacles = [
        {
            "id": 9,
            "x": 417.0,
            "y": 128.0,
            "vx": 0.0,
            "vy": 0.0,
            "confidence": 0.96,
            "source": "lidar",
        },
        {
            "id": 10,
            "x": 426.0,
            "y": 118.5,
            "vx": 0.0,
            "vy": 0.0,
            "confidence": 0.82,
            "source": "radar",
        },
    ]

    description = (
        "Alpamayo proposes a left-turning maneuver at a moderately aggressive speed. "
        "A high-confidence obstacle lies near the predicted curved path, so the safety "
        "enforcer should detect the conflict and reduce speed or override the action."
    )

    return description, ego_state, proposed_action, obstacles

# Scenario 5:
# Multiple obstacles with different confidence levels. The validator should respond
# conservatively to the most restrictive credible obstacle, not just the nearest raw detection.
def scenario_multiple_obstacles_confidence_arbitration():
    ego_state = {
        "x": 12.0,
        "y": -88.0,
        "speed": 10.5,
        "heading": -0.12,
    }
    proposed_action = {
        "target_speed": 12.5,
        "steering_angle": -0.05,
    }
    obstacles = [
        {"id": 11, "x": 28.0, "y": -89.5, "vx": 5.0, "vy": 0.0, "confidence": 0.97, "source": "radar"},
        {"id": 12, "x": 24.0, "y": -84.0, "vx": 0.0, "vy": 0.0, "confidence": 0.30, "source": "lidar"},
        {"id": 13, "x": 35.0, "y": -90.0, "vx": 1.0, "vy": 0.0, "confidence": 0.91, "source": "lidar"},
        {"id": 14, "x": 20.0, "y": -100.0, "vx": 0.0, "vy": 0.0, "confidence": 0.85, "source": "radar"},
    ]
    description = (
        "There are multiple detections ahead with different positions, speeds, and confidences. "
        "The validator should aggregate across relevant obstacles and choose a safe action based "
        "on the most restrictive credible threat."
    )
    return description, ego_state, proposed_action, obstacles

'''
Get all scenarios in a list for iteration in main
'''
def get_scenarios():
    return [
        ("Scenario 1: Alpamayo Action Confirmed", scenario_action_is_correct()),
        ("Scenario 2: Alpamayo Speed Reduced", scenario_action_too_fast()),
        ("Scenario 3: Alpamayo Overridden with Emergency Brake", scenario_action_requires_emergency_brake()),
        ("Scenario 4: Steering-Aware Turning Conflict", scenario_turning_conflict()),
        ("Scenario 5: Multi-Obstacle Confidence Arbitration", scenario_multiple_obstacles_confidence_arbitration()),
    ]

# # Run all demo scenarios.
# def main():
#     validator = SafetyValidator()
#     scenarios = [
#         ("Scenario 1: Alpamayo Action Confirmed", scenario_action_is_correct()),
#         ("Scenario 2: Alpamayo Speed Reduced", scenario_action_too_fast()),
#         ("Scenario 3: Alpamayo Overridden with Emergency Brake", scenario_action_requires_emergency_brake()),
#         ("Scenario 4: Steering-Aware Turning Conflict", scenario_turning_conflict()),
#         ("Scenario 5: Multi-Obstacle Confidence Arbitration", scenario_multiple_obstacles_confidence_arbitration()),
#     ]
#     for name, scenario_data in scenarios:
#         description, ego_state, proposed_action, obstacles = scenario_data
#         result = validator.validate(ego_state, proposed_action, obstacles)
#         print_result(name, description, ego_state, proposed_action, obstacles, result)


# if __name__ == "__main__":
#     main()