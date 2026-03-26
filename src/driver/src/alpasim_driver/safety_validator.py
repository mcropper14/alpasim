import math
from physics import (
    world_to_ego,
    obstacle_forward_speed,
    relative_forward_speed,
    compute_time_to_collision,
    compute_stopping_distance,
    compute_required_deceleration,
    compute_max_safe_speed,
)


'''
- lane_half_width:
  nominal half-width of the ego lane/path corridor in meters
- prediction_horizon:
  how far ahead to reason about obstacles
- reaction_time:
  delay before braking begins
- max_decel:
  magnitude of max feasible deceleration
- min_ttc / emergency_ttc:
  TTC thresholds for nominal slowdown vs emergency brake
- safety_buffer:
  extra margin beyond exact stopping distance
- min_confidence:
  min obstacle confidence to even consider
'''
class SafetyValidator:
    def __init__(
        self,
        lane_half_width=1.75,
        prediction_horizon=60.0,
        reaction_time=0.5,
        max_decel=6.0,
        min_ttc=2.0,
        emergency_ttc=1.0,
        safety_buffer=2.0,
        min_confidence=0.25,
        steering_path_gain=12.0,
        confidence_weight_power=2.0,
    ):
        self.lane_half_width = lane_half_width
        self.prediction_horizon = prediction_horizon
        self.reaction_time = reaction_time
        self.max_decel = max_decel
        self.min_ttc = min_ttc
        self.emergency_ttc = emergency_ttc
        self.safety_buffer = safety_buffer
        self.min_confidence = min_confidence
        self.steering_path_gain = steering_path_gain
        self.confidence_weight_power = confidence_weight_power

    '''
    Validate the proposed action against physics-based safety constraints given the current ego state and list of perceived obstacles
    '''
    def validate(self, ego_state, proposed_action, obstacles):
        ego_speed = max(ego_state.get("speed", 0.0), 0.0)
        validated_action = {
            "target_speed": max(proposed_action.get("target_speed", ego_speed), 0.0),
            "steering_angle": proposed_action.get("steering_angle", 0.0),
            "brake": False,
            "override_reason": None,
        }
        violations, trace_lines = [], []
        trace_data = {
            "inputs": {
                "ego_state": ego_state,
                "proposed_action": proposed_action,
                "num_obstacles_received": len(obstacles),
            },
            "relevant_obstacles": [],
            "per_obstacle_assessment": [],
            "summary_metrics": {},
            "actions_taken": [],
        }
        trace_lines.append(
            "Safety validator invoked with proposed action "
            f"(target_speed={validated_action['target_speed']:.2f} m/s, "
            f"steering_angle={validated_action['steering_angle']:.3f} rad)."
        )
        relevant_obstacles = self._collect_relevant_obstacles(ego_state, validated_action, obstacles)
        trace_lines.append(
            f"Received {len(obstacles)} obstacles; "
            f"{len(relevant_obstacles)} considered relevant to predicted path."
        )

        trace_data["relevant_obstacles"] = [
            {
                "id": obs.get("id"),
                "source": obs.get("source", "unknown"),
                "confidence": obs.get("confidence", 1.0),
                "longitudinal_distance": obs["longitudinal_distance"],
                "lateral_distance": obs["lateral_distance"],
                "path_lateral_error": obs["path_lateral_error"],
                "forward_speed": obs["forward_speed"],
                "risk_weight": obs["risk_weight"],
            }
            for obs in relevant_obstacles
        ]

        if not relevant_obstacles:
            trace_lines.append("No relevant obstacles detected along predicted path. Action accepted.")
            trace_data["summary_metrics"] = {
                "worst_ttc": float("inf"),
                "max_required_decel": 0.0,
                "min_safe_speed": validated_action["target_speed"],
            }
            result = self._build_result(
                        is_safe=True,
                        proposed_action=proposed_action,
                        validated_action=validated_action,
                        violations=violations,
                        trace_lines=trace_lines,
                        trace_data=trace_data,
                    )
            return result

        assessments = []
        worst_ttc = float("inf")
        max_required_decel = 0.0
        min_safe_speed = validated_action["target_speed"]
        emergency_triggered = False
        for obs in relevant_obstacles:
            distance = obs["longitudinal_distance"]
            obstacle_speed = obs["forward_speed"]
            confidence = obs.get("confidence", 1.0)
            risk_weight = obs["risk_weight"]
            closing_speed = max(relative_forward_speed(validated_action["target_speed"], obstacle_speed), 0.0)
            ttc = compute_time_to_collision(distance, closing_speed)
            stopping_distance = compute_stopping_distance(
                validated_action["target_speed"],
                reaction_time=self.reaction_time,
                max_decel=self.max_decel,
            )
            required_decel = compute_required_deceleration(
                validated_action["target_speed"],
                distance - self.safety_buffer,
                reaction_time=self.reaction_time,
            )
            safe_speed = compute_max_safe_speed(
                distance=distance,
                reaction_time=self.reaction_time,
                max_decel=self.max_decel,
                safety_buffer=self.safety_buffer,
            )
            confidence_adjusted_ttc = ttc / max(risk_weight, 1e-6)
            assessment = {
                "id": obs.get("id"),
                "source": obs.get("source", "unknown"),
                "confidence": confidence,
                "risk_weight": risk_weight,
                "distance": distance,
                "path_lateral_error": obs["path_lateral_error"],
                "forward_speed": obstacle_speed,
                "closing_speed": closing_speed,
                "ttc": ttc,
                "confidence_adjusted_ttc": confidence_adjusted_ttc,
                "stopping_distance": stopping_distance,
                "required_decel": required_decel,
                "safe_speed": safe_speed,
                "flags": [],
            }

            if confidence_adjusted_ttc < worst_ttc:
                worst_ttc = confidence_adjusted_ttc
            if required_decel > max_required_decel:
                max_required_decel = required_decel
            if safe_speed < min_safe_speed:
                min_safe_speed = safe_speed
            if confidence >= self.min_confidence and confidence_adjusted_ttc < self.emergency_ttc:
                assessment["flags"].append("EMERGENCY_TTC_VIOLATION")
                emergency_triggered = True
            if confidence >= self.min_confidence and ((required_decel > self.max_decel) or (stopping_distance + self.safety_buffer > distance)):
                assessment["flags"].append("STOPPING_DISTANCE_VIOLATION")
            if confidence >= self.min_confidence and confidence_adjusted_ttc < self.min_ttc:
                assessment["flags"].append("LOW_TTC_VIOLATION")

            assessments.append(assessment)
        trace_data["per_obstacle_assessment"] = assessments
        trace_data["summary_metrics"] = {
            "worst_confidence_adjusted_ttc": worst_ttc,
            "max_required_decel": max_required_decel,
            "min_safe_speed_across_obstacles": min_safe_speed,
        }

        # Sort logged obstacle assessments by severity for easier debugging/demo purposes
        assessments_sorted = sorted(
            assessments,
            key=lambda a: (
                a["confidence_adjusted_ttc"],
                a["distance"],
            )
        )
        for a in assessments_sorted:
            trace_lines.append(
                "Obstacle assessment: "
                f"id={a['id']}, source={a['source']}, confidence={a['confidence']:.2f}, "
                f"distance={a['distance']:.2f} m, "
                f"path_error={a['path_lateral_error']:.2f} m, "
                f"ttc={a['ttc']:.2f} s, "
                f"adj_ttc={a['confidence_adjusted_ttc']:.2f} s, "
                f"required_decel={a['required_decel']:.2f} m/s^2, "
                f"safe_speed={a['safe_speed']:.2f} m/s, "
                f"flags={a['flags']}."
            )

        # Emergency brake if any sufficiently confident obstacle looks immediately dangerous!
        if emergency_triggered:
            violations.append("EMERGENCY_TTC_VIOLATION")
            validated_action["target_speed"] = 0.0
            validated_action["brake"] = True
            validated_action["override_reason"] = "emergency_ttc"
            trace_lines.append(
                "At least one high-confidence obstacle triggered the emergency TTC rule. "
                "Emergency braking override issued."
            )
            trace_data["actions_taken"].append({
                "action": "emergency_brake",
                "reason": "emergency_ttc",
            })
            result = self._build_result(
                        is_safe=False,
                        proposed_action=proposed_action,
                        validated_action=validated_action,
                        violations=violations,
                        trace_lines=trace_lines,
                        trace_data=trace_data,
                    )
            return result

        # Slowdown based on the most restrictive safe speed across all relevant obstacles
        if min_safe_speed < validated_action["target_speed"]:
            violations.append("STOPPING_DISTANCE_VIOLATION")
            old_speed = validated_action["target_speed"]
            validated_action["target_speed"] = min_safe_speed
            validated_action["override_reason"] = "stopping_distance"
            trace_lines.append(
                f"Aggregated stopping-distance analysis reduced target speed from "
                f"{old_speed:.2f} to {validated_action['target_speed']:.2f} m/s."
            )
            trace_data["actions_taken"].append({
                "action": "speed_reduction",
                "reason": "stopping_distance",
                "old_speed": old_speed,
                "new_speed": validated_action["target_speed"],
            })

        # if any confident obstacle still yields low TTC, avoid outputting a speed above the slowest relevant obstacle's forward speed.
        low_ttc_obstacles = [
            a for a in assessments
            if "LOW_TTC_VIOLATION" in a["flags"]
        ]

        if low_ttc_obstacles:
            violations.append("LOW_TTC_VIOLATION")
            slowest_relevant_speed = min(
                max(a["forward_speed"], 0.0) for a in low_ttc_obstacles
            )
            old_speed = validated_action["target_speed"]
            validated_action["target_speed"] = min(
                validated_action["target_speed"],
                slowest_relevant_speed,
            )
            if validated_action["target_speed"] <= 1e-6:
                validated_action["brake"] = True
            if validated_action["override_reason"] is None:
                validated_action["override_reason"] = "low_ttc"

            trace_lines.append(
                f"Low-TTC condition detected across one or more obstacles. "
                f"Target speed capped from {old_speed:.2f} to "
                f"{validated_action['target_speed']:.2f} m/s."
            )
            trace_data["actions_taken"].append({
                "action": "speed_cap",
                "reason": "low_ttc",
                "old_speed": old_speed,
                "new_speed": validated_action["target_speed"],
            })

        is_safe = len(violations) == 0

        if is_safe:
            trace_lines.append("All physics-based safety checks passed. Proposed action validated.")
        else:
            trace_lines.append(
                "Proposed action was modified based on aggregated physics-based safety constraints."
            )
        result = self._build_result(
                    is_safe=is_safe,
                    proposed_action=proposed_action,
                    validated_action=validated_action,
                    violations=self._deduplicate_preserve_order(violations),
                    trace_lines=trace_lines,
                    trace_data=trace_data,
                ) 
        return result

    '''
    Convert the obstacle list into a list of path-relevant obstacles (judged against the predicted curved path)
    '''
    def _collect_relevant_obstacles(self, ego_state, proposed_action, obstacles):
        relevant = []

        for obs in obstacles:
            confidence = obs.get("confidence", 1.0)
            if confidence < self.min_confidence:
                continue

            longitudinal, lateral = world_to_ego(
                ego_state,
                obs["x"],
                obs["y"],
            )
            if longitudinal <= 0 or longitudinal > self.prediction_horizon:
                continue

            predicted_path_lateral = self._predicted_path_lateral_offset(
                longitudinal,
                proposed_action.get("steering_angle", 0.0),
            )
            path_lateral_error = lateral - predicted_path_lateral
            if abs(path_lateral_error) > self.lane_half_width:
                continue

            fwd_speed = obstacle_forward_speed(ego_state, obs)
            risk_weight = self._confidence_risk_weight(confidence)
            relevant.append({
                **obs,
                "longitudinal_distance": longitudinal,
                "lateral_distance": lateral,
                "predicted_path_lateral": predicted_path_lateral,
                "path_lateral_error": path_lateral_error,
                "forward_speed": fwd_speed,
                "risk_weight": risk_weight,
            })
        return relevant

    '''
    Approximate the ego vehicle's future lateral offset at a given longitudinal distance
    pos/neg steering angles will cause the same lateral shift at the same longitudinal distance, just in opposite directions
    '''
    def _predicted_path_lateral_offset(self, longitudinal_distance, steering_angle):
        return self.steering_path_gain * math.tan(steering_angle) * ((longitudinal_distance / max(self.prediction_horizon, 1e-6)) ** 2)
    
    '''
    Convert raw confidence into a risk weight
    Higher confidence -> larger risk weight -> smaller adjusted TTC -> more conservative behavior
    '''
    def _confidence_risk_weight(self, confidence):
        confidence = max(0.0, min(confidence, 1.0))
        return max(confidence ** self.confidence_weight_power, 1e-3)

    # Convert steering angle into a human-friendly phrase.
    def _describe_steering_action(self, steering_angle):
        abs_angle = abs(steering_angle)
        if abs_angle < 0.02:
            direction = "continue straight"
        elif steering_angle > 0:
            if abs_angle < 0.08:
                direction = "nudge left"
            elif abs_angle < 0.18:
                direction = "turn left slightly"
            else:
                direction = "turn left"
        else:
            if abs_angle < 0.08:
                direction = "nudge right"
            elif abs_angle < 0.18:
                direction = "turn right slightly"
            else:
                direction = "turn right"
        return direction


    # Generate a short, presentation-friendly decision trace.
    def _generate_human_readable_trace(self, proposed_action, validated_action, violations):
        proposed_speed = proposed_action.get("target_speed", 0.0)
        proposed_steering = proposed_action.get("steering_angle", 0.0)
        validated_speed = validated_action.get("target_speed", 0.0)
        validated_steering = validated_action.get("steering_angle", 0.0)
        brake = validated_action.get("brake", False)
        proposed_direction = self._describe_steering_action(proposed_steering)
        validated_direction = self._describe_steering_action(validated_steering)

        if brake:
            return (
                f"Alpamayo indicated the vehicle should {proposed_direction} "
                f"at {proposed_speed:.1f} m/s. "
                f"Our safety enforcer overrode this action with emergency braking "
                f"due to an immediate collision risk."
            )
        if not violations:
            return (
                f"Alpamayo indicated the vehicle should {proposed_direction} "
                f"at {proposed_speed:.1f} m/s. "
                f"Our safety enforcer agreed that this action is physically safe."
            )
        if abs(validated_speed - proposed_speed) > 1e-6 and proposed_direction == validated_direction:
            if "STOPPING_DISTANCE_VIOLATION" in violations:
                return (
                    f"Alpamayo indicated the vehicle should {proposed_direction} "
                    f"at {proposed_speed:.1f} m/s. "
                    f"Our safety enforcer reduced the speed to {validated_speed:.1f} m/s "
                    f"to satisfy stopping-distance constraints."
                )
            if "LOW_TTC_VIOLATION" in violations:
                return (
                    f"Alpamayo indicated the vehicle should {proposed_direction} "
                    f"at {proposed_speed:.1f} m/s. "
                    f"Our safety enforcer reduced the speed to {validated_speed:.1f} m/s "
                    f"because the predicted time-to-collision was too low."
                )
        return (
            f"Alpamayo indicated the vehicle should {proposed_direction} "
            f"at {proposed_speed:.1f} m/s. "
            f"Our safety enforcer modified the action to {validated_direction} "
            f"at {validated_speed:.1f} m/s based on physics-based safety checks."
        )

    '''
    Build final returned object
    '''
    def _build_result(self, is_safe, proposed_action, validated_action, violations, trace_lines, trace_data):
        return {
            "is_safe": is_safe,
            "validated_action": validated_action,
            "violations": violations,
            "decision_trace": "\n".join(trace_lines),
            "human_readable_trace": self._generate_human_readable_trace(
                proposed_action,
                validated_action,
                violations,
            ),
            "trace_data": trace_data,
        }

    '''
    Remove duplicate violations but still preserve order
    '''
    def _deduplicate_preserve_order(self, items):
        seen = set()
        result = []
        for item in items:
            if (item not in seen):
                seen.add(item)
                result.append(item)
        return result