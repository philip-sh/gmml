with (obj_agent_drone) {
	var _pol = other.policies[$ behavior_name];
	if (!is_undefined(_pol)) gmml_policy_step(_pol);
}