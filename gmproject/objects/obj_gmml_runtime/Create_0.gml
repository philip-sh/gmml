/// @desc
policies = {};

/// @desc Load and register a policy.
register_policy = function(_b, _f) {
	policies[$ _b] = gmpolicy_load(_f);
}

/// @desc Unload a registered policy to free memory.
free_policy = function(_b) {
	if (variable_struct_exists(policies, _b)) variable_struct_remove(policies, _b);
}

register_policy("drone", "final_policy.gmpolicy");