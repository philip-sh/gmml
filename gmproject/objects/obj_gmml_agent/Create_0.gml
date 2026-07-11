/// Base object for every ML behavior. Do not place this directly.
/// Make a chold object and in its Create event:
///		1. event_inherited();							// run this event
///		2. behavior_name = "your_behavior_name";		// name your behavior
///		3. gmml_set_spec(13, gmml_discrete([3, 2]));	// declare obs size + action spec
///		4. override collect_observations() and on_action(action).
///
/// Two instances of the same behavior in a room -> self-play automatically: the
/// academy tags each by id and Python assigns learner vs. opponent.

behavior_name	= "unnamed";
obs_size		= 0;
action_spec		= gmml_continuous(0);

reward			= 0;
episode_done	= false;
__obs			= [];

/// @desc Declare this behavior's observation size and action spec.
gmml_set_spec = function(_obs_size, _action_spec) {
	obs_size	= _obs_size;
	action_spec = _action_spec;
};

// --- API used in collect_observations() ------------------------------
add_observation = function(_v) { array_push(__obs, _v); };

// --- API used inside on_action() (or anywhere during the episode) ----
set_reward	= function(_r) { reward = _r; };
add_reward	= function(_r) { reward += _r; };
end_episode	= function() { episode_done = true; };

// --- Virtual hooks - overridden by the behavior object ---------------
/// @desc Fill observations by using add_observation(). Called by obj_gmml_academy each step.
collect_observations = function() {};
/// @desc Apply a decoded action vector to the game. '_action' is an array.
on_action = function(_action) {};
/// @desc Re-init THIS agent for a new episode (per-agent auto-reset / vectorized mode).
///       No need to override this if you're using the global reset path.
agent_reset = function() {};
/// @desc Optional, override these if your environment needs to handle the pause during gradient update,
///		  such as freezing physics if using Box2D physics.
agent_pause  = function() {};
agent_resume = function() {};

// --- Internal, the academy calls this to gather one step record ------
__gmml_collect = function() {
    __obs = [];
    collect_observations();
    var _rec = {
        agent_id: real(id),
        behavior: behavior_name,
        obs:      __obs,
        reward:   reward,
        done:     episode_done,
    };
    // Auto-reset (vectorized): the agent ended this step, so reset it NOW and report the
    // fresh obs, stashing the ended episode's final obs as terminal_obs. Global-reset mode
    // leaves the global flag false -> this branch is skipped and the record is unchanged.
    if (global.__gmml_auto_reset && episode_done) {
        _rec.terminal_obs = __obs;
        agent_reset();
        __obs = [];
        collect_observations();
        _rec.obs = __obs;
    }
    reward       = 0; // reset per-step accumulators after reporting
    episode_done = false;
    return _rec;
};

// --- For in-engine inference -----------------------------------------

gmml_observe = function() { 
	__obs = [];
	collect_observations();
	return __obs;
}

gmml_policy_step = function(_policy) {
	on_action(gmpolicy_predict(_policy, gmml_observe()));
	if (episode_done) {
		agent_reset();
		episode_done = false;
		reward = 0;
	}
}
