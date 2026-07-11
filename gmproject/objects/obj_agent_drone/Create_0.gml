/// @desc
/// obj_agent_drone: a Box2D physics drone that flies an ordered waypoint course.
///
/// Generates lift with two thrusters sitting at +- momentArm.

#region Tuning

thrustMax = 50;        // force per step per thruster at full throttle
momentArm = 8;         // thruster offset along local x

#endregion
#region Observation normalization

maxSpeed  = 6;			// px/step
maxAngVel = 360;		// deg/sec
RAY_RANGE = GRIDW * 12;

#endregion
#region Reward tuning

PROGRESS_SCALE		= 0.1;		// per-pixel progress reward
WAYPOINT_REWARD		= 2.0;
COMPLETE_REWARD		= 8.0;		// keep high so finishing beats going fast and crashing

CRASH_PENALTY		= -5.0;		// keep high so going fast and crashing can't out-earn completing the course

WP_TIMEOUT_PENALTY	= -4.0;
THRUST_COST			= 0.001;
STEP_PENALTY		= -0.002;
CRASH_RAY			= 0.08;		// nearest-ray fraction that counts as a wall hit

UPRIGHT_PENALTY		= -0.01;	// 0 upright -> -0.02 inverted
ANGVEL_PENALTY		= -0.01;
THRUST_JERK_COST	= -0.01;	// * (|d_thrustL| + |d_thrustR|)

CAPTURE_RADIUS		= GRIDW * 1.5;

waypoint_time		= 0;
waypoint_time_max	= 10.0;		// per waypoint time

#endregion
#region State

thrustL = 0;
thrustR = 0;

prev_thrustL = 0;
prev_thrustR = 0;

// waypoint course
WAYPOINT_COUNT	= 4;
BORDER_MARGIN	= GRIDW * 2;
waypoints		= [];
wp_index		= 0;
arena			= { x1: 0, y1: 0, x2: room_width, y2: room_height };
prev_dist		= 0;

is_player = false;

// freeze on gradient update
paused     = false;
frozen_vx  = 0;
frozen_vy  = 0;
frozen_av  = 0;

// raycast sensor to detect the environment, local space
raycast_sensor = new Raycaster(id, make_even_directions(8), RAY_RANGE, obj_solid, SOLIDMAP, false);

#endregion
#region Helpers

/// @desc Rotate a world-space direction into the drone's body frame.
///       Rotation only for directions/velocities, not points.
world_dir_to_local = function(_dx, _dy) {
	var _v = new Vec2(_dx, _dy);
	_v.rotate(phy_rotation);
	return _v;
};

/// @desc Rotate a body-space direction into world space. Inverse of world_dir_to_local.
local_dir_to_world = function(_dx, _dy) {
	var _v = new Vec2(_dx, _dy);
	_v.rotate(-phy_rotation);
	return _v;
};

/// @desc Transform a body-space point to world coordinates (rotate, then translate
///       by the body position).
local_point_to_world = function(_lx, _ly) {
	var _v = local_dir_to_world(_lx, _ly);
	_v.set(_v.x + phy_position_x, _v.y + phy_position_y);
	return _v;
};

/// @desc 8 body-frame ray depths. 1.0 = clear, 0 = wall at the sensor.
get_ray_observations = function() {
	raycast_sensor.directions = make_even_directions(8, -phy_rotation);
	raycast_sensor.cast_all();
	var _n = raycast_sensor.ray_count;
	var _rays = array_create(_n, 1.0);
	for (var i = 0; i < _n; i++) {
		if (raycast_sensor.results[i] == noone) continue;
		_rays[i] = clamp(raycast_sensor.results[i] / raycast_sensor.range, 0.0, 1.0);
	}
	return _rays;
};

/// @desc Apply both thruster forces.
apply_thrust = function() {
	physics_apply_local_force(momentArm, 0, 0, -thrustMax * thrustR);
	physics_apply_local_force(-momentArm, 0, 0, -thrustMax * thrustL);
};

/// @desc Simulate air resistance on velocity and angular velocity.
apply_resistance = function() {
	phy_speed_x *= 0.999;
	phy_speed_y *= 0.999;
	phy_angular_velocity *= 0.999;
}

/// @desc Ordered course of WAYPOINT_COUNT points, each kept BORDER_MARGIN from arena edges.
scatter_waypoints = function() {
	var _wps = [];
	repeat (WAYPOINT_COUNT) {
		array_push(_wps, {
			x: random_range(arena.x1 + BORDER_MARGIN, arena.x2 - BORDER_MARGIN),
			y: random_range(arena.y1 + BORDER_MARGIN, arena.y2 - BORDER_MARGIN),
		});
	}
	return _wps;
};

/// @desc Reset pose to a random spot in the arena (with a 15% margin to edges), zero velocities.
reset_drone = function(_arena) {
	arena = _arena;
	phy_position_x       = lerp(arena.x1, arena.x2, random_range(0.15, 1 - 0.15));
	phy_position_y       = lerp(arena.y1, arena.y2, random_range(0.15, 1 - 0.15));
	phy_speed_x          = 0;
	phy_speed_y          = 0;
	phy_angular_velocity = 0;
	phy_rotation         = 0;
	thrustL = 0;
	thrustR = 0;
};

/// @desc Advance the waypoint target if within capture range. Returns "none" | "captured" | "complete".
try_capture = function() {
	var _cur = waypoints[min(wp_index, array_length(waypoints) - 1)];
	if (point_distance(phy_position_x, phy_position_y, _cur.x, _cur.y) >= CAPTURE_RADIUS) return "none";
	wp_index++;
	if (wp_index >= array_length(waypoints)) return "complete";
	var _new = waypoints[wp_index];
	prev_dist = point_distance(phy_position_x, phy_position_y, _new.x, _new.y);
	return "captured";
};

#endregion
#region GMML

event_inherited();
behavior_name = "drone";
gmml_set_spec(17, gmml_continuous(2));   // [left, right] thruster

agent_reset = function() {
	reset_drone(arena);
	waypoints     = scatter_waypoints();
	wp_index      = 0;
	waypoint_time = 0;
	prev_thrustL  = 0;
	prev_thrustR  = 0;
	prev_dist     = point_distance(phy_position_x, phy_position_y, waypoints[0].x, waypoints[0].y);
};

agent_pause = function() {
	frozen_vx  = phy_speed_x;
	frozen_vy  = phy_speed_y;
	frozen_av  = phy_angular_velocity;
	phy_active = false;
	paused     = true;
};

agent_resume = function() {
	phy_active           = true;
	phy_speed_x          = frozen_vx;
	phy_speed_y          = frozen_vy;
	phy_angular_velocity = frozen_av;
	paused               = false;
};

collect_observations = function() {
	// angular velocity
	add_observation(clamp(phy_angular_velocity / maxAngVel, -1.0, 1.0));

	// velocity in body frame
	var _v = world_dir_to_local(phy_speed_x, phy_speed_y);
	add_observation(clamp(_v.x / maxSpeed, -1.0, 1.0));
	add_observation(clamp(_v.y / maxSpeed, -1.0, 1.0));

	// world-up in body frame
	var _up = world_dir_to_local(0, -1);
	add_observation(_up.x);
	add_observation(_up.y);

	// 8 body-frame rays
	var _rays = get_ray_observations();
	for (var i = 0; i < 8; i++) add_observation(_rays[i]);

	// delta to current + next waypoint in body frame, normalized by arena diagonal
	var _diag = point_distance(arena.x1, arena.y1, arena.x2, arena.y2);
	var _last = array_length(waypoints) - 1;

	var _cur = waypoints[min(wp_index, _last)];
	var _nxt = waypoints[min(wp_index + 1, _last)];

	var _dc  = world_dir_to_local(_cur.x - phy_position_x, _cur.y - phy_position_y);
	var _dn  = world_dir_to_local(_nxt.x - phy_position_x, _nxt.y - phy_position_y);
	
	add_observation(clamp(_dc.x / _diag, -1.0, 1.0));
	add_observation(clamp(_dc.y / _diag, -1.0, 1.0));

	add_observation(clamp(_dn.x / _diag, -1.0, 1.0));
	add_observation(clamp(_dn.y / _diag, -1.0, 1.0));
};

on_action = function(_action) {
	thrustL = (clamp(_action[0], -1, 1) + 1) * 0.5;
	thrustR = (clamp(_action[1], -1, 1) + 1) * 0.5;

	set_reward(0);

	// energy cost: discourage blasting both thrusters at full
	add_reward(-THRUST_COST * (thrustL + thrustR));

	// progress toward the current waypoint
	var _cur  = waypoints[min(wp_index, array_length(waypoints) - 1)];
	var _dist = point_distance(phy_position_x, phy_position_y, _cur.x, _cur.y);
	add_reward((prev_dist - _dist) * PROGRESS_SCALE);
	prev_dist = _dist;

	var _cap = try_capture();
	if (_cap != "none") {
		add_reward(WAYPOINT_REWARD);
		waypoint_time = 0;
	}
	if (_cap == "complete") {
		add_reward(COMPLETE_REWARD);
		on_success();
		end_episode();
		return;
	}

	// crash detection
	var _rays = get_ray_observations();
	var _near = 1.0;
	for (var i = 0; i < array_length(_rays); i++) _near = min(_near, _rays[i]);
	var _oob = (phy_position_x < arena.x1 || phy_position_x > arena.x2
				|| phy_position_y < arena.y1 || phy_position_y > arena.y2);
	if (_near < CRASH_RAY || _oob) {
		add_reward(CRASH_PENALTY);
		on_failure(true);
		end_episode();
		return;
	}

	// per-waypoint timeout
	waypoint_time += DT;
	if (waypoint_time >= waypoint_time_max) {
		add_reward(WP_TIMEOUT_PENALTY);
		on_failure(false);
		end_episode();
		return;
	}

	add_reward(STEP_PENALTY); // encourage speed

	// flight quality
	var _up = world_dir_to_local(0, -1);
	add_reward(UPRIGHT_PENALTY * (1 + _up.y));
	add_reward(ANGVEL_PENALTY * abs(phy_angular_velocity / maxAngVel));
	add_reward(THRUST_JERK_COST * (abs(thrustL - prev_thrustL) + abs(thrustR - prev_thrustR)));
	prev_thrustL = thrustL;
	prev_thrustR = thrustR;
};

on_success = function() {
	with (obj_drone_controller) {
		successes ++;
		log_standings();
	}
}

on_failure = function(crashed) {
	with (obj_drone_controller) {
		if (crashed) failures_crash ++;
		else failures_timeout ++;
		log_standings();
	}
}

#endregion
#region FX

particle_thrust = new Particle();
particle_thrust.alpha0 = 0;
particle_thrust.alpha1 = 1;
particle_thrust.color0 = c_red;
particle_thrust.color1 = make_color_hsv(40, 255, 255);
particle_thrust.life = sec_to_steps(0.5);
particle_thrust.life_init = sec_to_steps(0.5);
particle_thrust.size0 = 2;
particle_thrust.size1 = 4;
particle_thrust.damp = 0.9;
particle_thrust.vel.x = 2;

emit_thrust_particles = function(_pos, _thrust_factor) {
	if (!instance_exists(obj_particle_manager)) return;
	
	_thrust_factor *= _thrust_factor;
	
	var _p = particle_thrust.duplicate(),
		_v = _p.vel.get_magnitude() * (0.5 + _thrust_factor),
		_pv = local_dir_to_world(0, _v);
	_pv.set(_pv.x + phy_speed_x, _pv.y + phy_speed_y);
	_p.vel.copy(_pv);
	_p.size1 *= 1 + _thrust_factor;
	_p.life = _p.life_init * (_thrust_factor*0.9 + 0.1);
	_p.life_init = _p.life;
	_p.pos.copy(_pos);
	
	PARTICLE_MANAGER.add_particle(_p);
}

handle_thrust_particles = function() {
	var _pl = local_point_to_world(-momentArm - 0.5 - 2, 0),
		_pr = local_point_to_world( momentArm - 0.5 + 2, 0);
	
	emit_thrust_particles(_pl, thrustL);
	emit_thrust_particles(_pr, thrustR);
}

#endregion
