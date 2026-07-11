if (PLAYER_OVERRIDE && is_player) {
    if (keyboard_check_pressed(ord("R"))) agent_reset();

	var _f = keyboard_check(ord("S")) || keyboard_check(ord("W")) ? 0.5 : 0;
	thrustL = real(keyboard_check(ord("A")))*0.5 + _f;
	thrustR = real(keyboard_check(ord("D")))*0.5 + _f;

	if (try_capture() == "complete") agent_reset();

	get_ray_observations();
}

if (!paused) {
	apply_thrust();
	//apply_resistance();
}

handle_thrust_particles();