depth += 1;

#region Params

DRONE_COUNT		= (room_width * room_height) div (288 * 288);
WAYPOINT_COUNT	= 4;
BORDER_MARGIN	= GRIDW * 2; // for waypoints
CAPTURE_RADIUS	= GRIDW * 1.5;
WALL_INSET		= GRIDW;

#endregion
#region Arenas

var _cells	= ceil(sqrt(DRONE_COUNT));
var _cell_w = room_width  div _cells;
var _cell_h = room_height div _cells;

drones = [];
var _made = 0;
for (var _cy = 0; _cy < _cells && _made < DRONE_COUNT; _cy++) {
	for (var _cx = 0; _cx < _cells && _made < DRONE_COUNT; _cx++) {
		var _ox = _cx * _cell_w;
		var _oy = _cy * _cell_h;

		var _arena = {
			x1: _ox + WALL_INSET,
			y1: _oy + WALL_INSET,
			x2: _ox + _cell_w - WALL_INSET,
			y2: _oy + _cell_h - WALL_INSET,
		};

		// solids
		var _wcols = _cell_w div GRIDW;
		var _wrows = _cell_h div GRIDH;
		for (var i = 0; i < _wcols; i++) {
			for (var j = 0; j < _wrows; j++) {
				if (i == 0 || j == 0 || i == _wcols - 1 || j == _wrows - 1) {
					instance_create_layer(_ox + i * GRIDW, _oy + j * GRIDH, "Instances", obj_solid);
				}
			}
		}

		var _d = instance_create_layer(
			(_arena.x1 + _arena.x2) * 0.5, (_arena.y1 + _arena.y2) * 0.5,
			"Instances", obj_agent_drone);
		_d.arena			= _arena;
		_d.WAYPOINT_COUNT	= WAYPOINT_COUNT;
		_d.BORDER_MARGIN	= BORDER_MARGIN;
		_d.CAPTURE_RADIUS	= CAPTURE_RADIUS;
		_d.is_player		= (_made == 0);   // first drone can be played with PLAYER_OVERRIDE
		_d.agent_reset();

		array_push(drones, _d);
		_made++;
	}
}

#endregion
#region GMML

with (obj_gmml_academy) connect(5555);

#endregion

reset_stats_every = 1000; // set to -1 to never reset
successes = 0;
failures_crash = 0;
failures_timeout = 0;

log_standings = function() {
	var _total = successes + failures_crash + failures_timeout;
	if (reset_stats_every > 0 && _total > reset_stats_every) {
		successes = 0;
		failures_crash = 0;
		failures_timeout = 0;
		return;
	}
	
	var _str = string(
		"-----------------------------"
		+ "\nsuccesses: {0}, {1}%\ncrashes: {2}, {3}%\ntimeouts: {4}, {5}%",
		successes,
		round((successes/_total)*100),
		failures_crash,
		round((failures_crash/_total)*100),
		failures_timeout,
		round((failures_timeout/_total)*100)
	);
	show_debug_message(_str);
}