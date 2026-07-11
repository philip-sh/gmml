// arena bounds
draw_rectangle_color(arena.x1, arena.y1, arena.x2, arena.y2, c_maroon, c_maroon, c_maroon, c_maroon, true);

// waypoints
var _ring = merge_color(c_lime, c_black, 0.75);
draw_circle_color(waypoints[wp_index].x, waypoints[wp_index].y, CAPTURE_RADIUS, _ring, c_black, false);

var _n = array_length(waypoints);
var _col0, _col1;
for (var i = 0; i < _n - 1; i++) {
	var _c = (i == wp_index);
	
	if (i < wp_index) {
		_col0 = make_color_hsv(0,0,40);
		_col1 = _col0;
	} else {
		_col0 = _c ? merge_color(c_lime, c_black, 0.75) : c_dkgray;
		_col1 = _c ? merge_color(c_yellow, c_black, 0.75) : c_dkgray;
	}
	
    draw_line_color(waypoints[i].x, waypoints[i].y, waypoints[i + 1].x, waypoints[i + 1].y, _col0, _col1);
}
for (var i = 0; i < _n; i++) {
    var _is_current = (i == wp_index);
    var _col = _is_current ? c_lime : ((i == wp_index + 1) ? c_yellow : (i < wp_index ? make_color_hsv(0,0,40) : c_dkgray));
    draw_circle_color(waypoints[i].x, waypoints[i].y, 3, _col, _col, !_is_current);
}

// drone
draw_self();

if (DEBUG) {
	//var _up = local_dir_to_world(0, -GRIDW*2);
	//draw_line(x, y, x + _up.x, y + _up.y);

	raycast_sensor.draw();
}
