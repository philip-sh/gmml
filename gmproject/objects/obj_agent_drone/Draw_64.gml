if (point_in_rectangle(mouse_x, mouse_y, arena.x1, arena.y1, arena.x2, arena.y2)) {
	var _str_l = string("vel:\nvel mag:\nang vel:\nwaypoint time:"),
		_str_r = string("{0}, {1}\n{2}\n{3}\n{4}", 
			string_format(phy_speed_x, 3, 1), string_format(phy_speed_y, 3, 1),
			string_format(point_distance(0, 0, phy_speed_x, phy_speed_y), 3, 1),
			string_format(phy_angular_velocity, 3, 1),
			string_format(waypoint_time_max - waypoint_time, 2, 1)
		);
	
	var _w = string_width(_str_l) + string_width(_str_r) + string_width(" "),
		_h = max(string_height(_str_l), string_height(_str_r)),
		_pad = 10,
		_gui_w = display_get_gui_width(),
		_sprite_backdrop = spr_12x12;
	
	// Backdrop
	draw_sprite_ext(
		_sprite_backdrop, 
		0,
		_gui_w - _w - 2*_pad,
		0,
		(_w + 2*_pad) / sprite_get_width(_sprite_backdrop),
		(_h + 2*_pad) / sprite_get_height(_sprite_backdrop),
		0,
		c_black,
		1
	);
	
	// Debug table
	var _halign = draw_get_halign();
	draw_set_halign(fa_left);
	draw_text(_gui_w - _w - _pad, _pad, _str_l);
	draw_set_halign(fa_right);
	draw_text(_gui_w - _pad, _pad, _str_r);
	draw_set_halign(_halign);
}