if (DEBUG) {
	var _str = string("fps: {0}\nfps real: {1}\nroom_speed: {2}\ngamespeed: {3}\ntraining mode: {4}",
		fps, fps_real, room_speed, game_get_speed(gamespeed_fps), TRAINING_MODE ? "ON" : "OFF");
	
	draw_text(2, 2, _str);
}