DT = (delta_time / 1000000) * timeScale;

if (keyboard_check_pressed(vk_f6)) DEBUG = !DEBUG;
if (keyboard_check_pressed(vk_f7)) toggleTrainingMode();
if (keyboard_check_pressed(vk_f12)) game_end();
if (keyboard_check_pressed(ord("P"))) PLAYER_OVERRIDE = !PLAYER_OVERRIDE;