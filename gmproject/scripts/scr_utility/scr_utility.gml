function make_even_directions(count, offset) {
    offset = (is_undefined(offset)) ? 0 : offset;
    var dirs = array_create(count);
    var step = 360 / count;
    for (var i = 0; i < count; i++) {
        dirs[i] = (offset + step * i) % 360;
    }
    return dirs;
}

/// @description Rotates an angle toward a target angle by at most maxTurnDegrees.
///              Correctly handles wraparound (e.g. 350 degrees -> 10 degrees).
/// @param {real}	_current	Current angle in degrees.
/// @param {real}	_target		Desired angle in degrees.
/// @param {real}	_maxTurn	Maximum degrees to rotate this call.
/// @returns {real}				The new angle in degrees.
function angle_rotate_towards(_current, _target, _maxTurn) {
    var _delta = angle_difference(_target, _current);
    return _current + clamp(_delta, -_maxTurn, _maxTurn);
}

function sec_to_steps(_seconds) {
	return game_get_speed(gamespeed_fps)*_seconds;
}

function curve_value_get(curveAsset, channelName, curvePosition){
	var _curveStruct = animcurve_get(curveAsset);
	var _channelY = animcurve_get_channel(_curveStruct, channelName);
	var _curveY = animcurve_channel_evaluate(_channelY, curvePosition);
	
	return _curveY;
}