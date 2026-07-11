/// In-engine inference for '.gmpolicy' policies exported by 'gmml export --format gmpolicy'.
///
/// Runs a trained policy entirely in GML with no Python, no DLLs and no network. Load the
/// .gmpolicy once, then call predict every step with the same observation vector your
/// agent's collect_observations() builds.
///
///     policy = gmpolicy_load("drone.gmpolicy");   // .gmpolicy added as an Included File
///     ...
///     var _action = gmpolicy_predict(policy, _obs_array);
///     on_action(_action);                          // reuse your existing action handler
///
/// Acting is deterministic and matches 'gmml infer': continuous behaviors return the
/// Gaussian mean clipped to the action range (an array of floats), discrete/multi-discrete
/// behaviors return one argmax index per branch.

#macro GMPOLICY_ACTION_DISCRETE   0
#macro GMPOLICY_ACTION_CONTINUOUS 1
#macro GMPOLICY_ACT_TANH          0
#macro GMPOLICY_ACT_RELU          1

/// @desc Load a .gmpolicy file into a policy struct usable by gmpolicy_predict.
/// @param	{String} _filename  path to the .gmpolicy file (an Included File in a build)
/// @return {Struct}
function gmpolicy_load(_filename) {
	var _buf = buffer_load(_filename);
	if (_buf < 0) show_error("gmpolicy: could not load '" + string(_filename) + "'", true);

	var _magic = "";
	repeat (8) _magic += chr(buffer_read(_buf, buffer_u8));
	if (_magic != "GMPOLICY") {
		buffer_delete(_buf);
		show_error("gmpolicy: '" + string(_filename) + "' is not a .gmpolicy file", true);
	}

	var _version = buffer_read(_buf, buffer_u32);
	if (_version != 1) {
		buffer_delete(_buf);
		show_error("gmpolicy: unsupported format version " + string(_version), true);
	}

	var _obs_size    = buffer_read(_buf, buffer_u32);
	var _action_type = buffer_read(_buf, buffer_u8);

	var _branches   = [];
	var _action_size = 0;
	var _low = -1, _high = 1;
	if (_action_type == GMPOLICY_ACTION_DISCRETE) {
		var _n = buffer_read(_buf, buffer_u32);
		for (var b = 0; b < _n; b++) array_push(_branches, buffer_read(_buf, buffer_u32));
	} else {
		_action_size = buffer_read(_buf, buffer_u32);
		_low  = buffer_read(_buf, buffer_f32);
		_high = buffer_read(_buf, buffer_f32);
	}

	var _activation = buffer_read(_buf, buffer_u8);
	var _n_layers   = buffer_read(_buf, buffer_u32);

	var _layers = [];
	for (var l = 0; l < _n_layers; l++) {
		var _in  = buffer_read(_buf, buffer_u32);
		var _out = buffer_read(_buf, buffer_u32);
		// Weights are row-major: element (j, i) lives at index j * _in + i, where j is the
		// output neuron. Matches PyTorch's Linear weight of shape (out, in).
		var _w = array_create(_out * _in);
		for (var k = 0; k < _out * _in; k++) _w[k] = buffer_read(_buf, buffer_f32);
		var _bias = array_create(_out);
		for (var j = 0; j < _out; j++) _bias[j] = buffer_read(_buf, buffer_f32);
		array_push(_layers, { in_features: _in, out_features: _out, w: _w, b: _bias });
	}

	buffer_delete(_buf);

	return {
		obs_size:    _obs_size,
		action_type: _action_type,
		branches:    _branches,
		action_size: _action_size,
		low:         _low,
		high:        _high,
		activation:  _activation,
		layers:      _layers,
	};
}

/// @desc Run the policy on one observation vector and return an action.
/// @param	{Struct}		_policy  a struct returned by gmpolicy_load
/// @param	{Array<Real>}	_obs  observation vector of length _policy.obs_size
/// @return {Array<Real>}	continuous -> the mean vector, discrete -> one index per branch
function gmpolicy_predict(_policy, _obs) {
	var _layers  = _policy.layers;
	var _n       = array_length(_layers);
	var _act     = _policy.activation;
	var _vec     = _obs;

	for (var l = 0; l < _n; l++) {
		var _layer = _layers[l];
		var _in    = _layer.in_features;
		var _out   = _layer.out_features;
		var _w     = _layer.w;
		var _b     = _layer.b;

		var _next = array_create(_out);
		for (var j = 0; j < _out; j++) {
			var _sum = _b[j];
			var _row = j * _in;
			for (var i = 0; i < _in; i++) _sum += _w[_row + i] * _vec[i];
			_next[j] = _sum;
		}

		// Activation after every layer except the last (the action_net output head).
		if (l < _n - 1) {
			for (var j = 0; j < _out; j++) {
				_next[j] = (_act == GMPOLICY_ACT_RELU) ? max(0, _next[j]) : __gmpolicy_tanh(_next[j]);
			}
		}
		_vec = _next;
	}

	if (_policy.action_type == GMPOLICY_ACTION_CONTINUOUS) {
		// Match 'gmml infer', which clips Box actions to the declared range.
		// The engine's on_action typically clamps again, bit of a hack but keeps 
		// the in-engine output identical to the Python evaluation path.
		var _lo = _policy.low, _hi = _policy.high;
		for (var j = 0; j < array_length(_vec); j++) _vec[j] = clamp(_vec[j], _lo, _hi);
		return _vec;
	}

	// Discrete / multi-discrete: the output holds concatenated per-branch logits. Take the
	// argmax within each branch's contiguous slice.
	var _branches = _policy.branches;
	var _actions  = array_create(array_length(_branches));
	var _offset   = 0;
	for (var b = 0; b < array_length(_branches); b++) {
		var _size = _branches[b];
		var _best = 0, _best_val = _vec[_offset];
		for (var c = 1; c < _size; c++) {
			if (_vec[_offset + c] > _best_val) { _best_val = _vec[_offset + c]; _best = c; }
		}
		_actions[b] = _best;
		_offset += _size;
	}
	return _actions;
}

/// @desc Numerically stable tanh, so this does not depend on a runtime tanh() builtin.
function __gmpolicy_tanh(_x) {
	if (_x >  20) return  1;
	if (_x < -20) return -1;
	var _e = exp(2 * _x);
	return (_e - 1) / (_e + 1);
}
