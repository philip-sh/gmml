/// Singleton transport/orchestrator.
/// Place one in your room and call connect().
///
/// Agents are discovered by iterating 'with (obj_gmml_agent)', so there is no 
/// registration order to get wrong. A controller that owns spawning/level layout 
/// can set 'on_reset' to rebuild the scene when Python requests a reset.

server   = -1;
client   = -1;
ready    = false;
on_reset = undefined; // optional function for global reset

inbox     = buffer_create(1024, buffer_grow, 1);
inbox_len = 0;

// Reset model, set from the handshake. false = global reset (on_reset rebuilds the whole
// scene, used by single-agent + self-play). true = per-agent auto-reset (each agent resets
// itself via agent_reset(), used by GMVecEnv / vectorized training).

// reset model: true	- per-agent auto-reset (agents reset themselves with their own agent_reset function)
//				false	- global reset with this object's on_reset function
global.__gmml_auto_reset = false;

connect = function(_port = 5555) {
	server = network_create_server_raw(network_socket_tcp, _port, 1);
	if (server < 0) show_error("gmml: failed to create server on port " + string(_port), true);
};

send = function(_data) {
	var _packet = json_stringify(_data) + "\n";
	var _buf = buffer_create(string_byte_length(_packet), buffer_fixed, 1);
	buffer_write(_buf, buffer_text, _packet);
	network_send_raw(client, _buf, buffer_get_size(_buf));
	buffer_delete(_buf);
};

// Handshake: one entry per distinct behavior present in the room.
build_spec = function() {
	var _behaviors = {};
	with (obj_gmml_agent) {
		if (!variable_struct_exists(_behaviors, behavior_name)) {
			_behaviors[$ behavior_name] = { obs_size: obs_size, action: action_spec };
		}
	}
	return { type: "spec", protocol: GMML_PROTOCOL, behaviors: _behaviors };
};

// One step packet: a record per agent currently in the room.
build_step = function() {
	var _agents = [];
	with (obj_gmml_agent) {
		array_push(_agents, __gmml_collect());
	}
	return { type: "step", agents: _agents };
};

// Dispatch actions {agent_id: [..]} to the matching agent instances.
apply_actions = function(_actions) {
	var _keys = variable_struct_get_names(_actions);
	for (var i = 0; i < array_length(_keys); i++) {
		var _key  = _keys[i];
		var _inst = real(_key);            // agent_id is the instance id
		if (instance_exists(_inst)) {
			with (_inst) on_action(_actions[$ _key]);
		}
	}
};

// Read '_len' bytes at '_off' of '_buf' as a UTF-8 string. Copy into a NUL-terminated temp so
// buffer_string reconstructs multibyte content correctly.
__gmml_line_to_string = function(_buf, _off, _len) {
	var _tmp = buffer_create(_len + 1, buffer_fixed, 1);
	buffer_copy(_buf, _off, _len, _tmp, 0);
	buffer_poke(_tmp, _len, buffer_u8, 0);   // NUL terminator
	buffer_seek(_tmp, buffer_seek_start, 0);
	var _s = buffer_read(_tmp, buffer_string);
	buffer_delete(_tmp);
	return _s;
};

// Handle one fully-framed protocol message. Called in networking event per complete line,
// so the event handler stays pure transport and the protocol logic lives here.
dispatch = function(_msg) {
	switch (_msg.type) {
		case "handshake":
			global.__gmml_auto_reset = _msg[$ "auto_reset"] ?? false;
			send(build_spec());
			break;

		case "reset":
			if (global.__gmml_auto_reset) {
				with (obj_gmml_agent) agent_reset();
			} else if (on_reset != undefined) {
				on_reset();
			}
			send(build_step());
			break;

		case "action":
			apply_actions(_msg.actions);
			send(build_step());
			break;

		case "pause":
			with (obj_gmml_agent) agent_pause();
			send({ type: "ack" });					// Don't forget ack, raw TCP
			break;

		case "resume":
			with (obj_gmml_agent) agent_resume();
			send({ type: "ack" });
			break;

		case "close":
			game_end();
			break;
	}
};
