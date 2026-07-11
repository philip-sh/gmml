depth --;

global.particle_manager = id;
#macro PARTICLE_MANAGER global.particle_manager

active_particles = [];

add_particle = function(_particle) {
	array_push(active_particles, _particle);
}

update = function() {
	for (var i=array_length(active_particles)-1; i>=0; --i) {
		if (!active_particles[i].update()) array_delete(active_particles, i, 1);
	}
}

draw = function() {
	for (var i=0; i<array_length(active_particles); ++i) {
		active_particles[i].draw();
	}
}