function Raycaster(origin_obj, directions, range, target_obj, grid, prec) constructor {
    self.origin_obj  = origin_obj;   // The object instance casting rays (usually 'id')
    self.directions  = directions;   // Array of angles to cast, e.g. [0, 90, 180, 270]
    self.range       = range;        // Max ray length in pixels
    self.target_obj  = target_obj;   // Object to collide against
	self.grid		 = grid;		 // Grid to use if using _cast_dda
    self.prec        = prec;         // Precise collision (true/false)

    // Results: one entry per direction, either a distance or 'noone' if no hit
    self.results = array_create(array_length(directions), noone);
	
	self.ray_count = array_length(directions);
	
	prev_cast_offset_x = 0;
	prev_cast_offset_y = 0;
		
    static _cast_bisect = function(ox, oy, dir) {
        var sx = lengthdir_x(range, dir);
        var sy = lengthdir_y(range, dir);
        var dx = ox + sx;
        var dy = oy + sy;

        if (collision_line(ox, oy, dx, dy, target_obj, prec, origin_obj) == noone) {
            return noone;
        }

        while ((abs(sx) >= 1) || (abs(sy) >= 1)) {
            sx /= 2;
            sy /= 2;
            if (collision_line(ox, oy, dx, dy, target_obj, prec, origin_obj) == noone) {
                dx += sx;
                dy += sy;
            } else {
                dx -= sx;
                dy -= sy;
            }
        }

        return point_distance(ox, oy, dx, dy);
    };
	
	static _cast_dda = function(ox, oy, dir) {
	    var cell = GRIDW; // cell size in pixels

	    // Ray direction vector
	    var rdx = lengthdir_x(1, dir);
	    var rdy = lengthdir_y(1, dir);

	    // Current cell
	    var cell_x = floor(ox / cell);
	    var cell_y = floor(oy / cell);

	    // How far along the ray we must travel to cross one full cell in each axis
	    var delta_x = (rdx == 0) ? infinity : abs(cell / rdx);
	    var delta_y = (rdy == 0) ? infinity : abs(cell / rdy);

	    // Step direction and initial boundary distance
	    var step_x, step_y, side_x, side_y;

	    if (rdx < 0) {
	        step_x = -1;
	        side_x = (ox - cell_x * cell) / abs(rdx) * -1 + delta_x;
	        // Distance to left boundary
	        side_x = ((ox - cell_x * cell) / cell) * delta_x;
	    } else {
	        step_x = 1;
	        side_x = (((cell_x + 1) * cell - ox) / cell) * delta_x;
	    }

	    if (rdy < 0) {
	        step_y = -1;
	        side_y = ((oy - cell_y * cell) / cell) * delta_y;
	    } else {
	        step_y = 1;
	        side_y = (((cell_y + 1) * cell - oy) / cell) * delta_y;
	    }

	    // March
	    var max_steps = ceil(range / cell) + 1;
	    var hit_dist = noone;

	    repeat (max_steps) {
	        if (hit_dist != noone) break;

	        // Step to nearest boundary
	        var hit_x, hit_y;
	        if (side_x < side_y) {
	            hit_dist_candidate = side_x;
	            side_x += delta_x;
	            cell_x += step_x;
	            // Boundary hit is on the vertical cell edge
	            hit_x = cell_x * cell + (step_x < 0 ? cell : 0);
	            hit_y = oy + rdy * hit_dist_candidate;
	        } else {
	            hit_dist_candidate = side_y;
	            side_y += delta_y;
	            cell_y += step_y;
	            // Boundary hit is on the horizontal cell edge
	            hit_x = ox + rdx * hit_dist_candidate;
	            hit_y = cell_y * cell + (step_y < 0 ? cell : 0);
	        }

	        // Range check
	        if (hit_dist_candidate > range) break;

	        // Grid bounds check
	        if (cell_x < 0 || cell_y < 0
	        ||  cell_x >= array_length(grid)
	        ||  cell_y >= array_length(grid[0])) break;

	        // Grid lookup
	        if (grid[cell_x][cell_y]) {
	            hit_dist = hit_dist_candidate;
	        }
	    }

	    return hit_dist;
	};
	
	// --- Cast a single ray, returns distance or noone ---
	static _cast_single = function(ox, oy, dir) {
		return _cast_dda(ox, oy, dir);
	}

    // --- Cast all rays from origin_obj's current position ---
    static cast_all = function() {
		prev_cast_offset_x = 0;
		prev_cast_offset_y = 0;
		
        var ox = origin_obj.x;
        var oy = origin_obj.y;
        var i = 0;
        repeat (array_length(directions)) {
            results[i] = _cast_single(ox, oy, directions[i]);
            i++;
        }
    };

    // --- Cast from an explicit offset relative to origin_obj ---
    static cast_all_from_offset = function(offset_x, offset_y) {
		prev_cast_offset_x = offset_x;
		prev_cast_offset_y = offset_y;
		
        var ox = origin_obj.x + offset_x;
        var oy = origin_obj.y + offset_y;
        var i = 0;
        repeat (array_length(directions)) {
            results[i] = _cast_single(ox, oy, directions[i]);
            i++;
        }
    };

    // --- Draw all rays based on last cast_all() results ---
    // hit_color:  color of rays that struck something  (default: c_red)
    // miss_color: color of rays that reached max range (default: c_green)
    // alpha:      draw alpha                           (default: 1)
    static draw = function(hit_color, miss_color, alpha) {
        hit_color  = (is_undefined(hit_color))  ? c_red   : hit_color;
        miss_color = (is_undefined(miss_color)) ? c_green : miss_color;
        alpha      = (is_undefined(alpha))      ? 1       : alpha;

        var ox = origin_obj.x + prev_cast_offset_x - 1;
        var oy = origin_obj.y + prev_cast_offset_y - 1;
        var prev_color = draw_get_color();
        var prev_alpha = draw_get_alpha();
        draw_set_alpha(alpha);

        var i = 0;
        repeat (array_length(directions)) {
            var dist = results[i];
            var dir  = directions[i];

            if (dist == noone) {
                // No hit - draw full-range ray
                draw_set_color(miss_color);
                draw_line(ox, oy,
                          ox + lengthdir_x(range, dir),
                          oy + lengthdir_y(range, dir));
            } else {
                // Hit - draw up to the hit point, then a small cross
                draw_set_color(hit_color);
                var hx = ox + lengthdir_x(dist, dir);
                var hy = oy + lengthdir_y(dist, dir);
                draw_line(ox, oy, hx, hy);
                draw_line(hx - 3, hy - 3, hx + 3, hy + 3);
                draw_line(hx + 3, hy - 3, hx - 3, hy + 3);
            }
            i++;
        }

        draw_set_color(prev_color);
        draw_set_alpha(prev_alpha);
    };

    // --- Convenience getters ---

    // Returns the distance for a given direction index, or noone
    static get_distance = function(index) {
        return results[index];
    };

    // Returns true if the ray at index hit something
    static hit = function(index) {
        return (results[index] != noone);
    };

    // Returns the actual hit position for a given index, or undefined if no hit
    static get_hit_pos = function(index) {
        if (results[index] == noone) return undefined;
        var ox  = origin_obj.x;
        var oy  = origin_obj.y;
        var dir = directions[index];
        return {
            x: ox + lengthdir_x(results[index], dir),
            y: oy + lengthdir_y(results[index], dir)
        };
    };
}