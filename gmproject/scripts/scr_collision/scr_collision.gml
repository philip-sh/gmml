/// @description Predicts tile collision for using the instance's velocity (Vec2)
///              using a DDA step algorithm.
/// @param {Id.TileMapElement}	tilemap  The collision tilemap layer ID.
/// @param {real}				bbox_l   Left offset of collision box from origin.
/// @param {real}				bbox_r   Right offset of collision box from origin.
/// @param {real}				bbox_t   Top offset of collision box from origin.
/// @param {real}				bbox_b   Bottom offset of collision box from origin.
function collide_tilemap(tilemap, bbox_l, bbox_r, bbox_t, bbox_b) {

    var dx = velocity.x * DT;
    var dy = velocity.y * DT;

    var sx = sign(dx);
    var sy = sign(dy);

    if (sx != 0 && tilemap_overlap(tilemap, x + sx, y, bbox_l, bbox_r, bbox_t, bbox_b))
        velocity.x = 0;
    if (sy != 0 && tilemap_overlap(tilemap, x, y + sy, bbox_l, bbox_r, bbox_t, bbox_b))
        velocity.y = 0;

    dx = velocity.x * DT;
    dy = velocity.y * DT;

    if (!tilemap_overlap(tilemap, x + dx, y + dy, bbox_l, bbox_r, bbox_t, bbox_b)) exit;

    var magnitude = sqrt(dx * dx + dy * dy);
    if (magnitude == 0) exit;

    var step_x = dx / magnitude;
    var step_y = dy / magnitude;

    var walked = 0;
    var curr_x = x;
    var curr_y = y;

    while (walked < magnitude) {
        var next_x = curr_x + step_x;
        var next_y = curr_y + step_y;

        if (tilemap_overlap(tilemap, next_x, next_y, bbox_l, bbox_r, bbox_t, bbox_b)) {
            var hit_x = tilemap_overlap(tilemap, next_x, curr_y, bbox_l, bbox_r, bbox_t, bbox_b);
            var hit_y = tilemap_overlap(tilemap, curr_x, next_y, bbox_l, bbox_r, bbox_t, bbox_b);

            if (hit_x && hit_y) {
                if (abs(velocity.x) >= abs(velocity.y))
                    hit_y = false;
                else
                    hit_x = false;
            }

            if (hit_x) velocity.x = 0;
            if (hit_y) velocity.y = 0;

            x = curr_x;
            y = curr_y;
            return;
        }

        curr_x  = next_x;
        curr_y  = next_y;
        walked += 1;
    }
}

/// @description Checks if a given bbox in world position overlaps a solid tile.
/// @param {Id.TileMapElement}	tilemap		The collision tilemap layer ID.
/// @param {real}				px			World x position to test.
/// @param {real}				py			World y position to test.
/// @param {real}				bbox_l		Left offset of collision box from origin.
/// @param {real}				bbox_r		Right offset of collision box from origin.
/// @param {real}				bbox_t		Top offset of collision box from origin.
/// @param {real}				bbox_b		Bottom offset of collision box from origin.
function tilemap_overlap(tilemap, px, py, bbox_l, bbox_r, bbox_t, bbox_b) {
    return (tilemap_get_at_pixel(tilemap, px + bbox_l, py + bbox_t) & tile_index_mask) ||
           (tilemap_get_at_pixel(tilemap, px + bbox_r, py + bbox_t) & tile_index_mask) ||
           (tilemap_get_at_pixel(tilemap, px + bbox_l, py + bbox_b) & tile_index_mask) ||
           (tilemap_get_at_pixel(tilemap, px + bbox_r, py + bbox_b) & tile_index_mask);
}