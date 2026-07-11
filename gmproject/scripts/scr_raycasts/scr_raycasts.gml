/// @description Casts a ray from a point in a direction and returns a RaycastHit.
/// @param {real}			xx       Ray origin x.
/// @param {real}			yy       Ray origin y.
/// @param {real}			dir      Direction in degrees.
/// @param {real}			range    Maximum ray distance.
/// @param {Asset.GMObject} object   Object to check collision against.
/// @param {bool}			prec     Precise collision check.
/// @param {bool}			notme    Exclude calling instance.
/// @returns {Struct.RaycastHit}
function raycast(xx, yy, dir, range, object, prec, notme) {
    prec = true;
    var ox, oy, dx, dy, sx, sy;
    ox = xx;
    oy = yy;
    sx = lengthdir_x(range, dir);
    sy = lengthdir_y(range, dir);
    dx = ox + sx;
    dy = oy + sy;

    var _instance = collision_line(ox, oy, dx, dy, object, prec, notme);

    if (_instance == noone) {
        return new RaycastHit(-1, -1, -1, noone);
    }

    while ((abs(sx) >= 1) || (abs(sy) >= 1)) {
        sx /= 2;
        sy /= 2;
        if (collision_line(ox, oy, dx, dy, object, prec, notme) == noone) {
            dx += sx;
            dy += sy;
        } else {
            dx -= sx;
            dy -= sy;
        }
    }

    // Re-fetch the instance at the refined hit point
    _instance = collision_line(ox, oy, dx, dy, object, prec, notme);

    return new RaycastHit(point_distance(ox, oy, dx, dy), dx, dy, _instance);
}

