/// @description A raycast hit result.
/// @param {real}			distance   Distance from origin to hit point, -1 if no hit.
/// @param {real}			x          World x of hit point, -1 if no hit.
/// @param {real}			y          World y of hit point, -1 if no hit.
/// @param {Id.Instance}	instance   The instance hit, noone if no hit.
function RaycastHit(distance, x, y, instance) constructor {
    self.distance	= distance;
    self.x			= x;
    self.y			= y;
    self.instance	= instance;
    self.hit		= (instance != noone);
}