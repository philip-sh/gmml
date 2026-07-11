var _tilemapCollision = layer_tilemap_get_id("Tilemap_Collision"),
	_useTilemap = (_tilemapCollision != -1);

var _w = (_useTilemap) ? tilemap_get_width(_tilemapCollision)	: room_width div GRIDW,
	_h = (_useTilemap) ? tilemap_get_height(_tilemapCollision)	: room_height div GRIDH;

array_resize(SOLIDMAP, _w);
for (var i=0; i<_w; ++i) {
	SOLIDMAP[i] = array_create(_h);
	for (var j=0; j<_h; ++j) {
		if (_useTilemap) SOLIDMAP[i][j] = tilemap_get(_tilemapCollision, i, j);
		else SOLIDMAP[i][j] = instance_position((i + 0.5)*GRIDW, (j + 0.5)*GRIDH, obj_solid) == noone ? false : true;
	}
}