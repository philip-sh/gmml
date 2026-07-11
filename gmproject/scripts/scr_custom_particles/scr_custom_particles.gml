function Particle() constructor {
	self.life = 1;
	self.life_init = life;
	self.color0 = c_black;
	self.color1 = c_white;
	self.alpha0 = 0;
	self.alpha1 = 1;
	self.size0 = 0;
	self.size1 = 1;
	self.sprite = -1;
	self.image = 0;
	self.damp = 1;
	self.pos = new Vec2();
	self.vel = new Vec2();
		
	static duplicate = function() {
		var _p = new Particle();
		
		_p.life			= life;
		_p.life_init	= life_init;
		_p.color0		= color0;
		_p.color1		= color1;
		_p.alpha0		= alpha0;
		_p.alpha1		= alpha1;
		_p.size0		= size0;
		_p.size1		= size1;
		_p.sprite		= sprite;
		_p.image		= image;
		_p.damp			= damp;
		_p.pos.copy(pos);
		_p.vel.copy(vel);
		
		return _p;
	}
	
	static update = function() {
		life --;
		
		pos.add(vel);
		vel.multiply(damp);
		
		return life > 0;
	}
	
	static draw = function() {
		var _l = life/life_init,
			_sz = lerp(size0, size1, _l),
			_col = merge_color(color0, color1, _l),
			_a = lerp(alpha0, alpha1, _l);
		
		if (sprite != -1) draw_sprite_ext(sprite, 0, pos.x, pos.y, _sz, _sz, 0, _col, _a);
		else {
			var _pa = draw_get_alpha();
			draw_set_alpha(_a);
			draw_circle_color(pos.x, pos.y, _sz*0.5, _col, _col, false);
			draw_set_alpha(_pa);
		}
	}
}
