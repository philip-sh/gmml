// Simple 2D Vector Script Files

// A collection of 2D vector scripts geared towards movement

function Vec2(_x = 0, _y = 0) constructor {
	
	x = _x;
	y = _y;
	
	static set = function(_x, _y) {
		x = _x;
		y = _y;
	}
	
	static add = function(_other) {
        x += _other.x;
        y += _other.y;
    }
	
	static subtract = function(_other) {
        x -= _other.x;
        y -= _other.y;
    }
    
    static negate = function() { 
        x = -x;
        y = -y;
    }
	
	static multiply = function(_scalar) {
        x *= _scalar;
        y *= _scalar;
    }
    
	static divide = function(_scalar) {
        x /= _scalar;
        y /= _scalar;
    }
	
	static normalize = function() {
		if ((x != 0) || (y != 0)) {
			var _factor = 1/sqrt((x * x) + (y *y));
			x = _factor * x;
			y = _factor * y;	
		}
	}
	
	static normalized = function() {
		var _norm = new Vec2(x, y);
		_norm.normalize();
		return _norm;
	}
       
	static get_magnitude = function() {
		return point_distance(0, 0, x, y);
    }

	static set_magnitude = function(_magnitude) {
		normalize();
		multiply(_magnitude);
	}
	
	static limit_magnitude = function(_limit) {
		if (get_magnitude() >= _limit) {
			set_magnitude(_limit);
		}
	}
		
	static dot_prod = function(_other) {
		return dot_product(x, y, _other.x, _other.y);
	}

	static distance_from_position = function(_x, _y) {
		var _other = new vector(_x, _y);
		return point_distance(x, y, _other.x, _other.y);
	}
	
	static distance_from_vector = function(_other) {
		return point_distance(x, y, _other.x, _other.y);
	}
    
    static dir = function() {
        return point_direction(0, 0, x, y);
    }
    
    static rotate = function(_degrees) {
        //var _theta, _c, _s, _tx, _ty
		//_theta = (dir() + _degrees) * pi / 180.0; 
		//_c = cos(_theta);
		//_s = sin(_theta);
		//_tx = x * _c - y * _s;
		//_ty = x * _s + y * _c;
		//x = _tx;
		//y = _ty;   
		var _dir = dir() + _degrees,
			_len = get_magnitude();
		set(lengthdir_x(_len, _dir), lengthdir_y(_len, _dir));
    }
	
	static set_rotation = function(_angle) {
		var _theta, _c, _s, _tx, _ty
		_theta = _angle * pi / 180.0; 
		_c = cos(_theta);
		_s = sin(_theta);
		_tx = x * _c - y * _s;
		_ty = x * _s + y * _c;
		x = _tx;
		y = _ty; 
	}
    
	static project_onto = function(_onto) {
		var _magnitudeSquared = _onto.x*_onto.x + _onto.y*_onto.y;
		var _ontoCopy = new Vec2();
		_ontoCopy.copy(_onto);
		
		if (_magnitudeSquared == 0) {
			multiply(0);
			return false;
		}
		
		_ontoCopy.multiply(dot_prod(_onto)/_magnitudeSquared);
		copy(_ontoCopy);
		
		return true;
	}
    static copy = function(_other) {
        x = _other.x;
        y = _other.y;
    }
    
    static to_string = function() {
		return "Vector <" + string(x) + ", " + string(y) + ">"; 
	}
	
	static draw = function(_from, _width, _color) {
		draw_line_width_color(_from.x, _from.y, x, y, _width, _color, _color);
	}

}
