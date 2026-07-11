function Counter(threshold, counter = 0) constructor {
	self.counter = counter;
	self.threshold = threshold;
	
	static manualCountLoop = function(amount) {
		counter += amount;
		while (counter > threshold) {
			counter -= threshold;
		}
	}
	static manualCount = function(amount) {
		counter = clamp(counter + amount, 0, threshold);
	}
	static setCounter = function(counter) {
		self.counter = counter;
	}
	static getCounter = function() {
		return counter;
	}
	static setThreshold = function(threshold) {
		self.threshold = threshold;
	}
	static getThreshold = function() { 
		return threshold; 
	}
	static setFac = function(fac) {
		counter = lerp(0, threshold, fac);
	}
	static getFac = function() {
		return counter / threshold;
	}
	static reset = function() { 
		counter = 0; 
	}
	static isDone = function() {
		return (counter >= threshold);
	}
	/// @func	countUp()
	/// @desc	Counts up to threshold but no further. Returns true if the count reaches threshold.
	/// @return	{bool} True if counter has reached threshold, false if not.
	static countUp = function() {
		counter = min(counter + DT, threshold);
		return (counter == threshold);
	}
	static countDown = function() {
		counter = max(counter - DT, 0);
		return (counter == 0);
	}
	/// @func	isCounting()
	/// @desc	{bool}	True if counter < threshold.
	static isCounting = function() { return (counter < threshold); }
	/// @desc	Counts up and resets if threshold is reached.
	/// @return {bool}	True if counter reaches threshold, otherwise false.
	static update = function() {
		counter += DT;
		if (counter >= threshold) {
			reset();
			return true;
		}
		return false;
	}
	/// @func	countUpWithMultiplier()
	/// @desc	Counts up to threshold but no further with a given time multiplier.
	/// @return	{bool} True if counter has reached threshold, false if not.
	static countUpWithMultiplier = function(_multiplier) {
		counter = min(counter + DT * _multiplier, threshold);
		return (counter >= threshold);
	}
}

function Stopwatch() constructor {
	counter = 0;
	
	static getCounter = function() { return counter; }
	static reset = function() { counter = 0; }
	static update = function() {
		counter += DT;
	}
}