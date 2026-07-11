/// @desc Discrete / multi-discrete action spec. One branch per independent choice.
/// @param {Array<Real>} _branches  e.g. [3, 2] = a 3-way move + a 2-way jump
function gmml_discrete(_branches) {
    return { type: "discrete", branches: _branches };
}

/// @desc Continuous action spec: a vector of _size floats in [_low, _high].
/// @param {Real} _size
/// @param {Real} _low
/// @param {Real} _high
function gmml_continuous(_size, _low = -1, _high = 1) {
    return { type: "continuous", size: _size, low: _low, high: _high };
}
