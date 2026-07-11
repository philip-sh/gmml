/// Async - Networking: drives the gmml protocol.

var _type = async_load[? "type"];

if (_type == network_type_connect) {
    client = async_load[? "socket"];
    ready  = true;
}

if (_type == network_type_data) {
    // Transport only: append the received bytes, then hand each complete '\n'-terminated line
    // to dispatch(). Framing lives here so a coalesced/split delivery can't desync the
    // protocol. Mirrors connection.py's stateful recv on the Python side.
    var _in   = async_load[? "buffer"];
    var _size = async_load[? "size"];          // bytes actually received (buffer may be larger)

    buffer_copy(_in, 0, _size, inbox, inbox_len);   // append to the accumulator
    inbox_len += _size;

    // Pull out every complete '\n'-terminated line. Leave any partial tail for next time.
    var _scan = 0, _line_start = 0;
    while (_scan < inbox_len) {
        if (buffer_peek(inbox, _scan, buffer_u8) == 10) {   // '\n' == 0x0A
            var _len = _scan - _line_start;
            if (_len > 0) {
                try {
                    dispatch(json_parse(__gmml_line_to_string(inbox, _line_start, _len)));
                } catch (_e) {
                    show_debug_message("gmml: dropped malformed message: " + string(_e));
                }
            }
            _line_start = _scan + 1;
        }
        _scan++;
    }

    // Compact the unparsed remainder to the front (temp copy avoids overlapping buffer_copy).
    var _rem = inbox_len - _line_start;
    if (_rem > 0 && _line_start > 0) {
        var _tmp = buffer_create(_rem, buffer_fixed, 1);
        buffer_copy(inbox, _line_start, _rem, _tmp, 0);
        buffer_copy(_tmp, 0, _rem, inbox, 0);
        buffer_delete(_tmp);
    }
    inbox_len = _rem;
}
