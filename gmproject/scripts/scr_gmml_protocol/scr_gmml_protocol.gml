// gmml wire protocol - GameMaker side. Mirror of gmml/protocol.py.
//
// GameMaker is the TCP server, Python is the client. Messages are newline-delimited
// JSON. The engine owns all domain logic. Python only sees generic vectors described
// by the self-reported behavior spec sent at handshake.
//
// This script holds only the shared protocol version. Packet assembly lives on
// obj_gmml_academy. Action-spec constructors live in scr_gmml_agent.

#macro GMML_PROTOCOL 1
