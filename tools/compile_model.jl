using ACEhamiltonians, Serialization
using ACEbase; read_dict, load_json

serialize(ARGS[2], read_dict(load_json(ARGS[1])))
