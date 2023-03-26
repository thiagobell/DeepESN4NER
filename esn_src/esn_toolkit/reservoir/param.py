def init_reservoir_from_param(param_dict):
    """
    Initializes an instance of the reservoir with hyperparameters parameters taken from param_dict. An additional
    key "type" provides the type of the reservoir. If it does not match raises an error
    Args:
        param_dict: a dictionary with parameters (same keys as __init__)

    Returns: a reservoir instance
    """
    # importing here to avoid nasty import loops
    from esn_toolkit.reservoir.bidirectional import Bi
    from esn_toolkit.reservoir.identity import Identity
    from esn_toolkit.reservoir.standard import Reservoir
    from esn_toolkit.reservoir.deep import DeepReservoir

    if param_dict["type"] == "DeepReservoir":
        return DeepReservoir.init_from_dict(param_dict)
    if param_dict["type"] == "Bi":
        return Bi.init_from_dict(param_dict)
    if param_dict["type"] == "Reservoir":
        return Reservoir.init_from_dict(param_dict)
    if param_dict["type"] == "Identity":
        return Identity.init_from_dict(param_dict)

    raise TypeError(f"unknown reservoir type {param_dict['type']}")