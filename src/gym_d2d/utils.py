def merge_dicts(original: dict, other: dict) -> dict:
    """Merge two dicts (in place), overwriting values in original.

    :param original: The original dict values being used.
    :param other: The dict to overwrite with.
    :return: The overwritten dict.
    """
    for k, v in other.items():
        if k in original:
            if isinstance(original[k], dict) and isinstance(other[k], dict):
                merge_dicts(original[k], other[k])
            else:
                original[k] = other[k]  # overwrite
        else:
            original[k] = other[k]
    return original
