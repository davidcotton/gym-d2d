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


def plot_devices(env, out_file: str = ''):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('`plot_devices()` requires matplotlib')

    cue_xs, cue_ys = [], []
    for cue_id, cue in env.simulator.devices.cues.items():
        cue_xs.append(cue.position.x)
        cue_ys.append(cue.position.y)
    due_tx_xs, due_tx_ys = [], []
    due_rx_xs, due_rx_ys = [], []
    for due_tx_id, due_rx_id in env.simulator.devices.due_pairs.items():
        due_tx = env.simulator.devices[due_tx_id]
        due_tx_xs.append(due_tx.position.x)
        due_tx_ys.append(due_tx.position.y)
        due_rx = env.simulator.devices[due_rx_id]
        due_rx_xs.append(due_rx.position.x)
        due_rx_ys.append(due_rx.position.y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mbs_pos = env.simulator.devices.bs.position
    ax.add_artist(plt.Circle(mbs_pos.as_tuple(), env.simulator.config.cell_radius_m, color='b', alpha=0.1))
    ax.scatter(due_tx_xs, due_tx_ys, c='r', label='DUE_TX')
    ax.scatter(due_rx_xs, due_rx_ys, c='m', label='DUE_RX')
    ax.scatter(cue_xs, cue_ys, c='b', label='CUE')
    ax.scatter(mbs_pos.x, mbs_pos.y, c='k', label='MBS')
    ax.legend()
    if out_file:
        plt.savefig(out_file)
    plt.show()
