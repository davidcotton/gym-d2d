from collections import namedtuple

Action = namedtuple('Action', ['tx_id', 'rx_id', 'mode', 'rb', 'tx_pwr_dBm'])
