from collections import namedtuple
from typing import Dict, Tuple

from gym_d2d.id import Id


Action = namedtuple('Action', ['tx_id', 'rx_id', 'mode', 'rb', 'tx_pwr_dBm'])


Actions = Dict[Tuple[Id, Id], Action]
