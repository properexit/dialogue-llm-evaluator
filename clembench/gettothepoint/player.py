import logging
import random
from abc import ABC

from clemcore.clemgame import Player
from clemcore.backends import Model
from typing import List, Dict

logger = logging.getLogger(__name__)


class Seeker(Player):
    def __init__(self, model: Model, name: str):
        super().__init__(model, name)

    def _custom_response(self, context: Dict) -> str:
        return f'GUESS:ocean\nCOT: The provided words strongly suggest something related to the sea.'


class Helper(Player):
    def __init__(self, model: Model, name: str):
        super().__init__(model, name)

    def _custom_response(self, context: Dict) -> str:
        return f'CLUE:navigator charts ocean currents\nCOT: The clue hints at someone who uses maps and understands water flow, directly relating to a navigator profession.'