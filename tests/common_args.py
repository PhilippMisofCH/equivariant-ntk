from dataclasses import dataclass
from itertools import product
import pytest


@dataclass
class RotoTranslation:
    r: int
    t: tuple[int, int]


def generate_roto_trans():
    trafos = [
        RotoTranslation(r, t)
        for r, t in product([1], product(range(-1, 2), range(-1, 2)))
    ]
    trafos += [RotoTranslation(r, t) for r, t in product(range(4), [(1, 2)])]
    return trafos


roto_trans_params = pytest.mark.parametrize("roto_trans", generate_roto_trans())
