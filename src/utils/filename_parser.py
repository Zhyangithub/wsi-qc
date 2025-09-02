# -*- coding: utf-8 -*-
"""
Utility helpers for tile filename/coordinate handling.
English comments.
"""
import os, re
from typing import Optional, Tuple

_XY_RE = re.compile(r"x(?P<x>[0-9]+)_y(?P<y>[0-9]+)", re.IGNORECASE)

def parse_xy_from_name(filename: str) -> Optional[Tuple[int,int]]:
    """
    Parse x/y from filenames like 'x0003072_y0025600.png'.
    Returns (x, y) as integers if found, else None.
    """
    name = os.path.basename(filename)
    m = _XY_RE.search(name)
    if not m:
        return None
    return int(m.group("x")), int(m.group("y"))
