# -*- coding: utf-8 -*-
"""
Normalizes the Python 2/3 API for internal use.
"""
import sys

_ver = sys.version_info
is_py2 = _ver[0] == 2
is_py3 = _ver[0] == 3

if is_py2:

    unicode = unicode
    bytes = str

elif is_py3:

    unicode = str
    bytes = bytes
