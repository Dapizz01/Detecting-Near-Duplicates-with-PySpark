from typing import Literal

_FINGERPRINT_SIZES = Literal[32, 64, 128, 256, 512]
"""Contains all of the possible fingerprint sizes. It must be a multiple of 2.
"""
# TODO: check if fingerprint_sizes must really be a multiple of 2.
