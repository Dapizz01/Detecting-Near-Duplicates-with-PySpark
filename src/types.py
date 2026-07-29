from typing import Literal

'''
class Shingles_params(TypedDict):
    """Additional params for the Shingles feature model"""

    overlap: bool
    """ *True* if the shingles can overlap, *False* otherwise. \n
        Example: "This is a test", K = 3, over_words = False \n
        overlap = *True* -> ["Thi", "his", "is ", ...] \n
        overlap = *False* -> ["Thi", "s i", "s a", ...]
    """
    over_words: bool
    """ *True* if the shingles are word-based, *False* if the shingles are character-based. \n
        Example: "This is a test", K = 2, overlap = True \n
        over_words = *True* -> ["This is", "is a", "a test"]
        over_words = *False* -> ["Th", "hi", "is", ...]
    """
'''

_FINGERPRINT_SIZES = Literal[64, 128, 256, 512]
"""Contains all of the possible fingerprint sizes. It must be a multiple of 2.
"""
# TODO: check if fingerprint_sizes must really be a multiple of 2.
