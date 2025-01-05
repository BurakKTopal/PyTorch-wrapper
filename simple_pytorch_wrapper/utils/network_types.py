from enum import Enum

class NetworkType(Enum):
    """
    Enum type for the transformation of image data (current use only).
    Can be easily extended with more network types
    """
    FNN = "FNN"
    CNN = "CNN"
