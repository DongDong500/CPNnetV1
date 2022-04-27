import numpy as np
from PIL import Image, ImageOps

class HistEqualization(object):
    """Histogram Equalization
    
    Args:
        ...
    """

    def __init__(self, lbl=None):
        self.mask = lbl

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image):
            lbl (PIL Image):
        Returns:
            PIL Image:
            PIL Image:
        """
        return ImageOps.equalize(img, mask=self.mask), lbl

    def __repr__(self):
        return self.__class__.__name__ + '()'

    