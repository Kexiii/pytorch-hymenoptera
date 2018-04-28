from PIL import Image

"""
Define your own data augmentation method here
"""

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size,self.size), resample=self.interpolation)