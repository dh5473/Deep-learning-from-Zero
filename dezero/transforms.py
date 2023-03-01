import numpy as np


class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms
    
    def __call__(self, img):
        if not self.transforms:
            return img
        for transform in self.transforms:
            img = transform(img)
        return img