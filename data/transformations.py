from typing import Callable, List, Optional, Dict, Tuple
import cv2
import numpy as np
from numpy import random

def _loop_all(sample, sample_setup, fn):

    process_sample, *args = sample_setup(sample)

    for key, value in sample.items():
        sample[key] = fn(value, *args)

    return sample

def _process_all(s: Dict[str, np.ndarray]) -> Tuple[bool, None]:
    return True, None


class RandomCrop:
    def __init__(
        self, probability: float = 1 / 2, scales: Optional[List[float]] = None, square = True,
    ):
        self._probability = probability
        if scales is None:
            scales = [0.8, 0.9, 0.95]
        self._scales = scales
        self.square = square

    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        return _loop_all(sample, self._process_s, self._f)

    def _process_s(self, s: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        scale = random.choice(self._scales)
        _, input_height, input_width = s['image'].shape

        if self.square:
            t_size = int(input_height * scale) if input_height < input_width else int(input_width * (scale))
            target_height,target_width  = (t_size, t_size)
        else:
            target_height, target_width = (
                int(input_height * scale),
                int(input_width * scale),
            )

        top = np.random.randint(0, input_height - target_height)
        left = np.random.randint(0, input_width - target_width)
        return (
            random.random() < self._probability,
            target_height,
            target_width,
            top,
            left,
        )

    @staticmethod
    def _f(value: np.ndarray, *args) -> np.ndarray:
        target_height, target_width, top, left = args
        return value[:, top : top + target_height, left : left + target_width]


class Resize:
    def __init__(self, imsize):
        self._imsize = imsize
    def __call__(self, sample):
        return _loop_all(sample, self._process_s, self._fn)

    def _process_s(self, sample):

        return True, self._imsize

    @staticmethod
    def _fn(value, *args):

        imsize = args[0]
        if (value % 1 == 0).all(): # label
            interpolation = cv2.INTER_NEAREST

        else: # image
            interpolation = cv2.INTER_AREA
        out = []
        for c in range(value.shape[0]):
            out.append(cv2.resize(value[c],  (imsize[0] , imsize[1]), interpolation = interpolation ))
        out = np.array(out)
        return out

class RandomHorizontalFlip:
    """RandomHorizontalFlip should be applied to all n images together, not just one
    """

    def __init__(self, probability: float = 0.5):
        self._probability = probability

    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        if random.random() < self._probability:
            return _loop_all(sample, _process_all, self._fn)
        else:
            return sample

    @staticmethod
    def _fn(value: np.ndarray, *args) -> np.ndarray:
        ret_array = value[:,:, ::-1].copy()
        return ret_array

class RandomVerticalFlip:

    def __init__(self, probability: float = 0.5):
        self._probability = probability

    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        if random.random() < self._probability:
            return _loop_all(sample, _process_all, self._fn)
        else:
            return sample

    @staticmethod
    def _fn(value: np.ndarray, *args) -> np.ndarray:
        ret_array = value[:,::-1].copy()
        return ret_array

class RandomRotation:
    def __init__(self, probability: float = 0.5, angle = 10):
        self._probability = probability
        self.angle = angle

    def __call__(
            self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        if random.random() < self._probability:
            return _loop_all(sample, self._process_s, self._fn)
        else:
            return sample

    def _process_s(self, sample):
        return True, np.random.uniform(-self.angle, self.angle)

    @staticmethod
    def _fn(value: np.ndarray, *args) -> np.ndarray:
        value = value.transpose(1,2,0) #CHW -> HWC
        rows, cols = value.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), args[0], 1)

        if value.dtype == np.uint8: # label
            interp = cv2.INTER_NEAREST
        else: # image
            interp = cv2.INTER_LINEAR
        ret_array = cv2.warpAffine(value, rotation_matrix, (cols, rows), flags=interp)

        if len(ret_array.shape) < len(value.shape):
            ret_array = np.expand_dims(ret_array,0)
        else:
            ret_array = ret_array.transpose(2, 0, 1) #HWC -> CHW
        return ret_array

class Crop:
    def __init__(self, imsize = None):
        self._imsize = imsize

    def __call__(self, sample):
        return _loop_all(sample, self._process_s, self._fn)

    def _process_s(self, sample):
        _, input_height, input_width = sample['image'].shape
        t_size = self._imsize

        target_height, target_width = (
            t_size[0],
            t_size[1],
        )

        # get a center cropping
        top = (input_height - target_height)//2
        left = (input_width - target_width)//2

        return (
            True,
            target_height,
            target_width,
            top,
            left,
        )

    @staticmethod
    def _fn(value, *args):
        target_height, target_width, top, left = args
        out = value[:, top: top + target_height, left: left + target_width]

        return out