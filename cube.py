# -*- coding: utf-8 -*-
import numpy as np
import random
import OpenEXR
import Imath
import array
from forest import shannon_array


def get_maps(filename):
    input_file = OpenEXR.InputFile("cube.exr")
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B) = [np.array(array.array('f', input_file.channel(channel, FLOAT)))
        for channel in ("R", "G", "B")]
    truth = np.array(R + (2 * G) + (3 * B)).astype(np.int)
    depth = np.array(array.array('f', input_file.channel("Z", FLOAT)))
    truth = truth.reshape(128, 128)
    depth = depth.reshape(128, 128)
    return truth, depth

if __name__ == '__main__':
    truth, depth = get_maps("cube.exr")
    sample_pixels = np.array(random.sample(xrange(128 * 128), 100))
    sample_depths = depth.flat[sample_pixels]
    sample_truths = truth.flat[sample_pixels]
    sample_coords = np.array([sample_pixels / 128, sample_pixels % 128])
    print shannon_array(depth), shannon_array(truth)
    print shannon_array(sample_depths), shannon_array(sample_truths)