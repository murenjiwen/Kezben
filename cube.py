# -*- coding: utf-8 -*-
import numpy as np
import random
import OpenEXR
import Imath
import array

input_file = OpenEXR.InputFile("cube.exr")
print input_file.header()

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
(R,G,B) = [np.array(array.array('f', input_file.channel(Chan, FLOAT))) for Chan in ("R", "G", "B") ]
truth = np.array(R+2*G+3*B).astype(int).reshape(128,128)
depth = np.array(array.array('f', input_file.channel("Z", FLOAT))).reshape(128,128)
print "Depth range: %dm to %dm" % (depth.min(), depth.max())
sample_pixels = np.array(random.sample(xrange(128*128),100))
sample_depths = depth.flat[sample_pixels]
sample_truths = truth.flat[sample_pixels]
sample_coords = np.array([sample_pixels/128,sample_pixels%128])
