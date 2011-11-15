import numpy as np
import scipy
import pylab
import pymorph
import mahotas
from scipy import ndimage
import random
from math import log


infinity = 16

class DepthPixel:
    def __add__(self, other):
        return self.coordinate+other
    def __init__(self, coordinate, depth_image, truth_image):
        self.coordinate = coordinate
        self.depth_image = depth_image
        self.truth_image = truth_image
    def depth(self):
        return self.depth_image[tuple(self.coordinate)]
    def depth_at(self, offset):
        coordinate = tuple(self + offset)
        for i, i_max in zip(coordinate, depth_image.shape):
            if i < 0 or i >= i_max:
                return infinity
        return self.depth_image[coordinate]
    def truth(self):
        return self.truth_image[self.coordinate]
    def __repr__(self):
        return "DepthPixel(%r,\n%r,\n%r)\n" % (self.coordinate, self.depth_image, self.truth_image)
    def __str__(self):
        return "DepthPixel(%s, %s)" % (self.coordinate, self.depth())

def shannon(pixels):
    frequency = {}
    for pixel in pixels:
        depth = pixel.depth()
        if depth in frequency:
            frequency[depth] += 1
        else:
            frequency[depth] = 1
    result = 0
    n = float(len(pixels))
    for depth in frequency:
        p_i = frequency[depth] / n
        result += p_i*log(p_i,2)
    return -result
        

class Node:
    def __init__(self):
        self.pixels = set()

class Split(Node):
    def __init__(self, u_pxmm=(0,0), v_pxmm=(0,0), threshold_mm=1, left=None, right=None):
        Node.__init__(self)
        self.left = left
        self.right = right
        self.u_pxmm, self.v_pxmm = map(np.array, (u_pxmm, v_pxmm))
        self.threshold_mm = threshold_mm
    def feature(self, pixel):
        depth_x_mm = pixel.depth()
        return pixel.depth_at(self.u_pxmm/depth_x_mm) - pixel.depth_at(self.v_pxmm/depth_x_mm)
    def is_left(self, pixel):
        return self.feature(pixel) < self.threshold_mm    
    def __repr__(self):
        return "Split(%r,%r,%r,%r,%r))" % (self.u_pxmm, self.v_pxmm, self.threshold_mm, self.left, self.right)
        
class Leaf(Node):
    def __init__(self, pixels):
        Node.__init__(self)
        self.prediction = {}
        frequency = {}
        for pixel in pixels:
            depth = pixel.depth()
            if depth in frequency:
                frequency[depth] += 1
            else:
                frequency[depth] = 1
        for depth in frequency:
            self.prediction[depth] = frequency[depth] / float(len(pixels))
    def __str__(self):
        return str(self.prediction)
        
        
class DecisionTree():
    def train(self, training_set, entropy_threshold, max_depth, depth=0):
        if depth == max_depth:
            return Leaf(training_set)
        max_gain = 0
        best_split = None
        best_left, best_right = None, None
        for i in xrange(100):
            u_pxmm = 5*(np.array((random.random(),random.random()))-0.5)
            v_pxmm = 5*(np.array((random.random(),random.random()))-0.5)
            tau_mm = random.randint(-2,2)
            split = Split(u_pxmm, v_pxmm, tau_mm)
            left = set()
            right = set()
            for pixel in training_set:
                if split.is_left(pixel):
                    left.add(pixel)
                else:
                    right.add(pixel)
            if len(left) == 0 or len(right) == 0:
                continue
            gain = shannon(training_set) - (shannon(left)*len(left) + shannon(right)*len(right))/len(training_set)
            if gain > max_gain:
                max_gain = gain
                best_split = split
                best_left, best_right = left, right
        if max_gain > entropy_threshold:
#            print "Max gain: ", max_gain
#            print "Left: %s" % map(lambda x: x.depth(), best_left)
#            print "Right: %s" % map(lambda x: x.depth(), best_right)
            best_split.left = self.train(best_left, entropy_threshold, max_depth, depth + 1)
            best_split.right = self.train(best_right, entropy_threshold, max_depth, depth + 1)
            return best_split
        else:
            return Leaf(training_set)
    def test(self, pixel):
        at = self.root
        while at.__class__ is not Leaf:
            if at.is_left(pixel):
                at = at.left
            else:
                at = at.right
        return at
            
    def __init__(self, training_set, entropy_threshold=0, max_depth=1):
        self.root = self.train(training_set, entropy_threshold, max_depth)
        
class DecisionForest():
    def __init__(self, training_sets, tree_count = 1, max_depth = 1):
        self.trees=[]
        for training_set in training_sets:
            self.trees.append(DecisionTree(training_set, max_depth = max_depth))
        
        

if __name__ == '__main__':
    image_shape = (2,2)
    image_count = 6
    training_pixels = 20
    depth_images = []
    for i in range(image_count):
        depth_images.append(np.random.random_integers(1,4,image_shape))
    #depth_images = [np.array([[1,2],[2,1]])]*3
    truth_images = depth_images
    pixels = set()
    for depth_image,truth_image in zip(depth_images,truth_images):
        for i in range(depth_image.shape[0]):
            for j in range(depth_image.shape[1]):
                pixels.add(DepthPixel((i,j), depth_image, truth_image))
    training_set = random.sample(pixels, training_pixels)
    test_set = list(pixels - set(training_set))
    print shannon(pixels), shannon(training_set), shannon(test_set)
    training_sets = [training_set]*3
    forest = DecisionForest(training_sets, max_depth = 5)
    for pixel in test_set:
        print pixel.depth(), forest.trees[0].test(pixel).prediction, forest.trees[1].test(pixel).prediction, forest.trees[2].test(pixel).prediction
#    pylab.imshow(depth_images[0])
    