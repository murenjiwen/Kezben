import numpy as np
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
        coordinate = self + offset
        try:
            return self.depth_image[tuple(coordinate)]
        except IndexError:
            return infinity
    def truth(self):
        return self.truth_image[tuple(self.coordinate)]
    def __repr__(self):
        return "DepthPixel(%r,\n%r,\n%r)\n" % (self.coordinate, self.depth_image, self.truth_image)
    def __str__(self):
        return "DepthPixel(%s, %s)" % (self.coordinate, self.depth())

def shannon_array(a):
    entropy=0.0
    n=float(a.size)
    for value in np.unique(a):
        p_value = np.sum(a == value)/n
        entropy += -p_value*np.log2(p_value)
    return entropy

def shannon(values):
    frequency = {}
    for value in values:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    result = 0
    n = float(len(values))
    for value in frequency:
        p_i = frequency[value] / n
        result += p_i*log(p_i,2)
    return -result

def shannon_depthpixels(pixels):
    return shannon(map(lambda x : x.truth(), pixels))

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
            truth = pixel.truth()
            if truth in frequency:
                frequency[truth] += 1
            else:
                frequency[truth] = 1
        for truth in frequency:
            self.prediction[truth] = frequency[truth] / float(len(pixels))
    def __repr__(self):
        return repr(self.prediction)

class DecisionTree():
    def train(self, training_set, max_depth, depth=0, entropy_threshold = 0):
        entropy = shannon_depthpixels(training_set)
        if depth == max_depth or entropy <= entropy_threshold:
            return Leaf(training_set)
        max_gain = 0
        best_split = None
        best_left, best_right = None, None
        for i in xrange(100):
            u_pxmm = 200*(np.array((random.random(),random.random()))-0.5)
            v_pxmm = 200*(np.array((random.random(),random.random()))-0.5)
            tau_mm = random.randint(-5,5)
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
            gain = entropy - (shannon_depthpixels(left)*len(left) + shannon_depthpixels(right)*len(right))/len(training_set)
            if gain > max_gain:
                max_gain = gain
                best_split = split
                best_left, best_right = left, right
        if max_gain > entropy_threshold:
#            print "Max gain: ", max_gain
#            print "Left: %s" % map(lambda x: x.depth(), best_left)
#            print "Right: %s" % map(lambda x: x.depth(), best_right)
            best_split.left = self.train(best_left, max_depth, depth + 1, entropy_threshold)
            best_split.right = self.train(best_right, max_depth, depth + 1, entropy_threshold)
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
    def __repr__(self):
        return repr(self.root)

    def __init__(self, training_set, entropy_threshold=0, max_depth=1):
        self.root = self.train(training_set, max_depth, 0, entropy_threshold)

class DecisionForest():
    def __init__(self, training_sets, tree_count = 1, **kwargs):
        self.trees=[]
        for training_set in training_sets:
            self.trees.append(DecisionTree(training_set, **kwargs))
    def test(self, pixel):
        result = {}
        for tree in self.trees:
            prediction = tree.test(pixel).prediction
            for depth in prediction:
                if depth in result:
                    result[depth] += prediction[depth]
                else:
                    result[depth] = prediction[depth]
        for depth in result:
            result[depth] = result[depth]/len(self.trees)
        return result
    def classify(self, pixel):
        maximum = 0
        label = None
        classification = self.test(pixel)
        for depth in classification:
            if classification[depth] > maximum:
                maximum = classification[depth]
                label = depth
        return label

if __name__ == '__main__':
    image_shape = (64,48)
    image_count = 12
    training_pixels = 400
    set_count = 4
    training_sets = []
    for j in range(set_count):
        depth_images = []
        for i in range(image_count):
            depth_images.append(np.random.random_integers(1,4,image_shape))
        truth_images = depth_images
        pixels = set()
        for depth_image,truth_image in zip(depth_images,truth_images):
            for i in range(depth_image.shape[0]):
                for j in range(depth_image.shape[1]):
                    pixels.add(DepthPixel(np.array((i,j)), depth_image, truth_image))
        training_set = random.sample(pixels, training_pixels)
        training_sets.append(training_set)
    forest = DecisionForest(training_sets[1:], max_depth = 4)
    correct = 0.0
    for pixel in training_sets[0]:
        if pixel.truth() == forest.classify(pixel):
            correct += 1.0
    print "Accuracy:", correct/len(training_sets[0])
