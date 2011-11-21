import numpy as np
import random


def add_border(image, border_value):
    for axis in range(len(image.shape)):
        shape = image.shape[:axis] + (1, ) + image.shape[(axis + 1):]
        border = border_value * np.ones(shape)
        image = np.concatenate((border, image, border), axis)
    return image


def shannon_array(a):
    entropy = 0.0
    n = float(a.size)
    for value in np.unique(a):
        p_value = np.sum(a == value) / n
        entropy += -p_value * np.log2(p_value)
    return entropy


class Split:
    def __init__(self,
                 u_pxmm=np.array((0, 0)),
                 v_pxmm=np.array((0, 0)),
                 threshold_mm=0):
        self.left = None
        self.right = None
        self.u_pxmm = u_pxmm.reshape((2, 1))
        self.v_pxmm = v_pxmm.reshape((2, 1))
        self.threshold_mm = threshold_mm

    def features(self, pixels, images, image_shape):
        pixels = pixels.reshape((3, -1))    # add a dimension if necessary
        depths = images[pixels[0],
                        pixels[1],
                        pixels[2]].reshape((1, -1))    # add a dimension
        u_px = (pixels[1:] + self.u_pxmm / depths).astype(np.int)
        v_px = (pixels[1:] + self.v_pxmm / depths).astype(np.int)
        du_mm = images[pixels[0], u_px[0].clip(0, image_shape[0]),
                                 u_px[1].clip(0, image_shape[1])]
        dv_mm = images[pixels[0], v_px[0].clip(0, image_shape[0]),
                                 v_px[1].clip(0, image_shape[1])]
        return du_mm - dv_mm

    def are_left(self, *args, **kwargs):
        return self.features(*args, **kwargs) < self.threshold_mm

    def __repr__(self):
        return "Split(%r, %r, %r, %r, %r))" % (self.u_pxmm,
                                               self.v_pxmm,
                                               self.threshold_mm,
                                               self.left,
                                               self.right)


class Leaf:
    def __init__(self, pixels, truth_images):
        self.prediction = {}
        frequency = {}
        for pixel in pixels:
            truth = truth_images[pixel[0], pixel[1], pixel[2]]
            if truth in frequency:
                frequency[truth] += 1
            else:
                frequency[truth] = 1
        for truth in frequency:
            self.prediction[truth] = frequency[truth] / float(len(pixels))

    def __repr__(self):
        return repr(self.prediction)


class DecisionTree():
    def train(self, pixels, depth_images, truth_images, image_shape, infinity,
              max_depth, candidate_count, entropy_threshold=0, depth=0):
        entropy = shannon_array(pixels)
        if depth == max_depth or entropy <= entropy_threshold:
            return Leaf(pixels, truth_images)
        max_gain = 0
        best_split = None
        best_left, best_right = None, None
        tau_limit = infinity - depth_images.min()
        parameters_theta = 2 * max(image_shape) * max_depth * (
                               np.random.random((candidate_count, 4)) - 0.5)
        thresholds_tau = np.random.random_integers(-tau_limit,
                                                   tau_limit,
                                                   (candidate_count, 1))
        candidates = np.hstack([parameters_theta, thresholds_tau])
        pixel_rows = pixels.transpose()
        for candidate in candidates:
            split = Split(candidate[:2], candidate[2:4], candidate[4])
            division = split.are_left(pixel_rows, depth_images, image_shape)
            if division.all() or (-division).all():
                continue
            left = pixels[division]
            right = pixels[-division]
            left_rows = pixel_rows[:, division]
            left_truth = truth_images[left_rows[0],
                                      left_rows[1],
                                      left_rows[2]]
            entropy_left = shannon_array(left_truth)
            right_rows = pixel_rows[:, -division]
            right_truth = truth_images[right_rows[0],
                                       right_rows[1],
                                       right_rows[2]]
            entropy_right = shannon_array(right_truth)
            gain = entropy - (entropy_left * left_truth.size
                              + entropy_right * right_truth.size) / len(pixels)
            if gain > max_gain:
                max_gain = gain
                best_split = split
                best_left, best_right = left, right
        if max_gain > entropy_threshold:
            best_split.left = self.train(np.array(best_left),
                                         depth_images,
                                         truth_images,
                                         image_shape,
                                         infinity,
                                         max_depth,
                                         candidate_count,
                                         entropy_threshold,
                                         depth + 1)
            best_split.right = self.train(np.array(best_right),
                                          depth_images,
                                          truth_images,
                                          image_shape,
                                          infinity,
                                          max_depth,
                                          candidate_count,
                                          entropy_threshold,
                                          depth + 1)
            return best_split
        else:
            return Leaf(pixels, truth_images)

    def test(self, pixel, *args, **kwargs):
        at = self.root
        while at.__class__ is not Leaf:
            if at.are_left(pixel, *args, **kwargs):
                at = at.left
            else:
                at = at.right
        return at

    def __repr__(self):
        return repr(self.root)

    def __init__(self, *args, **kwargs):
        self.root = self.train(*args, **kwargs)


class DecisionForest():
    def __init__(self, training_sets, *args, **kwargs):
        self.trees = []
        for training_set in training_sets:
            self.trees.append(DecisionTree(training_set, *args, **kwargs))

    def test(self, *args, **kwargs):
        result = {}
        for tree in self.trees:
            prediction = tree.test(*args, **kwargs).prediction
            for depth in prediction:
                if depth in result:
                    result[depth] += prediction[depth]
                else:
                    result[depth] = prediction[depth]
        for depth in result:
            result[depth] = result[depth] / len(self.trees)
        return result

    def classify(self, *args, **kwargs):
        maximum = 0
        label = None
        classification = self.test(*args, **kwargs)
        for depth in classification:
            if classification[depth] > maximum:
                maximum = classification[depth]
                label = depth
        return label

if __name__ == '__main__':
    training_pixels = 32
    image_shape = np.array((2, 2))
    image_count = 24
    min_depth, max_depth = 1, 4
    infinity = 2 * max_depth

    depth_images_shape = (image_count, ) + tuple(image_shape)
    depth_images = np.random.random_integers(min_depth,
                                             max_depth,
                                             depth_images_shape)
    depth_images = add_border(depth_images, infinity)[1:-1]
    min_truth, max_truth = min_depth, max_depth
    truth_images = depth_images
    candidate_count = 200
    trees = 2

    training_sets = []
    images_per_tree = image_count / (trees + 1)
    for j in range(trees + 1):
        pixel_indices = np.arange(images_per_tree
                                  * image_shape[1]
                                  * image_shape[0])
        image_indices = (pixel_indices
                         / image_shape[1]
                         / image_shape[0]) + j * images_per_tree
        x_coords = ((pixel_indices / image_shape[1]) % image_shape[0]) + 1
        y_coords = (pixel_indices % image_shape[1]) + 1
        pixel_array = np.vstack([image_indices,
                                 x_coords,
                                 y_coords]).transpose()
        training_list = random.sample(pixel_array, training_pixels)
        sample_pixels = np.array(training_list)
        training_sets.append(sample_pixels)
    forest = DecisionForest(np.array(training_sets[1:]),
                            depth_images,
                            truth_images,
                            image_shape,
                            infinity,
                            max_depth=5,
                            candidate_count=candidate_count,
                            entropy_threshold=0)
    correct = 0.0
    for pixel in training_sets[0]:
        if truth_images[pixel[0], pixel[1], pixel[2]] == forest.classify(pixel, depth_images, image_shape):
            correct += 1.0
    test_pixel_count = len(training_sets[0])
    incorrect = test_pixel_count - correct
    score = correct - incorrect / (max_truth - min_truth)
    print "Accuracy:", correct / test_pixel_count
    print "Score:", score
    print "Adjusted Accuracy:", score / test_pixel_count