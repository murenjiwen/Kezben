# -*- coding: utf-8 -*-
import numpy as np
import OpenEXR
import Imath
import array
import forest


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
    training_pixels = 100
    candidate_count = 400
    trees = 3

    truth_map, depth_map = get_maps("cube.exr")
    assert truth_map.shape == depth_map.shape
    image_shape = np.array(truth_map.shape)
    print forest.shannon_array(depth_map),
    print forest.shannon_array(truth_map)
    images_shape = (1, ) + tuple(image_shape)
    truth_images = forest.add_border(truth_map.reshape(images_shape),
                                     0)[1:-1]
    depth_images = forest.add_border(depth_map.reshape(images_shape),
                                     np.max(depth_map))[1:-1]

    training_sets = []
    for j in range(trees):
        sample_pixels = np.random.random_integers(0,
                                                  truth_map.size - 1,
                                                  training_pixels)
        sample_depths = depth_map.flat[sample_pixels]
        sample_truths = truth_map.flat[sample_pixels]
        x_coords = (sample_pixels / image_shape[0]) + 1
        y_coords = (sample_pixels % image_shape[0]) + 1

        sample_coords = np.vstack([np.zeros(training_pixels, dtype=np.int),
                                   x_coords,
                                   y_coords]).transpose()
        training_sets.append(sample_coords)
    near_infinity = np.max(depth_map * (depth_map < np.max(depth_map)))
    rdf = forest.DecisionForest(training_sets,
                                depth_images,
                                truth_images,
                                image_shape,
                                2 * near_infinity,
                                max_depth=4,
                                candidate_count=candidate_count,
                                entropy_threshold=0.1)

    test_pixels = np.arange(0, truth_map.size)
    test_set = np.vstack([np.zeros(test_pixels.size, dtype=np.int),
                                   (test_pixels / image_shape[0]) + 1,
                                   (test_pixels % image_shape[0]) + 1]).transpose()
    correct = 0.0
    test_pixel_count = 0.0
    prediction_map = np.zeros(truth_images[0].shape, dtype=np.int)
    for pixel in test_set:
        truth = truth_images[pixel[0], pixel[1], pixel[2]]
        prediction = rdf.classify(pixel, depth_images, image_shape)
        prediction_map[pixel[1], pixel[2]] = prediction
        test_pixel_count += 1.0
        if truth == prediction:
            correct += 1.0
    incorrect = test_pixel_count - correct
    score = correct - incorrect / (np.max(truth_images) - 1)
    print "Accuracy:", correct / test_pixel_count
    print "Score:", score
    print "Adjusted Accuracy:", score / test_pixel_count
