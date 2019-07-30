import time
import random
import math

import numpy as np
import cv2

import torch
from torchvision.transforms import Normalize


def distanceL2(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def sleep(prediction, ground_truth, frame_number):
    time.sleep(0.3)


def compute_centroids(image_batch):
    """
    Computes the centroids of the connected components in each image of image_batch.
    Each image is eroded and thresholded before the connected components are computed;
    in this way two colliding balls are not detected as one blob.
    Input:
        image_batch: numpy array of shape (batch_size, height, width) and type numpy.float32
    Output:
        list of shape (batch_size, num of connected components) of numpy array of shape (2,)
        and type numpy.int
    """
    batch_range = range(image_batch.shape[0])
    centroids_batch = [ [] for _ in batch_range ]
    eroding_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    for batch_index in batch_range:
        eroded_image = cv2.erode(image_batch[batch_index], eroding_kernel, iterations=3)
        _, image_thr = cv2.threshold(eroded_image, 0.5, 1, cv2.THRESH_BINARY)
        image_thr = np.uint8(image_thr)
        _, _, _, centroids = cv2.connectedComponentsWithStats(image_thr)
        centroids_batch[batch_index] = np.around(centroids[1:]).astype(np.int)  # remove background connected component
    return centroids_batch


##############################################
#                   Collectors               #
##############################################

class Collector:
    """
    WARNING! Collectors should not modify the underlying data of 'prediction' and 'ground_truth' params
    """

    def __init__(self, name=''):
        self.tester = None
        self.name = name

    def __call__(self, model_output, target):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self):
        pass

    def collect_results(self):
        pass


class MultiCollector(Collector):
    """
    Utility class to be used inside ModelTester to group many collectors
    together and call them easily
    """

    def __init__(self, name='Error computer collection'):
        super(MultiCollector, self).__init__(name)
        self.collectors = dict()

    def add_collector(self, name, collector):
        self.collectors[name] = collector
        return self

    def __call__(self, model_output, target):
        for test in self.collectors.values():
            test(model_output, target)

    def on_batch_begin(self):
        for test in self.collectors.values():
            test.on_batch_begin()

    def on_batch_end(self):
        for test in self.collectors.values():
            test.on_batch_end()

    def on_test_begin(self):
        for test in self.collectors.values():
            test.on_test_begin()

    def on_test_end(self):
        for test in self.collectors.values():
            test.on_test_end()

    def collect_results(self):
        results = dict()
        for name, test in self.collectors.items():
            results[name] = test.collect_results()
        return results


class BallsL2ErrorCollector(Collector):
    """
    Collects in self.error the errors with shape (sequence length, batch size)
    """

    def __init__(self, name=''):
        super(BallsL2ErrorCollector, self).__init__(name)
        self.error = []

    def __call__(self, output, target):
        assert output.size() == target.size()
        assert len(output.size()) == 5

        output_np = output.numpy()
        target_np = target.numpy()

        sequence_len = output.size(1)
        frame_errors = []
        for t in range(sequence_len):
            batch_errors = BallsL2ErrorCollector.compute_frame_error(output_np[:, t, :],
                                                                     target_np[:, t, :])
            frame_errors.append(batch_errors)

        self.error.append(frame_errors)

    @staticmethod
    def compute_frame_error(model_output, target):
        gt_centroids_batch = compute_centroids(target[:, 0, :, :])
        pr_centroids_batch = compute_centroids(model_output[:, 0, :, :])

        diagonal = distanceL2((0, 0), target.shape[2:])
        # for each element in the batch...
        batch_errors = []
        for i, centroids in enumerate(zip(gt_centroids_batch, pr_centroids_batch)):
            gt_centroids, pr_centroids = centroids
            # ... the error is computed as the L2 distance between ground truth centroids and predicted centroids
            error = [np.min([distanceL2(pr, gt) for gt in gt_centroids]) for pr in pr_centroids]

            # if there's a mismatch between number of predicted balls and ground truth 
            # balls add maximum possible error (image diagonal) for each missing or exceeding ball
            error = error + [diagonal for _ in range(abs(len(gt_centroids) - len(pr_centroids)))]

            batch_errors.append(np.sum(error))
        return batch_errors

    def on_test_end(self):
        self.error = []

    def collect_results(self):
        return np.array(self.error)


class FrameMSECollector(Collector):
    """
    Collects in self.error the errors with shape (sequence length, batch size)
    """

    def __init__(self, name=''):
        super(FrameMSECollector, self).__init__(name)
        self.error = []
        self.compute_error = torch.nn.MSELoss(reduction='sum')
        self.norm = Normalize([-1.0], [2.0])

    def __call__(self, output, target):
        assert output.size() == target.size()
        assert len(output.size()) == 5

        sequence_len = output.size(1)
        sequence_errors = []
        for t in range(sequence_len):  # cycle trough the sequence...
            batch_error = self.compute_frame_error(output[:, t, :], target[:, t, :])
            sequence_errors.append(batch_error)

        self.error.append(sequence_errors)

    def compute_frame_error(self, output, target):
        """
        Compute error for each element of the frame batch
        """
        batch_errors = []
        # for batch_index in range(output.size(0)):
        for output_batch_element, target_batch_element in zip(output, target):
            output_normalized = self.norm(output_batch_element).reshape(-1)
            target_normalized = self.norm(target_batch_element).reshape(-1)
            error = self.compute_error(output_normalized, target_normalized)
            
            batch_errors.append(error.item())
        return batch_errors

    def on_test_end(self):
        self.error = []

    def collect_results(self):
        return np.array(self.error)


class VideoCollector(Collector):
    """
    Collects one of the batch items as video frame
    """

    def __init__(self, name='', sampling_ratio=0.1):
        super(VideoCollector, self).__init__(name)
        self._video = []
        self.sampling_ratio = sampling_ratio

    def __call__(self, output, target):
        assert output.size() == target.size()
        assert len(output.size()) == 5

        if not self.sample_sequence():
            return

        shape = output.size()

        frames = []
        for t in range(shape[1]):
            frame = VideoCollector.get_frame(output[:, t, :], target[:, t, :],
                                             0, shape[2], shape[3])
            frames.append(frame)

        self._video.append(frames)

    def on_test_end(self):
        self._video = []

    def collect_results(self):
        return [torch.stack(video) for video in self._video]

    @staticmethod
    def get_frame(output, target, batch_dim, channels, height):
        assert channels == 1 or channels == 3
        
        pred_frame  = output[batch_dim, 0, :, :]
        gt_frame    = target[batch_dim, 0, :, :]
        middle_line = torch.ones((height, 2))
        frame = torch.cat([pred_frame, middle_line, gt_frame], dim=1)
        
        return frame

    def sample_sequence(self):
        return random.random() < self.sampling_ratio
