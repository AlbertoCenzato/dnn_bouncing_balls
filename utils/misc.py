import json
import math
from enum import Enum
import random

import numpy as np

import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Normalize

from .transforms import RepeatAlongAxis, CutSequence, get_dim


###################################################
#                  FUNCTIONS                      #
###################################################

def get_ball_centroid(image):
    centroid = np.zeros(2)
    tot_weight = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            weight = image[i,j]
            centroid = centroid + weight * np.array([i,j])
            tot_weight += weight
    return (centroid / tot_weight).astype(np.int32)


def reshape_for_cross_entropy(video_batch):
    zeros = torch.ones_like(video_batch) - video_batch
    ones  = video_batch
    return torch.nn.functional.softmax(torch.stack([zeros, ones], dim=1), dim=1)


def compare_sequence(seq1, seq2):
    seq1_frames = seq1.unbind(0)
    seq2_frames = seq2.unbind(0)

    seq1_unfolded = torch.cat(seq1_frames, dim=2)
    seq2_unfolded = torch.cat(seq2_frames, dim=2)

    return torch.cat([seq1_unfolded, seq2_unfolded], dim=1)


def sample_frame(image_sequence):
    index = random.randint(0, image_sequence.shape[0] - 1)
    return image_sequence[index]
    

###################################################
#                  MISC                           #
###################################################

class ModelType(Enum):
    LSTM             = 'lstm'
    CONVLSTM         = 'convlstm'
    SEQ2SEQ_LSTM     = 'seq2seq_lstm'
    SEQ2SEQ_CONVLSTM = 'seq2seq_convlstm'
    SEQ2SEQ_CONVLSTM_MULTIDEC = 'seq2seq_convlstm_multidecoder'

    CPC_AUTOENCODER = 'cpc_autoencoder'


class LossFunction(Enum):
    MSE           = 'mse'
    CROSS_ENTROPY = 'cross_entropy'
    ADVERSARIAL   = 'adversarial'


class TrainingStrategy(Enum):
    TEACHER_FORCED = 'teacher_forced'
    CURRICULUM     = 'curriculum'


class ParamsEncoder(json.JSONEncoder):

    def default(self, o):
        try:
            if isinstance(o, Enum):
                encoding = o.name
            else:
                model_params    = o.model_params.__dict__
                training_params = o.training_params.__dict__
                dataset_params  = o.dataset_params.__dict__
                encoding = {'model_params': model_params,
                            'training_params': training_params,
                            'dataset_params': dataset_params}
        except TypeError:
            pass
        else:
            return encoding
        return json.JSONEncoder.default(self, o)
                

class Params:

    def __init__(self, model_params, training_params, dataset_params):
        self.model_params    = model_params
        self.training_params = training_params
        self.dataset_params  = dataset_params


###################################################
#                  TRANSFORMS                     #
###################################################

class TransformSutsk:

    def __init__(self):
        self.transform = Compose([
                            CutSequence(0, 40),
                            RepeatAlongAxis(
                                Compose([
                                    ToPILImage(), 
                                    Resize((60,60)), 
                                    ToTensor()
                                ]), 0)
                        ])

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        return self.transform(array)

                   
class TransformBB:

    def __init__(self):
        self.transform = RepeatAlongAxis(Compose([ToTensor(), Normalize([0.5], [0.5])]), 0)

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        return self.transform(array)


class SplitSequence:

    def __call__(self, tensor_or_ndarray):
        half_len = int(get_dim(tensor_or_ndarray, 0)/2)
        return tensor_or_ndarray[:half_len, :], tensor_or_ndarray[half_len:, :]


###################################################
#                  CALLBACKS                      #
###################################################


def cross_entropy_tanh(a, target):
    error = torch.mean(-0.5 * ( (1.0-target)*torch.log(1.0-a) + (1.0+target)*torch.log(1.0+a) )) + math.log(2.0)
    return error
