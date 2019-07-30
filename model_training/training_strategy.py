import random

import torch

from .model_trainer import TrainingCallback
from utils.misc import LossFunction


# ----------------- utility functions -------------------------------

def reshape_for_cross_entropy(data):
    zeros = torch.ones_like(data) - data
    ones = data
    return torch.stack([zeros, ones], dim=1)


# ---------------------------------- Training callbacks --------------------------

class GenericCurriculumTrainer(TrainingCallback):
    """
    Once every 5 epochs increases the task difficulty asking the model
    to predict one more frame ahead
    """

    def __init__(self, max_predicted_frames: int, period: int = 10):
        super(GenericCurriculumTrainer, self).__init__()
        self.max_predicted_frames = max_predicted_frames
        self.n_of_predicted_frames = 1
        self.old_epoch = 0
        self.period = period

    def _raise_difficulty(self):
        if self.trainer.current_epoch != self.old_epoch and self.trainer.current_epoch % self.period == 0:
            self.n_of_predicted_frames = min(self.n_of_predicted_frames + 1, self.max_predicted_frames)
            if self.n_of_predicted_frames == self.max_predicted_frames:
                print('Max prediction length reached!')
            else:
                print('Epoch {}: Increasing number of predicted frames'.format(self.trainer.current_epoch))
            self.old_epoch = self.trainer.current_epoch

    def _naive_curriculum_learning(self, buffering_frames: int, sequence_len: int):
        """
            Returns begin and end indexes for the n-frames ahead predictions
            following the naive curriculum learning strategy used in
            Zaremba and Sustskever, 'Learning to execute', 2014.

            Output: (begin_index, end_index).
            buffering_frames <= begin_index <= end_index < sequence_len
        """
        begin_index = random.randint(buffering_frames, sequence_len - self.n_of_predicted_frames - 1)
        end_index = begin_index + self.n_of_predicted_frames
        return begin_index, end_index

    def _combined_curriculum_learning(self, buffering_frames: int, sequence_len: int):
        """
            Returns begin and end indexes for the n-frames ahead predictions
            following the combined curriculum learning strategy used in
            Zaremba and Sustskever, 'Learning to execute', 2014.

            Output: (begin_index, end_index).
            buffering_frames <= begin_index <= end_index < sequence_len
        """

        strategy = 'naive' if random.randint(0, 99) < 80 else 'mixed'
        if strategy == 'naive':
            begin_index, end_index = self._naive_curriculum_learning(buffering_frames, sequence_len)
        else:
            begin_index = random.randint(buffering_frames, sequence_len - 1)
            end_index = random.randint(begin_index, sequence_len - 1)

        return begin_index, end_index


class RNNCellTrainer(TrainingCallback):
    """
    Trains a recurrent model that receives as input a single time frame
    """
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def __call__(self, model, data_batch, loss_fn):
        # checking data.size(0) at each call instead of storing self.batch_size
        # because if (training data len % self.batch_size != 0) then the last
        # batch does not have self.batch_size elements
        state = model.init_hidden(data_batch.size(0))
        ground_truth = data_batch[:, 1:, :] # remove first frame

        predicted_sequence = []
        for t in range(data_batch.size(1) - 1):
            output, state = model(data_batch[:, t, :], state)
            predicted_sequence.append(output)
        prediction = torch.stack(predicted_sequence, dim=1)

        if self.loss_function == LossFunction.CROSS_ENTROPY:
            prediction   = reshape_for_cross_entropy(prediction)
            ground_truth = ground_truth.long()

        return loss_fn(prediction, ground_truth)


class RNNCellCurriculumTrainer(GenericCurriculumTrainer):
    """
    Once every 5 epochs increases the task difficulty asking the model
    to predict one more frame ahead
    """

    def __init__(self, loss_function, max_predicted_frames: int, period: int = 10):
        super(RNNCellCurriculumTrainer, self).__init__(max_predicted_frames, period)
        self.loss_function = loss_function

    def __call__(self, model, data_batch: torch.Tensor, loss_fn):
        # using data.size(0) instead of self.batch_size because
        # if training_data_len % self.batch_size != 0 then the last
        # batch returned by enumerate() does not have self.batch_size elements
        state = model.init_hidden(data_batch.size(0))
        sequence_len = data_batch.size(1)
        ground_truth = data_batch[:, 1:, :]

        gen_start, gen_end = self._naive_curriculum_learning(buffering_frames=4, sequence_len=sequence_len)

        predicted_sequence = []
        for t in range(sequence_len - 1):
            if gen_start <= t <= gen_end:
                output, state = model(output, state)
            else:
                output, state = model(data_batch[:, t, :], state)
            predicted_sequence.append(output)
        prediction = torch.stack(predicted_sequence, dim=1)

        if self.loss_function == LossFunction.CROSS_ENTROPY:
            prediction   = reshape_for_cross_entropy(prediction)
            ground_truth = ground_truth.long()

        self._raise_difficulty()

        return loss_fn(prediction, ground_truth)


class RNNAutoencoder(TrainingCallback):

    def __init__(self, loss_function):
        super(RNNAutoencoder, self).__init__()
        self.loss_function = loss_function

    def __call__(self, model, data_batch: torch.Tensor, loss_fn) -> torch.Tensor:
        data, prediction_ground_truth = data_batch

        prediction = model(data)

        return loss_fn(prediction, prediction_ground_truth)


class RNNAutoencoderMultiDec(TrainingCallback):

    def __init__(self, loss_function, loss_weights=[1, 1]):
        super(RNNAutoencoderMultiDec, self).__init__()
        self.loss_function = loss_function
        self.loss_weights = loss_weights

    def __call__(self, model, data_batch: torch.Tensor, loss_fn) -> torch.Tensor:
        data, label = data_batch

        reconstruction_ground_truth = data.flip(1)
        prediction_ground_truth = label

        reconstruction, prediction = model(data)

        if self.loss_function == LossFunction.CROSS_ENTROPY:
            reconstruction = reshape_for_cross_entropy(reconstruction)
            prediction     = reshape_for_cross_entropy(prediction)
            reconstruction_ground_truth = reconstruction_ground_truth.long()
            prediction_ground_truth     = prediction_ground_truth.long()

        loss_rec  = loss_fn(reconstruction, reconstruction_ground_truth)
        loss_pred = loss_fn(prediction, prediction_ground_truth)

        return self.loss_weights[0] * loss_rec + self.loss_weights[1] * loss_pred


class RNNAutoencoderCurriculumTrainer(GenericCurriculumTrainer):
    """
    Once every 5 epochs increases the task difficulty asking the model
    to predict one more frame ahead
    """

    def __init__(self, loss_function, max_predicted_frames: int, period: int = 10):
        super(RNNAutoencoderCurriculumTrainer, self).__init__(max_predicted_frames, period)
        self.loss_function = loss_function

    def __call__(self, model, data_batch: torch.Tensor, loss_fn):
        data, label = data_batch
        model.decoding_steps = self.n_of_predicted_frames

        prediction_ground_truth = label[:, :self.n_of_predicted_frames, :]

        prediction = model(data)

        if self.loss_function == LossFunction.CROSS_ENTROPY:
            prediction = reshape_for_cross_entropy(prediction)
            prediction_ground_truth = prediction_ground_truth.long()

        self._raise_difficulty()

        return loss_fn(prediction, prediction_ground_truth)


class RNNAutoencoderMultiDecCurriculumTrainer(GenericCurriculumTrainer):
    """
        Once every 5 epochs increases the task difficulty asking the model
        to predict one more frame ahead
        """

    def __init__(self, loss_function, max_predicted_frames: int, loss_weights=[1,1], period: int = 10):
        super(RNNAutoencoderMultiDecCurriculumTrainer, self).__init__(max_predicted_frames, period)
        self.loss_function = loss_function
        self.loss_weights = loss_weights

    def __call__(self, model, data_batch: torch.Tensor, loss_fn):
        data, label = data_batch
        model.decoding_steps = self.n_of_predicted_frames

        reconstruction_ground_truth = data.flip(1)[:, :self.n_of_predicted_frames, :]
        prediction_ground_truth = label[:, :self.n_of_predicted_frames, :]

        reconstruction, prediction = model(data)

        if self.loss_function == LossFunction.CROSS_ENTROPY:
            reconstruction = reshape_for_cross_entropy(reconstruction)
            prediction     = reshape_for_cross_entropy(prediction)
            reconstruction_ground_truth = reconstruction_ground_truth.long()
            prediction_ground_truth     = prediction_ground_truth.long()

        self._raise_difficulty()

        loss_rec  = loss_fn(reconstruction, reconstruction_ground_truth)
        loss_pred = loss_fn(prediction,     prediction_ground_truth)

        return self.loss_weights[0] * loss_rec + self.loss_weights[1] * loss_pred


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
class SupervisedTraining(TrainingCallback):
    """
    Trains a model that receives as input an (input data, label) tuple.
    Useful in most supervised training scenarios.
    """

    def __call__(self, model, data_batch, loss_fn):
        data, label = data_batch
        output = model(data)
        return loss_fn(output, label)


class AutoencoderTraining(TrainingCallback):

    def __call__(self, model, data, loss_fn):
        reconstruction = model(data)
        return loss_fn(reconstruction, data)


class RNNSequenceTrainer(TrainingCallback):
    """
    Trains a recurrent model that receives as input the whole sequence
    """

    def __call__(self, model, data_batch, loss_fn):
        output, _ = model(data_batch)
        return loss_fn(output[:-1], data_batch[1:])



class LSTMPrediction(TrainingCallback):

    def __call__(self, model, data_batch: torch.Tensor, loss_fn) -> torch.Tensor:
        batch_size = data_batch.size(0)
        sequence_len = data_batch.size(1)
        state = model.init_hidden(batch_size)

        loss = torch.zeros((1,))
        for t in range(sequence_len - 1):
            input = data_batch[:, t, :].view(batch_size, -1)
            output, state = model(input, state)

            ground_truth = data_batch[:, t + 1, :].view(batch_size, -1)
            loss += loss_fn(output, ground_truth)

        return loss


class LSTMPredictionCE(TrainingCallback):

    def __call__(self, model, data_batch, loss_fn):
        batch_size = data_batch[0].size(0)
        sequence_len = data_batch[0].size(1)
        state = model.init_hidden(batch_size)

        loss = torch.zeros((1,)).to(self.trainer.device)
        for t in range(sequence_len - 1):
            input = data_batch[:, t, :].view(batch_size, -1)
            output, state = model(input, state)

            output_ce = reshape_for_cross_entropy(output)
            ground_truth = data_batch[:, t + 1, :].view(batch_size, -1).long()
            loss += loss_fn(output_ce, ground_truth)

        return loss
