import time
import datetime

import matplotlib.pyplot as plt

from .model_trainer import MTCallback, Event


class TrainingTimeEstimation(MTCallback):
    """
    This callback estimates the remaining training time
    """

    def __init__(self):
        super(MTCallback, self).__init__()
        self.event = Event.ON_EPOCH_BEGIN
        self.epoch_start_time = None
        self.cumulative_epochs_times = 0.0

    def __call__(self):
        if not self.epoch_start_time:
            self.epoch_start_time = time.time()
        else:
            end = time.time()
            self.cumulative_epochs_times += end - self.epoch_start_time
            estimated_time_per_epoch = self.cumulative_epochs_times / self.trainer.current_epoch
            remaining_epochs = self.trainer.epochs - self.trainer.current_epoch

            eta = estimated_time_per_epoch * remaining_epochs
            time_delta = datetime.timedelta(seconds=int(eta))
            print('ETA: {}'.format(time_delta))

            self.epoch_start_time = end


class BatchStatistics(MTCallback):
    """
    Post-batch callback that plots the training loss
    """

    def __init__(self, period):
        """
        :param period: plotting period expressed in number of batches
        """
        super(BatchStatistics, self).__init__()
        self.event = Event.ON_BATCH_END
        self.logging_period = period
        self.running_loss = 0.0
        self.dataset_size = 0
        self.loss_plot = None
        self.last_point = None

        _, self.loss_plot = plt.subplots()
        self.loss_plot.set_title("Training loss")
        self.loss_plot.set_xlabel("Epoch")

    def __call__(self):
        batch = self.trainer.current_batch
        if batch == 0:
            self.running_loss = 0.0    # reset running loss at epoch begin

        self.running_loss += self.trainer.last_batch_loss
        if (batch % self.logging_period) == self.logging_period-1:
            epoch = self.trainer.current_epoch
            dataset_size = len(self.trainer.data_loader_tr)
            mean_loss = self.running_loss / self.logging_period  # print mean loss over the processed batches
            print('[epoch: {:.0f}, batch: {:.0f}] - loss: {:.6f}'.format(epoch + 1, batch + 1, mean_loss))
            x = epoch + batch / dataset_size
            y = mean_loss
            if self.last_point is None:
                self.last_point = (x, y)
            x_pts = [self.last_point[0], x]
            y_pts = [self.last_point[1], y]
            self.loss_plot.plot(x_pts, y_pts, 'b-')
            plt.draw()
            plt.pause(0.001)
            self.running_loss = 0.0
            self.last_point = (x, y)