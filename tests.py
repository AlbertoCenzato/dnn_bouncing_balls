import os
import traceback
import tempfile
import subprocess

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
from torchvision import utils as trchutil
from torchvision.transforms import ToTensor, Normalize, Compose

import models
from dataset import SutskeverDataset

from model_testing import ModelTester, Test
from model_testing import FrameMSECollector, BallsL2ErrorCollector, VideoCollector
from model_testing import LotterErrorProcessor, LotterErrorProcessorAuto, LotterErrorProcessorAutoMultiDec
from model_testing import LongTermProcessor, LongTermProcessorAuto, LongTermProcessorAutoMultiDec

from utils.misc import ModelType
from utils.transforms import RepeatAlongAxis

from settings import Str


DEVICE                      = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
VIDEO_OUTPUT_SAMPLING_RATIO = 1  # 0.2


# ------------------------------------- utility functions -----------------------------------

def init_dataset(configuration):
    test_set_path = configuration[Str.TESTING_SET_PATH]
    transform = RepeatAlongAxis(Compose([
                    ToTensor(),
                    Normalize([0.5], [0.5])
                ]))
    testing_set = SutskeverDataset(test_set_path, transform=transform)
    
    return testing_set


def init_model(configuration):
    model_dir  = configuration[Str.MODEL_SAVE_PATH]
    model_path = os.path.join(model_dir, 'model.pt')
    model = torch.load(model_path)

    model.to(DEVICE)
    
    model_type = configuration[Str.MODEL]
    if model_type == ModelType.CONVLSTM:
        model.set_mode(models.ConvLSTM.STEP_BY_STEP)

    return model


def save_video(video: torch.Tensor, video_file):
    """
    :param video:
    :param video_file:
    :return:
    """
    digits = len(str(video.size(0)))
    filename_template = 'frame_{:0' + str(digits) + 'd}.png'
    tempdir = tempfile.TemporaryDirectory()
    for i, frame in enumerate(video):
        filename = os.path.join(tempdir.name, filename_template.format(i))
        trchutil.save_image(frame, filename)
    template = os.path.join(tempdir.name, 'frame_')
    ffmpeg = 'ffmpeg -f image2 -framerate 5 -i {}%0{}d.png -vcodec mpeg4 -y {}.mp4'.format(template, digits, video_file)
    subprocess.run(ffmpeg)


def save_image_sequence(frames: torch.Tensor, image_file):
    """
    :param video:
    :param video_file:
    :return:
    """
    stacked_frames = []
    border_h = torch.ones((1,60))
    for frame in frames:
        stacked_frames.append(torch.cat([frame[:, :60], border_h, frame[:, -60:]]))
    border_v = torch.ones((121,1))
    frames = []
    for f in stacked_frames:
        frames.append(f)
        frames.append(border_v)
    image_sequence = torch.cat(frames, dim=1)
    trchutil.save_image(image_sequence, '{}.jpg'.format(image_file))


def mean_confidence_plot(axes: plt.Axes, x, mean, std, label, color):
    plt.autoscale()
    axes.fill_between(x, mean+std, mean-std, color=color, alpha=0.2)
    axes.plot(x, mean, color=color, label=label)

    axes.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)


def histogram_plot(error, savedir, name):
    axes = plt.figure().gca()
    flattened_error = error.reshape((-1))
    axes.hist(flattened_error)
    figure = axes.get_figure()
    figure.savefig(os.path.join(savedir, '{}.png'.format(name)), bbox_inches='tight')

# ---------------------------------------- Tests ---------------------------------------------

class TestNIPS(Test):
    """
    Base class for testing NIPS models. It provides common setup
    and configuration for the tests
    """

    def __init__(self, configurations, test_name, output_dir):
        super(TestNIPS, self).__init__()
        self.configurations = configurations
        self.test_name  = test_name
        self.test_dir = os.path.join(output_dir, test_name)
        self.collectors = {}
        self.figure   = None
        self.plot_L2  = None
        self.plot_MSE = None
        self.save_videos = True

        os.makedirs(self.test_dir, exist_ok=True)
    
    def setUp(self):
        # define collectors
        self.save_videos = True
        try:
            subprocess.run('ffmpeg --version')
        except FileNotFoundError as err:
            print('Warning: to save .mp4 predictions install ffmpeg and ensure it is in path.')
            self.save_videos = False

        self.collectors = { 'L2 error': BallsL2ErrorCollector('L2 error'),
                            'MSE':      FrameMSECollector('MSE'),
                            'Videos':   VideoCollector('Videos', sampling_ratio=VIDEO_OUTPUT_SAMPLING_RATIO)}

        # define plots
        self.plot_L2  = plt.figure().gca()
        self.plot_MSE = plt.figure().gca()

        chartBox = self.plot_L2.get_position()
        self.plot_L2.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
        chartBox = self.plot_MSE.get_position()
        self.plot_MSE.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])

        self.plot_L2.set_title(self.test_name + ' - centroid distance')
        self.plot_L2.set_xlabel('Frame')
        self.plot_L2.set_ylabel('Distance (px)')

        self.plot_MSE.set_title(self.test_name + ' - MSE')
        self.plot_MSE.set_xlabel('Frame')
        self.plot_MSE.set_ylabel('MSE')

    def test(self):
        for i, config in enumerate(self.configurations):
            try:
                # initialization
                model_name = os.path.basename(config[Str.MODEL_SAVE_PATH])
                test_set = init_dataset(config)
                model    = init_model  (config)
                batch_processor = self.init_batch_processor(config)

                # add collectors to ModelTester
                model_tester = ModelTester(config[Str.BATCH_SIZE], DEVICE)
                for name, collector in self.collectors.items():
                    model_tester.add_collector(name, collector)

                # run test
                results = model_tester.run(model, test_set, batch_processor)

                # plot results
                self.plot_error('L2',  results['L2 error'], model_name, i)
                self.plot_error('MSE', results['MSE'],      model_name, i)

                with open(os.path.join(self.test_dir, 'results.txt'), 'a') as text_file:
                    text_file.write('{}\n'.format(model_name))
                    text_file.write("Average L2 error:  {} +/- {}std\n"  .format(np.mean(results['L2 error']), np.mean(np.std(results['L2 error'], axis=(0,2)))))
                    text_file.write("Average MSE error: {} +/- {}std\n\n".format(np.mean(results['MSE']), np.mean(np.std(results['MSE'], axis=(0,2)))))

                for video_index, video in enumerate(results['Videos']):
                    video_file = os.path.join(self.test_dir, '{}_{}'.format(model_name, video_index))
                    if self.save_videos: save_video(video, video_file)
                    save_image_sequence(video, video_file)

            except Exception as e:
                print(e)
                traceback.print_exc()
                print('\n\nSkipping to the next model.\n\n')

        figure = self.plot_L2.get_figure()
        figure.savefig(os.path.join(self.test_dir, 'results_L2.png'), bbox_inches='tight')

        figure = self.plot_MSE.get_figure()
        figure.savefig(os.path.join(self.test_dir, 'results_MSE.png'), bbox_inches='tight')

    def plot_error(self, metric, error, model_name, i):
        config_number = len(self.configurations)
        colormap = matplotlib.cm.get_cmap()
        color = colormap(1. * i / config_number)

        x = np.arange(error.shape[1])
        mean_error = np.mean(error, axis=(0, 2))
        error_std  = np.std (error, axis=(0, 2))

        plot = self.plot_L2 if metric == 'L2' else self.plot_MSE
        mean_confidence_plot(plot, x, mean_error, error_std, model_name, color=color)
        histogram_plot(error, self.test_dir, model_name+metric)

    def init_batch_processor(self, config):
        raise NotImplementedError('TestNIPS subclasses must override init_batch_processor method!')


class TestLotterOneStep(TestNIPS):

    def __init__(self, configurations, output_dir):
        super(TestLotterOneStep, self).__init__(configurations, '1-step-ahead', output_dir)

    def init_batch_processor(self, config):
        model_type = config[Str.MODEL]
        if model_type == ModelType.SEQ2SEQ_CONVLSTM or model_type == ModelType.SEQ2SEQ_LSTM:
            return LotterErrorProcessorAuto()
        elif model_type == ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC:  # or model_type == ModelType.LSTM_AUTO_MULTIDEC:
            return LotterErrorProcessorAutoMultiDec()
        return LotterErrorProcessor()


class TestGeneration20(TestNIPS):

    def __init__(self, configurations, output_dir):
        super(TestGeneration20, self).__init__(configurations, '20-step-ahead', output_dir)

    def init_batch_processor(self, config):
        model_type = config[Str.MODEL]
        if model_type == ModelType.SEQ2SEQ_CONVLSTM or model_type == ModelType.SEQ2SEQ_LSTM:
            return LongTermProcessorAuto(prediction_length=20)
        elif model_type == ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC:  # or model_type == ModelType.LSTM_AUTO_MULTIDEC:
            return LongTermProcessorAutoMultiDec(prediction_length=20)
        return LongTermProcessor(prediction_length=20)


class TestGeneration200(TestNIPS):

    def __init__(self, configurations, output_dir):
        super(TestGeneration200, self).__init__(configurations, 'generation_200', output_dir)

    def init_batch_processor(self, config):
        model_type = config[Str.MODEL]
        if model_type == ModelType.SEQ2SEQ_CONVLSTM or model_type == ModelType.SEQ2SEQ_LSTM:
            return LongTermProcessorAuto(prediction_length=200)
        elif model_type == ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC:  # or model_type == ModelType.LSTM_AUTO_MULTIDEC:
            return LongTermProcessorAutoMultiDec(prediction_length=30)
        return LongTermProcessor(prediction_length=200)
