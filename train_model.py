import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize

from model_training import ModelTrainer, TrainingTimeEstimation, BatchStatistics
from model_training.training_strategy import RNNCellTrainer, RNNCellCurriculumTrainer
from model_training.training_strategy import RNNAutoencoder, RNNAutoencoderMultiDec
from model_training.training_strategy import RNNAutoencoderCurriculumTrainer, RNNAutoencoderMultiDecCurriculumTrainer

from dataset import SutskeverDataset
from models import ConvLSTM, Seq2SeqConvLSTM, Seq2SeqConvLSTMMultidecoder
from models import ImgLSTMCellStack, ImgLSTMAutoencoder

from utils.transforms import RepeatAlongAxis, Binarize, CutSequence

from utils.misc import ParamsEncoder, SplitSequence
from utils.misc import ModelType, LossFunction, TrainingStrategy
from utils.misc import cross_entropy_tanh

from settings import Str


# -------------- Const globals -------------------
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ----------------- initialization functions --------------------------

def init_dataset(configuration):
    model_type    = configuration[Str.MODEL]
    loss_function = configuration[Str.LOSS_FUNCTION]
    if model_type == ModelType.LSTM or model_type == ModelType.CONVLSTM:
        if loss_function == LossFunction.MSE:
            transform = RepeatAlongAxis(Compose([
                                            ToTensor(), 
                                            Normalize([0.5], [0.5])
                                        ]))
        elif loss_function == LossFunction.CROSS_ENTROPY:
            transform = RepeatAlongAxis(Compose([
                                            ToTensor(),
                                            Binarize(0.3),
                                            Normalize([0.5], [0.5])
                                        ]))
    elif model_type == ModelType.SEQ2SEQ_LSTM     or \
         model_type == ModelType.SEQ2SEQ_CONVLSTM or \
         model_type == ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC:
        trans_list = [
            CutSequence(0, 16),
            RepeatAlongAxis(
                ToTensor(),
            )
        ]
        if loss_function == LossFunction.MSE:
            transform = Compose(trans_list + [RepeatAlongAxis(Normalize([0.5], [0.5])), SplitSequence()])
        elif loss_function == LossFunction.CROSS_ENTROPY:
            transform = Compose(trans_list + [Binarize(0.3), RepeatAlongAxis(Normalize([0.5], [0.5])), SplitSequence()])
    else:
        raise ValueError('Unknown model type {}'.format(model_type))

    training_set = SutskeverDataset(configuration[Str.TRAINING_SET_PATH], transform=transform)
    validation_set = None

    return training_set, validation_set


def init_model(configuration, image_shape, n_channels):
    model_type = configuration[Str.MODEL]
    if model_type == ModelType.CONVLSTM:
        model = ConvLSTM(
                    input_size=image_shape, input_dim=n_channels,
                    hidden_dim=configuration[Str.HIDDEN_DIM],
                    kernel_size=configuration[Str.KERNEL_SIZE],
                    num_layers=configuration[Str.NUM_LAYERS], batch_first=True,
                    bias=True, mode=ConvLSTM.STEP_BY_STEP
                ).to(DEVICE)
    elif model_type == ModelType.SEQ2SEQ_CONVLSTM:
        model = Seq2SeqConvLSTM(
                    input_size=image_shape, input_ch=n_channels,
                    hidden_ch=configuration[Str.HIDDEN_DIM],
                    kernel_size=configuration[Str.KERNEL_SIZE],
                ).to(DEVICE)
    elif model_type == ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC:
        model = Seq2SeqConvLSTMMultidecoder(
                    input_size=image_shape, input_ch=n_channels,
                    hidden_ch=configuration[Str.HIDDEN_DIM],
                    kernel_size=configuration[Str.KERNEL_SIZE],
                ).to(DEVICE)
    elif model_type == ModelType.LSTM:
        model = ImgLSTMCellStack(
                    image_size=(n_channels, *image_shape),
                    hidden_size=configuration[Str.HIDDEN_DIM]
                )
    elif model_type == ModelType.SEQ2SEQ_LSTM:
        model = ImgLSTMAutoencoder(
            image_size=(n_channels, *image_shape),
            hidden_size=configuration[Str.HIDDEN_DIM],
            batch_first=True,
        )
    else:
        raise ValueError('Unknown model type {}'.format(model_type))

    return model


def init_trainer(configuration, model):
    loss_function = configuration[Str.LOSS_FUNCTION]
    if loss_function == LossFunction.MSE:
        loss_fn = nn.MSELoss()
    elif loss_function == LossFunction.CROSS_ENTROPY:
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = cross_entropy_tanh
    else:
        raise ValueError('Unknown loss function {}'.format(loss_function))

    model_trainer = ModelTrainer(loss_fn=loss_fn,
                                 epochs=configuration[Str.EPOCHS],
                                 optimizer=torch.optim.Adam(
                                     model.parameters(),
                                     lr=configuration[Str.LEARNING_RATE]
                                 ),
                                 device=DEVICE)
    model_trainer.attach_callback(TrainingTimeEstimation())
    model_trainer.attach_callback(BatchStatistics(period=10))

    return model_trainer


def init_training_strategy(configuration):
    tr_strategy   = configuration[Str.TRAINING_STRATEGY]
    model_type    = configuration[Str.MODEL]
    loss_function = configuration[Str.LOSS_FUNCTION]

    if tr_strategy == TrainingStrategy.TEACHER_FORCED:
        if model_type == ModelType.LSTM or model_type == ModelType.CONVLSTM:
            training_strategy = RNNCellTrainer(loss_function)

        elif model_type == ModelType.SEQ2SEQ_CONVLSTM:
            training_strategy = RNNAutoencoder(loss_function)

        elif model_type == ModelType.SEQ2SEQ_LSTM or model_type == ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC:
            training_strategy = RNNAutoencoderMultiDec(loss_function)

        else:
            raise ValueError('Invalid configuration')

    elif tr_strategy == TrainingStrategy.CURRICULUM:
        if model_type == ModelType.CONVLSTM or model_type == ModelType.LSTM:
            training_strategy = RNNCellCurriculumTrainer(loss_function, max_predicted_frames=8, period=12)

        elif model_type == ModelType.SEQ2SEQ_CONVLSTM:
            training_strategy = RNNAutoencoderCurriculumTrainer(loss_function, max_predicted_frames=8)

        elif model_type == ModelType.SEQ2SEQ_LSTM or ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC:
            training_strategy = RNNAutoencoderMultiDecCurriculumTrainer(loss_function, max_predicted_frames=8)

        else:
            raise ValueError('Invalid configuration')
    else:
        raise ValueError('Unknown training strategy {}'.format(tr_strategy))
    
    return training_strategy


def save_training_params(configuration):
    if not os.path.exists(configuration[Str.MODEL_SAVE_PATH]):
        os.makedirs(configuration[Str.MODEL_SAVE_PATH])

    with open(os.path.join(configuration[Str.MODEL_SAVE_PATH], 'params.json'), 'w') as file:
        json.dump(configuration, file, cls=ParamsEncoder)


def train(configuration):
    print('Initializing dataset')
    training_set, validation_set = init_dataset(configuration)
    training_data   = DataLoader(training_set, configuration[Str.BATCH_SIZE], shuffle=True)
    validation_data = DataLoader(validation_set, configuration[Str.BATCH_SIZE])

    sample = training_set[0]
    shape = sample.size() if torch.is_tensor(sample) else sample[0].size()
    image_shape = shape[-2:]
    n_channels  = shape[1]

    print('Building model')
    model = init_model(configuration, image_shape, n_channels)

    print('Setting-up training')
    model_trainer = init_trainer(configuration, model)

    save_training_params(configuration)
    
    # train model
    training_strategy = init_training_strategy(configuration)
    print('Training model')
    model_trainer.run(model, training_data, training_callback=training_strategy, validation_set=validation_data)

    # save trained model
    print('Saving model')
    model_save_path = os.path.join(configuration[Str.MODEL_SAVE_PATH], 'model.pt')
    torch.save(model, model_save_path)
