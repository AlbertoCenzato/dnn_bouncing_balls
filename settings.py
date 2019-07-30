from utils.misc import ModelType, LossFunction, TrainingStrategy


class Str:
    """
    Constant strings to be used as keys for model configuration dictionaries
    """
    TRAINING_SET_PATH   = 'training_set_path'
    TESTING_SET_PATH    = 'testing_set_path'
    VALIDATION_SET_PATH = 'validation_set_path'
    MODEL_SAVE_PATH     = 'model_save_path'

    MODEL = 'model'
    LOSS_FUNCTION = 'loss_function'
    TRAINING_STRATEGY = 'training_strategy'

    NUM_LAYERS  = 'num_layers'
    KERNEL_SIZE = 'kernel_size'
    HIDDEN_DIM  = 'hidden_dim'

    BATCH_SIZE = 'batch_size'
    EPOCHS     = 'epochs'
    LEARNING_RATE = 'learning_rate'


MODELS_CONFIG = [
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/LSTM_1',
  Str.MODEL : ModelType.LSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.CURRICULUM,
  # Str.NUM_LAYERS  : 4,
  # Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [2048, 1024, 1024, 1024, 3600],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/LSTM_2',
  Str.MODEL : ModelType.LSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.CURRICULUM,
  # Str.NUM_LAYERS  : 4,
  # Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [2048, 1024, 1024, 1024, 3600],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/LSTM_3',
  Str.MODEL : ModelType.LSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.CURRICULUM,
  # Str.NUM_LAYERS  : 4,
  # Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [2048, 1024, 1024, 1024, 3600],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},


{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/ConvLSTM_1',
  Str.MODEL : ModelType.CONVLSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.TEACHER_FORCED,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/ConvLSTM_2',
  Str.MODEL : ModelType.CONVLSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.TEACHER_FORCED,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/ConvLSTM_3',
  Str.MODEL : ModelType.CONVLSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.TEACHER_FORCED,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},



{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/seq2seq_ConvLSTM_1',
  Str.MODEL : ModelType.SEQ2SEQ_CONVLSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.TEACHER_FORCED,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/seq2seq_ConvLSTM_2',
  Str.MODEL : ModelType.SEQ2SEQ_CONVLSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.TEACHER_FORCED,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/seq2seq_ConvLSTM_3',
  Str.MODEL : ModelType.SEQ2SEQ_CONVLSTM,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.TEACHER_FORCED,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},



{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/seq2seq_ConvLSTM_multidecoder_1',
  Str.MODEL : ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.CURRICULUM,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/seq2seq_ConvLSTM_multidecoder_2',
  Str.MODEL : ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.CURRICULUM,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
{
  Str.TRAINING_SET_PATH   : './data/training',
  Str.TESTING_SET_PATH    : './data/testing',
  Str.VALIDATION_SET_PATH : './data/validation',
  Str.MODEL_SAVE_PATH     : './trained_models/seq2seq_ConvLSTM_multidecoder_3',
  Str.MODEL : ModelType.SEQ2SEQ_CONVLSTM_MULTIDEC,
  Str.LOSS_FUNCTION : LossFunction.MSE,
  Str.TRAINING_STRATEGY : TrainingStrategy.CURRICULUM,
  Str.NUM_LAYERS  : 4,
  Str.KERNEL_SIZE : [(7,7), (5,5), (3,3), (3,3)],
  Str.HIDDEN_DIM  : [10, 10, 10, 1],
  Str.BATCH_SIZE  : 16,
  Str.EPOCHS      : 100,
  Str.LEARNING_RATE : 0.001
},
]