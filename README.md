This repository contains various Deep Learning models built to predict the dynamics of the [bouncing balls](https://github.com/zhegan27/TSBN_code_NIPS2015/blob/master/bouncing_balls/data/data_handler_bouncing_balls.py) dataset and a series of tests to benchmark these models.

### Prerequisites
PyTorch is required, install it from their [website](https://pytorch.org/get-started/locally/). The code was tested with PyTorch 1.0, but 1.1 should be fine as well.

Other required Python 3 packages are listed in requirements.txt file.

NOTE: if you want the tests to produce .mp4 videos of the model's predictions you have to install ffmpeg and ensure it is in PATH.

### Dataset
To generate the dataset execute the following command on your terminal:
```
python dataset_generation.py --res 60 --frames 40 --samples 6000
```
to generate a dataset with 6000 samples of 40 frames, height and width = 60.

### Run 
You can configure your training/testing in settings.py. It contains a list of dictionaries which define the parameters of each model to train/test. It is already configured for the best hyperparameters for each architecture.

After hyperparameters configuration run your training with:
```
python train_and_test.py --train
```

and test the trained models:
```
python train_and_test.py --test
```

You can also use both ```--train``` and ```--test``` switches together.