import argparse
import traceback
import signal

from settings import MODELS_CONFIG
from train_model import train
from tests import TestLotterOneStep, TestGeneration20, TestGeneration200


class SIGINTError(RuntimeError):
    pass

def signal_handler(sig, frame):
    raise SIGINTError()


def main(train_models: bool, test_models: bool) -> None:
    signal.signal(signal.SIGINT, signal_handler)

    if train_models:
        print('---------------- TRAINING -----------------')
        for config in MODELS_CONFIG:
            try:
                train(config)
            except SIGINTError:
                print('\n\nTraining stopped. Skipping to the next model.\n\n')
            except Exception as e:
                print(e)
                traceback.print_exc()
                print('\n\nSkipping to the next model.\n\n')
    if test_models:
        test_output_dir = './test_results'

        print('---------------- TESTING ----------------')
        print('Warning! test_generation_200 disabled!')
        test_one_step         = TestLotterOneStep(MODELS_CONFIG, test_output_dir)
        test_generation_20    = TestGeneration20(MODELS_CONFIG, test_output_dir)
        test_generation_200   = TestGeneration200(MODELS_CONFIG, test_output_dir)

        print('Test one-step-ahead')
        test_one_step.run()
        print('Test 20-step-ahead')
        test_generation_20.run()
        print('Test generation 200')
        test_generation_200.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and/or tests bouncing ball trajectory prediction models.')
    parser.add_argument('--train', action='store_true', help='if true trains models')
    parser.add_argument('--test',  action='store_true', help='if true tests trained models')

    args = parser.parse_args()

    main(args.train, args.test)
