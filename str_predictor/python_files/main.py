import training
import testing
import sys
from optparse import OptionParser

def get_args():
    parser = OptionParser()
    parser.add_option('-a', '--training', action='store_true', dest='train_mode', default=False, 
                      help='for training the model')
    parser.add_option('-e', '--test', action='store_true', dest='test_mode', default=False,
                      help='for testing the model')

    parser.add_option('-m', '--model', dest='model', default=0, type=int, help='for specifying config file')
    (options, args) = parser.parse_args()
    return options

def print_warning():
    print('#########################################################################################')
    print('Before starting please make sure to edit the config.py to apply the desire configuration.')
    continue_str = input('Want to continue? (Y/y)')
    if continue_str == 'Y' or continue_str == 'y':
        print('Continuing...')
    else:
        print('Aborting...')
        sys.exit()
    


if __name__ == '__main__':

    print_warning()
    
    args = get_args()

    if args.train_mode:
        print('Training model with configuration in config.py')
        training.main(args.model)

    if args.test_mode:
        print('Testing model with configuration in config.py');
        testing.main(args.model)


