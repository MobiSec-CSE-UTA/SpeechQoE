import sys
import argparse
import time
import os

import numpy as np
import torch

import options
from data_loading import data_loader
from calibration import distri_cali
from learner.voice_qoe import VoiceQOE

from tensorboard_logger import Tensorboard


# A helping function to realize bool in "parse_arguments"
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str2bool, default=True, required=True,
                        help='Train or test only, if test only, load_checkpoint_path cannot be missing')
    parser.add_argument('--dataset', type=str, default='voice', required=True,
                        help='Dataset to be used in [voice]')
    parser.add_argument('--model', type=str, default='model', required=True,
                        help='Model to be used in [model, model_shallow, model_deep]')
    parser.add_argument('--method', type=str, default='voice_qoe', required=True,
                        help='Method to be used in [source, target, src_tgt, voice_qoe]')
    parser.add_argument('--src', type=str, default='rest', required=True,
                        help='Specify source dataset in [rest, all].'
                             'Rest: training without target domain'
                             'All: training with target domain')
    parser.add_argument('--tgt', type=str, default='', required=True,
                        help='Specific test domain. Support only one domain')
    parser.add_argument('--nshot', type=int, default=1,
                        help='Specify the number of shots for VoiceQOE training')
    parser.add_argument('--ntask', type=int, default=1,
                        help='Specify the multiplier of tasks for VoiceQOE, the number of task is domain '
                             'number multiply ntask')

    parser.add_argument('--calibrate', type=str2bool, default=False, required=True,
                        help='Specify if the target domain distribution need to be calibrated')
    parser.add_argument('--num_aug_shot', type=int, default=25,
                        help='Specify the number of shot for target dataset from calibrated distribution')
    parser.add_argument('--k', type=int, default=1,
                        help='Specify the number of source domain to calibrate target domain')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Specify ')

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Specify the learning rate')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Specify the number of epochs for training')
    # parser.add_argument('--dynamic_constant', type=str2bool, default=False,
    #                     help='Set a dynamic constant multiplied to the gradient reversal layer')

    parser.add_argument('--train_max_rows', type=int, default=np.inf,
                        help='Specify the maximum number of training data instance')
    parser.add_argument('--valid_max_rows', type=int, default=np.inf,
                        help='Specify the maximum number of valid data instance')
    parser.add_argument('--test_max_rows', type=int, default=np.inf,
                        help='Specify the maximum number of test data instance')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and starting training from checkpoint or test from it')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='Suffix of log file path')

    parser.add_argument('--shuffle_label', type=str2bool, default=False,
                        help='Shuffle labels for VoiceQOE')
    parser.add_argument('--num_source', type=int, default=np.inf,
                        help='Specify the maximum number of available sources')
    parser.add_argument('--num_bin', type=int, default=np.inf,
                        help='Specify the number of bins in Empirical CDF')

    parser.add_argument('--emodel', type=str, default="./emodel/emodel.pt",
                        help='Specify the path for E-model network')

    return parser.parse_args()


def get_path(args):
    # information about target domain augmentation dataset
    augment_path = 'calibration/target_augmentation/' + args.tgt + "_augmentation.csv"

    path = 'log/'
    # information about used data type
    # path += args.dataset + '/'
    # information about used method type
    path += args.method + '/'
    # information about domain(condition) of training data
    if args.src == 'rest':
        path += 'src_rest' + '/'
    else:
        path += 'src_all' + '/'

    if args.tgt:
        path += 'tgt_' + args.tgt + '/'
    else:
        path = path

    path += args.log_suffix + '/'

    checkpoint_path = path + 'check_point/'
    log_path = path + 'log/'
    result_path = path

    print('Saving path: {}'.format(path))
    return augment_path, result_path, checkpoint_path, log_path


def main(args):
    # Check if GPU available on the running machine
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Processing on " + str(device))

    # Get Working directory
    augment_path, result_path, checkpoint_path, log_path = get_path(args)

    # Initialize tensorboard
    tensorboard = Tensorboard(log_path)

    # Get voice profile
    if args.dataset == 'voice':
        opt = options.voice_option
    else:
        opt = {}
        print("Invalid dataset")

    # Load model
    if args.model == 'model':
        import models.model as model
    elif args.model == 'model_shallow':
        import models.model_shallow as model
    elif args.model == 'model_deep':
        import models.model_deep as model
    elif args.model == 'model_lstm':
        import models.model_deepConvLSTM as model
    else:
        model = None
        print("No model specified, stopped")

    """Target domain distribution calibration"""
    if args.calibrate:
        domain_number, mean, cov = distri_cali.mean_cov_calculate(file_path=opt['file_path'], seq_len=opt['seq_len'],
                                                                  target_domain=args.tgt)
        distri_cali.resampling(file_path=opt['file_path'], output=augment_path, target_domain=args.tgt,
                               base_means=mean, base_covs=cov, num_aug_shot=args.num_aug_shot, num_class=5,
                               k=args.k, alpha=args.alpha, seq_len=opt['seq_len'], source_domain=domain_number)
    else:
        pass

    """Dataset loading"""
    if args.method == 'voice_qoe':
        print('#################Source Data Loading...#################')
        source_data_loader = data_loader.domain_data_loader(args, domains=args.src, file_path=opt['file_path'],
                                                            batch_size=args.nshot, valid_split=0, test_split=0.5,
                                                            separate_domains=True, num_workers=0)
        print('#################Target Data Loading...#################')
        if args.calibrate:
            'Load calibrated dataset for adaption, batch size need to be 10 or above'
            # Augmented dataset for adaption, real data for validation and testing.
            target_data_loader = data_loader.calibrated_domain_data_loader(args, domains=args.tgt,
                                                                           file_path=opt['file_path'],
                                                                           augment_file_path=augment_path,
                                                                           batch_size=10, valid_split=0.2,
                                                                           num_workers=0)
            # target_data_loader = data_loader.calibrated_domain_data_loader2(args, domains=args.tgt,
            #                                                             file_path=opt['file_path'],
            #                                                             augment_file_path=augment_path,
            #                                                             batch_size=10, valid_split=0.2,
            #                                                             test_split=0.4)
        else:
            'Load original target dataset for adaption, batch size need to be 10 or above'
            target_data_loader = data_loader.domain_data_loader(args, domains=args.tgt, file_path=opt['file_path'],
                                                                batch_size=10, valid_split=0.2, test_split=0.4,
                                                                separate_domains=True, num_workers=0)

        learner = VoiceQOE(args, opt, model, tensorboard=tensorboard, source_dataloaders=source_data_loader,
                           target_dataloader=target_data_loader[0], lr=args.lr)

    else:
        learner = None
        print("Please select the method to be used in [source, target, src_tgt, voice_qoe]")

    """Training"""
    if args.train:
        start_time = time.time()

        start_epoch = 1
        lowest_loss = 100.00

        valid_every_epoch = 5

        # Load trained model if it is a contingent training
        if args.load_checkpoint_path:
            resume = args.load_checkpoint_path + 'cp_best.pth.tar'
            learner.load_checkpoint(resume)
        else:
            pass

        # Start training
        for epoch in range(start_epoch, args.epoch + 1):
            # Create directory if it doesn't exist
            if epoch == start_epoch:
                if not os.path.exists(result_path):
                    oldumask = os.umask(0)
                    os.makedirs(result_path, 0o777)
                    os.umask(oldumask)
                if not os.path.exists(checkpoint_path):
                    oldumask = os.umask(0)
                    os.makedirs(checkpoint_path, 0o777)
                    os.umask(oldumask)
                for arg in vars(args):
                    tensorboard.log_text('args/' + arg, getattr(args, arg), 0)
                script = ' '.join(sys.argv[1:])
                tensorboard.log_text('args/script', script, 0)

            # learner.train(epoch)

            # Validate the trained model every "valid_every_epoch" epoch
            if epoch % valid_every_epoch == 0:
                epoch_loss = learner.validation(epoch)
                # Save the  model with best accuracy
                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    learner.save_checkpoint(epoch=epoch, lowest_loss=lowest_loss,
                                            checkpoint_path=checkpoint_path + 'cp_best.pth.tar')

        time_elapsed = time.time() - start_time

        print('Time spend on training: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Lowest validation MAE: {:4f}'.format(lowest_loss))

    """Test model"""
    if args.train:
        resume = checkpoint_path + 'cp_best.pth.tar'
    else:
        resume = args.load_checkpoint_path + 'cp_best.pth.tar'

    # Loading model for testing
    checkpoint = learner.load_checkpoint(resume)
    # Test by target dataset from 1 to 10 shots
    accuracy_of_test_data = learner.test(epoch=1)

    # Write testing result into log
    if checkpoint is not None:
        lowest_loss = checkpoint['lowest_loss']
    else:
        lowest_loss = np.inf

    if isinstance(accuracy_of_test_data, (list,)):
        result_text = 'Lowest MAE in Validation:\t{:f}\n'.format(lowest_loss)
        for shot, loss in enumerate(accuracy_of_test_data):
            result_text += 'Test MAE {:d} shot:\t{:f}\n'.format(shot+1, loss)
    else:
        # Record args, script, result in Tensorboard log
        result_text = 'Lowest MAE in Validation:\t{:f}\nTest Class Accuracy:\t{:f}'.format(
            lowest_loss, accuracy_of_test_data)

    f = open(result_path + 'result.txt', 'w')
    f.write(result_text)
    f.close()

    tensorboard.log_text('result/', result_text, 0)
    tensorboard.close()


if __name__ == '__main__':
    parsed_arguments = parse_arguments(sys.argv[1:])
    main(parsed_arguments)
