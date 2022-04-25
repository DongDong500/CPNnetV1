import os
import argparse

import torch
import socket
import network
from datetime import datetime
from train import train

def get_argparser():
    parser = argparse.ArgumentParser()

    default_path = '/data/sdi/MUnetPlus-result/'
    dataset_path = '/data/sdi/datasets/'
    
    # Tensorboard Options
    parser.add_argument("--Tlog_dir", type=str, default=default_path, help="path to Tensorboard log")
    parser.add_argument("--save_log", action='store_true', default=False, 
                        help='save tensorboard logs to {}'.format(default_path))
    # Model Options
    available_models = sorted(name for name in network.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.model.__dict__[name]) 
                             )
    parser.add_argument("--model", type=str, default='unet_rgb',
                        choices=available_models, help='model name')
    # Dataset Options
    parser.add_argument("--data_root", type=str, default=dataset_path,
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default="CPN", 
                        choices=['CPN', 'CPN_all', 'CPN_crop', 'median', 'CTS', 'muscleUS'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num class (default: 2")
    parser.add_argument("--is_rgb", action='store_false', default=True)

    # Train Options
    parser.add_argument("--gpus", type=str, default='6,7')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    # Save best model checkpoint
    parser.add_argument("--save_model", action='store_true', default=False,
                        help='save best model param to \"./best_param\"')
    parser.add_argument("--save_ckpt", type=str, default=default_path,
                        help="save best model param to \"./best_param\"")

    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=(512, 448))
    parser.add_argument("--scale_factor", type=float, default=5e-1)

    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)") 
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'entropy_dice_loss', 
                                'dice_loss', 'ap_cross_entropy', 'ap_entropy_dice_loss'], 
                        help="loss type")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')

    # Validate Options
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 10)")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help='save segmentation results to \"./val_results\"')
    parser.add_argument("--save_last_results", action='store_true', default=False,
                        help='save segmentation results to \"./val_results\"')
    parser.add_argument("--save_val_dir", type=str, default=default_path,
                        help="save segmentation results to \"./results\"")

    # Save result images Options
    ctime = 'default'
    parser.add_argument("--save_train_results", action='store_true', default=False,
                        help='save segmentation results to \"./train_results\"')
    parser.add_argument("--save_train_dir", type=str, default=default_path,
                        help="save segmentation results to \"./train_results\"")
    parser.add_argument("--current_time", type=str, default=ctime,
                        help="results images folder name (default: current time)")

    # Run Demo
    parser.add_argument("--run_demo", action='store_true', default=False)

    return parser


if __name__ == '__main__':

    opts = get_argparser().parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    total_time = datetime.now()
    try:
        for model_choice in ['deeplabv3_resnet101', 'deeplabv3_resnet50']:
            for lr_policy_choice in ['poly', 'step']:
                for loss_name in ['cross_entropy', 'ap_cross_entropy', 'entropy_dice_loss', 'ap_entropy_dice_loss']:
                    for lr in [1e-3, 1e-4, 1e-5]:
                        
                        opts.model = model_choice
                        opts.lr_policy = lr_policy_choice
                        if lr_policy_choice == 'poly':
                            lr = lr
                        if model_choice == 'deeplabv3_resnet101':
                            opts.batch_size = 64
                        else:
                            opts.batch_size = 128

                        opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                        opts.loss_type = loss_name

                        if loss_name == 'focal_loss':
                            opts.lr = lr*1e+4
                        elif loss_name == 'entropy_dice_loss':
                            opts.lr = lr
                        else:
                            opts.lr = lr

                        start_time = datetime.now()
                        train(devices=device, opts=opts)
                        time_elapsed = datetime.now() - start_time

                        logdir = os.path.join(opts.Tlog_dir, opts.model, opts.current_time + '_' + socket.gethostname())
                        with open(os.path.join(logdir, 'summary.txt'), 'a') as f:
                            f.write('Time elapsed (h:m:s) {}'.format(time_elapsed))

    except KeyboardInterrupt:
        print("Stop !!!")
    total_time = datetime.now() - total_time

    print('Time elapsed (h:m:s.ms) {}'.format(total_time))