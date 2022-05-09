import os
import json
import argparse
import schedule
import socket
from datetime import datetime
import time

import torch
from mail import MailSend
from alphaTrain import train
import network

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--default_path", type=str, default="/data1/sdi/CPNnetV1-result/", 
                        help="path to results")
    parser.add_argument("--current_time", type=str, default=None,
                        help="results images folder name (default: current time)")
    
    # Tensorboard options
    parser.add_argument("--Tlog_dir", type=str, default=None,
                        help="path to tensorboard log")
    parser.add_argument("--save_log", action='store_true', default=False, 
                        help="save log to default path (default: False)")

    # Model options
    available_models = sorted(name for name in network.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.model.__dict__[name])
                             )
    parser.add_argument("--model", type=str, default='unet_rgb', choices=available_models,
                        help='model name (default: Unet RGB)')
    # DeeplabV3+ options
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp (default: False)")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32, 64],
                        help="output stride (default: 16)")

    # Dataset options
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers (default: 4)")
    parser.add_argument("--data_root", type=str, default="/data1/sdi/datasets/",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default="CPN_six",
                        help='Name of dataset (default: CPN_six)')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes (default: 2)")
    parser.add_argument("--is_rgb", action='store_false', default=True,
                        help="choose True: RGB, False: grey (default: True)")

    # Augmentation options
    parser.add_argument("--resize", default=(496, 468))
    parser.add_argument("--crop_size", default=(512, 448))
    parser.add_argument("--scale_factor", type=float, default=5e-1)

    # Train options
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--gpus", type=str, default='6,7',
                        help="GPU IDs (default: 6,7)")
    parser.add_argument("--total_itrs", type=int, default=10000,
                        help="epoch number (default: 10k)")
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)")
    parser.add_argument("--loss_type", type=str, default='dice_loss',
                        help="criterion (default: dice loss)")
    parser.add_argument("--lr_policy", type=str, default='step',
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1000, 
                        help="step size (default: 1000)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

    # Validate options
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validate (default: 4)')

    # Outcome options
    parser.add_argument("--save_model", action='store_true', default=False,
                        help="save best model param to \"./best_param\" (default: False)")
    parser.add_argument("--save_ckpt", type=str, default=None,
                        help="save best model param to \"./best_param\"")
    parser.add_argument("--val_results", action='store_true', default=False,
                        help='save validate segmentation results to \"./val_results\" (default: False)')
    parser.add_argument("--val_results_dir", type=str, default=None,
                        help="save segmentation results to \"./results\"")
    # Model checkpoint options
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    # Run Demo
    parser.add_argument("--run_demo", action='store_true', default=False)

    return parser

def smail(subject: str = 'default subject', body: dict = {}):
    ''' send short report mail (smtp) 
    '''
    to_addr = ['sdimivy014@korea.ac.kr']
    from_addr = ['donotreply@korea.ac.kr']

    ms = MailSend(subject=subject, msg=body, to_addr=to_addr, from_addr=from_addr)
    ms()


if __name__ == '__main__':

    opts = get_argparser().parse_args()
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    if os.path.exists(os.path.join(opts.default_path, 'log.json')):
        resume = True
        with open(os.path.join(opts.default_path, 'log.json'), "r") as f:
            jog = json.load(f)
    else:
        resume = False
        jog = {
                'loss_choice' : 0,
                'model_choice' : 0,
                'output_stride_choice' : 0,
                'current_working_dir' : 0
                }
    I = jog['loss_choice']
    J = jog['model_choice']
    K = jog['output_stride_choice']

    if os.path.exists(os.path.join(opts.default_path, 'mlog.json')):
        with open(os.path.join(opts.default_path, 'mlog.json'), "r") as f:
            mlog = json.load(f)
    else:
        mlog = {}

    total_time = datetime.now()
    try:
        loss_choice = ['dice_loss', 'ap_entropy_dice_loss', 'entropy_dice_loss', 
                                'ap_cross_entropy', 'cross_entropy', 'focal_loss']
        model_choice = ['deeplabv3plus_resnet101', 'deeplabv3plus_resnet50']
        output_stride_choice = [8, 16, 32, 64]

        for i in range(len(loss_choice)):
            if i < I:
                continue  
            mid_time = datetime.now()
            for j in range(len(model_choice)):
                if j < J:
                    continue
                for k in range(len(output_stride_choice)):
                    if k < K:
                        continue
                    opts.Tlog_dir = opts.default_path
                    opts.loss_type = loss_choice[i]
                    opts.model = model_choice[j]
                    opts.output_stride = output_stride_choice[k]
                    print("i: {}, j: {}, k: {}".format(i, j, k))

                    if resume:
                        resume = False
                        logdir = jog['current_working_dir']
                        opts.current_time = "resume"
                        opts.ckpt = logdir
                    else:
                        opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                        logdir = os.path.join(opts.Tlog_dir, opts.model, opts.current_time + '_' + opts.dataset)

                    # leave log
                    with open(os.path.join(opts.default_path, 'log.json'), "w") as f:
                        jog['loss_choice'] = i
                        jog['model_choice'] = j
                        jog['output_stride_choice'] = k
                        jog['current_working_dir'] = logdir
                        json.dump(jog, f, indent=2)

                    start_time = datetime.now()
                    key = str(i*len(model_choice)*len(output_stride_choice) + j*len(output_stride_choice) + k)
                    ''' 
                        Ex) {"Model" : model_choice[j], "F1-0" : "0.9", "F1-1" : "0.1"}
                    '''
                    mlog[key] = {"Model" : model_choice[j], "F1-0" : "0.9", "F1-1" : "0.1"}
                    #mlog[key] = train(devices=device, opts=opts, REPORT=ms)
                    time.sleep(5)
                    time_elapsed = datetime.now() - start_time

                    with open(os.path.join(opts.default_path, 'mlog.json'), "w") as f:
                        ''' JSON treats keys as strings
                        '''
                        json.dump(mlog, f, indent=2)
                    
                    if os.path.exists(os.path.join(logdir, 'summary.txt')):
                        with open(os.path.join(logdir, 'summary.txt'), 'a') as f:
                            f.write('Time elapsed (h:m:s) {}'.format(time_elapsed))

            mlog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(datetime.now() - mid_time)
            smail(subject="Short report-{}".format(loss_choice[i]), body=mlog)
            mlog = {}
            os.remove(os.path.join(opts.default_path, 'mlog.json'))

        os.remove(os.path.join(opts.default_path, 'log.json'))

    except KeyboardInterrupt:
        print("Stop !!!")
    total_time = datetime.now() - total_time

    print('Time elapsed (h:m:s.ms) {}'.format(total_time))