import os
import json
import time
import socket
from datetime import datetime

import torch

import utils
from mail import MailSend
from alphaTrain import train
from args import get_argparser


LOGIN = {
    3 : "/mnt/server5/sdi/login.json",
    4 : "/mnt/server5/sdi/login.json",
    5 : "/data1/sdi/login.json"
}
DEFAULT_DIR = {
    3 : "/mnt/server5/sdi",
    4 : "/mnt/server5/sdi",
    5 : "/data1/sdi"
}
DATA_DIR = {
    3 : "/mnt/server5/sdi/datasets",
    4 : "/mnt/server5/sdi/datasets",
    5 : "/data1/sdi/datasets"
}

def smail(subject: str = 'default subject', body: dict = {}, login_dir: str = ''):
    ''' send short report mail (smtp) 
    '''
    # Mail options
    to_addr = ['sdimivy014@korea.ac.kr']
    from_addr = ['singkuserver@korea.ac.kr']

    MailSend(subject=subject, msg=body, login_dir=login_dir, 
                ID='singkuserver', to_addr=to_addr, from_addr=from_addr)()

if __name__ == '__main__':

    print('basename:    ', os.path.basename(__file__)) # main.py
    print('dirname:     ', os.path.dirname(__file__)) # Empty in server3
    print('abspath:     ', os.path.abspath(__file__)) # /data1/sdi/CPNKD/main.py
    print('abs dirname: ', os.path.dirname(os.path.abspath(__file__))) # /data1/sdi/CPNKD

    opts = get_argparser().parse_args()

    ''' (.) Get hostname of server
    '''
    if socket.gethostname() == "server3":
        opts.cur_work_server = 3
        opts.login_dir = LOGIN[3]
        opts.default_path = os.path.join(DEFAULT_DIR[3], 
                                            os.path.dirname(os.path.abspath(__file__)).split('/')[-1] \
                                                + '-result')
        opts.data_root = DATA_DIR[3]
    elif socket.gethostname() == "server4":
        opts.cur_work_server = 4
        opts.login_dir = LOGIN[4]
        opts.default_path = os.path.join(DEFAULT_DIR[5], 
                                            os.path.dirname(__file__).split('/')[-1] \
                                                +'-result')
        opts.data_root = DATA_DIR[4]
    elif socket.gethostname() == "server5":
        opts.cur_work_server = 5
        opts.login_dir = LOGIN[5]
        opts.default_path = os.path.join(DEFAULT_DIR[5],
                                             os.path.dirname(__file__).split('/')[-1]\
                                                 +'-result')
        opts.data_root = DATA_DIR[5]
    else:
        raise NotImplementedError

    ''' (.) Verify ID & PW of G-mail
    '''
    if not os.path.exists(opts.login_dir):
        raise FileNotFoundError("login.json file not found (path: {})".format(opts.login_dir))
    
    ''' (.) GPUs setting
    '''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    ''' (.) Resume from log.json
    '''
    if os.path.exists(os.path.join(opts.default_path, 'log.json')):
        resume = True
        jog = utils.Params(os.path.join(opts.default_path, 'log.json')).__dict__
    else:
        resume = False
        jog = {
                'dataset_choice' : 0,
                'loss_choice' : 0,
                'model_choice' : 0,
                'output_stride_choice' : 0,
                'current_working_dir' : 0
                }
    H = jog['dataset_choice']
    I = jog['loss_choice']
    J = jog['model_choice']
    K = jog['output_stride_choice']

    ''' (.) Load message log
    '''
    if os.path.exists(os.path.join(opts.default_path, 'mlog.json')):
        with open(os.path.join(opts.default_path, 'mlog.json'), "r") as f:
            mlog = json.load(f)
    else:
        mlog = {}

    ''' (.) Start train
    '''
    total_time = datetime.now()
    try:
        dataset_choice = ['CPN_six', 'Median']
        loss_choice = ['ap_entropy_dice_loss', 'ap_cross_entropy', 'dice_loss']
        model_choice = ['deeplabv3plus_resnet101', 'deeplabv3plus_resnet50']
        output_stride_choice = [8, 16, 32, 64]

        for h in range(len(dataset_choice)):
            if h < H:
                continue
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
                        opts.dataset = dataset_choice[h]
                        opts.loss_type = loss_choice[i]
                        opts.model = model_choice[j]
                        opts.output_stride = output_stride_choice[k]
                        print("h: {}, i: {}, j: {}, k: {}".format(h, i, j, k))

                        if resume:
                            resume = False
                            logdir = jog['current_working_dir']
                            opts.current_time = "resume"
                            opts.ckpt = os.path.join(logdir, 'best_param', 'checkpoint.pt')
                        else:
                            opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                            if opts.run_demo:
                                logdir = os.path.join(opts.Tlog_dir, opts.model, 
                                                        opts.current_time + '_' + opts.dataset + '_demo')
                            else:
                                logdir = os.path.join(opts.Tlog_dir, opts.model, 
                                                        opts.current_time + '_' + opts.dataset)

                        # leave log
                        with open(os.path.join(opts.default_path, 'log.json'), "w") as f:
                            jog['loss_choice'] = i
                            jog['model_choice'] = j
                            jog['output_stride_choice'] = k
                            jog['current_working_dir'] = logdir
                            json.dump(jog, f, indent=2)

                        start_time = datetime.now()
                        key = str(h*len(dataset_choice)*len(model_choice)*len(output_stride_choice) \
                                    + i*len(model_choice)*len(output_stride_choice) \
                                        + j*len(output_stride_choice) + k)
                        ''' 
                            Ex) {"Model" : model_choice[j], "F1-0" : "0.9", "F1-1" : "0.1"}
                        '''
                        #mlog[key] = {"Model" : model_choice[j], "F1-0" : "0.9", "F1-1" : "0.1"}
                        mlog[key] = train(devices=device, opts=opts, LOGDIR=logdir)
                        #time.sleep(5)
                        time_elapsed = datetime.now() - start_time

                        with open(os.path.join(opts.default_path, 'mlog.json'), "w") as f:
                            ''' JSON treats keys as strings
                            '''
                            json.dump(mlog, f, indent=2)
                        
                        if os.path.exists(os.path.join(logdir, 'summary.json')):
                            params = utils.Params(json_path=os.path.join(logdir, 'summary.json')).dict
                            params["time_elpased"] = str(time_elapsed)
                            utils.save_dict_to_json(d=params, json_path=os.path.join(logdir, 'summary.json'))
                    K = 0
                J = 0

                mlog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(datetime.now() - mid_time)
                smail(subject="Short report-{}".format(loss_choice[i]), body=mlog, login_dir=opts.login_dir)
                mlog = {}
                os.remove(os.path.join(opts.default_path, 'mlog.json'))


        os.remove(os.path.join(opts.default_path, 'log.json'))

    except KeyboardInterrupt:
        print("Stop !!!")
    total_time = datetime.now() - total_time

    print('Time elapsed (h:m:s.ms) {}'.format(total_time))