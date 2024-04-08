import os, random
from torchvision import transforms
from PIL import Image
from trainer_data_generate import DGNetpp_Trainer
from utils import __write_images, get_config
import argparse

import numpy.random as random
def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def image_train(resize_size=64):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def casme_balance():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='DGNet++', help="DGNet++")
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    opts = parser.parse_args()
    config = get_config(opts.config)
    dict_path = "outputs/CASME2_stage2_NIR/"
    output_directory = os.path.join(opts.output_path + "/outputs_final", 'CASME2_new')

    if not os.path.exists(output_directory):
        print("Creating directory: {}".format(output_directory))
        os.makedirs(output_directory)

    train_tran = image_train()
    data_list = open('data/CASME2.txt').readlines()
    gen_list = open('data/CASME2.txt').readlines()

    trainer = DGNetpp_Trainer(config)
    image_path, labels = [], []
    positive = 41
    surprise = 48
    z = -1
    for i in range(1000):
        z = z + 1
        if z % len(data_list) == 0:
            z = 0
        j = random.randint(0,len(data_list))
        Onset_source_path, Apex_source_path, _, _, labels_source = data_list[z].split()
        Onset_target_path, Apex_target_path, _, _, labels_gen = gen_list[j].split()
        if positive==0 and surprise==0:
            print('结束：', positive, surprise)
            break
        if (labels_source == '2') or (positive==0 and labels_source=='0') or (surprise==0 and labels_source=='1'):
            continue
        else:
            Onset_source, Apex_source = rgb_loader(Onset_source_path), rgb_loader(Apex_source_path)
            Onset_target, Apex_target = rgb_loader(Onset_target_path), rgb_loader(Apex_target_path)

            Onset_source, Apex_source = train_tran(Onset_source), train_tran(Apex_source)
            Onset_target, Apex_target = train_tran(Onset_target), train_tran(Apex_target)

            trainer.load_dict(dict_path)
            gen_neu_target, gen_source_target = trainer.data_generate(Onset_source, Apex_source, labels_source,
                                                                      Onset_target, Apex_target)

            save_dir = output_directory.strip('./') + '/' + Onset_source_path.split('/')[-1].rstrip(
                '\\' + Onset_source_path.split('\\')[-1])
            print(save_dir)
            if not os.path.exists(save_dir):
                print("Creating directory: {}".format(save_dir))
                os.makedirs(save_dir)
            gen_neu, gen_source = [], []
            gen_neu.append(gen_neu_target)
            gen_source.append(gen_source_target)

            __write_images(gen_neu, 224, save_dir + '/' + Onset_source_path.split('\\')[-1])
            __write_images(gen_source, 224, save_dir + '/' + Apex_source_path.split('\\')[-1])
            image_path.append([save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1],
                               save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1]])
            labels.append(labels_source)
            if labels_source == '0':
                positive = positive - 1
            else:
                surprise = surprise - 1
    with open('data/CASME2_new.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(image_path)):
            f.write(image_path[i][0] + '\t' + image_path[i][1] + '\t' +
                    image_path[i][2] + '\t' + image_path[i][3] + '\t'+ str(labels[i]) + '\n')
    return

def SMIC_HS_balance():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='DGNet++', help="DGNet++")
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    opts = parser.parse_args()
    config = get_config(opts.config)
    dict_path = "outputs/SMIC_NIR_SMIC_HS/15000/checkpoints/"
    output_directory = os.path.join(opts.output_path + "/outputs_final", 'SMIC_HS_new')

    if not os.path.exists(output_directory):
        print("Creating directory: {}".format(output_directory))
        os.makedirs(output_directory)

    train_tran = image_train()
    data_list = open('data/SMIC_HS.txt').readlines()
    gen_list = open('data/SMIC_HS.txt').readlines()

    trainer = DGNetpp_Trainer(config)
    image_path, labels = [], []
    positive = 19
    surprise = 27
    z = -1
    for i in range(1000):
        z = z + 1
        if z % len(data_list) == 0:
            z = 0
        j = random.randint(0,len(data_list))
        Onset_source_path, Apex_source_path, _, _, labels_source = data_list[z].split()
        Onset_target_path, Apex_target_path, _, _, labels_gen = gen_list[j].split()
        if positive==0 and surprise==0:
            print('结束：', positive, surprise)
            break
        if (labels_source == '2') or (positive==0 and labels_source=='0') or (surprise==0 and labels_source=='1'):
            continue
        else:
            Onset_source, Apex_source = rgb_loader(Onset_source_path), rgb_loader(Apex_source_path)
            Onset_target, Apex_target = rgb_loader(Onset_target_path), rgb_loader(Apex_target_path)

            Onset_source, Apex_source = train_tran(Onset_source), train_tran(Apex_source)
            Onset_target, Apex_target = train_tran(Onset_target), train_tran(Apex_target)

            trainer.load_dict(dict_path)
            gen_neu_target, gen_source_target = trainer.data_generate(Onset_source, Apex_source, labels_source,
                                                                      Onset_target, Apex_target)

            save_dir = output_directory.strip('./') + '/' + Onset_source_path.split('/')[-1].rstrip(
                '\\' + Onset_source_path.split('\\')[-1])
            print(save_dir)
            if not os.path.exists(save_dir):
                print("Creating directory: {}".format(save_dir))
                os.makedirs(save_dir)
            gen_neu, gen_source = [], []
            gen_neu.append(gen_neu_target)
            gen_source.append(gen_source_target)

            __write_images(gen_neu, 224, save_dir + '/' + Onset_source_path.split('\\')[-1])
            __write_images(gen_source, 224, save_dir + '/' + Apex_source_path.split('\\')[-1])
            image_path.append([save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1],
                               save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1]])
            labels.append(labels_source)
            if labels_source == '0':
                positive = positive - 1
            else:
                surprise = surprise - 1
    with open('data/SMIC_HS_new.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(image_path)):
            f.write(image_path[i][0] + '\t' + image_path[i][1] + '\t' +
                    image_path[i][2] + '\t' + image_path[i][3] + '\t'+ str(labels[i]) + '\n')
    return
def SMIC_VIS_balance():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='DGNet++', help="DGNet++")
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    opts = parser.parse_args()
    config = get_config(opts.config)
    dict_path = "outputs/SMIC_NIR_SMIC_VIS/15000/checkpoints/"
    output_directory = os.path.join(opts.output_path + "/outputs_final", 'SMIC_VIS_new')

    if not os.path.exists(output_directory):
        print("Creating directory: {}".format(output_directory))
        os.makedirs(output_directory)

    train_tran = image_train()
    data_list = open('data/SMIC_VIS.txt').readlines()
    gen_list = open('data/SMIC_VIS.txt').readlines()

    trainer = DGNetpp_Trainer(config)
    image_path, labels = [], []
    positive = 5
    surprise = 8
    z = -1
    for i in range(1000):
        z = z + 1
        if z % len(data_list) == 0:
            z = 0
        j = random.randint(0,len(data_list))
        Onset_source_path, Apex_source_path, _, _, labels_source = data_list[z].split()
        Onset_target_path, Apex_target_path, _, _, labels_gen = gen_list[j].split()
        if positive==0 and surprise==0:
            print('结束：', positive, surprise)
            break
        if (labels_source == '2') or (positive==0 and labels_source=='0') or (surprise==0 and labels_source=='1'):
            continue
        else:
            Onset_source, Apex_source = rgb_loader(Onset_source_path), rgb_loader(Apex_source_path)
            Onset_target, Apex_target = rgb_loader(Onset_target_path), rgb_loader(Apex_target_path)

            Onset_source, Apex_source = train_tran(Onset_source), train_tran(Apex_source)
            Onset_target, Apex_target = train_tran(Onset_target), train_tran(Apex_target)

            trainer.load_dict(dict_path)
            gen_neu_target, gen_source_target = trainer.data_generate(Onset_source, Apex_source, labels_source,
                                                                      Onset_target, Apex_target)

            save_dir = output_directory.strip('./') + '/' + Onset_source_path.split('/')[-1].rstrip(
                '\\' + Onset_source_path.split('\\')[-1])
            print(save_dir)
            if not os.path.exists(save_dir):
                print("Creating directory: {}".format(save_dir))
                os.makedirs(save_dir)
            gen_neu, gen_source = [], []
            gen_neu.append(gen_neu_target)
            gen_source.append(gen_source_target)

            __write_images(gen_neu, 224, save_dir + '/' + Onset_source_path.split('\\')[-1])
            __write_images(gen_source, 224, save_dir + '/' + Apex_source_path.split('\\')[-1])
            image_path.append([save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1],
                               save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1]])
            labels.append(labels_source)
            if labels_source == '0':
                positive = positive - 1
            else:
                surprise = surprise - 1
    with open('data/SMIC_VIS_new.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(image_path)):
            f.write(image_path[i][0] + '\t' + image_path[i][1] + '\t' +
                    image_path[i][2] + '\t' + image_path[i][3] + '\t'+ str(labels[i]) + '\n')
    return

def SMIC_NIR_balance():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='DGNet++', help="DGNet++")
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    opts = parser.parse_args()
    config = get_config(opts.config)
    dict_path = "outputs/CASME2_stage2_NIR/"
    output_directory = os.path.join(opts.output_path + "/outputs_final", 'SMIC_NIR_new')

    if not os.path.exists(output_directory):
        print("Creating directory: {}".format(output_directory))
        os.makedirs(output_directory)

    train_tran = image_train()
    data_list = open('data/SMIC_NIR.txt').readlines()
    gen_list = open('data/SMIC_NIR.txt').readlines()

    trainer = DGNetpp_Trainer(config)
    image_path, labels = [], []
    positive = 19
    surprise = 27
    z = -1
    for i in range(1000):
        z = z + 1
        if z % len(data_list) == 0:
            z = 0
        j = random.randint(0,len(data_list))
        Onset_source_path, Apex_source_path, _, _, labels_source = data_list[z].split()
        Onset_target_path, Apex_target_path, _, _, labels_gen = gen_list[j].split()
        if positive==0 and surprise==0:
            print('结束：', positive, surprise)
            break
        if (labels_source == '2') or (positive==0 and labels_source=='0') or (surprise==0 and labels_source=='1'):
            continue
        else:
            Onset_source, Apex_source = rgb_loader(Onset_source_path), rgb_loader(Apex_source_path)
            Onset_target, Apex_target = rgb_loader(Onset_target_path), rgb_loader(Apex_target_path)

            Onset_source, Apex_source = train_tran(Onset_source), train_tran(Apex_source)
            Onset_target, Apex_target = train_tran(Onset_target), train_tran(Apex_target)

            trainer.load_dict(dict_path)
            gen_neu_target, gen_source_target = trainer.data_generate(Onset_source, Apex_source, labels_source,
                                                                      Onset_target, Apex_target)

            save_dir = output_directory.strip('./') + '/' + Onset_source_path.split('/')[-1].rstrip(
                '\\' + Onset_source_path.split('\\')[-1])
            print(save_dir)
            if not os.path.exists(save_dir):
                print("Creating directory: {}".format(save_dir))
                os.makedirs(save_dir)
            gen_neu, gen_source = [], []
            gen_neu.append(gen_neu_target)
            gen_source.append(gen_source_target)

            __write_images(gen_neu, 224, save_dir + '/' + Onset_source_path.split('\\')[-1])
            __write_images(gen_source, 224, save_dir + '/' + Apex_source_path.split('\\')[-1])
            image_path.append([save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1],
                               save_dir + '/' + Onset_source_path.split('\\')[-1],
                               save_dir + '/' + Apex_source_path.split('\\')[-1]])
            labels.append(labels_source)
            if labels_source == '0':
                positive = positive - 1
            else:
                surprise = surprise - 1
    with open('data/SMIC_NIR_new.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(image_path)):
            f.write(image_path[i][0] + '\t' + image_path[i][1] + '\t' +
                    image_path[i][2] + '\t' + image_path[i][3] + '\t'+ str(labels[i]) + '\n')
    return

def data_generate(args, times):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='DGNet++', help="DGNet++")
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    opts = parser.parse_args()
    config = get_config(opts.config)
    dict_path = "outputs/CASME2_SMIC_NIR/15000/checkpoints/"
    output_directory = os.path.join(opts.output_path + "/outputs_new", 'CASME2_SMIC_NIR')

    if not os.path.exists(output_directory):
        print("Creating directory: {}".format(output_directory))
        os.makedirs(output_directory)

    train_tran = image_train()

    ## prepare data

    source_list = open(args.s_dset_path).readlines()
    target_list = open(args.t_dset_path).readlines()

    trainer = DGNetpp_Trainer(config)
    image_path, labels = [], []
    j = -1

    for a in range(0, times):
        for i in range(len(source_list)):
            print(i)
            j = random.randint(1, len(target_list))
            Onset_source_path, Apex_source_path, _, _, labels_source = source_list[i].split()
            Onset_target_path, Apex_target_path, _, _, _ = target_list[j].split()

            Onset_source_path = Onset_source_path.replace('\\', '/')
            Apex_source_path = Apex_source_path.replace('\\', '/')
            Onset_target_path = Onset_target_path.replace('\\', '/')
            Apex_target_path = Apex_target_path.replace('\\', '/')

            Onset_source, Apex_source = rgb_loader(Onset_source_path), rgb_loader(Apex_source_path)
            Onset_target, Apex_target = rgb_loader(Onset_target_path), rgb_loader(Apex_target_path)

            Onset_source, Apex_source = train_tran(Onset_source), train_tran(Apex_source)
            Onset_target, Apex_target = train_tran(Onset_target), train_tran(Apex_target)
            Onset_source = Onset_source.cuda()
            Apex_source = Apex_source.cuda()
            Onset_target = Onset_target.cuda()
            Apex_target = Apex_target.cuda()

            trainer.load_dict(dict_path)
            gen_neu_target, gen_source_target = trainer.data_generate(Onset_source, Apex_source, labels_source,
                                                                      Onset_target, Apex_target)

            print("image name ", Onset_source_path.split('/')[-1])
            save_dir = output_directory.strip('./') + '/' + Onset_source_path.rstrip(Onset_source_path.split('/')[-1])
            print("save_dir", save_dir)
            if not os.path.exists(save_dir):
                print("Creating directory: {}".format(save_dir))
                os.makedirs(save_dir)
            gen_neu, gen_source = [], []
            gen_neu.append(gen_neu_target)
            gen_source.append(gen_source_target)
            print('save path ', save_dir + Onset_source_path.split('/')[-1])

            __write_images(gen_neu, 224, save_dir + str(a) + '_' + Onset_source_path.split('/')[-1])
            __write_images(gen_source, 224, save_dir + str(a) + '_' + Apex_source_path.split('/')[-1])

            image_path.append([save_dir + str(a) + '_' + Onset_source_path.split('/')[-1],
                               save_dir + str(a) + '_' + Apex_source_path.split('/')[-1],
                               save_dir + str(a) + '_' + Onset_source_path.split('/')[-1],
                               save_dir + str(a) + '_' + Apex_source_path.split('/')[-1]])
            labels.append(labels_source)

    with open('data_new/CASME2_SMIC_NIR_5.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
        for i in range(0, len(image_path)):
            f.write(image_path[i][0] + '\t' + image_path[i][1] + '\t' +
                    image_path[i][2] + '\t' + image_path[i][3] + '\t' + str(labels[i]) + '\n')

    return
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='run')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=4, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50", "VGG16"])

    parser.add_argument('--dset', type=str, default='micro-expression')
    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--mu', type=int, default=4, help="init augmentation size = batch_size//mu")
    parser.add_argument('--ent_weight', type=float, default=0.1)
    parser.add_argument('--cot_weight', type=float, default=1.0, choices=[0, 1, 5, 10])
    parser.add_argument('--weight_aug', type=bool, default=True)
    parser.add_argument('--weight_cls', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=1)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    k = 25
    args.class_num = 3

    data_folder1 = 'data_new/'
    data_folder2 = 'data/'
    args.s_dset_path = data_folder2 + 'CASME2.txt'
    args.t_dset_path = data_folder2 + 'SMIC_NIR.txt'

    # casme_balance()
    # SMIC_HS_balance()
    # SMIC_VIS_balance()
    # SMIC_NIR_balance()
    data_generate(args, 5)