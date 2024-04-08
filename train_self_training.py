import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import data_list
from trainer_self import DGNetpp_Trainer
from utils import get_mix_data_loaders, get_data_loader_folder, prepare_sub_folder_pseudo, write_html, write_loss, get_config, write_2images, Timer
import argparse
import torch
import numpy.random as random


def image_train(resize_size=64):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def image_test(resize_size=64):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/CASME2SMIC_HS.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
    parser.add_argument('--trainer', type=str, default='DGNet++', help="DGNet++")
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    opts = parser.parse_args()
    config = get_config(opts.config)
    count_dis_update = config['dis_update_iter']
    subiterations = 0
    countaa, countab, countba, countbb = 1, 1, 1, 1
    output_directory = os.path.join(opts.output_path + "/outputs", 'CASME2_SMIC_HS')



    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())
    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker)

    # random.permutation随机排序
    train_a_rand = random.permutation(len(dset_loaders["source"]))[0:1]
    train_b_rand = random.permutation(len(dset_loaders["target"]))[0:1]

    # torch.cat(): 用于连接两个相同大小的张量
    # torch.stack(): 用于连接两个相同大小的张量，并扩展维度
    train_display_images_a_p = torch.stack([dset_loaders["source"].dataset[i][0] for i in train_a_rand]).cuda()
    train_display_images_a = torch.stack([dset_loaders["source"].dataset[i][1] for i in train_a_rand]).cuda()
    train_display_images_a_l = torch.stack([dset_loaders["source"].dataset[i][2] for i in train_a_rand]).cuda()

    train_display_images_b_p = torch.stack([dset_loaders["target"].dataset[i][0] for i in train_b_rand]).cuda()
    train_display_images_b = torch.stack([dset_loaders["target"].dataset[i][1] for i in train_b_rand]).cuda()
    train_display_images_b_l = torch.stack([dset_loaders["target"].dataset[i][2] for i in train_b_rand]).cuda()


    trainer = DGNetpp_Trainer(config)
    trainer.cuda()



    for i in range(args.max_iterations + 1):

        trainer.update_learning_rate()
        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])


        Onset_source, Apex_source, land_source, labels_source = next(iter_source)
        Onset_target, Apex_target, land_target, labels_target = next(iter_target)
        Onset_source, Apex_source, labels_source, Onset_target, Apex_target = Onset_source.cuda(), Apex_source.cuda(), labels_source.cuda(), Onset_target.cuda(), Apex_target.cuda()
        land_source, land_target, labels_target = land_source.cuda(), land_target.cuda(), labels_target.cuda()
        labels_source = torch.flatten(labels_source, 0)
        labels_target = torch.flatten(labels_target, 0)

        with Timer("Elapsed time in update: %f"):
            # Main training code
            if config['aa']:
                trainer.gen_update(Apex_source, labels_source, Onset_source, land_source, Apex_target, labels_target, Onset_target, land_target, config, subiterations)
        # test in target domain in every epoch

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):

            train_temp_acc, train_class_weight, train_mean_ent = trainer.train_image_classification(dset_loaders)

            train_log_str = "iter: {:06d}, train precision: {:.5f}, train_mean_entropy: {:.5f}".format(i,train_temp_acc,train_mean_ent)
            print(train_log_str, '\n')

            round_output_directory = os.path.join(output_directory, str(i))
            checkpoint_directory, image_directory, pseudo_directory = prepare_sub_folder_pseudo(
                round_output_directory)
            trainer.save(checkpoint_directory, i)
            with torch.no_grad():
                image_outputs = trainer.sample_ab(train_display_images_a_p, train_display_images_a,
                                                  train_display_images_a_l, train_display_images_b_p,
                                                  train_display_images_b, train_display_images_b_l)

            write_2images(image_outputs, 224, image_directory, 'train_ab_%08d' % (i + 1))
            del image_outputs







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

    args.name = ['CASME2', 'SAMM']
    k = 25
    args.class_num = 3
    args.max_iterations = 20000
    args.test_interval = 500
    args.lr = 1e-3

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_folder = 'data/'
    args.s_dset_path = data_folder + 'CASME2.txt'
    args.t_dset_path = data_folder + 'SMIC_HS.txt'


    train(args)