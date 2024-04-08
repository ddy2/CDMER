from networks import MsImageDis, IdDis
from utils import get_scheduler
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from random_erasing import RandomErasing
import yaml
from models import *
import argparse
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:3")

parser = argparse.ArgumentParser(description='FaceCycle')
parser.add_argument('--loadmodel', default= 'FaceCycleModel.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()

idcodegen = codegeneration()
Swap_Norm = normalizer()

codegeneration = codegeneration().to(device)
exptoflow = exptoflow().to(device)
Swap_Generator = generator().to(device)

def denorm(x):
    x[:, 0, :, :] = x[:, 0, :, :] * 0.229 + 0.485
    x[:, 1, :, :] = x[:, 1, :, :] * 0.224 + 0.456
    x[:, 2, :, :] = x[:, 2, :, :] * 0.225 + 0.406
    return x.clamp(0, 1)


def denorm_reto(x):
    x[:, 0, :, :] = ((x[:, 0, :, :] * 0.229 + 0.485) - 0.5) * 0.5
    x[:, 1, :, :] = ((x[:, 1, :, :] * 0.224 + 0.456) - 0.5) * 0.5
    x[:, 2, :, :] = ((x[:, 2, :, :] * 0.225 + 0.406) - 0.5) * 0.5
    return x.clamp(0, 1)



def to_gray(half=False):  # simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x

    return forward

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i, :, :, :])  # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY)  # 256x128x1
        xx = cv2.Canny(xx, 10, 200)  # 256x128
        xx = xx / 255.0 - 0.5  # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0], xx.shape[1]) * 0.1  # add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i, :, :] = xx
    out = out.unsqueeze(1)
    return out


def scale2(x):
    if x.size(2) > 128:  # do not need to scale the input
        return x
    x = torch.nn.functional.upsample(x, scale_factor=2, mode='nearest')  # bicubic is not available for the time being.
    return x


def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp


def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(device)  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

######################################################################
# Load model
# ---------------------------
def load_network(network, name):
    save_path = os.path.join('./models', name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network


def load_config(name):
    config_path = os.path.join('./models', name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def norm(f, dim=1):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=dim, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f

class classifier(torch.nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.classify = nn.Sequential(nn.Linear(256, 3))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.classify(x)

class DGNetpp_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(DGNetpp_Trainer, self).__init__()
        lr_g = hyperparameters['lr_g']
        lr_d = hyperparameters['lr_d']
        lr_id_d = hyperparameters['lr_id_d']
        self.fc = classifier()
        # Initiate the networks
        # We do not need to manually set fp16 in the network. So here I set fp16=False.
        self.gen_encode = nn.Sequential(codegeneration,
                                          exptoflow
                                          )
        self.gen_decode = Swap_Generator
        self.swap_norm = Swap_Norm

        if not 'ID_stride' in hyperparameters.keys():
            hyperparameters['ID_stride'] = 2

        self.id_a = idcodegen
        self.id_b = self.id_a
        self.dis_a = MsImageDis(3, hyperparameters['dis'], fp16=False)  # discriminator for domain a
        self.dis_b = self.dis_a

        self.id_dis = IdDis(hyperparameters['gen']['id_dim'], hyperparameters['dis'], fp16=False)  # ID discriminator
        # RGB to one channel
        if hyperparameters['single'] == 'edge':
            self.single = to_edge
        else:
            self.single = to_gray(False)

        # Random Erasing when training
        if not 'erasing_p' in hyperparameters.keys():
            hyperparameters['erasing_p'] = 0
        self.single_re = RandomErasing(probability=hyperparameters['erasing_p'], mean=[0.0, 0.0, 0.0])

        if not 'T_w' in hyperparameters.keys():
            hyperparameters['T_w'] = 1
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_a_params = list(self.dis_a.parameters())
        gen_encoder_params = list(self.gen_encode.parameters())
        gen_decoder_params = list(self.gen_decode.parameters())
        id_dis_params = list(self.id_dis.parameters())

        self.dis_a_opt = torch.optim.Adam([p for p in dis_a_params if p.requires_grad],
                                          lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.id_dis_opt = torch.optim.Adam([p for p in id_dis_params if p.requires_grad],
                                           lr=lr_id_d, betas=(beta1, beta2),
                                           weight_decay=hyperparameters['weight_decay'])
        self.gen_encoder_opt = torch.optim.Adam([p for p in gen_encoder_params if p.requires_grad],
                                          lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_decoder_opt = torch.optim.Adam([p for p in gen_decoder_params if p.requires_grad],
                                          lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.fc_opt = torch.optim.SGD([{"params": self.fc.parameters()}], lr=0.0001, momentum=0.9, weight_decay=0.0001,
                                      nesterov=False)
        # id params
        ignored_params = (list(map(id, self.id_a.parameters()))
                          + list(map(id, self.id_a.parameters())))
        base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
        lr2 = hyperparameters['lr2']
        self.id_opt = torch.optim.SGD([
            {'params': base_params, 'lr': lr2},
            {'params': self.id_a.parameters(), 'lr': lr2 * 10}
        ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)

        self.dis_a_scheduler = get_scheduler(self.dis_a_opt, hyperparameters)
        self.id_dis_scheduler = get_scheduler(self.id_dis_opt, hyperparameters)
        self.id_dis_scheduler.gamma = hyperparameters['gamma2']
        self.gen_encoder_scheduler = get_scheduler(self.gen_encoder_opt, hyperparameters)
        self.gen_decoder_scheduler = get_scheduler(self.gen_decoder_opt, hyperparameters)
        self.id_scheduler = get_scheduler(self.id_opt, hyperparameters)
        self.id_scheduler.gamma = hyperparameters['gamma2']

        # ID Loss
        self.id_criterion = nn.CrossEntropyLoss()
        self.criterion_teacher = nn.KLDivLoss(size_average=False)

    def to_re(self, x):
        out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
        out = out
        for i in range(x.size(0)):
            out[i, :, :, :] = self.single_re(x[i, :, :, :])
        return out

    def forward(self, x_a, x_b):
        self.eval()
        s_a = self.gen_a_encode(self.single(x_a))
        s_b = self.gen_b_encode(self.single(x_b))
        f_a, _, _ = self.id_a(scale2(x_a))
        f_b, _, _ = self.id_b(scale2(x_b))
        x_ba = self.gen_b_decode(s_b, f_a)
        x_ab = self.gen_a_decode(s_a, f_b)
        self.train()
        return x_ab, x_ba

    def load_dict(self, dict_path):
        self.gen_encode.load_state_dict(
            torch.load(dict_path + 'gen_encode_00069001.pt')['a'])
        self.gen_decode.load_state_dict(
            torch.load(dict_path + 'gen_decoder_00069001.pt')['a'])

    def data_generate(self, Onset_source, Apex_source, labels_source, Onset_target, Apex_target):
        p_a, s_a, bf_a = self.gen_encode(Apex_source.unsqueeze(dim=0) - Onset_source.unsqueeze(dim=0))
        p_b, s_b, bf_b = self.gen_encode(Apex_target.unsqueeze(dim=0) - Onset_target.unsqueeze(dim=0))
        n_b = self.gen_decode(Apex_target.unsqueeze(dim=0), p_b)
        x_ab = self.gen_decode(n_b, bf_a)
        return n_b, x_ab

    def data_generate_test(self, Onset_source, Apex_source, labels_source, Onset_target, Apex_target):
        p_a, s_a, bf_a = self.gen_encode(Apex_source.unsqueeze(dim=0) - Onset_source.unsqueeze(dim=0))
        p_b, s_b, bf_b = self.gen_encode(Apex_target.unsqueeze(dim=0) - Onset_target.unsqueeze(dim=0))
        n_a = self.gen_decode(Apex_source.unsqueeze(dim=0), p_a)
        n_b = self.gen_decode(Apex_target.unsqueeze(dim=0), p_b)
        x_a_recon = self.gen_decode(n_a, bf_a)
        x_b_recon = self.gen_decode(n_b, bf_b)
        x_ba = self.gen_decode(n_a, bf_b)
        x_ab = self.gen_decode(n_b, bf_a)
        _, s_b_recon, fd_b_recon = self.gen_encode(x_ba - Onset_source.unsqueeze(dim=0))
        _, s_a_recon, fb_a_recon = self.gen_encode(x_ab - Onset_target.unsqueeze(dim=0))
        x_aba = self.gen_decode(n_a, fb_a_recon)
        x_bab = self.gen_decode(n_b, fd_b_recon)
        return n_a, n_b, x_a_recon, x_b_recon, x_ba, x_ab, x_aba, x_bab

    def image_classification(self, loader):
        start_test = True
        with torch.no_grad():
            bingo_cnt = 0
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                Onset_target, Apex_target, land_target, labels = next(iter_test)
                Onset_target = Onset_target.to(device)
                Apex_target = Apex_target.to(device)
                land_target = land_target.to(device)
                _, x, _ = self.gen_encode(Apex_target-Onset_target)
                outputs = self.fc(torch.flatten(x, 1))
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                _, predicts = torch.max(outputs, 1)
                # print('target_predicts:', predicts.cpu())
                correct_num = torch.eq(predicts.cpu(), labels.squeeze(dim=1))
                bingo_cnt += correct_num.sum().cpu()
        _, predict = torch.max(all_output, 1)
        matrix = confusion_matrix(all_label.squeeze(1), predict)
        print(matrix)
        positive_F1 = 2 * (matrix[0, 0] / matrix.sum(1)[0]) * (matrix[0, 0] / matrix.sum(0)[0]) / (
                (matrix[0, 0] / matrix.sum(1)[0]) + (matrix[0, 0] / matrix.sum(0)[0]))
        surprise_F1 = 2 * (matrix[1, 1] / matrix.sum(1)[1]) * (matrix[1, 1] / matrix.sum(0)[1]) / (
                (matrix[1, 1] / matrix.sum(1)[1]) + (matrix[1, 1] / matrix.sum(0)[1]))
        negtive_F1 = 2 * (matrix[2, 2] / matrix.sum(1)[2]) * (matrix[2, 2] / matrix.sum(0)[2]) / (
                (matrix[2, 2] / matrix.sum(1)[2]) + (matrix[2, 2] / matrix.sum(0)[2]))
        f1 = (positive_F1 + surprise_F1 + negtive_F1) / 3
        accuracy = bingo_cnt / float(all_label.size()[0])
        mean_ent = torch.mean(Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

        hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
        hist_tar = hist_tar / hist_tar.sum()
        return accuracy, f1, hist_tar, mean_ent

