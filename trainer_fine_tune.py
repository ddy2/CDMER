from networks import MsImageDis, IdDis
from utils import get_model_list, vgg_preprocess, load_vgg16, get_scheduler
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from random_erasing import RandomErasing
from shutil import copyfile, copytree
import yaml
from models import *
import argparse
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='FaceCycle')
parser.add_argument('--loadmodel', default= 'FaceCycleModel.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()

idcodegen = codegeneration().cuda()
Swap_Norm = normalizer().cuda()

codegeneration = codegeneration().cuda()
exptoflow = exptoflow().cuda()
Swap_Generator = generator().cuda()
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    codegeneration.load_state_dict(state_dict['codegeneration'])
    exptoflow.load_state_dict(state_dict['exptoflow'])
    Swap_Generator.load_state_dict(state_dict['Swap_Generator'])
    idcodegen.load_state_dict(state_dict['idcodegen'])
    Swap_Norm.load_state_dict(state_dict['Swap_Norm'])

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
    return out.cuda()


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

def norm(f, dim=1):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=dim, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f

class classifier(torch.nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.classify = nn.Sequential(nn.Linear(256, 3)
                                      # nn.LeakyReLU(),
                                      # nn.Linear(64, 3)
        )

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
        self.fc = classifier().cuda()
        # self.fc.load_state_dict(torch.load('outputs/CASME2_stage2/fc_00010001.pt')['a'])

        # Initiate the networks
        self.gen_encode = nn.Sequential(codegeneration,
                                          exptoflow
                                          )
        # self.gen_encode.load_state_dict(torch.load('outputs/CASME2_stage2_HS_NIR/gen_encode_00011001.pt')['a'])
        self.gen_decode = Swap_Generator
        # self.gen_decode.load_state_dict(torch.load('outputs/CASME2_stage2_HS_NIR/gen_decoder_00011001.pt')['a'])

        self.gen_encode.load_state_dict(
            torch.load('outputs/CASME2_SMIC_VIS/15000/checkpoints/gen_encode_00015001.pt')['a'])

        # self.gen_decode.load_state_dict(
        #     torch.load('outputs/SMIC_NIR_CASME2/gen_decoder_00015001.pt')['a'])
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
        self.fc_opt = torch.optim.SGD([{"params": self.fc.parameters()}], lr=0.008, momentum=0.9, weight_decay=0.0005,
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
        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def to_re(self, x):
        out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
        out = out.cuda()
        for i in range(x.size(0)):
            out[i, :, :, :] = self.single_re(x[i, :, :, :])
        return out

    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))

    def recon_criterion_sqrt(self, input, target):
        diff = input - target
        return torch.mean(torch.sqrt(torch.abs(diff[:]) + 1e-8))

    def recon_criterion2(self, input, target):
        diff = input - target
        return torch.mean(diff[:] ** 2)

    def recon_cos(self, input, target):
        cos = torch.nn.CosineSimilarity()
        cos_dis = 1 - cos(input, target)
        return torch.mean(cos_dis[:])

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

    def stage3_gen_update_aa(self, x_a, l_a, xp_a, ld_a, x_b, l_b, xp_b, ld_b, hyperparameters, iteration):
        # ppa, ppb is the same person
        self.gen_encoder_opt.zero_grad()
        self.fc_opt.zero_grad()
        self.gen_encode.train()
        self.fc.train()
        # encode, s_a,s_b表情
        p_a, s_a, fb_a = self.gen_encode(x_a-xp_a)
        p_b, s_b, fb_b = self.gen_encode(x_b-xp_b)
        # 基于表情信息s_a进行表情预测
        y_a = self.fc(torch.flatten(s_a, 1))

        l_a = l_a.long()
        # 表情损失
        self.loss_id = self.id_criterion(y_a, l_a)
        self.loss_gen_total = hyperparameters['id_w'] * self.loss_id


        self.loss_gen_total.backward()
        self.gen_encoder_opt.step()
        self.fc_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample_ab(self, x_a_p, x_a, x_a_l, x_b_p, x_b, x_b_l):
        self.eval()
        neu_a, neu_b, x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2, x_aba, x_bab = [], [], [], [], [], [], [], [], [], []

        for i in range(x_a.size(0)):

            p_a, s_a, bf_a = self.gen_encode(x_a[i].unsqueeze(dim=0)-x_a_p[i].unsqueeze(dim=0))
            p_b, s_b, bf_b = self.gen_encode(x_b[i].unsqueeze(dim=0)-x_b_p[i].unsqueeze(dim=0))
            n_a = self.gen_decode(x_a, p_a)
            n_b = self.gen_decode(x_b, p_b)
            neu_a.append(n_a)
            neu_b.append(n_b)
            x_a_recon.append(self.gen_decode(n_a, bf_a))
            x_b_recon.append(self.gen_decode(n_b, bf_b))

            x_ba = self.gen_decode(n_a, bf_b)
            x_ab = self.gen_decode(n_b, bf_a)
            x_ba1.append(x_ba)
            # x_ba2.append(self.gen_b_decode(s_b, f_a))
            x_ab1.append(x_ab)
            # x_ab2.append(self.gen_a_decode(s_a, f_b))
            # cycle
            _, s_b_recon, fd_b_recon = self.gen_encode(x_ba-x_a_p[i].unsqueeze(dim=0))
            _, s_a_recon, fb_a_recon = self.gen_encode(x_ab-x_b_p[i].unsqueeze(dim=0))
            x_aba.append(self.gen_decode(n_a, fb_a_recon))
            x_bab.append(self.gen_decode(n_b, fd_b_recon))



        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ba1 = torch.cat(x_ba1)
        x_ab1 = torch.cat(x_ab1)
        neu_a = torch.cat(neu_a)
        neu_b = torch.cat(neu_b)
        self.train()

        return x_a, neu_a, x_a_recon, x_aba, x_ab1, x_b, neu_b, x_b_recon, x_bab, x_ba1


    def dis_update_aa(self, x_a, p_a, ld_a,  x_b, p_b, ld_b, hyperparameters):
        self.id_dis_opt.zero_grad()
        # encode
        # x_a_single = self.single(x_a)
        f_a, s_a, bf_a = self.gen_a_encode(x_a)
        f_b, s_b, bf_b = self.gen_a_encode(x_b)
        n_a = Swap_Generator(x_a, f_a)
        n_b = Swap_Generator(x_b, f_b)
        # has gradient x_ba：b表情a身份   x_ab：a表情b身份
        x_ba = self.gen_a_decode(n_a, bf_b)
        x_ab = self.gen_a_decode(n_b, bf_a)
        f_b_recon, s_b_recon, _ = self.gen_a_encode(x_ba)
        f_a_recon, s_a_recon, _ = self.gen_a_encode(x_ab)
        print(s_a.shape)
        self.loss_exp_dis_ab, _, _ = self.id_dis.calc_dis_loss_ab(torch.flatten(s_a, 1).detach(), torch.flatten(s_b, 1).detach())
        self.loss_exp_dis_ab_recon, _, _ = self.id_dis.calc_dis_loss_ab(torch.flatten(s_a_recon, 1).detach(), torch.flatten(s_b_recon, 1).detach())
        self.loss_id_dis_total = hyperparameters['id_adv_w'] * self.loss_exp_dis_ab + hyperparameters['id_adv_w'] * self.loss_exp_dis_ab_recon

        print("DLoss: %.4f" % self.loss_id_dis_total)
        self.loss_id_dis_total.backward()
        # check gradient norm
        self.id_dis_opt.step()


    def update_learning_rate(self):
        if self.dis_a_scheduler is not None:
            self.dis_a_scheduler.step()
        # if self.dis_b_scheduler is not None:
        #     self.dis_b_scheduler.step()
        if self.gen_encoder_scheduler is not None:
            self.gen_encoder_scheduler.step()
        if self.gen_decoder_scheduler is not None:
            self.gen_decoder_scheduler.step()
        if self.id_scheduler is not None:
            self.id_scheduler.step()
        if self.id_dis_scheduler is not None:
            self.id_dis_scheduler.step()

    def scale_learning_rate(self, lr_decayed, lr_recover, hyperparameters):
        if not lr_decayed:
            if lr_recover:
                for g in self.dis_a_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma']
                for g in self.gen_a_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma']
                for g in self.gen_b_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma']
                for g in self.id_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma2']
                for g in self.id_dis_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma2']
            elif not lr_recover:
                for g in self.id_opt.param_groups:
                    g['lr'] = g['lr'] * hyperparameters['lr2_ramp_factor']
        elif lr_decayed:
            for g in self.dis_a_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma'] * hyperparameters['lr2_ramp_factor']
            for g in self.gen_a_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma'] * hyperparameters['lr2_ramp_factor']
            for g in self.gen_b_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma'] * hyperparameters['lr2_ramp_factor']
            for g in self.id_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma2'] * hyperparameters['lr2_ramp_factor']
            for g in self.id_dis_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma2'] * hyperparameters['lr2_ramp_factor']

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_encoder_name = os.path.join(snapshot_dir, 'gen_encode_%08d.pt' % (iterations + 1))
        gen_decoder_name = os.path.join(snapshot_dir, 'gen_decoder_%08d.pt' % (iterations + 1))
        dis_a_name = os.path.join(snapshot_dir, 'dis_a_%08d.pt' % (iterations + 1))
        dis_b_name = os.path.join(snapshot_dir, 'dis_b_%08d.pt' % (iterations + 1))
        id_name = os.path.join(snapshot_dir, 'id_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        fc_name = os.path.join(snapshot_dir, 'fc_%08d.pt' % (iterations + 1))
        torch.save({'a': self.gen_encode.state_dict()}, gen_encoder_name)
        torch.save({'a': self.gen_decode.state_dict()}, gen_decoder_name)
        torch.save({'a': self.dis_a.state_dict()}, dis_a_name)
        torch.save({'a': self.id_a.state_dict()}, id_name)
        torch.save({'a': self.fc.state_dict()}, fc_name)
        torch.save(
            {'gen_encoder': self.gen_decoder_opt.state_dict(), 'gen_decoder': self.gen_decoder_opt.state_dict(), 'id': self.id_opt.state_dict(),
             'fc':self.fc_opt.state_dict(), 'dis_a': self.dis_a_opt.state_dict(), 'dis_b': self.dis_a_opt.state_dict()},
            opt_name)

    def image_classification(self, loader):
        # model = copy.deepcopy(self.gen_a.enc_content)
        start_test = True
        with torch.no_grad():
            bingo_cnt = 0
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                Onset_target, Apex_target, land_target, labels = next(iter_test)
                Onset_target = Onset_target.cuda()
                Apex_target = Apex_target.cuda()
                land_target = land_target.cuda()
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
