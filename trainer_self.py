from networks import MsImageDis, IdDis
from utils import get_model_list, vgg_preprocess, load_vgg16, get_scheduler
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from random_erasing import RandomErasing
from models import *
import argparse

parser = argparse.ArgumentParser(description='FaceCycle')
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
    def gen_update(self, x_a, l_a, xp_a, ld_a, x_b, l_b, xp_b, ld_b, hyperparameters, iteration):
        # ppa, ppb is the same person

        self.gen_encoder_opt.zero_grad()
        self.gen_decoder_opt.zero_grad()
        self.id_opt.zero_grad()
        self.id_dis_opt.zero_grad()

        self.gen_encode.train()
        self.gen_decode.train()
        self.id_a.train()
        # encode, s_a,s_b表情
        p_a, s_a, fb_a = self.gen_encode(x_a-xp_a)
        p_b, s_b, fb_b = self.gen_encode(x_b-xp_b)
        # 生成neural face
        n_a = self.gen_decode(x_a, p_a)
        n_b = self.gen_decode(x_b, p_b)
        # autodecode 基于Apex帧提取身份信息，使用原表情和原图像，生成图像应该与原图apex帧相同
        i_a = self.id_a(x_a)
        i_b = self.id_a(x_b)
        global_mean_a = self.swap_norm(n_a, True, i_a)
        global_mean_b = self.swap_norm(n_b, True, i_b)

        n_a1 = Swap_Norm(global_mean_a, False, i_a)
        n_b1 = Swap_Norm(global_mean_b, False, i_b)
        # 基于中性人脸重建表情
        x_a_recon = self.gen_decode(n_a, fb_a)
        x_b_recon = self.gen_decode(n_b, fb_b)
        # 基于onset帧重建表情，x_a_recon=x_a_recon_p
        x_a_recon_p = self.gen_decode(xp_a, fb_a)
        x_b_recon_p = self.gen_decode(xp_b, fb_b)
        # 交叉重建图像 x_ba：b表情a身份   x_ab：a表情b身份
        x_ba = self.gen_decode(n_a, fb_b)
        x_ab = self.gen_decode(n_b, fb_a)

        ##################################################################
        # encode structure
        # 对重建后的图像a表情b身份重新识别表情信息
        f_b_recon, s_b_recon, fb_b_recon = self.gen_encode(x_ba-xp_a)
        f_a_recon, s_a_recon, fb_a_recon = self.gen_encode(x_ab-xp_b)
        i_a_recon = self.id_a(x_ba)
        i_b_recon = self.id_a(x_ab)
        # 对重建后的图像重新生成中性人脸
        n_a_recon = self.gen_decode(x_ba, f_b_recon)
        n_b_recon = self.gen_decode(x_ab, f_a_recon)
        ##################################################################
        # decode again (if needed)
        # 重新合成图像 x_aba：重新生成的a表情a身份图像(Apex)，x_bab：重新生成的b表情b身份图像(Apex)
        x_aba = self.gen_decode(n_a_recon, fb_a_recon)
        x_bab = self.gen_decode(n_b_recon, fb_b_recon)


        ##################################################################
        # auto-encoder image reconstruction
        # 使用原图表情和身份重建的图像应该与原图（Apex）相通，因此无论Onset帧还是Apex帧生成图像都与原图Apex帧相比
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_xp_a = self.recon_criterion(x_a_recon_p, x_a)
        self.loss_gen_recon_xp_b = self.recon_criterion(x_b_recon_p, x_b)
        self.loss_gen_recon_n_a = self.recon_criterion(n_a, xp_a) + self.recon_criterion(n_a1, xp_a)
        self.loss_gen_recon_n_b = self.recon_criterion(n_b, xp_b) + self.recon_criterion(n_b1, xp_b)

        # feature reconstruction
        #中间层生成的图像，其表情信息（s_a和s_a_recon）应该相同，其身份信息（i_a和i_a_recon）应该相同
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(i_a_recon, i_a) if hyperparameters['recon_f_w'] > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(i_b_recon, i_b) if hyperparameters['recon_f_w'] > 0 else 0


        # 重建后图像与原图像应该相同
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # # GAN loss 对表情信息进行域鉴别
        # self.loss_gen_id_adv = (self.id_dis.calc_gen_loss(torch.flatten(s_b, 1)) + self.id_dis.calc_gen_loss(torch.flatten(s_a, 1))) / 2
        self.loss_gen_id_adv = 0
        if iteration > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])

        if iteration > hyperparameters['warm_teacher_iter']:
            hyperparameters['teacher_w'] += hyperparameters['warm_scale']
            hyperparameters['teacher_w'] = min(hyperparameters['teacher_w'], hyperparameters['max_teacher_w'])
        hyperparameters['id_adv_w'] += hyperparameters['adv_warm_scale']
        hyperparameters['id_adv_w'] = min(hyperparameters['id_adv_w'], hyperparameters['id_adv_w_max'])

        # total loss
        self.loss_gen_total = hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_a + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_b + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b


        self.loss_gen_total.backward()
        self.gen_decoder_opt.step()
        self.gen_encoder_opt.step()
        self.id_opt.step()


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

    def train_image_classification(self, loader):
        # model = copy.deepcopy(self.gen_a.enc_content)
        start_test = True
        with torch.no_grad():
            bingo_cnt = 0
            iter_test = iter(loader["source"])
            for i in range(len(loader['source'])):
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
                correct_num = torch.eq(predicts.cpu(), labels.squeeze(dim=1))
                bingo_cnt += correct_num.sum().cpu()
        _, predict = torch.max(all_output, 1)
        accuracy = bingo_cnt / float(all_label.size()[0])
        mean_ent = torch.mean(Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

        hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
        hist_tar = hist_tar / hist_tar.sum()
        return accuracy, hist_tar, mean_ent



