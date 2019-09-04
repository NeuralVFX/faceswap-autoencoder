import math
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from util import helpers as helper
from util import loaders as load
from models import networks as n
import torch.nn as nn
import cv2
import os
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')
plt.switch_backend('agg')


############################################################################
# Train
############################################################################


class FaceSwap:
    """
    Example usage if not using command line:

    from reverse_track import *

    params = {'dataset_a': 'faceA',
              'dataset_b': 'faceB',
              'enc_att':True,
              'dec_att':True,
              'disc_att' :False,
              'batch_size': 8,
              'workers': 16,
              'res': 128,
              'res_layers_p': [2,6,9,14,17,20,23,26],
              'res_layers_p_weight': [1, 1, 1,1,1,1,1,1],
              'lr': 2e-4,
              'beta1': .5,
              'beta2': .999,
              'edge_weight':.1,
              'recon_weight': 3.,
              'train_epoch': 301,
              'save_every': 20,
              'save_img_every': 1,
              'lr_drop_start': 0,
              'lr_drop_every': 40,
              'save_root': 'gump'}

    rev = FaceSwap(params)
    rev.train()

    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.perc_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0
        self.current_epoch_iter = 0

        # Setup data loaders
        self.transform = load.NormDenorm([.5, .5, .5], [.5, .5, .5])

        self.train_data_a = load.FaceDataset(f'./data/{params["dataset_a"]}/',
                                             f'./data/{params["dataset_b"]}/',
                                             self.transform,
                                             output_res=params["res"])
        self.train_data_b = load.FaceDataset(f'./data/{params["dataset_b"]}/',
                                             f'./data/{params["dataset_a"]}/',
                                             self.transform,
                                             output_res=params["res"])

        self.train_loader_a = torch.utils.data.DataLoader(self.train_data_a,
                                                          batch_size=params["batch_size"],
                                                          num_workers=params["workers"],
                                                          shuffle=True,
                                                          drop_last=True)

        self.train_loader_b = torch.utils.data.DataLoader(self.train_data_b,
                                                          batch_size=params["batch_size"],
                                                          num_workers=params["workers"],
                                                          shuffle=True,
                                                          drop_last=True)

        print(f'Data Loaders Initialized,  Data A Len:{self.train_data_a.__len__()} '
              f' Data B Len:{self.train_data_b.__len__()}')

        # Setup models
        self.res_tran = n.TensorTransform(res=params["res"],
                                          mean=[91.4953, 103.8827, 131.0912],
                                          std=[1, 1, 1])
        self.res_tran.cuda()

        self.model_dict['ENC'] = n.Encoder(layers=int(math.log(params["res"], 2) - 2),
                                           attention=params['enc_att'])

        self.model_dict['DEC_A'] = n.Decoder(layers=int(math.log(params["res"], 2) - 4),
                                             min_filts=128,
                                             attention=params['dec_att'])
        self.model_dict['DEC_B'] = n.Decoder(layers=int(math.log(params["res"], 2) - 4),
                                             min_filts=128,
                                             attention=params['dec_att'])

        self.model_dict['DISC_A'] = n.Discriminator(attention=params['disc_att'],
                                                    channels=3)
        self.model_dict['DISC_B'] = n.Discriminator(attention=params['disc_att'],
                                                    channels=3)

        self.res_face = n.resnet_face()
        for param in self.res_face.parameters():
            param.requires_grad = False

        self.res_face.cuda()

        for i in self.model_dict.keys():
            self.model_dict[i].apply(helper.weights_init_normal)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        self.model_dict['ENC'].apply(helper.weights_init_icnr)
        self.model_dict['DEC_A'].apply(helper.weights_init_icnr)
        self.model_dict['DEC_B'].apply(helper.weights_init_icnr)

        print('Networks Initialized')

        # Setup loss
        face_children = list(self.res_face.children())

        res_face_hooks = [n.SetHook(face_children[i]) for i in params['res_layers_p']]

        self.perceptual_loss = n.PerceptualLoss(self.res_face,
                                                params['perceptual_weight'],
                                                params['res_layers_p'],
                                                params['res_layers_p_weight'],
                                                hooks=res_face_hooks,
                                                use_instance_norm=True)

        self.perceptual_loss.cuda()

        disc_a_convs = [list(self.model_dict['DISC_A'].children())[0][1],
                        list(list(self.model_dict['DISC_A'].children())[0][2].children())[0].conv,
                        list(list(self.model_dict['DISC_A'].children())[0][3].children())[0].conv,
                        list(list(self.model_dict['DISC_A'].children())[0][4].children())[0].conv]

        disc_a_hooks = [n.SetHook(i) for i in disc_a_convs]

        self.perc_dict['DISC_A'] = n.PerceptualLoss(self.model_dict['DISC_A'],
                                                    params['disc_perceptual_weight'],
                                                    [],
                                                    [1, 1, 1, 1],
                                                    hooks=disc_a_hooks,
                                                    use_instance_norm=True)
        self.perc_dict['DISC_A'].cuda()

        disc_b_convs = [list(self.model_dict['DISC_B'].children())[0][1],
                        list(list(self.model_dict['DISC_B'].children())[0][2].children())[0].conv,
                        list(list(self.model_dict['DISC_B'].children())[0][3].children())[0].conv,
                        list(list(self.model_dict['DISC_B'].children())[0][4].children())[0].conv]

        disc_b_hooks = [n.SetHook(i) for i in disc_b_convs]

        self.perc_dict['DISC_B'] = n.PerceptualLoss(self.model_dict['DISC_B'],
                                                    params['disc_perceptual_weight'],
                                                    [],
                                                    [1, 1, 1, 1],
                                                    hooks=disc_b_hooks,
                                                    use_instance_norm=True)
        self.perc_dict['DISC_B'].cuda()
        # Setup optimizers
        self.model_dict["DEC_A"].apply(helper.weights_init_icnr)
        self.model_dict["DEC_B"].apply(helper.weights_init_icnr)

        self.opt_dict["AE_A"] = optim.Adam(itertools.chain(self.model_dict["ENC"].parameters(),
                                                           self.model_dict["DEC_A"].parameters()),
                                           lr=params['lr'],
                                           betas=(params['beta1'],
                                                  params['beta2']),
                                           weight_decay=0.0)

        self.opt_dict["AE_B"] = optim.Adam(itertools.chain(self.model_dict["ENC"].parameters(),
                                                           self.model_dict["DEC_B"].parameters()),
                                           lr=params['lr'],
                                           betas=(params['beta1'],
                                                  params['beta2']),
                                           weight_decay=0.0)

        self.opt_dict["DISC_A"] = optim.Adam(self.model_dict["DISC_A"].parameters(),
                                             lr=params['lr'],
                                             betas=(params['beta1'],
                                                    params['beta2']),
                                             weight_decay=0.0)

        self.opt_dict["DISC_B"] = optim.Adam(self.model_dict["DISC_B"].parameters(),
                                             lr=params['lr'],
                                             betas=(params['beta1'],
                                                    params['beta2']),
                                             weight_decay=0.0)
        print('Losses Initialized')

        # Setup history storage
        self.losses = ['L1_A_Loss', 'L1_B_Loss', 'P_A_Loss',
                       'P_B_Loss', 'D_A_Loss', 'D_B_Loss',
                       'DP_A_Loss', 'DP_B_Loss', 'AE_A_Loss',
                       'AE_B_Loss', 'MV_A_Loss', 'MV_B_Loss',
                       'M_A_Loss', 'M_B_Loss']

        self.loss_batch_dict = {}
        self.loss_epoch_dict = {}
        self.train_hist_dict = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)

        for i in self.model_dict.keys():
            if i in state['models'].keys():
                self.model_dict[i].load_state_dict(state['models'][i], strict=False)

        for i in self.opt_dict.keys():
            if i in state['optimizers'].keys():
                self.opt_dict[i].load_state_dict(state['optimizers'][i])

        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1
        self.train_hist_dict = state['train_hist']

        self.display_history()

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()

        model_state = {'iter': self.current_iter,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict,
                       }

        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)

        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def show_result(self, which, distorted, real):

        # check if we have a stored test image (to compare easily one sample across all models)
        if not (os.path.exists(f'output/test_real_{which}.jpg') and os.path.exists(
                f'output/test_distorted_{which}.jpg')):
            cv2.imwrite(f'output/test_distorted_{which}.jpg',
                        cv2.cvtColor(self.transform.denorm(distorted[0], cpu=True) * 255,
                                     cv2.COLOR_BGR2RGB))
            cv2.imwrite(f'output/test_real_{which}.jpg',
                        cv2.cvtColor(self.transform.denorm(real[0], cpu=True) * 255,
                                     cv2.COLOR_BGR2RGB))

        # load our test images
        test_real = load.cv2_open(f'output/test_real_{which}.jpg')
        test_real_input = np.rollaxis(self.transform.norm(test_real), 2)
        test_real_input = torch.FloatTensor(test_real_input).cuda()
        real[0] = test_real_input
        test_distorted = load.cv2_open(f'output/test_distorted_{which}.jpg')
        test_distorted_input = np.rollaxis(self.transform.norm(test_distorted), 2)
        test_distorted_input = torch.FloatTensor(test_distorted_input).cuda()
        distorted[0] = test_distorted_input

        other = 'A'
        if which == other:
            other = 'B'

        self.set_grad(f"ENC", False)
        self.set_grad(f"DEC_{which}", False)
        self.set_grad(f"DEC_{other}", False)
        self.set_grad(f"DISC_A", False)
        self.set_grad(f"DISC_B", False)

        # generate fake
        enc = self.model_dict["ENC"](distorted)
        fake_this = self.model_dict[f"DEC_{which}"](enc)
        fake_other = self.model_dict[f"DEC_{other}"](enc)

        fake_this_raw = fake_this[:, 1:, :, :]
        fake_this_alpha = fake_this[:, :1, :, :]
        fake_this_comp = (fake_this_alpha * fake_this_raw) + ((1 - fake_this_alpha) * distorted)

        fake_other_raw = fake_other[:, 1:, :, :]
        fake_other_alpha = fake_other[:, :1, :, :]
        fake_other_comp = (fake_other_alpha * fake_other_raw) + ((1 - fake_other_alpha) * distorted)

        image_grid_len = real.shape[0]

        fig, ax = plt.subplots(image_grid_len, 8, figsize=(8 * 2.5, image_grid_len * 2.5))

        for idx in range(image_grid_len):
            ax[idx, 0].cla()
            ax[idx, 0].imshow(self.transform.denorm(distorted[idx], cpu=True))
            ax[idx, 1].cla()
            ax[idx, 1].imshow(self.transform.denorm(fake_this_raw[idx], cpu=True))
            ax[idx, 2].cla()
            ax[idx, 2].imshow(self.transform.denorm(fake_this_comp[idx], cpu=True))
            ax[idx, 3].cla()
            ax[idx, 3].imshow(self.transform.denorm((fake_this_alpha[idx] * 2.0) - 1.0, cpu=True))
            ax[idx, 4].cla()
            ax[idx, 4].imshow(self.transform.denorm(real[idx], cpu=True))
            ax[idx, 5].cla()
            ax[idx, 5].imshow(self.transform.denorm(fake_other_raw[idx], cpu=True))
            ax[idx, 6].cla()
            ax[idx, 6].imshow(self.transform.denorm(fake_other_comp[idx], cpu=True))
            ax[idx, 7].cla()
            ax[idx, 7].imshow(self.transform.denorm((fake_other_alpha[idx] * 2.0) - 1.0, cpu=True))

        title_dict = {0: f"Distorted {which}",
                      1: f"Recon {which}",
                      2: f"Comped {which}",
                      3: f"Mask {which}",
                      4: f"Original {which}",
                      5: f"Converted {other}",
                      6: f"Comped {other}",
                      7: f"Mask {other}"}
        count = 0
        for a in ax.flat:
            a.set_xticklabels('')
            a.set_yticklabels('')
            a.set_xticks([])
            a.set_yticks([])
            a.set_aspect('equal')
            if count in title_dict.keys():
                a.set_title(title_dict[count], fontsize=12)
            count += 1

        fig.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(f'output/{self.params["save_root"]}_{which}_{self.current_epoch}.jpg')
        plt.show()
        plt.close(fig)

    def train_ae(self, which, distorted, real):

        other = 'A'
        if which == other:
            other = 'B'

        self.set_grad(f"ENC", True)
        self.set_grad(f"DEC_{which}", True)
        self.set_grad(f"DEC_{other}", False)
        self.set_grad(f"DISC_A", False)
        self.set_grad(f"DISC_B", False)

        self.opt_dict[f"AE_{which}"].zero_grad()

        # generate fake
        fake = self.model_dict[f"DEC_{which}"](self.model_dict["ENC"](distorted))

        # comp result over input using mask
        fake_raw = fake[:, 1:, :, :]
        fake_alpha = fake[:, :1, :, :]
        fake_comp = (fake_alpha * fake_raw) + ((1 - fake_alpha) * distorted)

        # get perceptual loss, using mixup between comped and raw
        dist = torch.distributions.beta.Beta(.2, .2)
        lam = dist.sample().cuda()
        mixup = lam * fake_comp + (1 - lam) * fake_raw
        perc_losses_mixup = self.perceptual_loss(self.res_tran(mixup), self.res_tran(real))
        self.loss_batch_dict[f'P_{which}_Loss'] = sum(perc_losses_mixup)

        # edge loss
        edge = n.edge_loss(fake_raw, real, self.params['edge_weight'])

        # Recon loss
        l1_loss = n.recon_loss(fake_raw, real, self.params['recon_weight'])
        self.loss_batch_dict[f'L1_{which}_Loss'] = l1_loss

        # Discriminator loss
        # TODO - USE MIXUP INSTEAD OF RUNNING BOTH
        disc_perc_losses_fake, disc_result_losses_fake = self.perc_dict[f'DISC_{which}'](fake_raw, real,
                                                                                         disc_mode=True)
        disc_perc_losses_comp, disc_result_losses_comp = self.perc_dict[f'DISC_{which}'](fake_comp, real,
                                                                                         disc_mode=True)
        # Adversarial loss
        self.loss_batch_dict[f'AE_{which}_Loss'] = (-disc_result_losses_fake.mean() * .5) + (
                -disc_result_losses_comp.mean() * .5)

        # Perceptual loss from discriminator
        # TODO - USE MIXUP INSTEAD OF RUNNING BOTH
        self.loss_batch_dict[f'DP_{which}_Loss'] = (sum(disc_perc_losses_fake) * .5) +\
                                                   (sum(disc_perc_losses_comp) * .5)

        # Alpha Mask loss
        self.loss_batch_dict[f'M_{which}_Loss'] = 1e-2 * torch.mean(torch.abs(fake_alpha))

        # Alpha mask variation loss
        self.loss_batch_dict[f'MV_{which}_Loss'] = .1 * ( torch.mean(n.emboss(fake_alpha, axis=2)) + \
                                                          torch.mean(n.emboss(fake_alpha, axis=3)))

        total_loss = self.loss_batch_dict[f'L1_{which}_Loss'] +\
                     self.loss_batch_dict[f'P_{which}_Loss'] + \
                     self.loss_batch_dict[f'DP_{which}_Loss'] +\
                     self.loss_batch_dict[f'M_{which}_Loss'] + \
                     self.loss_batch_dict[f'MV_{which}_Loss'] + \
                     edge

        total_loss.backward()
        self.opt_dict[f"AE_{which}"].step()

        return fake.detach()

    def train_disc(self, which, real, fake, distorted):
        other = 'A'
        if which == other:
            other = 'B'

        self.set_grad(f"ENC", False)
        self.set_grad(f"DEC_A", False)
        self.set_grad(f"DEC_B", False)
        self.set_grad(f"DISC_{other}", False)
        self.set_grad(f"DISC_{which}", True)
        self.opt_dict[f"DISC_{which}"].zero_grad()

        fake_raw = fake[:, 1:, :, :]
        fake_alpha = fake[:, :1, :, :]
        comp = (fake_alpha * fake_raw) + ((1 - fake_alpha) * distorted)
        # discriminate fake samples
        # TODO - USE MIXUP INSTEAD OF RUNNING BOTH
        d_result_fake_comp = self.model_dict[f"DISC_{which}"](comp)
        d_result_fake_rgb = self.model_dict[f"DISC_{which}"](fake_raw)
        # discriminate real samples
        d_result_real = self.model_dict[f"DISC_{which}"](real)

        # add up disc a loss and step
        # TODO - USE MIXUP INSTEAD OF RUNNING BOTH
        comp_loss = nn.ReLU()(1.0 - d_result_real).mean() + nn.ReLU()(1.0 + d_result_fake_comp).mean()
        rgb_loss = nn.ReLU()(1.0 - d_result_real).mean() + nn.ReLU()(1.0 + d_result_fake_rgb).mean()
        self.loss_batch_dict[f'D_{which}_Loss'] = (comp_loss * .5) + (rgb_loss * .5)
        self.loss_batch_dict[f'D_{which}_Loss'].backward()
        self.opt_dict[f"DISC_{which}"].step()

    def set_grad(self, model, grad):
        for param in self.model_dict[model].parameters():
            param.requires_grad = grad

    def lr_lookup(self):
        # Determine proper learning rate multiplier for this iter, cuts in half every "lr_drop_every"
        div = max(0, ((self.current_epoch - self.params["lr_drop_start"]) // self.params["lr_drop_every"]))
        lr_mult = 1 / math.pow(2, div)
        return lr_mult

    def get_train_loop(self):
        if self.train_data_a.__len__() == self.train_data_b.__len__():
            return zip(self.train_loader_a, self.train_loader_b)
        if self.train_data_a.__len__() > self.train_data_b.__len__():
            return zip(self.train_loader_a, cycle(self.train_loader_b))
        if self.train_data_a.__len__() < self.train_data_b.__len__():
            return zip(cycle(self.train_loader_a), self.train_loader_b)

    def train_loop(self):
        # Train on train set
        self.current_epoch_iter = 0
        for key in self.model_dict.keys():
            self.model_dict[key].train()
            self.set_grad(key, True)

        for loss in self.losses:
            self.loss_epoch_dict[loss] = []

        lr_mult = self.lr_lookup()
        self.opt_dict["AE_A"].param_groups[0]['lr'] = self.params['lr'] * (lr_mult / 2)
        self.opt_dict["AE_B"].param_groups[0]['lr'] = self.params['lr'] * (lr_mult / 2)
        self.opt_dict["DISC_A"].param_groups[0]['lr'] = self.params['lr'] * lr_mult
        self.opt_dict["DISC_B"].param_groups[0]['lr'] = self.params['lr'] * lr_mult
        # print LR and weight decay
        print(f"Sched Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
        [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}",
               f" Weight Decay:{ self.opt_dict[opt].param_groups[0]['weight_decay']}")
         for opt in self.opt_dict.keys()]

        # Train loop

        for package_a, package_b in tqdm(self.get_train_loop()):
            warped_a = Variable(package_a[0]).cuda()
            target_a = Variable(package_a[1]).cuda()
            warped_b = Variable(package_b[0]).cuda()
            target_b = Variable(package_b[1]).cuda()
            # TRAIN GENERATOR, OR JUST GENERATE

            fake_a = self.train_ae('A', warped_a, target_a)
            fake_b = self.train_ae('B', warped_b, target_b)

            self.train_disc('A', target_a, fake_a, warped_a)
            self.train_disc('B', target_b, fake_b, warped_b)

            if self.current_epoch_iter == 0 and self.current_epoch % self.params["save_img_every"] == 0:
                self.show_result('A', warped_a, target_a)
                self.show_result('B', warped_b, target_b)

            # append all losses in loss dict
            [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].item()) for loss in self.losses]
            self.current_iter += 1
            self.current_epoch_iter += 1
        [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]

    def train(self):
        # Train following learning rate schedule
        params = self.params

        while self.current_epoch < params["train_epoch"]:
            epoch_start_time = time.time()

            # TRAIN LOOP
            self.train_loop()

            # save
            if self.current_epoch % params["save_every"] == 0:
                save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                tqdm.write(save_str)

                self.display_history()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print(f'Epoch Training Training Time: {per_epoch_ptime}')
            [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]
            # [print(f'Val {loss}: {helper.mft(self.loss_epoch_dict_test[loss])}') for loss in self.losses]
            print('\n')
            self.current_epoch += 1

        self.display_history()
        print('Hit End of Learning Schedule!')
