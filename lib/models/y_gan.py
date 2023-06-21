""" Y-GAN
"""
import torch
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
from lib.logger import Logger
from lib.models.networks import NetG, NetD, weights_init, NetLatD, NetResG
from lib.loss import l2_loss, latent_classifier_loss, l1_loss
from lib.evaluate import evaluate

class Y_GAN():
    def __init__(self, opt, train_dl, valid_dl, num_normal_classes):
        # Initalize variables.
        self.opt = opt
        self.data_train = train_dl
        self.data_valid = valid_dl
        self.num_normal_classes = num_normal_classes
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.logger=Logger(opt)

        self.opt.iter = 0
        self.times = []
        self.total_steps = 0

        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device) # encoder for semantic information and decoder
        self.netresg = NetResG(self.opt).to(self.device) # encoder for residual information
        self.netlatd = NetLatD(self.opt, nclass=self.num_normal_classes).to(self.device) # latent classifier
        self.netd = NetD(self.opt).to(self.device) # adversarial discriminator

        self.l_adv = l2_loss
        self.l_rec = l1_loss
        self.l_bce = torch.nn.BCELoss()
        self.l_lat_ce = latent_classifier_loss
        self.cos_perm = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=self.device)  # binary labels of reals, where 0 stands for non-anomalous data and 1 stands for anomalous data
        self.gt_classes = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)  # digit labels of reals, where 0-8 stands for non-anomalous data and 9 stands for anomalous data
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)  # labels of original samples (all labels equal 1)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)  # labels of reconstructed samples (all labels equal 0)

    def reinit_d(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        # self.netlatd.apply(weights_init())
        print('Reloading d net')

    def save_weights(self):
        """Save netG and netD weights for the current epoch.
        """
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': self.epoch+1, 'state_dict_g': self.netg.state_dict(), 'state_dict_resg': self.netresg.state_dict(), 'state_dict_latd': self.netlatd.state_dict(), 'state_dict_d': self.netd.state_dict()}, f'{weight_dir}/y_gan.pth')

    def load_weights(self, path=None):
        """ Load pre-trained weights
        Keyword Arguments:
            path {str}      -- Path to weight file (default: {None})
        """
        if path is None or path=='':
            path = f"./output/{self.opt.name}/train/weights/y_gan.pth"

        # Load the weights.
        print('>> Loading weights...')

        try:
            weights = torch.load(path)
            self.opt.iter = weights['epoch']-1
            self.netg.load_state_dict(weights['state_dict_g'])
            self.netresg.load_state_dict(weights['state_dict_resg'])
            self.netlatd.load_state_dict(weights['state_dict_latd'])
            self.netd.load_state_dict(weights['state_dict_d'])
        except IOError:
            raise IOError("Weights not found")
        print("   Done.")

    def forward_resg(self, alpha):
        """ Forward propagate through netEr (encoder for residual information)
        """
        self.input_resg = self.input.clone()
        self.latent_zres, self.latent_zres_with_reversal = self.netresg(self.input_resg, alpha=alpha)

    def forward_g(self, alpha):
        """ Forward propagate through netEs (encoder for semantic information), D (decoder) and again through netEr (encoder for residual information) form permuted samples
        """
        self.latent_zres_g = self.latent_zres.clone()
        self.input_g = self.input.clone()
        self.fake, self.latent_i, self.latent_zs, self.perm_fake, _, _ = self.netg(self.input_g, self.latent_zres_g, permute=True, permute_zs=False, permute_zres=False)

        error = torch.zeros(size=(self.opt.nperm,), dtype=torch.float32, device=self.device)
        error_zres = torch.zeros(size=(self.opt.nperm,), dtype=torch.float32, device=self.device)

        # permute zres vectors of the samples
        for i in range(self.opt.nperm):
            _, _, latent_zs_of_permuted_samples, _, _, _ = self.netg(self.latent_zs.detach(), self.latent_zres_g.detach(), permute=False, permute_zs=False, permute_zres=True)
            _, _, _, _, _, self.gen_img_permuted_zs = self.netg(self.latent_zs.detach(), self.latent_zres_g.detach(), permute=False, permute_zs=True, permute_zres=False)
            latent_zres_of_permuted_samples, _ = self.netresg(self.gen_img_permuted_zs, alpha=alpha)
            error[i] = 1 - torch.abs(self.cos_perm(latent_zs_of_permuted_samples, self.latent_zs)).mean()
            error_zres[i] = 1 - torch.abs(self.cos_perm(latent_zres_of_permuted_samples, self.latent_zres_g)).mean()

        self.error_perm = error.mean()
        self.error_perm_zres = error_zres.mean()

    def forward_d(self):
        """ Forward propagate through netDs (adversarial discriminator)
        """
        self.input_d = self.input.clone()
        self.pred_real, self.feat_real = self.netd(self.input_d)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    def forward_latd(self):
        """ Forward propagate through netC (latent classifier)
        """
        self.pred_s, self.pred_res, _ = self.netlatd(self.latent_zs, self.latent_zres_with_reversal)

    def backward_g(self):
        """ Backpropagate through netEs (encoder for semantic information), netC (latent classifier), D (decoder)
        """

        err_adv = self.l_adv(self.feat_fake, self.feat_real)
        err_rec = self.l_rec(self.fake, self.input)
        self.err_semantic, self.err_residual = self.l_lat_ce(self.pred_s, self.pred_res, self.gt_classes)
        self.err_g_adv = self.opt.w_adv * err_adv
        self.err_g_rec = self.opt.w_rec * err_rec
        self.err_g_sem = self.err_semantic * self.opt.w_sem
        self.err_g_perm = self.opt.w_perm * self.error_perm
        self.err_g_perm_zres = self.error_perm_zres * self.opt.w_perm
        self.err_g = self.err_g_adv + self.err_g_sem + self.err_g_rec + self.err_g_perm
        self.err_g.backward(retain_graph=True)

    def backward_resg(self):
        """ Backpropagate through netEr (encoder for residual information)
        """
        self.err_g_residual = self.opt.w_res * self.err_residual
        self.err_resg = self.err_g_residual + self.err_g_perm_zres
        self.err_resg.backward(retain_graph=True)

    def backward_d(self):
        """ Backpropagate through netDs (adversarial discriminator)
        """
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5

        self.err_d.backward()

    def set_learning_rate(self):
        if self.epoch == 0:
            for param_group in self.optimizer_d.param_groups:
                param_group['lr'] = 0.0002
            for param_group in self.optimizer_latd.param_groups:
                param_group['lr'] = 0.0002
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = 0.0002
            for param_group in self.optimizer_resg.param_groups:
                param_group['lr'] = 0.0002
        if self.epoch == 8:
            for param_group in self.optimizer_d.param_groups:
                param_group['lr'] = 0.00015
            for param_group in self.optimizer_latd.param_groups:
                param_group['lr'] = 0.00015
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = 0.00015
            for param_group in self.optimizer_resg.param_groups:
                param_group['lr'] = 0.00015
        if self.epoch == 25:
            for param_group in self.optimizer_d.param_groups:
                param_group['lr'] = 0.0001
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = 0.0001
            for param_group in self.optimizer_resg.param_groups:
                param_group['lr'] = 0.0001

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward pass
        self.forward_resg(alpha=self.alpha)
        self.forward_g(alpha=self.alpha)
        self.forward_latd()
        self.forward_d()

        # Reset gradients
        self.optimizer_resg.zero_grad()
        self.optimizer_g.zero_grad()
        self.optimizer_latd.zero_grad()
        self.optimizer_d.zero_grad()

        # Backward pass and parameter optimization
        self.backward_g()
        self.backward_resg()
        self.optimizer_g.step()
        self.optimizer_latd.step()
        self.optimizer_resg.step()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5:
            self.reinit_d()

    def train(self):
        """ Train the model
        """
        # Load pretrained model to resume training
        if self.opt.load_checkpoint != '':
            print("\nLoading pre-trained networks.")
            self.load_weights(path=self.opt.load_checkpoint)
        else:
            self.netg.apply(weights_init)
            self.netresg.apply(weights_init)
            self.netlatd.apply(weights_init)
            self.netd.apply(weights_init)

        self.optimizer_d = torch.optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_latd = torch.optim.Adam(self.netlatd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = torch.optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_resg = torch.optim.Adam(self.netresg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        self.total_steps = 0
        best_auc = 0
        # Train for niter epochs.
        print(f">> Training on {self.opt.dataset} to detect {self.opt.abnormal_class}")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            print(">> Training model. Epoch %d/%d" % (self.epoch + 1, self.opt.niter))

            self.netresg.train()
            self.netg.train()
            self.netlatd.train()
            self.netd.train()

            epoch_iter = 0
            train_dl_len=len(self.data_train)
            # allocate memory for all latent vector for clustering
            for i, data in enumerate(tqdm(self.data_train, leave=False, total=train_dl_len)):
                if i == 0:
                    self.set_learning_rate()

                p = float(i + self.epoch * train_dl_len) / self.opt.niter / train_dl_len
                # calculate alpha
                if self.opt.alpha != 0:
                    self.alpha = self.opt.alpha + (2. / (1. + np.exp(-10 * p)) - 1) * (1 - self.opt.alpha)
                else:
                    self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize

                self.input.resize_(data[0].size()).copy_(data[0])
                self.gt_classes.resize_(data[1].size()).copy_(data[1])
                self.gt.resize_(data[2].size()).copy_(data[2])

                self.optimize_params()

            performance = self.test()
            if performance['AUC'] > best_auc:
                best_auc = performance['AUC']
                self.save_weights()
            self.logger.print_current_performance(performance, best_auc)

        print(">> Training model.[Done]")

    def test(self):
        """ Test the model
        """
        with torch.no_grad():
            if self.opt.mode=="test":
                # Load the weights
                self.load_weights(path=self.opt.load_checkpoint)

            self.netg.eval()
            self.netresg.eval()
            self.netlatd.eval()

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data_valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data_valid.dataset),), dtype=torch.long,    device=self.device)
            self.gt_class_labels = torch.zeros(size=(len(self.data_valid.dataset),), dtype=torch.long,    device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0

            for i, data in enumerate(self.data_valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.input.resize_(data[0].size()).copy_(data[0])
                self.gt_classes.resize_(data[1].size()).copy_(data[1])
                self.gt.resize_(data[2].size()).copy_(data[2])

                latent_zres, _ = self.netresg(self.input)
                _, _, latent_zs, _, _, _ = self.netg(self.input, latent_zres, permute=True, permute_zs=False, permute_zres=False)

                self.pred_s, _, self.predicted_class = self.netlatd(latent_zs, latent_zres)

                predictions_s = 1-torch.max(torch.nn.functional.softmax(self.pred_s, dim=1).data, dim=1)[0]

                time_o = time.time()
                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + predictions_s.size(0)] = predictions_s
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + self.gt.size(0)] = self.gt # binary labels
                self.gt_class_labels[i * self.opt.batchsize: i * self.opt.batchsize + self.gt_classes.size(0)] = self.gt_classes # class labels

                self.times.append(time_o - time_i)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc_pred_zs = evaluate(self.gt_labels, self.an_scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc_pred_zs)])
            if self.opt.mode == "test":
                self.logger.print_test_performance(performance)

        return performance