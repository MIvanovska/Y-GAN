""" Network architectures.
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import functools
from torch.nn import init
from torchvision import models
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('InstanceNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=False))
        csize, cndf = isize / 2, ndf


        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat, affine=True, track_running_stats=False))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=False))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

class EncoderShape(nn.Module):

    def __init__(self, isize, nz, nc, ndf, ngpu, add_final_conv=True):
        super(EncoderShape, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=False)) # original

        csize, cndf = isize / 2, ndf


        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),nn.BatchNorm2d(out_feat, affine=True, track_running_stats=False))

            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=False))

            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class EncoderResidual(nn.Module):

    def __init__(self, isize, nz, nc, ndf, ngpu, add_final_conv=True):
        super(EncoderResidual, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=False))

        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),nn.BatchNorm2d(out_feat, affine=True, track_running_stats=False))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=False))

            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input, alpha=1.0):

        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output_reversal = ReverseLayerF.apply(output, alpha)
        else:
            output = self.main(input)
            output_reversal = ReverseLayerF.apply(output, alpha)
        return output, output_reversal

class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))

            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class LatentDiscriminator(nn.Module):
    """
    Discriminator in the latent space
    """
    def __init__(self, opt, nclass=9):
        super(LatentDiscriminator, self).__init__()
        self.ngpu = opt.ngpu
        latDis = nn.Sequential()
        nclass = nclass
        D_in=opt.nz
        H=30


        latDis.add_module('input_layer', torch.nn.Linear(D_in, H))
        latDis.add_module('initial-{0}-relu', torch.nn.LeakyReLU(inplace=True))
        latDis.add_module('hidden_layer', torch.nn.Linear(H, nclass))
        self.latDis = latDis

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            latDisoutput = nn.parallel.data_parallel(self.latDis, input, range(self.ngpu))
        else:
            latDisoutput = self.latDis(input)

        return latDisoutput

##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(isize=opt.isize, nz=1, nc=opt.nc, ndf=64, ngpu=opt.ngpu)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

class NetLatD(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt, nclass=9):
        super(NetLatD, self).__init__()
        self.latentDis = LatentDiscriminator(opt, nclass=nclass)

    def forward(self, latent_zs, latent_zres):
        latent_zs = latent_zs.contiguous().view(-1, latent_zs.size(1), )
        latent_zres = latent_zres.contiguous().view(-1, latent_zres.size(1), )

        pred_s = self.latentDis(latent_zs)
        pred_res = self.latentDis(latent_zres)
        predicted_class= torch.max(torch.nn.functional.softmax(pred_s.detach(), dim=1).data, dim=1)[1]

        return pred_s, pred_res, predicted_class

class NetResG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetResG, self).__init__()
        self.encoderRes = EncoderResidual(isize=opt.isize, nz=opt.nz, nc=opt.nc, ndf=64, ngpu=opt.ngpu)

    def forward(self, x, alpha=1.0):
        latent_zres, latent_zres_with_reversal = self.encoderRes(x, alpha=alpha)
        return latent_zres, latent_zres_with_reversal

class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoderShape = EncoderShape(isize=opt.isize, nz=opt.nz, nc=opt.nc, ndf=64, ngpu=opt.ngpu)
        self.decoder = Decoder(isize=opt.isize, nz=2*opt.nz, nc=opt.nc, ngf=64, ngpu=opt.ngpu)


    def forward(self, x, latent_zres, permute=True, permute_zs = False, permute_zres = False):
        gen_imag = None
        latent_i = None
        latent_zs = None
        perm_gen_imag = None
        gen_img_permuted_zres = None
        gen_img_permuted_zs = None
        device = torch.device(x.get_device())
        if permute_zs:
            latent_i_permuted_zs = torch.cat((torch.index_select(x, 0, torch.randperm(x.size(0)).to(device=device)), latent_zres), dim=1)  # ith permutation of zs vectors
            B = self.decoder(latent_i_permuted_zs)
            BB = B.clone().view(B.size(0), -1)
            BB -= BB.min(1, keepdim=True)[0]
            BB /= BB.max(1, keepdim=True)[0]
            BB = BB * 2 - 1
            BB = BB.view(-1, 3, 32, 32)
            gen_img_permuted_zs = BB
        elif permute_zres:
            latent_i_permuted_zres = torch.cat((x, torch.index_select(latent_zres, 0, torch.randperm(latent_zres.size(0)).to(device=device))), dim=1)  # ith permutation of zres vectors
            A = self.decoder(latent_i_permuted_zres)
            AA = A.clone().view(A.size(0), -1)
            AA -= AA.min(1, keepdim=True)[0]
            AA /= AA.max(1, keepdim=True)[0]
            AA = AA * 2 - 1
            AA = AA.view(-1, 3, 32, 32)
            gen_img_permuted_zres = AA
            latent_zs_of_permuted_samples = self.encoderShape(AA)
            latent_zs = latent_zs_of_permuted_samples
        else:
            latent_zs = self.encoderShape(x)
            latent_i = torch.cat((latent_zs, latent_zres), dim=1)
            gen_imag = self.decoder(latent_i)
            if permute:
                # random permutation of zres vectors
                perm_gen_imag = self.decoder(torch.cat((latent_zs, torch.index_select(latent_zres, 0, torch.randperm(latent_zres.size(0)).to(device=device))), dim=1))

        return gen_imag, latent_i, latent_zs, perm_gen_imag, gen_img_permuted_zres, gen_img_permuted_zs
