import argparse
from os import makedirs
from os.path import join, isdir
from torch.cuda import set_device

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--dataroot', default='./data', help='path to dataset')
        self.parser.add_argument('--dataset', default='CIFAR10', help='name of the dataset (eg. MNIST, FMNIST, CIFAR10)')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--isize', type=int, default=32, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent vectors of each encoder')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment', help='name of the experiment')
        self.parser.add_argument('--outf', default='./output', help='output root path for logging and model checkpoints')
        self.parser.add_argument('--load_checkpoint', default='', help="path to the checkpoint (to resume training or test the model)")
        self.parser.add_argument('--niter', type=int, default=15, help='number of training epochs')
        self.parser.add_argument('--nperm', type=int, default=64, help='number of permutations in the latent space')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='weight for adversarial loss (default=1)')
        self.parser.add_argument('--w_rec', type=float, default=50, help='weight for reconstruction loss (default=50)')
        self.parser.add_argument('--w_sem', type=float, default=1, help='weight for the semantic loss in the latent space (default=1)')
        self.parser.add_argument('--w_res', type=float, default=1, help='weight for the residual loss in the latent space (default=1)')
        self.parser.add_argument('--w_perm', type=float, default=50, help='weight for consistency loss (default=50)')
        self.parser.add_argument('--alpha', type=float, default=0, help='lambda constant for the gradient reversal layer (if alpha=0 then alpha is calculated as in their paper)')

    def parse(self, mode="train"):

        self.opt = self.parser.parse_args()
        self.opt.mode=mode
        self.opt.print_freq = self.opt.batchsize
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        # save to the disk
        expr_dir = join(self.opt.outf, self.opt.name, self.opt.mode)
        test_dir = join(self.opt.outf, self.opt.name, 'test')

        if not isdir(expr_dir):
            makedirs(expr_dir)
        if not isdir(test_dir):
            makedirs(test_dir)

        file_name = join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt