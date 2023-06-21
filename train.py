from options import Options
from lib.dataloader import load_data
from lib.models import y_gan

def main():
    """ Training
    """
    opt = Options().parse(mode="train")
    train_dl, valid_dl, num_normal_classes = load_data(opt)
    model = y_gan.Y_GAN(opt, train_dl, valid_dl, num_normal_classes)
    _=model.train()

if __name__ == '__main__':
    main()
