from options import Options
from lib.dataloader import load_data
from lib.models import y_gan


##
def main():
    """ Testing
    """
    opt = Options().parse(mode="test")
    _, valid_dl, num_normal_classes = load_data(opt)
    model = y_gan.Y_GAN(opt, None, valid_dl, num_normal_classes)
    model.test()

if __name__ == '__main__':
    main()
