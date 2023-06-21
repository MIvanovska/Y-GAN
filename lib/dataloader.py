from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from os import listdir, walk
from os.path import join, isdir

##
def load_data(opt):

    datapath='{}/{}'.format(opt.dataroot, opt.dataset)
    print("Loading data from ", datapath)

    transform = [Resize(opt.isize), CenterCrop(opt.isize), ToTensor()]

    if opt.dataset == 'MNIST':
        transform.append(Normalize((0.1307,), (0.3081,)))
    else:
        transform.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    normal_classes = listdir(join(datapath, 'train', '0.normal'))
    normal_classes.sort()
    num_normal_classes=len(normal_classes)
    abnormal_classes = listdir(join(datapath, 'test', '1.abnormal'))
    abnormal_classes.sort()
    opt.abnormal_class = abnormal_classes
    class_to_id={}
    class_to_bin={}
    for i, normal_class in enumerate(normal_classes):
        class_to_id[normal_class]= i
        class_to_bin[normal_class]= 0
    for j, abnormal_class in enumerate(abnormal_classes):
        class_to_id[abnormal_class]= j+i+1
        class_to_bin[abnormal_class]= 1

    transform = Compose(transform)
    if opt.mode=="train":
        train_ds = ImageFolder(join(datapath, 'train', '0.normal'), transform=transform, class_to_id=class_to_id, class_to_bin=class_to_bin)
        train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True, num_workers=8)
    else:
        train_dl = None
    valid_ds1 = ImageFolder(join(datapath, 'test', '0.normal'), transform=transform, class_to_id=class_to_id, class_to_bin=class_to_bin)
    valid_ds2 = ImageFolder(join(datapath, 'test', '1.abnormal'), transform=transform, class_to_id=class_to_id, class_to_bin=class_to_bin)
    valid_ds = ConcatDataset([valid_ds1, valid_ds2])
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False, num_workers=8)


    return train_dl, valid_dl, num_normal_classes


def has_file_allowed_extension(filename, extensions):

    return filename.lower().endswith(extensions)


def is_image_file(filename):

    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_id, class_to_bin, extensions=None, is_valid_file=None):
    images = []
    subdirs = listdir(dir)

    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for subdir in sorted(subdirs):
        d = join(dir, subdir)
        if not isdir(d):
            continue
        for root, _, fnames in sorted(walk(d)):
            for fname in sorted(fnames):
                path = join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_id[subdir], class_to_bin[subdir])
                    images.append(item)

    return images

class DatasetFolder(VisionDataset):

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, class_to_id = None, class_to_bin = None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        samples = make_dataset(self.root, class_to_id, class_to_bin, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    def __getitem__(self, index):
        path, class_id, binary_id = self.samples[index] # class_id is the id of the class, binary_id is 0 for normal and 1 for anomalies
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, class_id, binary_id, path

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, class_to_id = None, class_to_bin = None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, class_to_id = class_to_id, class_to_bin = class_to_bin)
        self.imgs = self.samples
