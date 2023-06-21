from torch import mean, pow, abs
from torch.nn import CrossEntropyLoss

cross_entropy_loss = CrossEntropyLoss()

def l1_loss(input, target):

    return mean(abs(input - target)) # original

def l2_loss(input, target, size_average=True):

    if size_average:
        return mean(pow((input-target), 2))
    else:
        return pow((input-target), 2)

def latent_classifier_loss(pred_s, pred_res, gt_classes):

    shape_loss = cross_entropy_loss(pred_s, gt_classes.clone())
    residual_loss = cross_entropy_loss(pred_res, gt_classes.clone())

    return shape_loss, residual_loss
