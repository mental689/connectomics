import torch


def jaccard_index(y_pred, y_true, smooth=100):
    """
    Jaccard index for Semantic Segmentation, following a Keras example:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    :param y_pred:
    :param y_true:
    :param smooth:
    :return:
    """
    intersection = torch.sum(torch.abs(y_pred*y_true), dim=-1)
    sum_ = torch.sum(torch.add(torch.abs(y_pred), torch.abs(y_true)), dim=-1)
    jac = (intersection + smooth) / (torch.sub(sum_, intersection) + smooth)
    return (1 - jac) *  smooth


