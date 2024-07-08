# ========================================================
#             Media and Cognition
#             Homework 3 Support Vector Machine
#             data_preprocess.py - Using pretrained convolutional layers to extract feature,
#                                   and using PCA for dimensionality reduction
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================

import os
import torchvision.transforms as transforms
import torch
from PIL import Image
from networks import Classifier
import matplotlib.pyplot as plt
import argparse


def preprocess(pre_conv, data_root, image_size, classes):
    # TODO 1: Using PCA to reduce the dimensionality of 2048 point features extracted by convolution

    # ===============  process training dataset ======================
    print("Start preprocessing the training dataset !!!")
    train_data, train_label = loaddata(pre_conv, data_root, 'train', image_size, classes)

    # calculate the mean and PCA projection matrix
    data_mean, u = PCA(train_data, 2)

    # TODO: using PCA to compress the dimensionality of the train_data after subtracting the mean vector
    train_data_pca =torch.matmul(u.t(), (train_data - data_mean).t()).t()

    visualize(train_data_pca, train_label, "train")
    savedata(train_data_pca, train_label, data_root+"/train.pt")
    print("training dataset saved !!!")

    # ===============  process validation dataset ======================
    print("Start preprocessing the validation dataset!!!")
    val_data, val_label = loaddata(pre_conv, data_root, 'val', image_size, classes)

    # TODO: using PCA to compress the dimensionality of the val_data after subtracting the mean vector
    val_data_pca = torch.matmul(u.t(), (val_data - data_mean).t()).t()

    visualize(val_data_pca, val_label, "val")
    savedata(val_data_pca, val_label, data_root+"/val.pt")
    print("validation dataset saved !!!")

    # ===============  process testing dataset ======================
    print("Start preprocessing the testing dataset!!!")
    test_data, test_label = loaddata(pre_conv, data_root, 'test', image_size, classes)

    # TODO: using PCA to compress the dimensionality of the test_data after subtracting the mean vector
    test_data_pca = torch.matmul(u.t(), (test_data - data_mean).t()).t()

    visualize(test_data_pca, test_label, "test")
    savedata(test_data_pca, test_label, data_root+"/test.pt")
    print("testing dataset saved !!!")


def savedata(data, label, save_path):
    save_dict = {
        'data': data,
        'label': label
    }
    torch.save(save_dict, save_path)


def visualize(datas, labels, mode):
    """
    Display feature points after dimensionality reduction
    -------------------------------
    :param datas: the samples after dimensionality reduction, with the shape of [N, 2]
    :param labels: the labels (chosen from {-1, +1}) corresponding to the samples
    :param mode: chosen from {'train', 'val', 'test'}
    :return:
    """
    plt.figure()
    for idx in range(datas.shape[1]):
        plt.scatter(datas[labels == 2*idx-1, 0], datas[labels == 2*idx-1, 1], label=(2*idx-1))
    plt.legend()
    plt.title(mode)
    plt.show()


def PCA(data, dim=2):
    """
    calculate the mean value of the data and the projection matrix for PCA
    :param data: the sample features extracted by the pretrained network in homework2, with the shape of [N, 2048]
    :param dim: the data dimension after projection
    :return:
        data_mean: the mean value of the data
        u: the projection matrix for PCA, with the shape of [2048, dim]
    """
    # TODO 2: complete the algorithm of PCA, calculate the mean value of the data and the projection matrix

    # TODO: compute the mean of train_data
    data_mean=torch.mean(data,0)
    # TODO: compute the covariance matrix of train_data
    data_cov = torch.zeros(data.shape[1], data.shape[1])
    for i in range(len(data)):
        data_diff=data[i]-data_mean
        data_cov += torch.outer(data_diff, data_diff)
    data_cov /= len(data)

    # TODO: compute the SVD decomposition of data_cov using torch.linalg.svd
    # reference: https://pytorch.org/docs/1.11/generated/torch.linalg.svd.html
    u, s, v = torch.linalg.svd(data_cov)
    #print(u[:,:dim].shape)
    # TODO: return the proper 'data_mean' and 'u[]'
    return  data_mean, u[:, :dim]


def loaddata(pre_conv, data_root, mode, image_size, classes):
    """
    load one dataset, and use pretrained network in homework 2 to extract feature
    :param pre_conv: pretrained network in homework 2
    :param data_root: the path of the dataset
    :param mode: chosen from {'train', 'val', 'test'}
    :param image_size: the preset size that each image try to zoom to
    :param classes: two classes that need to be classified
    :return:
        datas: the samples of extracted features with the shape of [N, 2048]
        labels: the corresponding labels for each sample (chosen from {-1, +1}), with the shape of [N]
    """
    assert len(classes) == 2
    datas = []
    labels = []
    for idx in range(len(classes)):
        for img in os.listdir(data_root + '/' + mode + '/' + classes[idx]):
            data = readimg(pre_conv, data_root + '/' + mode + '/' + classes[idx] + '/' + img, image_size)
            label = 2 * idx - 1
            datas.append(data)
            labels.append(label)
    return torch.stack(datas), torch.tensor(labels)


def readimg(pre_conv, filepath, image_size):
    """
    Read one image and use pretrained network to extract the feature
    --------------------------
    :param pre_conv: pretrained network in homework 2
    :param filepath: the file path of one image
    :param image_size: the preset size that each image try to zoom to
    :return:
        data: the extracted feature with the length of 2048
    """
    img_pil = Image.open(filepath).convert('RGB')
    img_pil = img_pil.resize(image_size)
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5),
                                        ])
    img_tensor = img_transform(img_pil)
    data = pre_conv(img_tensor.unsqueeze(0)).reshape(-1)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_net", type=str, default="checkpoints/bn/ckpt_epoch_15.pth",
                        help="the filepath of the pretrained network in homework 2")
    parser.add_argument("--data_root", type=str, default="data", help="the path of all datasets")
    parser.add_argument("--image_size", type=tuple, default=(32, 32),
                        help="the preset size that each image try to zoom to")
    parser.add_argument("--classes", default=["B", "C"], help="two classes that need to be classified")

    args = parser.parse_args()

    pretrained_checkpoint = torch.load(args.pretrained_net, map_location="cpu")
    configs = pretrained_checkpoint["configs"]
    cls = Classifier(
        configs["in_channels"],
        configs["num_classes"],
        configs["use_batch_norm"],
        configs["use_stn"],
        configs["dropout_prob"],
    )
    cls.load_state_dict(pretrained_checkpoint["model_state"],strict=False)
    for param in cls.parameters():
        param.requires_grad = False
    conv = cls.conv_net

    preprocess(conv, args.data_root, args.image_size, args.classes)
