# ========================================================
#             Media and Cognition
#             Homework 3 Support Vector Machine
#             test_svm.py - Test svm model for traffic sign
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================

# ==== Part 1: import libs
import argparse
import torch
from datasets import Traffic_Dataset
from svm_hw import SVM_HINGE
from torch.utils.data import DataLoader


# ==== Part 2: testing
def test(
    data_root,
    model_save_path,
    device,
):
    """
    The main testing procedure of SVM model
    ----------------------------
    :param data_root: path to the root directory of dataset
    :param model_save_path: path to pretrained SVM model
    :param device: device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    """

    # TODO 1: =================== load the pretrained SVM model ==================================

    # TODO: construct testing data loader with 'Traffic_Dataset' and DataLoader, and set 'batch_size=1' and 'shuffle=False'
    test_data =   Traffic_Dataset(data_root+'/test.pt')
    test_loader =  DataLoader(test_data, batch_size=1, shuffle=False)

    # TODO: load state dictionary of pretrained SVM model
    model_svm =   torch.load(model_save_path)

    # TODO: initialize the SVM model using 'model_svm["configs"]["feature_channel"]' and 'model_svm["configs"]["C"]'
    svm =   SVM_HINGE(model_svm["configs"]["feature_channel"], model_svm["configs"]["C"])

    # TODO: load model parameters (model_svm['state_dict']) we saved in model_path using svm.load_state_dict()
    svm.load_state_dict(model_svm['state_dict'])
    # TODO: put the model on CPU or GPU
    svm=  svm.to(device)
       

    # TODO 2 : ================================ testing ==============================================

    # TODO: set the model in evaluation mode
    svm.eval()
    
    # to calculate and save the testing accuracy
    n_correct = 0.  # number of images that are correctly classified
    n_feas = 0.  # number of total images

    with torch.no_grad():  # we do not need to compute gradients during validation
        # TODO: inference on the testing dataset, similar to the training stage but use 'test_loader'.
        for  step,(input,label)   in   enumerate(test_loader) :
            # TODO: set data type (.float()) and device (.to())
            input= input.float().to(device)
            label= label.float().to(device)

            # TODO: run the model; at the validation step, the model only needs one input: feas
            # _ refers to a placeholder, which means we do not need the second returned value during validating
               
            out, loss = svm(input)
            # TODO: sum up the number of images correctly recognized. note the shapes of 'out' and 'labels' are different
            n_correct += ((out > 0.0) == (label > 0.0)).sum().item()

            # TODO:sum up the total image number
            n_feas +=  input.size(0)

    # show prediction accuracy
    acc = 100 * n_correct / n_feas
    print('Test accuracy = {:.1f}%'.format(acc))


if __name__ == "__main__":
    # set configurations of the testing process
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="file list of training image paths and labels")
    parser.add_argument("--device", type=str, help="cpu or cuda")
    parser.add_argument("--model_save_path", type=str, default="checkpoints/svm.pth", help="path to save SVM model")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # run the testing procedure
    test(
        data_root=args.data_root,
        model_save_path=args.model_save_path,
        device=args.device,
    )









