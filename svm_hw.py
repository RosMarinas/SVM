# ========================================================
#             Media and Cognition
#             Homework 3 Support Vector Machine
#             svm_hw.py - The implementation of SVM using hinge loss
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO 1: complete the forward and backward propagation processes of the linear layer
class LinearFunction(torch.autograd.Function):
    '''
    we will implement the linear function:
    y = xW^T + b
    as well as its gradient computation process
    '''

    @staticmethod
    def forward(ctx, x, W, b):
        '''
        Input:
        :param ctx: a context object that can be used to stash information for backward computation
        :param x: input features with size [batch_size, input_size]
        :param W: weight matrix with size [output_size, input_size]
        :param b: bias with size [output_size]
        Return:
        y :output features with size [batch_size, output_size]
        '''

        # TODO
        y=torch.matmul(W,x.t())+b
        ctx.save_for_backward(x, W)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Input:
        :param ctx: a context object with saved variables
        :param grad_output: dL/dy, with size [batch_size, output_size]
        Return:
        grad_input: dL/dx, with size [batch_size, input_size]
        grad_W: dL/dW, with size [output_size, input_size], summed for data in the batch
        grad_b: dL/db, with size [output_size], summed for data in the batch
        '''

        x, W = ctx.saved_variables

        # calculate dL/dx by using dL/dy (grad_output) and W, e.g., dL/dx = dL/dy*W
        # calculate dL/dW by using dL/dy (grad_output) and x
        # calculate dL/db using dL/dy (grad_output)
        # you can use torch.matmul(A, B) to compute matrix product of A and B

        # TODO
        grad_input = torch.matmul(grad_output.t(), W)
        grad_W = torch.matmul(grad_output, x)
        grad_b = grad_output.sum(0)
        #print("1,",grad_input.shape, grad_W.shape, grad_b.shape)
      

        return grad_input, grad_W, grad_b


# TODO 2: complete the forward and backward propagation processes of the hinge loss
class Hinge(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, W, label, C):
        """
        Compute the hinge loss
        --------------------------------------
        :param ctx: a context object that can be used to stash information for backward computation
        :param output: the output of the linear layer with size [batch_size, 1], i.e. output = W^T*x + b
        :param W: weight matrix with size [1, input_size]
        :param label: the ground truth y in the equation for loss calculation, with size [batch_size]
        :param C: the regularization coefficient of hinge loss with size [1, 1]
        :return: the hinge loss with size [1, 1]
        """
        C = C.type_as(W)

        # TODO: compute the hinge loss (together with L2 norm for SVM): loss = 0.5*||w||^2 + C*\sum_i{max(0, 1 - y_i*output_i)}
        # you may need F.relu() to implement the max() function.
        # the hinge loss is the sum of the loss of each sample in the batch
        hinge_loss = 0.5*torch.norm(W)**2 + C*torch.sum(F.relu(1 - label*output))

        ctx.save_for_backward(output, W, label, C)

        return hinge_loss

    @staticmethod
    def backward(ctx, grad_loss):
        """
        Compute the gradient of hinge loss
        :param ctx: a context object with saved variables
        :param grad_loss: dL/dloss, with size [1, 1], the gradient of the final target loss with respect to the output (variable 'loss') of the forward function
        :return:
            grad_output: dL/doutput, with size [batch_size, 1]
            grad_W: dL/dW, with size [1, channels]
        """
        output, W, label, C = ctx.saved_tensors
      

        # TODO: compute the grad with respect to the output of the linear function and W: dL/doutput, dL/dW
        #print(label.shape,W.shape,output.shape)
        grad_output = C*(-label)*(output*label < 1).type_as(output)*grad_loss
        # print('grad_output:',grad_output.shape)
        grad_W = grad_loss*W
        return grad_output, grad_W, None, None


# TODO 3: complete the structure of SVM model
class SVM_HINGE(nn.Module):

    def __init__(self, in_channels, C):
        """
        :param in_channels: number of feature channels for SVM input
        :param C: regularization coefficient of hinge loss with size [1, 1]
        """
        super().__init__()

        # TODO: define the parameters W and b
        """
            the shape of W should be [1, channels] and the shape of b should be [1, ]
            you need to use nn.Parameter() to make W and b be trainable parameters, don't forget to set requires_grad=True for self.W and self.b
            please use torch.randn() to initialize W and b
        """

        self.W=nn.Parameter(torch.randn(1,in_channels),requires_grad=True)
        self.b=nn.Parameter(torch.randn(1),requires_grad=True)
        self.C = torch.tensor([[C]], requires_grad=False)

    def forward(self, x, label=None):
        # SVM calculation
        output = LinearFunction.apply(x, self.W, self.b)
        if label is not None:
            loss = Hinge.apply(output, self.W, label, self.C)
        else:
            loss = None
        output = (output > 0.0).type_as(x) * 2.0 - 1.0
        return output, loss
