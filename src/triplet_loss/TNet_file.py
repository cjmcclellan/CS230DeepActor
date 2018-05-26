# this will house the TNet nn module


import torch
import facenet.contributed.face as face
import tensorflow as tf
import facenet.src.facenet as facenet
import imageio
import os
import pickle as pkl

class TNet(torch.nn.Module):

    def __init__(self, input_size, layersSize):
        super(TNet, self).__init__()
        self.batchNorm = torch.nn.BatchNorm1d(input_size)
        self.lin1 = torch.nn.Linear(input_size, layersSize[0])
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(layersSize[0], layersSize[1])
        self.relu2 = torch.nn.ReLU()
        # self.lin3 = torch.nn.Linear(layersSize[1], layersSize[2])
        # self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.batchNorm(x)
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        # x = self.lin3(x)
        # x = self.relu3(x)
        return x

    def init_weights(self):
        torch.nn.init.xavier_normal(self.lin1.weight.data)
        torch.nn.init.xavier_normal(self.lin2.weight.data)
        # torch.nn.init.xavier_normal(self.lin3.weight.data)
