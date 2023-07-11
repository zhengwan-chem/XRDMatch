import numpy as np
import torch.nn as nn
import torch
import sys

class VGG16(nn.Module):
    def __init__(self,In_channel=1,num_classes=2):

        super(VGG16, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, 64, kernel_size=16, stride=2, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=16 , stride=1, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.5),

            torch.nn.Conv1d(64, 128, kernel_size=12, stride=2, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=12, stride=1, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.5),

            torch.nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
#            torch.nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
#            torch.nn.BatchNorm1d(256),
#            torch.nn.ReLU(),
#            torch.nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=1),
#            torch.nn.BatchNorm1d(256),
#            torch.nn.ReLU(),            
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.5),

            torch.nn.Conv1d(256, 512, kernel_size=8, stride=2,  padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=8, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
#            torch.nn.Conv1d(512, 512, kernel_size=4, padding=1),
#            torch.nn.BatchNorm1d(512),
#            torch.nn.ReLU(),
#            torch.nn.Conv1d(512, 512, kernel_size=8, padding=1),
#            torch.nn.BatchNorm1d(512),
#            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.5),

            torch.nn.Conv1d(512, 512, kernel_size=4, stride=2,  padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=4, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
#            torch.nn.Conv1d(512, 512, kernel_size=2, padding=1),
#            torch.nn.BatchNorm1d(512),
#            torch.nn.ReLU(),
#            torch.nn.Conv1d(512, 512, kernel_size=4, padding=1),
#            torch.nn.BatchNorm1d(512),
#            torch.nn.ReLU(),            
            torch.nn.MaxPool1d(2),        
            torch.nn.AdaptiveAvgPool1d(2), 
            torch.nn.Dropout(0.5)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),   
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """

        if only_fc:
            return self.classifer(x)

        x = self.feature(x)
        x = torch.flatten(x, 1)

        if only_feat:
            return x

        out = self.classifer(x)
        result_dict = {'logits':out, 'feat':x}
        return result_dict        

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd
        

def vgg_16(pretrained=False, pretrained_path=None, **kwargs):
    model = VGG16(**kwargs)
    return model


 
