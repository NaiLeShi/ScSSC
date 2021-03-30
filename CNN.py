import torch
from torch import nn
from torch.nn import functional as F

class Lenet(nn.Module):
    """
    for cifar10 dataset
    """
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 1, 100, 100] => [b,6,]
            nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            #
            nn.Conv2d(3,10, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(10, 24, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        # flatten
        # fc unit
        self.fc_unit1 = nn.Sequential(
            nn.Linear(24*10*10,1024),
            nn.ReLU(),
            nn.Linear(1024,184),
            nn.ReLU(),
            #nn.Linear(184,11)
        )
        self.fc_unit2 = nn.Sequential(

            nn.Linear(184, 11)
        )
        self.fc_unit3 = nn.Sequential(

            nn.Linear(184, 860),
            nn.ReLU(),
            nn.Linear(860, 2100),
            nn.ReLU(),
            nn.Linear(2100, 5200),
            nn.ReLU(),
            nn.Linear(5200, 10000),
            nn.Sigmoid(),
        )




        # use Cross Entropy Loss
        #self.criteon = nn.CrossEntropyLoss()


    def forward(self,x,label,train = True,):
        """

        :param x: [b,1,100,100]
        :return:
        """
        batchsz = x.size(0)
        # [b, 1, 100,100] => [b, 24, 10, 10]
        x = self.conv_unit(x)
        # [b, 24, 10, 10] => [b, 24*10*10]
        x = x.view(batchsz, 24*10*10)
        # [b, 24*10*10] => [b, 11]
        if train:
            if label == -1:
                logits = self.fc_unit3(self.fc_unit1(x))
                logits = logits.view(1,1,100,100)
            else:
                logits = self.fc_unit2(self.fc_unit1(x))
        else:
            #logits = self.fc_unit2(self.fc_unit1(x))
            logits = self.fc_unit1(x)
        return logits




def main():
    net = Lenet()
    # [b, 3, 32, 32]
    tmp = torch.randn(2, 1, 100, 100)
    out = net(tmp,3,False)
    # [b, 24, 10, 10]
    print('lenet out', out.shape)


if __name__ == '__main__':
    main()