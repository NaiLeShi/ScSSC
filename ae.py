import torch
from torch import nn
#from keras.layers import Dense, Input, GaussianNoise

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        # [b, 784]
        self.encoder = nn.Sequential(
            nn.Linear(10000, 5200),
            nn.ReLU(),
            nn.Linear(5200, 2100),
            nn.ReLU(),
            nn.Linear(2100,860),
            nn.ReLU(),
            nn.Linear(860, 100),
            nn.ReLU()
        )

        # [b, 20] => [b,784]
        self.decoder = nn.Sequential(
            nn.Linear(100, 860),
            nn.ReLU(),
            nn.Linear(860, 2100),
            nn.ReLU(),
            nn.Linear(2100, 5200),
            nn.ReLU(),
            nn.Linear(5200, 10000),
            nn.Sigmoid(),
        )

    # def isforward(self,x,encode):
    #     if encode :
    #         batchsz = x.size(0)
    #         # flatten
    #         x = x.view(batchsz, 784)
    #         x = self.encoder(x)
    #         x = x.view(batchsz, 1, 4, 5)
    #         return x
    #     else:
    #         self.x = x

    def forward(self, x, endcode):
        '''
        :param x: [b, 1, 28, 28]
        :return:
        '''
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, 10000)
        if endcode:
            x = self.encoder(x)
            x = x.view(batchsz,100)
            return x
        else:
            # encoder
            x = self.encoder(x)
            # decoder
            x = self.decoder(x)
            # reshape
            x = x.view(batchsz, 1, 100, 100)

            return x

def main():
    tmp = torch.randn(2,1,100,100)
    net = AE()
    tmp = net(tmp,False)
    print(tmp.shape)

if __name__ == '__main__':
    main()