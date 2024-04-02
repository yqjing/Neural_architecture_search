# from model import Block, Cell, Cell_input, Net
from model import *
from utils import *
from Network import Network


if __name__ == "__main__":
    # generate random encoding for an individual net
    # encoding for a cell:
    # [filter_size: int, num_blocks: int, action_list: ["identity", "3*3 avgpool", "1*7-7*1 conv"]]
    # encoding for a net:
    # [[inf, 1, ['identity', '5*5 conv', '3*3 conv']], 
    #  [256, 1, ['3*3 dconv', '5*5 dconv', '3*3 maxpool', '5*5 dconv']], 
    #  [24,  2, ['3*3 maxpool', '3*3 dil conv']], 
    #  [40,  3, ['identity', '5*5 dconv', '5*5 conv']], 
    #  [128, 4, ['3*3 dil conv', '5*5 conv', '3*3 maxpool', '5*5 dconv']]]
    net_ed_1 = full_ed_generator(0.5)
    print(net_ed_1)

    # create a net according to the generated encoding
    net_1 = Network(net_ed_1)
    x = torch.rand(32, 3, 32, 32)
    y = net_1(x)
    # the output of a net is the softmaxed classification output
    print(y.shape)
    # print the summary for the individual net
    net_1.net_summary()
    summary(net_1, (3, 32, 32))







              