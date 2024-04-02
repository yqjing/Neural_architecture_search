# from model import Block, Cell, Cell_input, Net
from model import *
import sys


action_ls_full = ["identity", "3*3 dconv",  "5*5 dconv", "3*3 conv", "5*5 conv", "1*7-7*1 conv", "3*3 dil conv", "3*3 maxpool", "3*3 avgpool"]
action_ls_input = ["identity", "3*3 dconv",  "5*5 dconv", "3*3 conv", "5*5 conv"]

def input_ed_generator(random_pre):
    ed_input = []
    action_ls = copy.deepcopy(action_ls_input)
    random_p = random.random()
    if random_p >= random_pre:
        ed_input.append('identity')
        del action_ls[0]
    else:
        del action_ls[0]
    len_remain = 3 - len(ed_input)
    ed_input = ed_input + [random.choice(action_ls) for _ in range(len_remain)]
    return ed_input

def normal_ed_generator(random_pre):
    ed_input = []
    action_ls = copy.deepcopy(action_ls_full)
    random_p = random.random()
    if random_p >= random_pre:
        ed_input.append('identity')
        del action_ls[0]
    num_ls = [2, 3, 4]
    num_actions = random.choice(num_ls)
    len_remain = num_actions - len(ed_input)
    ed_input = ed_input + [random.choice(action_ls) for _ in range(len_remain)]
    return ed_input

def full_ed_generator(random_pre):
    """
    Net encoding generator

    Parameters
    ----------

    random_pre : float
        [0, 1], 
        if randomly chosen probability random_p > random_pre, 
        assign "identity" operation to the first position of the action list

    Variables in the encoding of a cell
    -----------------------------------

    [num_channels: int, num_blocks: int, action_list: ["identity", "3*3 avgpool", "1*7-7*1 conv"]]

    num_channel : int
        randomly chosen from the list [24, 40, 64, 80, 128, 256]
    action_list : List[str]
        a list of string, e.g. ["identity", "3*3 avgpool", "1*7-7*1 conv"], 
        randomly initialized.

    """

    net_ed = []
    channel_list = [24, 40, 64, 80, 128, 256]
    ed_cell_0 = [np.inf, 1, input_ed_generator(random_pre)]
    ed_cell_1 = [random.choice(channel_list), 1, normal_ed_generator(random_pre)]
    ed_cell_2 = [random.choice(channel_list), 2, normal_ed_generator(random_pre)]
    ed_cell_3 = [random.choice(channel_list), 3, normal_ed_generator(random_pre)]
    ed_cell_4 = [random.choice(channel_list), 4, normal_ed_generator(random_pre)]

    net_ed.append(ed_cell_0)
    net_ed.append(ed_cell_1)
    net_ed.append(ed_cell_2)
    net_ed.append(ed_cell_3)
    net_ed.append(ed_cell_4)

    return net_ed


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
    net_1 = Net(net_ed_1)
    x = torch.rand(32, 3, 32, 32)
    y = net_1(x)
    # the output of a net is the softmaxed classification output
    print(y.shape)
    # print the summary for the individual net
    net_1.net_summary()
    summary(net_1, (3, 32, 32))

    def capture_summary(net, input_size, filename):
        original_stdout = sys.stdout
        with open(filename, 'w') as f:
            sys.stdout = f  
            net.net_summary()
            summary(net, input_size)
        sys.stdout = original_stdout  

    # print the network summary
    capture_summary(net_1, (3, 32, 32), 'network_summary.txt')









              