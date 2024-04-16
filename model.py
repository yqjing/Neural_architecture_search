from ops import *
# from ops import Identity, Sep_Conv, Conv, Stacked_conv, Pooling, Dil_Conv, Op 
from torchsummary import summary

class Block(Op):
    """ 
    Block object that inherits the Op
    """
    def __init__(self, num_actions, action_list, num_channels, strides):
        super().__init__()
        assert len(action_list) == num_actions, "the length of the action list must be equal to the number of actions for the block"
        if strides == 2:
            assert len(action_list) <= 3 and num_actions <= 3, "the input block must have less than or equal to 3 layers"
        self.num_actions = num_actions
        self.action_list = action_list 
        self.num_channels = num_channels
        self.strides = strides
        self.out_channels = num_channels
        self.identity = True
        self.build_block()


    def build_block(self):
        self.block = nn.ModuleList([])
        action_ls_tmp = copy.deepcopy(self.action_list)
        num_actions_tmp = copy.deepcopy(self.num_actions)

        if self.action_list[0] == "identity":
            self.identity = True
        else:
            self.identity = False

        if self.identity == True:
            if self.strides == 1:
                self.skip_layer = self.str_2_action(self.num_channels, self.num_channels, 'identity', 1)
                del action_ls_tmp[0]
                num_actions_tmp -= 1
            else:
                self.skip_layer = self.str_2_action(self.num_channels, 64, 'identity', 2)
                del action_ls_tmp[0]
                num_actions_tmp -= 1
        else:
            pass

        for idx in range(num_actions_tmp):
            action = action_ls_tmp[idx]
            if self.strides == 1:
                layer = self.str_2_action(self.num_channels, self.num_channels, action, 1)
                self.block.append(layer)
            else:
                if idx == 0:
                    layer = self.str_2_action(3, 32, action, 2)
                elif idx == 1:
                    layer = self.str_2_action(32, 64, action, 2)
                elif idx == 2:
                    layer = self.str_2_action(64, 128, action, 2)
                self.block.append(layer)

    def forward(self, inputs):
        x = inputs
        for op in self.block:
            x = op(x)
        if self.identity == True:
            skip = self.skip_layer(inputs)
            x = nn.functional.relu(x + skip)
        return x
    
        
    def str_2_action(self, in_channels, num_channels, action, strides):

        if action == "3*3 dconv":
            x = Sep_Conv(in_channels, num_channels, 3, strides)
            return x
        
        if action == "5*5 dconv":
            x = Sep_Conv(in_channels, num_channels, 5, strides)
            return x

        if action == "3*3 conv":
            x = Conv(num_channels, 3, strides)
            return x

        if action == "5*5 conv":
            x = Conv(num_channels, 5, strides)
            return x

        if action == "1*7-7*1 conv":
            x = Stacked_conv([num_channels, num_channels], [strides, strides])
            return x

        if action == "3*3 dil conv":
            x = Dil_Conv(in_channels, num_channels, strides)
            return x

        if action == "identity":
            x = Identity(num_channels, strides)
            return x

        if action == "3*3 maxpool":
            x = Pooling(in_channels, "max", strides)
            return x

        if action == "3*3 avgpool":
            x = Pooling(in_channels, "average", strides)
            return x


# # test   
# action_list = ["identity", "3*3 avgpool", "3*3 avgpool", "1*7-7*1 conv"]
# b_1 = Block(4, action_list, 128*2, 1)
# x1 = torch.randn(32, 128*2, 32, 32)
# y1 = b_1(x1)

# print(summary(b_1, (128*2, 32, 32)))
# print(y1.shape)

# action_list = ["identity", "5*5 dconv", "3*3 dconv"]
# b_2 = Block(3, action_list, 3, 2)
# x2 = torch.randn(32, 3, 32, 32)
# y2 = b_2(x2)

# print(summary(b_2, (3, 32, 32)))
# print(y2.shape)


class Cell(Op):
    """ 
    Cell that builds on the Op object. Cell is composed of Blocks. 

    Parameters
    ----------

    cell_encoding : List[List]
         [filter_size: int, num_blocks: int, action_list: ["identity", "3*3 avgpool", "1*7-7*1 conv"]]
    
    cell_idx : int
        idx of the cell
    
    """

    def __init__(self, cell_idx, cell_encoding, strides=1):
        super().__init__()
        self.action_list = cell_encoding[2]
        self.num_blocks = cell_encoding[1]
        self.strides = strides
        self.num_channels = cell_encoding[0]
        self.cell_idx = cell_idx
        self.build_cell()

    def build_cell(self):
        self.cell = nn.ModuleList([])
        self.first_layer = nn.LazyConv2d(self.num_channels, kernel_size=1, stride=1)
        self.cell.append(self.first_layer)
        for _ in range(self.num_blocks):
            block = Block(len(self.action_list), self.action_list, self.num_channels, 1)
            self.cell.append(block)
        
    def forward(self, inputs):
        x = inputs
        for block_op in self.cell:
                x = block_op(x)
        return x

    def cell_summary(self):
         print(f"For Cell {self.cell_idx} | the Resolution of the image is 32 * 32 | the channel size is {self.num_channels} | the number of blocks are {self.num_blocks}.")
         print("The summary of the block is")
         block_fake = copy.deepcopy(self.cell[1])
         print(summary(block_fake, (int(self.num_channels), 32, 32)))
         
# # testing 
# ed = [300, 3, ["identity", "3*3 avgpool", "1*7-7*1 conv", "5*5 dconv"]]
# c = Cell(3, ed)
# x = torch.randn(32, 200, 32, 32)
# y = c(x)
# print(c.cell_summary())
# print(y.shape)

class Cell_input(Op):
    """ 
    Cell that builds on the Op object. Cell is composed of Blocks. 

    Parameters
    ----------

    cell_encoding : List[List]
        [filter_size: int, num_blocks: int, action_list: ["identity", "3*3 avgpool", "1*7-7*1 conv"]]
    
    cell_idx : int
        idx of the cell
    
    """

    def __init__(self, cell_idx, cell_encoding, strides=2):
        super().__init__()
        self.action_list = cell_encoding[2]
        self.num_blocks = cell_encoding[1]
        self.strides = strides
        self.num_channels = cell_encoding[0]
        self.cell_idx = cell_idx
        self.build_cell()

    def build_cell(self):
        self.cell = nn.ModuleList([])
        for _ in range(self.num_blocks):
            block = Block(len(self.action_list), self.action_list, self.num_channels, 2)
            self.cell.append(block)
        
    def forward(self, inputs):
        x = inputs
        for block_op in self.cell:
                x = block_op(x)
        return x

    def cell_summary(self):
         print(f"For Cell {self.cell_idx} | the Resolution of the image is 32 * 32 | the channel size is 3 -> 32 -> 64 | the number of blocks are {self.num_blocks}.")
         print("The summary of the block is")
         block_fake = copy.deepcopy(self.cell[0])
         print(summary(block_fake, (3, 32, 32)))
         
# # testing
# ed = [np.inf, 1, ["identity", "5*5 dconv", "3*3 conv"]]
# c = Cell_input(0, ed)
# x = torch.randn(32, 3, 32, 32)
# y = c(x)
# print(y.shape)
# c.cell_summary()


class Net(Op):
    """ 
    Net object that inherits the Op that is the next level of Cell
    """
    def __init__(self, net_encoding):
        assert len(net_encoding) == 5, "the number of cell in an individual must be 5"
        super().__init__()
        self.net_ed = net_encoding
        self.post_process_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                                nn.Flatten(),
                                                nn.LazyLinear(10),
                                                nn.Softmax(dim=1))
        self.build_net()

    def build_net(self):
        self.net = nn.ModuleList([])
        cell_0 = Cell_input(0, self.net_ed[0])
        cell_1 = Cell(1, self.net_ed[1])
        cell_2 = Cell(2, self.net_ed[2])
        cell_3 = Cell(3, self.net_ed[3])
        cell_4 = Cell(4, self.net_ed[4])

        self.net.append(cell_0)
        self.net.append(cell_1)
        self.net.append(cell_2)
        self.net.append(cell_3)
        self.net.append(cell_4)

    def forward(self, inputs):
        x = inputs
        for cell in self.net:
            x = cell(x)
        output = self.post_process_layer(x)
        return output
    
    def net_summary(self):
        for cell in self.net:
            cell.cell_summary()
            print("\n")
        print("Plus the post processing layer.\n")





        











