# Neural_architecture_search

CS_679 Course project. 

## Ops.py
basic operation objects:  
Identity, Sep_Conv, Conv, Stacked_conv, Pooling, Dil_Conv, Op 

## model.py
building blocks objects:  
Block, Cell, Cell_input, Net

## initialization:
randomly generate encoding for creating a net:
 
### encoding for a cell:
[filter_size: int, num_blocks: int, action_list: ["identity", "3*3 avgpool", "1*7-7*1 conv"]]

### encoding for a net:
    [[inf, 1, ['identity', '5*5 conv', '3*3 conv']], 
     [256, 1, ['3*3 dconv', '5*5 dconv', '3*3 maxpool', '5*5 dconv']], 
     [24,  2, ['3*3 maxpool', '3*3 dil conv']], 
     [40,  3, ['identity', '5*5 dconv', '5*5 conv']], 
     [128, 4, ['3*3 dil conv', '5*5 conv', '3*3 maxpool', '5*5 dconv']]]
    




