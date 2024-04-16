# MiniGen: Efficient Evolutionary Search for Convolutional Neural Networks
*Code and experimental results for CS_679 Course project*

# Backend

## ops.py
Basic operation objects:  
Sep_Conv, Conv, Stacked_conv, Dil_Conv, Identity, Pooling.

## model.py
Blocks & Cell objects built on operations:  
Block, Cell, Cell_input.

## Network.py
Network objects built on Blocks & Cells:
Network 

    Methods  
    -------  

    assemble_model
        Assembles the neural network model
    update_performance
        Update performance after each generation
    build_net
        Building the torch network using the input encoding
    forward
        Forward process for the network
    net_summary
        Give the network summary using summary module
    average_performance
        Calculate the average performance of the nod
    assign_gen
        Assign generation number to the individual network
    train
        Train the network
    evaluate_node
        Test the accuracy of the network

# Frontend

## mini_gen.py
Evolutionary search implementation:
mutation, crossover, weighted population selection, generation decay.
 
# Encoding:
### Encoding for a cell:
    [filter_size: int, num_blocks: int, action_list: list = [action_str_1, action_str_2, action_str_3]]  

### Encoding example for a net:
    [[inf, 1, ['identity', '5*5 conv', '3*3 conv']], 
     [256, 1, ['3*3 dconv', '5*5 dconv', '3*3 maxpool', '5*5 dconv']], 
     [24,  2, ['3*3 maxpool', '3*3 dil conv']], 
     [40,  3, ['identity', '5*5 dconv', '5*5 conv']], 
     [128, 4, ['3*3 dil conv', '5*5 conv', '3*3 maxpool', '5*5 dconv']]]

### Encodings of 10 neural architectures mentioned in the Result:


1.The performance is 73.4%.

    [[24, 1, ['identity', '5*5 conv', '5*5 conv']], 
    [256, 1, ['3*3 dil conv', '3*3 conv']], 
    [40, 1, ['3*3 maxpool', '3*3 dil conv', '3*3 avgpool']], 
    [128, 1, ['5*5 conv', '3*3 maxpool', '3*3 avgpool']], 
    [64, 1, ['identity', '3*3 avgpool']]]

2.The performance is 74.74%.

    [[128, 1, ['identity', '3*3 conv', '3*3 conv']], 
    [256, 1, ['3*3 avgpool', '3*3 maxpool', '3*3 dil conv', '3*3 dil conv']], 
    [40, 1, ['3*3 avgpool', '3*3 dil conv', '5*5 dconv']], 
    [128, 1, ['5*5 dconv', '3*3 conv']], 
    [64, 1, ['1*7-7*1 conv', '3*3 avgpool', '3*3 avgpool']]]

3.The performance is 71.8%.

    [[128, 1, ['identity', '5*5 conv', '5*5 conv']], 
    [256, 1, ['3*3 dconv', '3*3 maxpool', '3*3 dil conv', '3*3 dconv']], 
    [40, 1, ['identity', '3*3 conv', '5*5 dconv']], 
    [128, 1, ['5*5 dconv', '3*3 conv']], 
    [40, 1, ['5*5 dconv', '3*3 dconv', '3*3 avgpool']]] 

4.The performance is 69.71%.

    [[128, 1, ['identity', '5*5 conv', '3*3 dconv']], 
    [128, 1, ['3*3 conv', '5*5 dconv']], 
    [80, 1, ['identity', '3*3 dconv']], 
    [256, 1, ['3*3 maxpool', '3*3 avgpool', '3*3 conv']], 
    [256, 1, ['5*5 dconv', '5*5 conv']]]

5.The performance is 70.44%.

    [[128, 1, ['identity', '3*3 conv', '5*5 dconv']], 
    [40, 1, ['3*3 conv', '5*5 dconv']], 
    [128, 1, ['5*5 dconv', '5*5 dconv', '5*5 dconv']], 
    [256, 1, ['1*7-7*1 conv', '5*5 dconv', '3*3 dconv']], 
    [256, 1, ['identity', '3*3 conv', '5*5 dconv']]]

6.The performance is 67.28%.

    [[40, 1, ['identity', '3*3 conv', '5*5 dconv']], 
    [40, 1, ['identity', '3*3 maxpool']], 
    [40, 1, ['3*3 dil conv', '3*3 avgpool']], 
    [256, 1, ['identity', '3*3 avgpool', '3*3 avgpool', '1*7-7*1 conv']], 
    [40, 1, ['5*5 dconv', '3*3 conv', '3*3 maxpool']]]

7.The performance is 71.61%.

    [[24, 1, ['identity', '3*3 conv', '5*5 conv']], 
    [256, 1, ['1*7-7*1 conv', '3*3 dconv']], 
    [80, 1, ['3*3 dil conv', '3*3 conv']], 
    [256, 1, ['3*3 conv', '3*3 maxpool', '3*3 avgpool', '1*7-7*1 conv']], 
    [24, 1, ['identity', '3*3 maxpool']]]


8.The performance is 64.56%.

    [[24, 1, ['identity', '3*3 dconv', '5*5 dconv']], 
    [128, 1, ['3*3 maxpool', '3*3 dil conv']], 
    [64, 1, ['5*5 dconv', '3*3 dil conv']], 
    [128, 1, ['identity', '1*7-7*1 conv', '3*3 conv']], 
    [64, 1, ['3*3 avgpool', '3*3 maxpool']]]

9.The performance is 71.63%.

    [[256, 1, ['identity', '3*3 conv', '5*5 conv']], 
    [40, 1, ['identity', '3*3 maxpool']], 
    [80, 1, ['3*3 avgpool', '3*3 avgpool']], 
    [128, 1, ['3*3 dil conv', '5*5 dconv', '3*3 maxpool']], 
    [80, 1, ['3*3 conv', '1*7-7*1 conv', '3*3 dil conv', '5*5 conv']]]

10.The performance is 74.99%.

    [[80, 1, ['identity', '3*3 conv', '5*5 conv']], 
    [128, 1, ['identity', '3*3 avgpool']], 
    [256, 1, ['5*5 dconv', '5*5 dconv']], 
    [256, 1, ['1*7-7*1 conv', '5*5 dconv', '3*3 avgpool']], 
    [64, 1, ['3*3 conv', '5*5 dconv', '3*3 avgpool']]]












