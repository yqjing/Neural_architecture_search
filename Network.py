import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
class Network(torch.nn.Module):
    """ 
    Net object that inherits the Op that is the next level of Cell
    """
    def __init__(self, net_encoding = None, learning_rate = 0.001, device='cpu'):
        assert len(net_encoding) == 5, "the number of cell in an individual must be 5"
        super().__init__()
        if net_encoding == None:
            self.net_ed = full_ed_generator(0.5)
        else:
            self.net_ed = net_encoding
        
        self.post_process_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                                nn.Flatten(),
                                                nn.LazyLinear(10),
                                                nn.Softmax(dim=1))
        self.build_net()
        self.performance_history=[]
        self.learning_rate = learning_rate
        self.device = device
        self.assemble_model()

    def assemble_model(self):
        """Assembles the neural network model."""
        self.model = nn.Sequential(*self.net, self.post_process_layer).to(self.device)

    def update_performance(self, new_score):
        self.performance_history.append(new_score)  

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

    @property
    def average_performance(self):
        """Calculate the average performance of the node."""
        if not self.performance_history:
            return 0 
        return sum(self.performance_history) / len(self.performance_history)

    def train(self, train_data, epochs=20, device='cpu'):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(epochs):
            for inputs, labels in DataLoader(train_data, batch_size=64, shuffle=True):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                print(loss)
    
    def evaluate_node(self, validation_data, device='cpu'):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in DataLoader(validation_data, batch_size=64, shuffle=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        self.performance_history.append(accuracy)
        return accuracy
        












