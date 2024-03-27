import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
class Node:
    
    def __init__(self, parent1=None, parent2=None, learning_rate=0.001, available_ops=[], device='cpu'):
        """
        Initializes a Node object. If two parents are provided, performs a crossover
        to generate the operations list for the child node.

        Parameters:
        - parent1, parent2 (Node or None): The parent nodes for crossover.
        - learning_rate (float): The learning rate for the node.
        """
        self.operations = []
        self.learning_rate = learning_rate
        
        self.performance_history = []
        if parent1 is not None and parent2 is not None:
            self.crossover(parent1, parent2)
        else:
            self.random_init(available_ops)
        
        self.model = self.build_pytorch_model().to(device)


    def build_pytorch_model(self):
        """Converts the node's operations into a sequential PyTorch model."""
        model_layers = nn.Sequential(*self.operations)
        return model_layers


    def update_performance(self, new_score):
        self.performance_history.append(new_score)
        
    @property
    def average_performance(self):
        """Calculate the average performance of the node."""
        if not self.performance_history:
            return 0 
        return sum(self.performance_history) / len(self.performance_history)

    def train(self, train_data, epochs=20, device='cpu'):
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(epochs):
            for inputs, labels in DataLoader(train_data, batch_size=64, shuffle=True):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate_node(self, validation_data, device='cpu'):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in DataLoader(validation_data, batch_size=64, shuffle=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    def random_init(self, available_ops):
        """
        Initializes the node with a random number (between 1 and 3) of operations
        from the available operations.

        Parameters:
        - available_ops (list of callable): List of callable operations (classes) that can be initialized.
        """
        if available_ops and len(available_ops) > 0:
            num_operations = random.randint(1, 3)

            for _ in range(num_operations):
                OpClass = random.choice(available_ops)
                self.operations.append(OpClass())

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent nodes to populate the child's operations list.

        Parameters:
        - parent1, parent2 (Node): The parent nodes.
        """
        max_len = max(len(parent1.operations), len(parent2.operations))
        self.learning_rate = random.choice([parent1.learning_rate, parent2.learning_rate])
        lr_adjustment = random.uniform(0.9, 1.1)
        self.learning_rate *= lr_adjustment
        for i in range(max_len):
            inherit_decision = random.choices(['none', 'parent1', 'parent2', 'both'], weights=[0.1, 0.4, 0.4, 0.1], k=1)[0]

            if inherit_decision == 'parent1' and i < len(parent1.operations):
                self.operations.append(parent1.operations[i])
            elif inherit_decision == 'parent2' and i < len(parent2.operations):
                self.operations.append(parent2.operations[i])
            elif inherit_decision == 'both':
                if i < len(parent1.operations):
                    self.operations.append(parent1.operations[i])
                if i < len(parent2.operations):
                    self.operations.append(parent2.operations[i])

    def mutate(self, available_ops, mutation_rate=0.1):
        """
        Mutates the node's operations list or learning rate based on a given mutation rate.

        Parameters:
        - available_ops (list of callable): List of callable operations for potential inclusion in the node.
        - mutation_rate (float): The probability of each mutation type occurring.
        """
        if random.random() < mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'modify', 'alter_lr'])
            
            if mutation_type == 'add' and len(available_ops) > 0:
                new_op = random.choice(available_ops)()
                position = random.randint(0, len(self.operations))
                self.operations.insert(position, new_op)
            
            elif mutation_type == 'remove' and len(self.operations) > 0:
                position = random.randint(0, len(self.operations) - 1)
                self.operations.pop(position)
            
            elif mutation_type == 'modify' and len(available_ops) > 0 and len(self.operations) > 0:
                new_op = random.choice(available_ops)()
                position = random.randint(0, len(self.operations) - 1)
                self.operations[position] = new_op

    def add_operation(self, operation):
        """
        Adds an operation to the list of operations.

        Parameters:
        - operation (Op object): The operation to add.
        """
        self.operations.append(operation)

    def remove_operation(self, index):
        """
        Removes an operation at a specified index from the list of operations.

        Parameters:
        - index (int): The index of the operation to remove.
        """
        if index < len(self.operations):
            self.operations.pop(index)

    def set_learning_rate(self, learning_rate):
        """
        Sets the learning rate for the node.

        Parameters:
        - learning_rate (float): The new learning rate.
        """
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Applies the sequence of operations to an input tensor.

        Parameters:
        - x (tensor): The input tensor.

        Returns:
        - tensor: The output tensor after applying all operations.
        """
        for op in self.operations:
            x = op(x)
        return x

    def describe(self):
        """Prints a description of the node's architecture and learning rate."""
        print("Node Architecture:")
        for op in self.operations:
            print(f"  {op}")  # Adjust based on how operations are represented/stored
        print(f"Learning Rate: {self.learning_rate}")