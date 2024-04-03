from Network import Network 
from utils import *
import random
from torch.utils.data import Subset
import numpy as np
from model import *


def initialize_population(size=10, device = 'cpu'):
    """Initialize a population of networks."""
    population = [Network(net_encoding=full_ed_generator(0.5), device = device) for _ in range(size)]
    return population

def evaluate_fitness(node, train_data, validation_data, epoch = 20, device ='cpu', save=True):
    """Evaluate the fitness of a node by training and then measuring performance."""
    node.train(train_data, epochs=epoch, device = device)
    performance_score = node.evaluate_node(validation_data, device = device)
    if save:
      node.update_performance(performance_score)
    return performance_score


def select_population(population, to_keep=5):
    """Select the best-performing nodes based on average performance."""
    sorted_population = sorted(population, key=lambda x: x.average_performance, reverse=True)
    return sorted_population[:to_keep]

def weighted_selection(nodes, number_of_parents=2):
    """Selects parents based on their average performance."""
    total_performance = sum(node.average_performance for node in nodes)
    if total_performance == 0:
        weights = [1/len(nodes)] * len(nodes)
    else:
        weights = [node.average_performance / total_performance for node in nodes]

    selected_parents = random.choices(nodes, weights=weights, k=number_of_parents)
    return selected_parents

def crossover_and_mutate(parents, available_ops, population_size=15, to_keep=5, device = 'cpu'):
    """Generate a new population, ensuring the best 'to_keep' nodes are included."""
    new_population = parents
    while len(new_population) < population_size:
        parent1, parent2 = weighted_selection(parents, number_of_parents=2)
        child = create_new_child(parent1, parent2, device = device)
        new_population.append(child)

    return new_population

def create_new_child(parent1, parent2, base_mutation_rate=0.05, mutation_increase=0.05, device = 'cpu', random_pre=0.2, channel_list = [24, 40, 64, 80, 128, 256]):
    child_net_ed = []
    num_cells = len(parent1.net_ed)

    for i in range(num_cells):
        if random.random() < 0.5:
            child_net_ed.append(copy.deepcopy(parent1.net_ed[i]))
        else:
            child_net_ed.append(copy.deepcopy(parent2.net_ed[i]))
        adj = float(mutation_increase) * float(i / (num_cells - 1))
        adjusted_mutation_rate = float(base_mutation_rate) + adj

        if random.random() < adjusted_mutation_rate:
            if i == 0:
                ed_cell = [np.inf, 1, input_ed_generator(random_pre)]
            else:
                ed_cell = [random.choice(channel_list), 1, normal_ed_generator(random_pre)]
            child_net_ed[i] = ed_cell

    child_net = Network(net_encoding=child_net_ed, learning_rate=random.choice([parent1.learning_rate, parent2.learning_rate]), device = device)
    return child_net

def select_data_subsets(train_data, subset_size=20):
    """
    Selects random subsets from the training data for training and validation.

    Parameters:
    - train_data: The dataset from which to select subsets.
    - subset_size (int): The size of the subsets to select for both training and validation.

    Returns:
    - A tuple containing the training and validation subsets.
    """
    indices = np.random.choice(len(train_data), 2 * subset_size, replace=False)
    train_indices = indices[:subset_size]
    val_indices = indices[subset_size:]
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(train_data, val_indices)

    return train_subset, val_subset

def genetic_algorithm(train_data, validation_data, generations=100, population_size=10, to_keep=5, subset_size=120, train_epoches = 10, device = 'cpu'):
    population = initialize_population(size=population_size, device = device)
    for generation in tqdm(range(generations)):
        print(f"Generation {generation + 1}")
        train_subset, test_subset = select_data_subsets(train_data, subset_size=subset_size)

        test_performances = []
        i = 0
        for node in tqdm(population):
            performance_score = evaluate_fitness(node, train_subset, test_subset, train_epoches, device)
            test_performances.append(performance_score)
            #print(f"training on node {i}, performance {performance_score}")
            i+=1

        print(f"Finish Generation {i}")
        avg_test_performance = sum(test_performances) / len(test_performances)
        best_child_index = test_performances.index(max(test_performances))
        best_child = population[best_child_index]
        best_child_test_score = test_performances[best_child_index]

        print(f"Average performance on test subset: {avg_test_performance}")
        print(f"Best child's performance on test subset: {best_child_test_score}")
        # print(f"Best child's performance on validation subset{evaluate_fitness(best_child, train_data, validation_data, 10, device, save=False)}")
        print(f"Best child's structure: {summary(best_child, (3, 32, 32))}")

        selected = select_population(population, to_keep=to_keep)
        population = crossover_and_mutate(selected, available_ops=available_ops, population_size=population_size, to_keep=to_keep, device = device)



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    train_data, validation_data, _ = load_cifar10_data() 
    genetic_algorithm(train_data, validation_data, device = device)
