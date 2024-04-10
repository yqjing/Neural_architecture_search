from Network import Network 
from utils import *
import random
from torch.utils.data import Subset
import numpy as np
from model import *
from gpu import get_gpu_status
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

def initialize_population(size=15, device = 'cpu'):
    """Initialize a population of networks."""
    population = [Network(net_encoding=full_ed_generator(0.5), device = device) for _ in range(size)]
    return population

def evaluate_fitness(node, validation_data, device ='cpu', save=True):
    """Evaluate the fitness of a node by training and then measuring performance."""
    # node.train(train_data, epochs=epoch, device = device)
    performance_score = node.evaluate_node(validation_data, device = device)
    if save:
      node.update_performance(performance_score)
    return performance_score

def model_train(node, train_data, epoch, device = 'cpu'):
    node.train(train_data, epoch , device = device)


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
        performance_ls = [node.average_performance for node in nodes]

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
    
        weights = softmax(performance_ls)
        # weights = [node.average_performance / total_performance for node in nodes]

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

def create_new_child(parent1, parent2, base_mutation_rate=0.1, mutation_increase=0.05, device = 'cpu', random_pre=0.5, channel_list = [24, 40, 64, 80, 128, 256]):
    child_net_ed = []
    num_cells = len(parent1.net_ed)

    channel_list = [24, 40, 64, 80, 128]
    action_ls_full = ["identity", "3*3 dconv",  "5*5 dconv", "3*3 conv", "5*5 conv", "1*7-7*1 conv", "3*3 dil conv", "3*3 maxpool", "3*3 avgpool"]
    action_ls_input = ["identity", "3*3 dconv",  "5*5 dconv", "3*3 conv", "5*5 conv"]

    for i in range(num_cells):
        # crossover
        if random.random() < 0.5:
            child_net_ed.append(copy.deepcopy(parent1.net_ed[i]))
        else:
            child_net_ed.append(copy.deepcopy(parent2.net_ed[i]))
        
        # mutation
        adj = float(mutation_increase) * float(i / (num_cells - 1))
        adjusted_mutation_rate = float(base_mutation_rate) + adj

        # filter size
        if random.random() < 3 * adjusted_mutation_rate:
            new_filter_size = random.choice(channel_list)
            child_net_ed[i][0] = new_filter_size

        # each gene mutate in the action_list
        action_ls = child_net_ed[i][2]
        if i == 0:
            a_ls = copy.deepcopy(action_ls_input)
        else:
            a_ls = copy.deepcopy(action_ls_full)
        for j in range(len(action_ls)):
            if j == 0:
                if random.random() < 2 * adjusted_mutation_rate:
                    action_ls[j] = random.choice(a_ls)
                del a_ls[0]
                continue   
            if random.random() < 2 * adjusted_mutation_rate:
                    action_ls[j] = random.choice(a_ls)
        child_net_ed[i][2] = action_ls

        # whole action list mutates
        if random.random() < adjusted_mutation_rate:
            if i == 0:
                new_action_ls = input_ed_generator(random_pre)
            else:
                new_action_ls = normal_ed_generator(random_pre)
            child_net_ed[i][2] = new_action_ls

    child_net = Network(net_encoding=child_net_ed, learning_rate=random.choice([parent1.learning_rate, parent2.learning_rate]), device = device)
    return child_net

def select_data_subsets(train_data, N_h = 10, K=6):
    """
    Selects random subsets from the training data for training and validation.

    Parameters:
    - train_data: The dataset from which to select subsets.
    - subset_size (int): The size of the subsets to select for both training and validation.

    Returns:
    - A tuple containing the training and validation subsets.
    """

    dataset = train_data.dataset

    label_classes = list(range(0, 10))
    random_classes = random.sample(label_classes, K)

    train = []
    label_to_be_selected = {i: N_h for i in random_classes}
    while sum(list(label_to_be_selected.values())) > 0:
        random_index = random.randint(0, len(dataset) - 1)
        element = dataset[random_index]
        if element[1] in list(label_to_be_selected.keys()) and label_to_be_selected[element[1]] > 0:
            label_to_be_selected[element[1]] -= 1
            train.append(element)

    test = []
    label_to_be_selected = [N_h]*10
    selected = 0
    while selected<=sum(label_to_be_selected):
        random_index = random.randint(0, len(dataset) - 1)
        element = dataset[random_index]
        label_to_be_selected[element[1]]-=1
        test.append(element)
        selected += 1
        
    train_subset = train
    val_subset = test

    return train_subset, val_subset

def genetic_algorithm(train_data, validation_data, generations=2000, population_size=15, to_keep=7, subset_size=30, train_epoches = 20, device = 'cpu', load=False):
    if load == False:
        population = initialize_population(size=population_size, device = device)
    else:
        best_id_ed = torch.load("saved_encodings/best_ed_100.pkl")
        best_id = [Network(ed, device = device) for ed in best_id_ed]
        population = crossover_and_mutate(best_id, available_ops=available_ops, population_size=population_size, to_keep=to_keep, device = device)
        print("population loaded")
    best_average_accuracy_across_generation = 0
    
    best_child_across_generation = population[0]
    best_child_across_generation_copy = population[0]
    best_child_epoches = 0
    for generation in tqdm(range(generations)):
        print(f"Generation {generation + 1}")
        best_child = population[0]
        best_child2 = population[0]
        test_performances = []
        i = 0
        best_child_average_performance_without_decay = 0
        best_child_average_performance = 0
        train_subset, test_subset = select_data_subsets(train_data, subset_size)

        for node in tqdm(population):
            model_train(node, train_subset, epoch=train_epoches)

            performance_score = evaluate_fitness(node, test_subset, device)

            test_performances.append(performance_score)
            #print(f"training on node {i}, performance {performance_score}")
            i+=1
            if node.average_performance_without_decay>=best_child_average_performance_without_decay:
                best_child_average_performance_without_decay = node.average_performance_without_decay
                best_child = node
            if node.average_performance>=best_child_average_performance:
                best_child_average_performance = node.average_performance
                best_child2 = node

        print(f"Finish Generation {generation}")
        avg_test_performance = sum(test_performances) / len(test_performances)
        best_child_index = test_performances.index(max(test_performances))
        best_child = population[best_child_index]
        best_child_test_score = test_performances[best_child_index]

        print(f"Average performance on test subset: {avg_test_performance}")

        print(f"Best child's performance on test subset: {best_child_test_score}")
        if best_child_average_performance_without_decay>=best_average_accuracy_across_generation:
            best_average_accuracy_across_generation = best_child_average_performance_without_decay
            best_child_epoches = generation

        print(f"Best child's average performance: {best_child_average_performance_without_decay}")

        
        # print(f"Best child's performance on validation subset{evaluate_fitness(best_child, train_data, validation_data, 10, device, save=False)}")     
        # if generation % 10 == 0:
        #     print(f"Best child's structure: {summary(best_child, (3, 32, 32))}")
        if generation % 10 == 0:
            print(f"Best individual net at generation {generation} is\n", best_child.net_ed)
            print(f"Best child's average performance across generation is:{best_average_accuracy_across_generation} which appear in generation {best_child_epoches}")
        
        if generation % 50 == 0 and generation != 0:
            best_child.train(train_data, epochs=10, device = device)
            best_child2.train(train_data, epochs=10, device = device)
            acc = best_child.evaluate_node(validation_data, device = device)
            acc2 = best_child2.evaluate_node(validation_data, device = device)
            real_acc = best_child_across_generation.evaluate_node(validation_data, device = device)
            real_acc_cpy = best_child_across_generation_copy.evaluate_node(validation_data, device = device)
            if acc>=real_acc:
                best_child_across_generation = best_child
                best_child_across_generation_copy = copy.deepcopy(best_child)

            print(f"The performance of the best individual with decay at generation {generation} is\n", acc)
            print(f"The performance of the best individual no dacay at generation {generation} is\n", acc2)
            print(f"Best Accuracy\n", real_acc)
            print(f"Best accuracy using the copied node\n", real_acc_cpy)
            
        
        selected = select_population(population, to_keep=to_keep)
        population = crossover_and_mutate(selected, available_ops=available_ops, population_size=population_size, to_keep=to_keep, device = device)

if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    train_data, validation_data, _ = load_cifar10_data() 
    genetic_algorithm(train_data, validation_data, device = device)
