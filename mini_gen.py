from Network import Network 
from utils import *
import random
from torch.utils.data import Subset
import numpy as np
from model import *
from gpu import get_gpu_status


def initialize_population(size=15, device = 'cpu'):
    """Initialize a population of networks."""
    torch.manual_seed(2809)
    population = [Network(net_encoding=full_ed_generator(0.5), device = device) for _ in range(size)]
    for node in population:
        node.assign_gen(0)
    return population

def evaluate_fitness(node, train_data, validation_data, epoch = 20, device ='cpu', save=True):
    """Evaluate the fitness of a node by training and then measuring performance."""
    # node.train(train_data, epochs=epoch, device = device)
    performance_score = node.evaluate_node(validation_data, device = device)
    if save:
      node.update_performance(performance_score)
    return performance_score

def model_train(node, train_data, epoch = 10, device = 'cpu'):
    node.train(train_data, epochs = 1, device = device)


def select_population(population, generation, to_keep=5):
    """Select the best-performing nodes based on average performance."""

    def eval_fcn(x, generation = generation):
        y = x.average_performance - 0.01 * abs(generation - x.gen)
        return y

    sorted_population = sorted(population, key=lambda x: eval_fcn(x), reverse=True)
    return sorted_population[:to_keep]

def weighted_selection(nodes, number_of_parents=2):
    """Selects parents based on their average performance."""
    total_performance = sum(node.average_performance for node in nodes)
    if total_performance == np.inf:
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

def crossover_and_mutate(parents, gen, population_size=15, to_keep=5, device = 'cpu'):
    """Generate a new population, ensuring the best 'to_keep' nodes are included."""
    new_population = parents
    while len(new_population) < population_size:
        parent1, parent2 = weighted_selection(parents[:to_keep], number_of_parents=2)
        child = create_new_child(parent1, parent2, gen, device = device)
        new_population.append(child)

    return new_population

def create_new_child(parent1, parent2, gen, base_mutation_rate=0.1, mutation_increase=0.05, device = 'cpu', random_pre=0.5, channel_list = [24, 40, 64, 80, 128, 256]):
    child_net_ed = []
    num_cells = len(parent1.net_ed)

    channel_list = [24, 40, 64, 80, 128, 256]
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

        if gen < 100:
            # each gene mutate in the action_list
            action_ls = child_net_ed[i][2]
            if i == 0:
                a_ls = copy.deepcopy(action_ls_input)
            else:
                a_ls = copy.deepcopy(action_ls_full)
            for j in range(len(action_ls)):
                if j == 0:
                    if random.random() < 5 * adjusted_mutation_rate:
                        action_ls[j] = random.choice(a_ls)
                    del a_ls[0]
                    continue   
                if random.random() < 5 * adjusted_mutation_rate:
                        action_ls[j] = random.choice(a_ls)
            child_net_ed[i][2] = action_ls

        if gen < 100:
            # whole action list mutates
            if random.random() < 2 * adjusted_mutation_rate:
                if i == 0:
                    new_action_ls = input_ed_generator(random_pre)
                else:
                    new_action_ls = normal_ed_generator(random_pre)
                child_net_ed[i][2] = new_action_ls

    torch.manual_seed(2809)
    child_net = Network(net_encoding=child_net_ed, learning_rate=random.choice([parent1.learning_rate, parent2.learning_rate]), device = device)
    child_net.assign_gen(gen)
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
    label_to_be_selected = {i: 100 for i in random_classes}
    while sum(list(label_to_be_selected.values())) > 0:
        random_index = int(np.random.random()*len(dataset))
        element = dataset[random_index]
        if element[1] in list(label_to_be_selected.keys()) and label_to_be_selected[element[1]] > 0:
            label_to_be_selected[element[1]] -= 1
            train.append(element)

    test = []
    label_to_be_selected = [10]*10
    selected = 0
    while selected<=sum(label_to_be_selected):
        random_index = int(np.random.random()*len(dataset))
        element = dataset[random_index]
        label_to_be_selected[element[1]]-=1
        test.append(element)
        selected += 1

    # test = []
    # label_to_be_selected = [10]*10
    # selected = 0
    # while selected<=sum(label_to_be_selected):
    #     random_index = int(np.random.random()*len(train))
    #     element = train[random_index]
    #     label_to_be_selected[element[1]]-=1
    #     test.append(element)
    #     selected += 1
        
    train_subset = train
    val_subset = test

    return train_subset, val_subset

def genetic_algorithm(train_data, validation_data, generations=200, population_size=15, to_keep=5, train_epoches = 20, gen_limit = 25, device = 'cpu', load=False):
    if load == False:
        population = initialize_population(size=population_size, device = device)
    else:
        best_id_ed = torch.load("saved_encodings/best_ed_20.pkl")
        torch.manual_seed(2809)
        best_id = [Network(ed, device = device) for ed in best_id_ed]
        for node in best_id:
            node.assign_gen(0)
        population = crossover_and_mutate(best_id, population_size=population_size, gen = 0, to_keep=to_keep, device = device)
        print("population loaded")
    
    best_gen_ls = []
    best_ed_20 = []
    best_ed_80 = []
    winners = []
    best_ed_20_perf = []
    inner = 100
    outer = 100
    patience = 0
    # train_subset, test_subset = select_data_subsets(train_data)
    # torch.save(train_subset, "saved_encodings/data_train")
    # torch.save(test_subset, "saved_encodings/data_test")

    train_subset = torch.load("saved_encodings/data_train")
    test_subset = torch.load("saved_encodings/data_test")
    while len(best_ed_20) <= to_keep and (outer, inner) != (1, 1):
        best_gen_ls = []
        best_perf_ls = []
        for generation in tqdm(range(generations)):
            print(f"Generation {generation}")

            test_performances = []
            i = 0
            for node in tqdm(population):
                for e in range(train_epoches):
                    model_train(node, train_subset)
                performance_score = evaluate_fitness(node, train_subset, test_subset, train_epoches, device)
                test_performances.append(performance_score)
                i+=1

            avg_test_performance = sum(test_performances) / len(test_performances)
            best_child_index = test_performances.index(max(test_performances))
            best_child = population[best_child_index]
            best_gen = best_child.gen
            best_child_test_score = test_performances[best_child_index]
            best_gen_ls.append(best_gen)
            best_perf_ls.append(best_child_test_score)

            if best_child_test_score > 0.4 and best_child.net_ed not in winners:
                winners.append(best_child.net_ed)

            print(f"\nPerformance list: \n{sorted(test_performances, reverse=True)}")
            print(f"Average performance on test subset: {avg_test_performance}")
            print(f"Best child's performance on test subset: {best_child_test_score}")
            print(f"Best child from generation: {best_gen}")
            print(f"Best individual net at generation {generation} outer {outer} inner {inner} is\n", best_child.net_ed)
            print(f"Finish Generation {generation}\n")
            plot_best_gen(best_gen_ls, best_perf_ls, outer, inner, patience)

            if generation > 2 and best_child_test_score > best_perf_ls[-2] and best_gen < max(best_gen_ls[:-1]):
                best_child.gen = generation

            if generation % 40 == 0 and generation != 0:
                train_data, validation_data, _ = load_cifar10_data() 
                best_child.train(train_data, epochs=100, device = device)
                acc = best_child.evaluate_node(validation_data, device = device)
                print(f"The performance of the best individual at generation {generation} is\n", acc)
                print(f"Best child's structure: {summary(best_child, (3, 32, 32))}")
            if generation % 7 == 0:
                print(get_gpu_status(device))

            if generation == gen_limit and len(best_ed_20) < 10000:
                if best_child_test_score > 0.35:

                    for node in population:
                        node.assign_gen(0)
                    best_ed_80.append(best_child.net_ed)
                    
                else:
                    # best_id_ed = torch.load("saved_encodings/best_ed_20.pkl")
                    # torch.manual_seed(2809)
                    # best_id = [Network(ed, device = device) for ed in best_id_ed]
                    # for node in best_id:
                    #     node.assign_gen(0)
                    # population = crossover_and_mutate(best_id, population_size=population_size, gen = 0, to_keep=to_keep, device = device)
                    population = initialize_population(size=population_size, device = device)

                
                patience += 1
                if len(winners) == 10:
                    pass
                else:
                    torch.save(winners, "saved_encodings/winners.pkl")
                    torch.save(best_ed_80, "saved_encodings/best_ed_80.pkl")
                    break

            if len(winners) == 10:
                
                print(f"FINISHED SEARCHING")
                torch.save(winners, "saved_encodings/winners.pkl")

                ed_ls = torch.load("saved_encodings/winners.pkl")
                torch.manual_seed(2809)
                winner_nets = [Network(ed, device = device) for ed in ed_ls]

                train_data, validation_data, _ = load_cifar10_data() 
                performance_list = []
                for net in winner_nets:
                    net.train(train_data, epochs=340, device = device)
                    acc = net.evaluate_node(validation_data, device = device)
                    performance_list.append(acc)
                    print(f"The performance of the \n{net.net_ed} is", acc)

                torch.save(performance_list, "saved_encodings/winners_ls.pkl")

                return "FINISHED"
            else:
                print("Current winners length is", len(winners))

        
            selected = select_population(population, generation, to_keep=to_keep)
            best_ed = [i.net_ed for i in selected]
            best_gen = [i.gen for i in selected]
            best_perf = [i.performance_history for i in selected]
            torch.save(best_ed, "saved_encodings/best_ed_100.pkl")
            torch.save(best_gen, "saved_encodings/best_gen_100.pkl")
            torch.save(best_perf, "saved_encodings/best_perf_100.pkl")

            
            best_id_ed = torch.load("saved_encodings/best_ed_100.pkl")
            best_id_gen = torch.load("saved_encodings/best_gen_100.pkl")
            best_id_perf = torch.load("saved_encodings/best_perf_100.pkl")
            torch.manual_seed(2809)
            best_id = [Network(ed, device = device) for ed in best_id_ed]
            for idx, node in enumerate(best_id):
                node.assign_gen(best_id_gen[idx])
                node.performance_history = best_id_perf[idx]
            population = crossover_and_mutate(best_id, population_size=population_size, to_keep=to_keep, gen = generation + 1, device = device)

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    train_data, validation_data, _ = load_cifar10_data() 
    genetic_algorithm(train_data, validation_data, device = device)
