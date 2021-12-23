import random
import numpy as np
import time

import data_loader


class GeneticAlgorithm:

    def __init__(self, mlp, population_size, crossover_rate, mutation_rate, elitism_size,
                 x_train_dataset, y_train_dataset, batch_size, upper_limit, lower_limit,
                 generation_limit=-1):
        self.mlp = mlp
        self.generation_limit = generation_limit
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        self.x_train_dataset = x_train_dataset
        self.y_train_dataset = y_train_dataset
        self.batch_size = batch_size
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.partition_list, self.chromosome_size = self.extract_weight_bias_details()

    def extract_weight_bias_details(self):
        # detail_list[0][0]: partitions of weights in a particular layer [partition_start, partition_end, weight_size]
        # detail_list[0][1]: partitions of bias in a particular layer [partition_start, partition_end, bias_size]
        # detail_list[1]: total size of weights and bias
        # e.g. for a single layer of 10 perceptrons,
        # detail_list = [[[[0, 7840], [7840, 7940]], [[7940, 7950], [7950, 7960]]], 7960]

        detail_list = []
        weight_partition = []
        bias_partition = []
        partition_start = 0
        partition_end = 0

        for weight_layer in self.mlp.coefs_:
            partition_end += weight_layer.size
            weight_partition.append([partition_start, partition_end, weight_layer[0].size])
            partition_start = partition_end

        for bias_layer in self.mlp.intercepts_:
            partition_end += bias_layer.size
            bias_partition.append(([partition_start, partition_end, bias_layer.size]))
            partition_start = partition_end

        detail_list.append([weight_partition, bias_partition])
        detail_list.append(partition_end)

        return detail_list

    def reshape_weights_bias(self, chromosome):
        weights = []
        bias = []

        # reshape the chromosome to weights and bias for the model
        for weight_partition in self.partition_list[0]:
            layer_parameters = []
            sublayer_parameters = []
            for i in range(weight_partition[0], weight_partition[1]):
                sublayer_parameters.append(chromosome[i])
                if (i - weight_partition[0]) % weight_partition[2] == weight_partition[2] - 1:
                    layer_parameters.append(sublayer_parameters)
                    sublayer_parameters = []
            weights.append(np.array(layer_parameters))

        for bias_partition in self.partition_list[1]:
            layer_parameters = []
            for i in range(bias_partition[0], bias_partition[1]):
                layer_parameters.append(chromosome[i])
                if (i - bias_partition[0]) % bias_partition[2] == bias_partition[2] - 1:
                    bias.append(np.array(layer_parameters))

        return [weights, bias]

    def evaluation_function(self, chromosome, x_train, y_train):
        weights_bias = self.reshape_weights_bias(chromosome)

        # update weights and bias in the model
        self.mlp.coefs_ = weights_bias[0]
        self.mlp.intercepts_ = weights_bias[1]

        # calculate the accuracy with input chromosome
        return self.mlp.score(x_train, y_train)

    def generate_population(self):
        population = []

        for i in range(0, self.population_size):
            chromosome = []
            for j in range(0, self.chromosome_size):
                chromosome.append(random.random() * (self.upper_limit - self.lower_limit) + self.lower_limit)
            population.append(chromosome)

        return population

    def selection_function(self, population, x_train, y_train):
        # sort the population in ascending order of score
        sorted_population = sorted(population,
                                   key=lambda chromosome: self.evaluation_function(chromosome, x_train, y_train),
                                   reverse=False)

        # chromosome with higher score has higher rank
        rank_prob = []
        for i in range(1, len(sorted_population) + 1):
            rank_prob.append(i / ((1 + len(sorted_population)) * len(sorted_population) / 2))

        return random.choices(population=sorted_population,
                              weights=rank_prob,
                              k=2)

    def crossover_function(self, parents):
        crossed_chromosome = []

        if random.random() > self.crossover_rate:
            for gene in parents:
                crossed_chromosome.append(gene)
        else:
            index = random.randint(1, self.chromosome_size - 1)
            crossed_chromosome.append(parents[0][:index] + parents[1][index:])
            crossed_chromosome.append(parents[1][:index] + parents[0][index:])
        return crossed_chromosome

    def mutation_function(self, chromosome):
        mutated_chromosome = []
        for i in range(0, self.chromosome_size):
            if random.random() < self.mutation_rate:
                mutated_chromosome.append(random.random() * (self.upper_limit - self.lower_limit) + self.lower_limit)
            else:
                mutated_chromosome.append(chromosome[i])
        return mutated_chromosome

    def elitism_function(self, parent_population, child_population, x_train, y_train):

        # sort parent population in descending order of score
        sorted_parent_population = sorted(parent_population,
                                          key=lambda chromosome: self.evaluation_function(chromosome, x_train, y_train),
                                          reverse=True)
        # sort child population in ascending order of score
        sorted_child_population = sorted(child_population,
                                         key=lambda chromosome: self.evaluation_function(chromosome, x_train, y_train),
                                         reverse=False)

        # replace worst chromosomes in child population by best chromosomes in parent population
        chromosome_count = 0
        while chromosome_count / len(parent_population) < self.elitism_size:
            sorted_child_population[chromosome_count] = sorted_parent_population[chromosome_count]
            chromosome_count += 1

        return sorted_child_population
