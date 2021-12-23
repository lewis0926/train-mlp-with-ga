import time
from sklearn.model_selection import train_test_split

import data_loader
import genetic_algorithm


def train(epochs, batch_size, mlp, x_train, y_train, resume_train=False):
    start_time = time.time()

    epoch_train_accs = []
    epoch_valid_accs = []

    for epoch in range(epochs):
        print("Epoch ", epoch + 1)

        test_size = 0.2
        x_train_dataset, x_valid_dataset, y_train_dataset, y_valid_dataset = train_test_split(x_train, y_train,
                                                                                              test_size=test_size,
                                                                                              train_size=1 - test_size,
                                                                                              shuffle=True)

        ga = genetic_algorithm.GeneticAlgorithm(mlp=mlp, generation_limit=10, population_size=100, crossover_rate=1,
                                                mutation_rate=0.05, elitism_size=0.1, x_train_dataset=x_train_dataset,
                                                y_train_dataset=y_train_dataset, batch_size=batch_size,
                                                upper_limit=1, lower_limit=-1)

        if epoch == 0:
            population = ga.generate_population()

        train_accs = []

        x_train_loader, y_train_loader = data_loader.data_loader(ga.x_train_dataset, ga.y_train_dataset,
                                                                 ga.batch_size)
        generation = 1
        i = 0
        step_count = ga.batch_size

        while True:
            new_population = []

            while len(new_population) < ga.population_size:
                parent_chromosomes = ga.selection_function(population, x_train_loader[i], y_train_loader[i])
                child_chromosomes = ga.crossover_function(parent_chromosomes)
                child_chromosomes[0] = ga.mutation_function(child_chromosomes[0])
                child_chromosomes[1] = ga.mutation_function(child_chromosomes[1])
                new_population += [child_chromosomes[0], child_chromosomes[1]]

            new_population = ga.elitism_function(population, new_population, x_train_loader[i], y_train_loader[i])
            population = sorted(new_population,
                                key=lambda chromosome: ga.evaluation_function(chromosome, x_train_loader[i],
                                                                              y_train_loader[i]), reverse=True)

            train_acc = ga.evaluation_function(population[0], x_train_loader[i], y_train_loader[i])
            train_accs.append(train_acc)

            print("Generation: {}, Step [{}/{}], Acc: {:.4f}"
                  .format(generation, step_count, len(ga.x_train_dataset), train_acc))

            generation += 1
            if generation % ga.generation_limit == 0:
                i += 1
                if i >= len(x_train_loader):
                    break
                step_count += len(x_train_loader[i])

        epoch_train_accs.append(sum(train_accs) / len(train_accs))

        epoch_valid_acc = mlp.score(x_valid_dataset, y_valid_dataset)
        epoch_valid_accs.append(epoch_valid_acc)
        print("Avg validation acc: {:.4f}"
              .format(epoch_valid_acc))

    end_time = time.time()

    results = {"mlp_model": mlp,
               "train_accs": train_accs,
               "epoch_train_accs": epoch_train_accs,
               "epoch_valid_accs": epoch_valid_accs,
               "computation_time": end_time - start_time}

    return results
