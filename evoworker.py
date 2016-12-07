
import random, time
import xmlrpclib

from deap import base
from deap import creator
from deap import tools
import evospace
import jsonrpclib
import cherrypy_server

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def getToolBox(config):
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool,config["CHROMOSOME_LENGTH"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Operator registering
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoints)
    toolbox.register("mutate", tools.mutFlipBit, indpb = config["MUTATION_FLIP_PB"])
    toolbox.register("select", tools.selTournament, tournsize=config["TOURNAMENT_SIZE"])
    return toolbox


def initialize(config):
    pop = getToolBox(config).population(n=config["POPULATION_SIZE"])
    print pop
    server = evospace.Population("pop")#jsonrpclib.Server(config["SERVER"])
    server.initialize()
    #server.initialize(None)

    sample = [{"chromosome":ind[:], "id":None, "fitness":{"DefaultContext":0.0}} for ind in pop]
    init_pop = {'sample_id': 'None' , 'sample':   sample}
    server.put_sample(init_pop)
    #server.putSample(init_pop)


def evalOneMax(individual):
    return sum(individual),


def evolve(sample_num, config):
    #random.seed(64)

    toolbox = getToolBox(config)
    start = time.time()
    server = evospace.Population("pop")
    #server = jsonrpclib.Server(config["SERVER"])

    evospace_sample = server.get_sample(config["SAMPLE_SIZE"])
    #evospace_sample = server.getSample(config["SAMPLE_SIZE"])
    pop = [ creator.Individual(cs['chromosome']) for cs in evospace_sample['sample']]

    begin =   time.time()
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    total_evals = len(pop)
    best_first   = None
    # Begin the evolution

    for g in range(config["WORKER_GENERATIONS"]):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config["CXPB"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < config["MUTPB"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        total_evals+=len(invalid_ind)
        #print "  Evaluated %i individuals" % len(invalid_ind),

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        #length = len(pop)
        #mean = sum(fits) / length
        #sum2 = sum(x*x for x in fits)
        #std = abs(sum2 / length - mean**2)**0.5

        best = max(fits)
        if not best_first:
            best_first = best

        if best >= config["CHROMOSOME_LENGTH"]:
            break

        #print  "  Min %s" % min(fits) + "  Max %s" % max(fits)+ "  Avg %s" % mean + "  Std %s" % std

    print "-- End of (successful) evolution --"

    putback =  time.time()

    sample = [ {"chromosome":ind[:],"id":None,
            "fitness":{"DefaultContext":ind.fitness.values[0]} }
                                                            for ind in pop]
    evospace_sample['sample'] = sample
    server.put_sample(evospace_sample)
    #server.putSample(evospace_sample)
    best_ind = tools.selBest(pop, 1)[0]

    return best >= config["CHROMOSOME_LENGTH"] , [config["CHROMOSOME_LENGTH"],best, sample_num, round(time.time() - start, 2),
                                        round(begin - start, 2), round(putback - begin, 2),
                                        round(time.time() - putback, 2), total_evals, best_first, best_ind]


def work(params):
    worker_id = params[0][0]
    config = params[0][1]
    #server = jsonrpclib.Server(config["SERVER"])
    server = evospace.Population("pop")
    results = []
    for sample_num in range(config["MAX_SAMPLES"]):
        #if int(server.found(None)):
        if int(server.found()):
            print server.found()
            break
        else:
            gen_data = evolve(sample_num, config)
            if gen_data[0]:
                server.found()
            results.append([worker_id] + gen_data[1])
    return results

