#Archivos importados para el algoritmo
import operator
import csv
import funcEval
import numpy as np
import neatGPLS
import neatGPLS_evospace
import init_conf
import os.path
from deap import base
from deap import creator
from deap import tools
from deap import gp
from speciation import getInd_perSpecie
import gp_conf as neat_gp
from my_operators import safe_div, mylog, mypower2, mypower3, mysqrt, myexp

#Imports de evospace
import random, time
import evospace
import xmlrpclib
import jsonrpclib
import cherrypy_server


pset = gp.PrimitiveSet("MAIN", 13)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(np.cos, 1)
pset.addPrimitive(np.sin, 1)
#pset.addPrimitive(myexp, 1)
pset.addPrimitive(mylog, 1)
pset.addPrimitive(mypower2, 1)
pset.addPrimitive(mypower3, 1)
pset.addPrimitive(mysqrt, 1)
pset.addPrimitive(np.tan, 1)
pset.addPrimitive(np.tanh, 1)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
pset.renameArguments(ARG0='x0',ARG1='x1', ARG2='x2', ARG3='x3', ARG4='x4', ARG5='x5', ARG6='x6', ARG7='x7',  ARG8='x8', ARG9='x9',  ARG10='x10',  ARG11='x11',  ARG12='x12')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessTest", base.Fitness, weights=(-1.0,))
creator.create("Individual", neat_gp.PrimitiveTree, fitness=creator.FitnessMin, fitness_test=creator.FitnessTest)

def getToolBox(config):
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=6)
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", init_conf.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # Operator registering
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", neat_gp.cxSubtree)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=6)
    toolbox.register("mutate", neat_gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    #toolbox.register("evaluate", evalSymbReg, points=data_[0])
    #toolbox.register("evaluate_test", evalSymbReg, points=data_[1])

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox


def initialize(config):
    pop = getToolBox(config).population(n=config["POPULATION_SIZE"])
    server = jsonrpclib.Server(config["SERVER"]) #evospace.Population("pop")
    server.initialize()
    #server.initialize(None)
    neat_alg = config["neat_alg"]
    if neat_alg:
        a,b=speciation_init(config, server, pop)
        return a,b
    else:
        sample = [{"chromosome":str(ind), "id":None, "fitness":{"DefaultContext":0.0}, "params":[0.0], "specie":1} for ind in pop]
        init_pop = {'sample_id': 'None' , 'sample':   sample}
        server.put_sample(init_pop)
        server.putZample(init_pop)
        return 1,1

def speciation_init(config,server, pop):
    neat_h=0.15
    num_Specie, specie_list = neatGPLS_evospace.evo_species(pop, neat_h)
    sample = [{"chromosome": str(ind), "id": None, "fitness": {"DefaultContext": 0.0}, "params": [0.0],  "specie":ind.get_specie()} for ind in pop]
    evospace_sample = {'sample_id': 'None', 'sample': sample}
    server.putZample(evospace_sample)
    return num_Specie, specie_list

def speciation(config):
    server = jsonrpclib.Server(config["SERVER"])
    #numsampl=server.getSampleNumber()
    evospace_sample = server.getPopulation()
    pop = [creator.Individual(neat_gp.PrimitiveTree.from_string(cs['chromosome'], pset)) for cs in
           evospace_sample['sample']]
    neat_h=0.15
    num_Specie, specie_list = neatGPLS_evospace.evo_species(pop, neat_h)
    sample = [{"chromosome": str(ind), "id": None, "fitness": {"DefaultContext": 0.0}, "params": [0.0],  "specie":ind.get_specie()} for ind in pop]
    evospace_sample = {'sample_id': 'None', 'sample': sample}
    server.putZample(evospace_sample)
    return num_Specie, specie_list



def evalSymbReg(individual, points, toolbox):
    func = toolbox.compile(expr=individual)
    vector = points[13]
    data_x=np.asarray(points)[:13]
    vector_x=func(*data_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(vector_x, np.ndarray):
            for e in range(len(vector_x)):
                if np.isnan(vector_x[e]) or np.isinf(vector_x[e]):
                    vector_x[e] = 0.
    result = np.sum((vector_x - vector)**2)
    return np.sqrt(result/len(points[0])),

def data_(n_corr,p, problem, name_database,toolbox):
    n_archivot='./data_corridas/%s/test_%d_%d.txt'%(problem,p,n_corr)
    n_archivo='./data_corridas/%s/train_%d_%d.txt'%(problem,p,n_corr)
    if not (os.path.exists(n_archivo) or os.path.exists(n_archivot)):
        direccion = "./data_corridas/%s/%s" % (problem, name_database)
        with open(direccion) as spambase:
            spamReader = csv.reader(spambase,  delimiter=' ', skipinitialspace=True)
            num_c = sum(1 for line in open(direccion))
            num_r = len(next(csv.reader(open(direccion), delimiter=' ', skipinitialspace=True)))
            Matrix = np.empty((num_r, num_c,))
            for row, c in zip(spamReader, range(num_c)):
                for r in range(num_r):
                    try:
                        Matrix[r, c] = row[r]
                    except ValueError:
                        print 'Line {r} is corrupt', r
                        break
        if not os.path.exists(n_archivo):
            long_train=int(len(Matrix.T)*.7)
            data_train1 = random.sample(Matrix.T, long_train)
            np.savetxt(n_archivo, data_train1, delimiter=",", fmt="%s")
        if not os.path.exists(n_archivot):
            long_test=int(len(Matrix.T)*.3)
            data_test1 = random.sample(Matrix.T, long_test)
            np.savetxt(n_archivot, data_test1, delimiter=",", fmt="%s")
    with open(n_archivo) as spambase:
        spamReader = csv.reader(spambase,  delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivo))
        num_r = len(next(csv.reader(open(n_archivo), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, range(num_c)):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print 'Line {r} is corrupt' , r
                    break
        data_train=Matrix[:]
    with open(n_archivot) as spambase:
        spamReader = csv.reader(spambase,  delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivot))
        num_r = len(next(csv.reader(open(n_archivot), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, range(num_c)):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print 'Line {r} is corrupt' , r
                    break
        data_test=Matrix[:]
    #return data_train,data_test
    toolbox.register("evaluate", evalSymbReg, points=data_train, toolbox=toolbox)
    toolbox.register("evaluate_test", evalSymbReg, points=data_test, toolbox=toolbox)

def evolve(sample_num, config):
    toolbox = getToolBox(config)
    start = time.time()
    problem=config["problem"]
    direccion=config["DIRECCION"]
    n_corr=config["n_corr"]
    n_prob=config["n_problem"]
    name_database=config["name_database"]



    #server = evospace.Population("pop")
    server = jsonrpclib.Server(config["SERVER"])

    #evospace_sample = server.get_sample(config["SAMPLE_SIZE"])
    evospace_sample = server.getSample(config["SAMPLE_SIZE"])

    #evospace_specie= server.getSample_specie(config["set_specie"])

    pop = [creator.Individual(neat_gp.PrimitiveTree.from_string(cs['chromosome'], pset)) for cs in evospace_sample['sample']]

    cxpb = config["CXPB"]#0.7  # 0.9
    mutpb = config["MUTPB"]#0.3  # 0.1
    ngen = config["WORKER_GENERATIONS"]#50000
    params = config["PARAMS"]
    neat_cx = config["neat_cx"]
    neat_alg = config["neat_alg"]
    neat_pelit = config["neat_pelit"]
    neat_h = config["neat_h"]
    funcEval.LS_flag = config["LS_FLAG"]
    LS_select = config["LS_SELECT"]
    funcEval.cont_evalp = 0
    num_salto = config["num_salto"]
    cont_evalf = config["cont_evalf"]
    SaveMatrix = config["save_matrix"]
    GenMatrix = config["gen_matrix"]
    version=3
    data_(n_corr, n_prob, problem,name_database,toolbox)

    begin =time.time()
    print "inicio del proceso"

    if neat_alg:
        num_Specie, specie_list = neatGPLS_evospace.evo_species(pop, neat_h)
        for specie in specie_list:
            pop_gpo=getInd_perSpecie(specie, pop)
            pop, log = neatGPLS.neat_GP_LS(pop_gpo, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h, neat_pelit,
                                           funcEval.LS_flag, LS_select, cont_evalf, num_salto, SaveMatrix, GenMatrix, pset,
                                           n_corr, n_prob, params, direccion, problem, stats=None, halloffame=None,
                                           verbose=True)
    else:
        pop, log = neatGPLS.neat_GP_LS(pop, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h, neat_pelit,
                                       funcEval.LS_flag, LS_select, cont_evalf, num_salto, SaveMatrix, GenMatrix, pset,
                                       n_corr, n_prob, params, direccion, problem, testing=config["TESTING"], version=version,
                                       stats=None, halloffame=None, verbose=True)


    putback =  time.time()
    #
    sample = [{"chromosome":str(ind),"id":None, "fitness":{"DefaultContext":[ind.fitness.values[0].item() if isinstance(ind.fitness.values[0], np.float64) else ind.fitness.values[0]]}, "params":[x for x in ind.get_params()]if funcEval.LS_flag else [0.0] } for ind in pop]
    #print sample
    evospace_sample = {'sample_id': 'None', 'sample': sample}
    #evospace_sample['sample'] = sample
    #server.put_sample(evospace_sample)
    server.putZample(evospace_sample)
    best_ind = tools.selBest(pop, 1)[0]
    #
    best = [len(best_ind), sample_num, round(time.time() - start, 2),
                                         round(begin - start, 2), round(putback - begin, 2),
                                         round(time.time() - putback, 2), best_ind]
    return best
    #

def work(params):
    worker_id = params[0][0]
    config = params[0][1]
    server = jsonrpclib.Server(config["SERVER"])
    results = []
    for sample_num in range(config["MAX_SAMPLES"]):
        # if int(server.found(None)):
        #      break
        # else:
        gen_data = evolve(sample_num, config)

            # if gen_data[0]:
            #      server.found_it(None)
        #if server.getSampleNumber()>4:
            #num_Specie, specie_list = neatGPLS_evospace.evo_species(pop, neat_h)
        results.append([worker_id] + gen_data)
    return results

