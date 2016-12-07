import random
import funcEval
import numpy as np
import copy
from deap import tools
from neat_operators import neatGP
from speciation import ind_specie, species, specie_parents_child, count_species, list_species
from fitness_sharing import SpeciesPunishment
from ParentSelection import p_selection
from tree_subt import add_subt, add_subt_cf
from tree2func import tree2f
from treesize_h import trees_h, specie_h, best_specie, best_pop_ls, all_pop, trees_h_wo, ls_bestset, ls_random, ls_randbestset
from my_operators import avg_nodes


def varOr(population, toolbox, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
        "probabilities must be smaller or equal to 1.0.")

    new_pop = [toolbox.clone(ind) for ind in population]
    offspring = []
    for i in range(1, len(new_pop), 2):
        new_pop[i-1].off_cx_set(0), new_pop[i].off_cx_set(0)
        if random.random() < cxpb and len(ind)>1:
            new_pop[i-1].off_cx_set(1)
            new_pop[i].off_cx_set(1)
            offspring1, offspring2 = toolbox.mate(new_pop[i-1], new_pop[i])
            del offspring1.fitness.values
            del offspring2.fitness.values
            offspring1.bestspecie_set(0), offspring2.bestspecie_set(0)
            offspring1.LS_applied_set(0), offspring2.LS_applied_set(0)
            offspring1.LS_fitness_set(None), offspring2.LS_fitness_set(None)
            offspring1.off_cx_set(1), offspring2.off_cx_set(1)
            # sizep = len(offspring1)+2
            # param_ones = np.ones(sizep)
            # param_ones[0] = 0
            # offspring1.params_set(param_ones)
            # sizep = len(offspring2)+2
            # param_ones = np.ones(sizep)
            # param_ones[0] = 0
            # offspring2.params_set(param_ones)
            offspring.append(offspring1)
            offspring.append(offspring2)
    for i in range(len(new_pop)):
        if new_pop[i].off_cx_get() != 1:
            if random.random() < (cxpb+mutpb):  # Apply mutation
                offspring1, = toolbox.mutate(new_pop[i])
                del offspring1.fitness.values
                offspring1.bestspecie_set(0)
                offspring1.LS_applied_set(0)
                offspring1.LS_fitness_set(None)
                offspring1.off_mut_set(1)
                # sizep = len(offspring1)+2
                # param_ones = np.ones(sizep)
                # param_ones[0] = 0
                # offspring1.params_set(param_ones)
                offspring.append(offspring1)

    if len(offspring) < len(population):
        for i in range(len(new_pop)):
            if new_pop[i].off_mut_get() != 1 and new_pop[i].off_cx_get() != 1:
                offspring1 = copy.deepcopy(new_pop[i])
                offspring.append(offspring1)

    return offspring

def evo_species(population, neat_h):
    species(population, neat_h)
    num_Species=count_species(population)
    specie_list=list_species(population)
    return  num_Species, specie_list

def evo_neat_GP_LS(population, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h,neat_pelit, LS_flag, LS_select, cont_evalf, num_salto, SaveMatrix, GenMatrix, pset,n_corr, num_p, params, direccion, problem,stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param neat_alg: wheter or not to use species stuff.
    :param neat_cx: wheter or not to use neatGP cx
    :param neat_h: indicate the distance allowed between each specie
    :param neat_pelit: probability of being elitist, it's used in the neat cx and mutation
    :param LS_flag: wheter or not to use LocalSearchGP
    :param LS_select: indicate the kind of selection to use the LSGP on the population.
    :param cont_evalf: contador maximo del numero de evaluaciones
    :param n_corr: run number just to wirte the txt file
    :param p: problem number just to wirte the txt file
    :param params:indicate the params for the fitness sharing, the diffetent
                    options are:
                    -DontPenalize(str): 'best_specie' or 'best_of_each_specie'
                    -Penalization_method(int):
                        1.without penalization
                        2.penalization fitness sharing
                        3.new penalization
                    -ShareFitness(str): 'yes' or 'no'
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.

    It uses :math:`\lambda = \kappa = \mu` and goes as follow.
    It first initializes the population (:math:`P(0)`) by evaluating
    every individual presenting an invalid fitness. Then, it enters the
    evolution loop that begins by the selection of the :math:`P(g+1)`
    population. Then the crossover operator is applied on a proportion of
    :math:`P(g+1)` according to the *cxpb* probability, the resulting and the
    untouched individuals are placed in :math:`P'(g+1)`. Thereafter, a
    proportion of :math:`P'(g+1)`, determined by *mutpb*, is
    mutated and placed in :math:`P''(g+1)`, the untouched individuals are
    transferred :math:`P''(g+1)`. Finally, those new individuals are evaluated
    and the evolution loop continues until *ngen* generations are completed.
    Briefly, the operators are applied in the following order ::

        evaluate(population)
        for i in range(ngen):
            offspring = select(population)
            offspring = mate(offspring)
            offspring = mutate(offspring)
            evaluate(offspring)
            population = offspring

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_file = open('./Results/%s/pop_file_%d_%d.txt' % (problem, num_p, n_corr), 'a')

    if SaveMatrix:  # Saving data in matrix
        num_r = 9
        if GenMatrix:
            num_salto=1
            num_c=ngen+1
            Matrix= np.empty((num_c, num_r,))
            vector = np.arange(0, num_c, num_salto)
        else:
            num_c = (cont_evalf/num_salto) + 1
            Matrix = np.empty((num_c, num_r,))
            vector = np.arange(1, cont_evalf+num_salto, num_salto)
        for i in range(len(vector)):
            Matrix[i, 0] = vector[i]
            #num_r-1
        Matrix[:, 6] = 0.

    #Creation of the species
    # if neat_alg:
    #     species(population,neat_h)
    #     ind_specie(population)

    if funcEval.LS_flag:
        for ind in population:
            sizep = len(ind)+2
            param_ones = np.ones(sizep)
            param_ones[0] = 0
            ind.params_set(param_ones)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        funcEval.cont_evalp += 1
        ind.fitness.values = fit

    best = open('./Results/%s/bestind_%d_%d.txt' % (problem, num_p, n_corr), 'a')  # save data

    best_ind = best_pop(population)  # best individual of the population
    fitnesst_best = toolbox.map(toolbox.evaluate_test, [best_ind])
    best_ind.fitness_test.values = fitnesst_best[0]
    best.write('\n%s;%s;%s;%s;%s;%s' % (0, funcEval.cont_evalp, best_ind.fitness_test.values[0], best_ind.fitness.values[0], len(best_ind), avg_nodes(population)))
    data_pop=avg_nodes(population)
    if SaveMatrix:
        idx = 0
        Matrix[idx, 1] = best_ind.fitness.values[0]
        Matrix[idx, 2] = best_ind.fitness_test.values[0]
        Matrix[idx, 3] = len(best_ind)
        Matrix[idx, 4] = data_pop[0]
        Matrix[idx, 5] = 0.
        Matrix[idx, 6] = 1  # just an id to know if the current row is full
        Matrix[idx, 7] = data_pop[1]  # max size
        Matrix[idx, 8] = data_pop[2]  # min size

        np.savetxt('./Matrix/%s/idx_%d_%d.txt' % (problem,num_p, n_corr), Matrix, delimiter=",", fmt="%s")

    if neat_alg:
        SpeciesPunishment(population,params,neat_h)

    out = open('./Results/%s/bestind_str_%d_%d.txt' % (problem, num_p, n_corr), 'a')

    if funcEval.LS_flag == 1:
        strg = best_ind.__str__()
        l_strg = add_subt_cf(strg, args=[])
        c = tree2f()
        cd = c.convert(l_strg)
        out.write('\n%s;%s;%s;%s;%s;%s' % (0, len(best_ind), best_ind.LS_applied_get(), best_ind.get_params(), cd, best_ind))
    else:
        out.write('\n%s;%s;%s' % (0, len(best_ind), best_ind))

    for ind in population:
        pop_file.write('\n%s;%s'%(ind.fitness.values[0], ind))

    ls_type = ''
    if LS_select == 1:
        ls_type = 'LSHS'
    elif LS_select == 2:
        ls_type = 'Best-Sp'
    elif LS_select == 3:
        ls_type = 'LSHS-Sp'
    elif LS_select == 4:
        ls_type = 'Best-Pop'
    elif LS_select == 5:
        ls_type = 'All-Pop'
    elif LS_select == 6:
        ls_type = 'LSHS-test'
    elif LS_select == 7:
        ls_type = 'Best set'
    elif LS_select == 8:
        ls_type = 'Random set'
    elif LS_select == 9:
        ls_type = "Best-Random set"

    print '---- Generation %d -----' % (0)
    print 'Problem: ', problem
    print 'Problem No.: ', num_p
    print 'Run No.: ', n_corr
    print 'neat-GP:', neat_alg
    print 'neat-cx:', neat_cx
    print 'Local Search:', funcEval.LS_flag
    if funcEval.LS_flag:
        print 'Local Search Heuristic: %s (%s)' % (LS_select,ls_type)
    print 'Best Ind.:', best_ind
    print 'Best Fitness:', best_ind.fitness.values[0]
    print 'Test fitness:',best_ind.fitness_test.values[0]
    print 'Avg Nodes:', avg_nodes(population)
    print 'Evaluations: ', funcEval.cont_evalp

    # Begin the generational process
    for gen in range(1, ngen+1):

        if funcEval.cont_evalp > cont_evalf:
            break

        print '---- Generation %d -----' % (gen)
        print 'Problem: ', problem
        print 'Problem No.: ', num_p
        print 'Run No.: ', n_corr
        print 'neat-GP:', neat_alg
        print 'neat-cx:', neat_cx
        print 'Local Search:', funcEval.LS_flag
        if funcEval.LS_flag:
            print 'Local Search Heuristic: %s (%s)' % (LS_select, ls_type)

        best_ind = copy.deepcopy(best_pop(population))
        if neat_alg:
            parents = p_selection(population, gen)
        else:
            parents = toolbox.select(population, len(population))

        if neat_cx:
            n = len(parents)
            mut = 1
            cx = 1
            offspring = neatGP(toolbox, parents, cxpb, mutpb, n, mut, cx, neat_pelit)
        else:
            offspring = varOr(parents, toolbox, cxpb, mutpb)

        if neat_alg:
            specie_parents_child(parents,offspring, neat_h)
            offspring[:] = parents+offspring
            ind_specie(offspring)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                funcEval.cont_evalp += 1
                ind.fitness.values = fit
        else:
            invalid_ind = [ind for ind in offspring]
            if funcEval.LS_flag:
                new_invalid_ind = []
                for ind in invalid_ind:
                    strg = ind.__str__()
                    l_strg = add_subt(strg, ind)
                    c = tree2f()
                    cd = c.convert(l_strg)
                    new_invalid_ind.append(cd)
                fitness_ls = toolbox.map(toolbox.evaluate, new_invalid_ind)
                for ind, ls_fit in zip(invalid_ind, fitness_ls):
                    funcEval.cont_evalp += 1
                    ind.fitness.values = ls_fit
            else:
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    funcEval.cont_evalp += 1
                    ind.fitness.values = fit

            orderbyfit = sorted(offspring, key=lambda ind:ind.fitness.values)
            print len(orderbyfit),len(best_ind)

            if best_ind.fitness.values[0] <= orderbyfit[0].fitness.values[0]:
                offspring[:] = [best_ind]+orderbyfit[:len(population)-1]

        if neat_alg:
            SpeciesPunishment(offspring, params, neat_h)

        population[:] = offspring  # population update

        cond_ind = 0
        cont_better=0
        if funcEval.LS_flag:
            for ind in population:
                ind.LS_applied_set(0)

            if   LS_select == 1:
                trees_h(population, num_p, n_corr,  pset, direccion)
            elif LS_select == 2:
                best_specie(population, num_p, n_corr, pset, direccion)
            elif LS_select == 3:
                specie_h(population, num_p, n_corr, pset, direccion)
            elif LS_select == 4:
                best_pop_ls(population, num_p, n_corr, pset, direccion)
            elif LS_select == 5:
                all_pop(population, num_p, n_corr, pset, direccion)
            elif LS_select == 6:
                trees_h_wo(population, num_p, n_corr, pset, direccion)
            elif LS_select == 7:
                ls_bestset(population, num_p, n_corr, pset, direccion)
            elif LS_select == 8:
                ls_random(population, num_p, n_corr, pset, direccion)
            elif LS_select == 9:
                ls_randbestset(population, num_p, n_corr, pset, direccion)
            #
            invalid_ind = [ind for ind in population]
            new_invalid_ind = []
            for ind in population:
                strg = ind.__str__()
                l_strg = add_subt(strg, ind)
                c = tree2f()
                cd = c.convert(l_strg)
                new_invalid_ind.append(cd)
            fitness_ls = toolbox.map(toolbox.evaluate, new_invalid_ind)
            print 'Fitness comp.:',
            for ind, ls_fit in zip(invalid_ind, fitness_ls):
                if ind.LS_applied_get() == 1:
                    cond_ind += 1
                    if ind.fitness.values[0] < ls_fit:
                        print '-',
                    elif ind.fitness.values[0] > ls_fit:
                        cont_better += 1
                        print '+',
                    elif ind.fitness.values[0] == ls_fit:
                        print '=',
                funcEval.cont_evalp += 1
                ind.fitness.values = ls_fit
            print ''

            pop_file.write('\n----------------------------------------%s'%(gen))
            for ind in population:
                pop_file.write('\n%s;%s;%s;%s'%(ind.LS_applied_get(),ind.fitness.values[0], ind, [x for x in ind.get_params()]))
        else:
            pop_file.write('\n----------------------------------------')
            for ind in population:
                pop_file.write('\n%s;%s;%s;%s;%s;%s'%(ind.LS_applied_get(),ind.LS_story_get(),ind.off_cx_get(),ind.off_mut_get(),ind.fitness.values[0], ind))

        best_ind = best_pop(population)
        if funcEval.LS_flag:
            strg = best_ind.__str__()
            l_strg = add_subt(strg, best_ind)
            c = tree2f()
            cd = c.convert(l_strg)
            new_invalid_ind.append(cd)
            fit_best = toolbox.map(toolbox.evaluate_test, [cd])
            best_ind.fitness_test.values = fit_best[0]
            best.write('\n%s;%s;%s;%s;%s;%s;%s' % (gen, funcEval.cont_evalp,  best_ind.fitness.values[0], best_ind.LS_fitness_get(), best_ind.fitness_test.values[0], len(best_ind), avg_nodes(population)))
            out.write('\n%s;%s;%s;%s;%s;%s' % (gen, len(best_ind), best_ind.LS_applied_get(), best_ind.get_params(), cd, best_ind))
        else:
            fitnesses_test = toolbox.map(toolbox.evaluate_test, [best_ind])
            best_ind.fitness_test.values = fitnesses_test[0]
            best.write('\n%s;%s;%s;%s;%s;%s' % (gen, funcEval.cont_evalp, best_ind.fitness_test.values[0], best_ind.fitness.values[0], len(best_ind), avg_nodes(population)))
            out.write('\n%s;%s;%s' % (gen, len(best_ind), best_ind))

        if funcEval.LS_flag:
            print 'Num. LS:', cond_ind
            print 'Ind. Improvement:', cont_better
            print 'Best Ind. LS:', best_ind.LS_applied_get()

        print 'Best Ind.:', best_ind
        print 'Best Fitness:', best_ind.fitness.values[0]
        print 'Test fitness:',best_ind.fitness_test.values[0]
        print 'Avg Nodes:', avg_nodes(population)
        print 'Evaluations: ', funcEval.cont_evalp

        if SaveMatrix:
            data_pop=avg_nodes(population)
            if GenMatrix:
                idx_aux = np.searchsorted(Matrix[:, 0], gen)
                Matrix[idx_aux, 1] = best_ind.fitness.values[0]
                Matrix[idx_aux, 2] = best_ind.fitness_test.values[0]
                Matrix[idx_aux, 3] = len(best_ind)
                Matrix[idx_aux, 4] = data_pop[0]
                Matrix[idx_aux, 5] = gen
                Matrix[idx_aux, 6] = 1
                Matrix[idx_aux, 7] = data_pop[1]  # max nodes
                Matrix[idx_aux, 8] = data_pop[2]  # min nodes
            else:
                if funcEval.cont_evalp >= cont_evalf:
                    num_c -= 1
                    idx_aux=num_c
                    Matrix[num_c, 1] = best_ind.fitness.values[0]
                    Matrix[num_c, 2] = best_ind.fitness_test.values[0]
                    Matrix[num_c, 3] = len(best_ind)
                    Matrix[num_c, 4] = data_pop[0]
                    Matrix[num_c, 5] = gen
                    Matrix[num_c, 6] = 1
                    Matrix[num_c, 7] = data_pop[1]  #max_nodes
                    Matrix[num_c, 8] = data_pop[2]  #min nodes
                else:
                    idx_aux = np.searchsorted(Matrix[:, 0], funcEval.cont_evalp)
                    Matrix[idx_aux, 1] = best_ind.fitness.values[0]
                    Matrix[idx_aux, 2] = best_ind.fitness_test.values[0]
                    Matrix[idx_aux, 3] = len(best_ind)
                    Matrix[idx_aux, 4] = data_pop[0]
                    Matrix[idx_aux, 5] = gen
                    Matrix[idx_aux, 6] = 1
                    Matrix[idx_aux, 7] = data_pop[1]  #max nodes
                    Matrix[idx_aux, 8] = data_pop[2]  #min nodes

                id_it = idx_aux-1
                id_beg = 0
                flag = True
                flag2 = False
                while flag:
                    if Matrix[id_it, 6] == 0:
                        id_it -= 1
                        flag2 = True
                    else:
                        id_beg = id_it
                        flag = False
                if flag2:
                    x = Matrix[id_beg, 1:8]
                    Matrix[id_beg:idx_aux, 1:] = Matrix[id_beg, 1:]

            np.savetxt('./Matrix/%s/idx_%d_%d.txt' % (problem, num_p, n_corr), Matrix, delimiter=",", fmt="%s")

    return population, logbook


def best_pop(population):
    orderbyfit=list()
    orderbyfit=sorted(population, key=lambda ind:ind.fitness.values)
    return orderbyfit[0]