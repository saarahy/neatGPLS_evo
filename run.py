import evoworker_gp as evoworker
import cloud, time, yaml


config = yaml.load(open("conf/conf.yaml"))

start = time.time()

#init_job = cloud.call(evoworker.initialize, config=config,  _type='s1', _env="deap")

params = [(i, config) for i in range(1)]


# jids = cloud.map(onemax.work, params, _type='s1',_depends_on= init_job )
# results_list = cloud.result(jids)
#
# print time.time()-start
#
# for r in results_list:
#     for a in r:
#         print a

evoworker.initialize(config)
#a,b=evoworker.speciation(config)

problem='Housing'
num_p=3
config["n_problem"]=num_p
config["problem"] = problem

best = open('./Timing/%s/time_%d.txt' % (problem, num_p), 'w')

for ci in range(17,18):
    print ci
    config["n_corr"]=ci
    config["set_specie"] = 3

    with open("conf/conf.yaml","w") as f:
        yaml.dump(config, f)

    params = [(i, config) for i in range(1)]
    result=evoworker.work(params)
    if result:
        best.write('\n%s;%s;%s'% (ci, result[0][2], result[0][3]))

        print 'finished'

