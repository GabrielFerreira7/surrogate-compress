from copy import Error
import sampling
import benchmarks
import numpy as np
import surrogate_optimization
import surrogate_selection
import infill_methods
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
#from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.util.plotting import plot
import time
import copy
import pickle
from pymoo.algorithms.moo.ctaea import CTAEA
import sys

'''
input: problema algoritmos
'''
if __name__ == "__main__":


    param = sys.argv
    # Define original problem
    # problem = benchmarks.mw1()

    # Define original problem
    

    filenameTestes = 'Resultados/'+ param[1] + 'A' + param[2]

    if param[1] == 'zdt1':
        print("problema zdt1")
        problem = benchmarks.zdt1()


    if param[1] == 'zdt2':
        print("problema zdt2")
        problem = benchmarks.zdt2()

    if param[1] == 'zdt3':
        print("problema zdt3")
        problem = benchmarks.zdt3()


    if param[2] == '1':
        print("opção 1")
        surrogate_ensemble = [
                        KNeighborsRegressor(),
                        LGBMRegressor(),
                        XGBRegressor(),
                        RandomForestRegressor(),

        ]

    if param[2] == '2':
        print("opção 2")
        surrogate_ensemble = [
                        KNeighborsRegressor(),
                        LinearRegression(),

        ]

    # Sample
    randomSample = sampling.rand(problem, 50)

    # Define surrogate ensemble
   

    # Define Optimizer
    filename33 = "ref"
    infile = open(filename33,'rb')
    ref_dirs = pickle.load(infile)
    infile.close()

    # Define termination criteria

    termination = get_termination("n_gen", 1000)
    optimizer = CTAEA(ref_dirs=ref_dirs)
   
    

    # Define infill criteria

    infill_methods_functions = [
        #infill_methods.distance_search_space,
        infill_methods.distance_objective_space,
        #infill_methods.rand,
        ]

    # Define surrogate selection

    surrogate_selection_functions = [
        surrogate_selection.mse,
        surrogate_selection.mape,
        # surrogate_selection.r2,
        surrogate_selection.spearman,
        surrogate_selection.rand,
    ]
    # Optimize 
    sampled = []
    for j, surrogate_selection_function in enumerate(surrogate_selection_functions):
        
        for i,infill_method in enumerate(infill_methods_functions):
            for z in range(0,1):
                start = time.time()
                samples = copy.deepcopy(randomSample)
                
                #print('index das parada -----------------------')
                #print(i)
                #print(j)
                res = surrogate_optimization.optimize(problem,optimizer,termination,
                                    surrogate_ensemble,samples,infill_method,
                                    surrogate_selection_function,surrogate_selection.spearman,n_infill=2,
                                    max_samples=50)
                end = time.time()

                #print(samples['X'].shape)
                #print(samples['F'].shape)
                #print(samples['G'].shape)

                #print('Elapsed time: {}'.format(end-start))
                print("funcao ", surrogate_selection_function)
                sampled.append(samples)
            

            #print(samples['X'].shape)
        
    
    outfile = open(filenameTestes,'wb')
    pickle.dump(sampled,outfile)
    outfile.close()
    #print(sampled.shape)



        #plt.plot(sampled[0]['F'][:,0],sampled[0]['F'][:,1],'ob')
        #plt.plot(sampled[1]['F'][:,0],sampled[1]['F'][:,1],'sg')
        #plt.plot(sampled[2]['F'][:,0],sampled[2]['F'][:,1],'xm')
        #plot(problem.pareto_front(), no_fill=True)
        #plt.show()

        #plt.plot(randomSample['F'][:,0],randomSample['F'][:,1],'ob')
        #plot(problem.pareto_front(), no_fill=True)
        #plt.show()