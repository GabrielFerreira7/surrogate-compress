from copy import Error
import sampling
#import benchmarks
import numpy as np
import surrogate_optimization
import surrogate_selection
import infill_methods
from compressPyTorch import problem_compress 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.util.plotting import plot
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
import time
import copy
from pymoo.factory import get_problem
import pickle
from pymoo.algorithms.moo.ctaea import CTAEA
import sys


if __name__ == "__main__":
    
    # Define original problem
    #problem_compress = get_problem("mw1")

    # Define original problem
    
    param = sys.argv
    op = param[1]

    if param[1] == 'vgg16':
        print("vgg16")
        infile = open('amostragemVGG','rb')
        randomSample = pickle.load(infile)
        infile.close()
        problem_compress = problem_compress.problemCNN([0,2]) #MW1
    elif param[1] == 'resnet50':
        print("resnet")
        infile = open('amostragem2','rb')
        randomSample = pickle.load(infile)
        infile.close()
        problem_compress = problem_compress.problemCNN([1,1]) #MW1


   
    # pontos de referencia
    filename33 = "ref"
    infile = open(filename33,'rb')
    ref_dirs = pickle.load(infile)
    infile.close()
    
    filenameTestes = 'Resultados/'+ param[1] + param[2] + 'T' + param[3] 
    filenameTempo = 'Resultados/tempo' + param[1] + param[2] + 'T' + param[3]
    print(filenameTestes)
    print(filenameTempo)


    #problem = get_problem("zdt3")
    #print(problem.pareto_front())
    
    # Sample
    mask = ["int", "int", "real", "real", "int", "int", "int"]

    '''
    randomSample = sampling.rand2(problem_compress, mask, 50)
    
    filename2 = 'amostragemVGG'
    outfile = open(filename2,'wb')
    pickle.dump(randomSample,outfile)
    outfile.close()
    '''
    
    #randomSample = sampling.rand(problem_compress, 50)#pickle.load(infile)
    '''
    filename2 = 'amostragemVGG'
    infile = open(filename2,'rb')
    randomSample = pickle.load(infile)
    infile.close()
    '''
    # Define surrogate ensemble
    surrogate_ensemble = [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

    # Define Optimizer
    
    
    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=0.9, eta=15),
        "int": get_crossover("int_sbx", prob=0.9, eta=15)
    })

    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=20),
        "int": get_mutation("int_pm", eta=20)
    })
    
    optimizer = CTAEA(
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    
   
    
    
    
    # Define termination criteria

    termination = get_termination("n_gen", 1000)

    # Define infill criteria

    infill_methods_functions = [ # estrategia 
        #infill_methods.distance_search_space,
        infill_methods.distance_objective_space,
        #infill_methods.rand,
        ]

    # Define surrogate selection
    if param[2] == '1': # combinação 1
        surrogate_selection_functions = [
            surrogate_selection.mse,
            #surrogate_selection.mape,
            # surrogate_selection.r2, esse não 
            #surrogate_selection.spearman,
            #surrogate_selection.rand,
        ]

        surrogate_selection_functions2 = [
            surrogate_selection.mse,
            surrogate_selection.mape,
            #surrogate_selection.r2,
            surrogate_selection.spearman,
            surrogate_selection.rand,
        ]
    elif param[2] == '2': # combinação 2
        print("foi certo")
        surrogate_selection_functions = [
            #surrogate_selection.mse,
            surrogate_selection.mape,
            # surrogate_selection.r2, esse não 
            #surrogate_selection.spearman,
            #surrogate_selection.rand,
        ]

        surrogate_selection_functions2 = [
            surrogate_selection.mse,
            surrogate_selection.mape,
            #surrogate_selection.r2,
            surrogate_selection.spearman,
            surrogate_selection.rand,
        ]
    elif param[2] == '3': # combinação 2
        surrogate_selection_functions = [
            #surrogate_selection.mse,
            #surrogate_selection.mape,
            # surrogate_selection.r2, esse não 
            surrogate_selection.spearman,
            #surrogate_selection.rand,
        ]

        surrogate_selection_functions2 = [
            surrogate_selection.mse,
            surrogate_selection.mape,
            #surrogate_selection.r2,
            surrogate_selection.spearman,
            surrogate_selection.rand,
        ]
    elif param[2] == '4': # combinação 4
        surrogate_selection_functions = [
            #surrogate_selection.mse,
            #surrogate_selection.mape,
            # surrogate_selection.r2, esse não 
            #surrogate_selection.spearman,
            surrogate_selection.rand,
        ]

        surrogate_selection_functions2 = [
            surrogate_selection.mse,
            surrogate_selection.mape,
            #surrogate_selection.r2,
            surrogate_selection.spearman,
            surrogate_selection.rand,
        ]


    
    # Optimize 
   # Optimize 
    sampled = []
    tempoT = []
    for j, surrogate_selection_function in enumerate(surrogate_selection_functions):
        for k, surrogate_selection_function2 in enumerate(surrogate_selection_functions2):
        
            for i in range(0,1):
                start = time.time()
                tempo = copy.deepcopy([])
                samples = copy.deepcopy(randomSample)
                surrogate_ensemble = [
                    KNeighborsRegressor(),
                    LGBMRegressor(),
                    XGBRegressor(),
                    RandomForestRegressor(),
                    

                ]
                #print('index das parada -----------------------')
                #print(i)
                #print(j)
                res= surrogate_optimization.optimize(problem_compress,optimizer,termination, op, tempo,
                                    surrogate_ensemble,samples,infill_methods.distance_objective_space,
                                    surrogate_selection_function, surrogate_selection_function2,
                                    n_infill=2,
                                    max_samples=50)
                end = time.time()

                print(samples['X'].shape)
                print(samples['F'].shape)
                print(samples['G'].shape)

                print('Elapsed time: {}'.format(end-start))
                sampled.append(samples)
                tempoT.append(tempo)
            

            print(samples['X'].shape)
        
    
    outfile = open(filenameTestes,'wb')
    pickle.dump(sampled,outfile)
    outfile.close()
    
    
    outfile = open(filenameTempo,'wb')
    pickle.dump(tempoT,outfile)
    outfile.close()
    

