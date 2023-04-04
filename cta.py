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
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
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

	param = sys.argv
	if param[1] == 'vgg16':
		print("vgg16")
		problem_compress = problem_compress.problemCNN([0,2]) #MW1

	elif param[1] == 'resnet50':
		print("resnet")
		problem_compress = problem_compress.problemCNN([1,1]) #MW1


	# pontos de referencia
	filenameTestes = 'Original/'+ param[1] + param[2] + param[3] 
	print(filenameTestes)
	mask = ["int", "int", "real", "real", "int", "int", "int"]

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

	if param[2] == '1':
		ger = 20
		filename33 = "ref5"
		infile = open(filename33,'rb')
		ref_dirs = pickle.load(infile)
		infile.close()

	elif param[2] == '2':
		ger = 10
		filename33 = "ref10"
		infile = open(filename33,'rb')
		ref_dirs = pickle.load(infile)
		infile.close()



	optimizer = CTAEA(
		ref_dirs=ref_dirs,
		sampling=sampling,
		crossover=crossover,
		mutation=mutation,
		eliminate_duplicates=True,
	)


	res2 = minimize(problem_compress,
			   optimizer,
			   ('n_gen', ger),
			   seed=1,
			   verbose=True
			   )

	outfile = open(filenameTestes,'wb')
	pickle.dump(res2,outfile)
	outfile.close()
