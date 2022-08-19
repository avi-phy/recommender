import numpy as np
import math






def xy_independent_model(table):
	prob_table=table/table.sum()
	table_model=np.zeros((table.shape[0],table.shape[1]))
	for i in range(prob_table.shape[0]):
		for j in range(prob_table.shape[1]):
			table_model[i,j]=prob_table[i,:].sum()*prob_table[:,j].sum()*table.sum()
	return(table_model)
	
	
	
def chi_square(table,table_model):
	chi_square=0
	for i in range(table.shape[0]):
		for j in range(table.shape[1]):
			chi_square=chi_square+((table[i,j]-table_model[i,j])**2)/table_model[i,j]
	return(chi_square)

def likelihood_ratio(table,table_model):
	likelihood_ratio=0
	for i in range(table.shape[0]):
		for j in range(table.shape[1]):
			likelihood_ratio=likelihood_ratio+\
			2*table[i,j]*math.log(table[i,j]/table_model[i,j])
	return(likelihood_ratio)
