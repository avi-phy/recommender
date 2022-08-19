import numpy as np
import cupy as cp
from numba import cuda
from datetime import datetime
from datetime import timedelta
import time
import matplotlib.pyplot as plt


def matmul(left,right,blockspergrid=(125,125),threadsperblock=(16,16)):
    C=cp.zeros((left.shape[0],right.shape[1]))
    A=left.copy()
    B=right.copy()
    gpu_matmul[blockspergrid,threadsperblock](A,B,C)
    
    out=cp.asarray(C)
    del A,B,C
    return(out)

@cuda.jit
def gpu_matmul(A,B,C):
    """Perform square matrix multiplication of C = A * B
       """
    i,j = cuda.grid(2)  
      
    if i < C.shape[0] and j < C.shape[1]:
        tmp=0
        for k in range(A.shape[1]):
            tmp =tmp + A[i,k] * B[k,j]

        C[i,j] = tmp  
        
        
        
def multiply(left,right,blockspergrid=(125,125),threadsperblock=(16,16)):
    C=cp.zeros((left.shape[0],right.shape[1]))
    A=left.copy()
    B=right.copy()
    gpu_multiply[blockspergrid,threadsperblock](A,B,C)
    out=cp.asarray(C)
    del A,B,C
    return(out)

def scalar_multiply(left,right,blockspergrid=(125,125),threadsperblock=(16,16)):
    C=cp.zeros((right.shape[0],right.shape[1]))
    A=cp.full((right.shape[0],right.shape[1]),left)
    B=right.copy()
    gpu_multiply[blockspergrid,threadsperblock](A,B,C)
    out=cp.asarray(C)
    del A,B,C
    return(out)

@cuda.jit
def gpu_multiply(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)  
      
    if i < C.shape[0] and j < C.shape[1]:

        C[i, j] = A[i,j]*B[i,j]
        
        
def divide(left,right,blockspergrid=(125,125),threadsperblock=(16,16)):
    C=cp.zeros((left.shape[0],right.shape[1]))
    A=left.copy()
    B=right.copy()
    gpu_divide[blockspergrid,threadsperblock](A,B,C)
    out=cp.asarray(C)
    del A,B,C
    return(out)

@cuda.jit
def gpu_divide(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)  
      
    if i < C.shape[0] and j < C.shape[1]:

        C[i, j] = A[i,j]/B[i,j]


def sub(left,right,blockspergrid=(125,125),threadsperblock=(16,16)):
    C=cp.zeros((left.shape[0],right.shape[1]))
    A=left.copy()
    B=right.copy()
    gpu_sub[blockspergrid,threadsperblock](A,B,C)
    out=cp.asarray(C)
    del A,B,C
    return(out)


@cuda.jit
def gpu_sub(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)  
      
    if i < C.shape[0] and j < C.shape[1]:

        C[i, j] = A[i,j]-B[i,j]            
        
def add(left,right,blockspergrid=(125,125),threadsperblock=(16,16)):
    C=cp.zeros((left.shape[0],right.shape[1]))
    A=left.copy()
    B=right.copy()
    gpu_add[blockspergrid,threadsperblock](A,B,C)
    out=cp.asarray(C)
    del A,B,C
    return(out)


@cuda.jit
def gpu_add(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)  
      
    if i < C.shape[0] and j < C.shape[1]:

        C[i, j] = A[i,j]+B[i,j]       
        
        
def sparser(matrix,fill=(0,1),blockspergrid=(125,125),threadsperblock=(16,16)):
    A=matrix.copy()
    B=fill[0]
    C=fill[1]
    gpu_sparser[blockspergrid,threadsperblock](A,B,C)
    out=cp.asarray(A)
    del A,B,C
    return(out)



@cuda.jit
def gpu_sparser(A,B,C):
    '''Finds the location of the non zero elements of a spase matrix and saves it as a list of tuples'''
    row,column=cuda.grid(2)
    if row<A.shape[0] and column<A.shape[1]: 
        if A[row,column]!=0:
            A[row,column]=C
        else:
            A[row,column]=B       
        
def transpose(matrix,blockspergrid=(125,125),threadsperblock=(16,16)):
    B=cp.zeros((matrix.shape[1],matrix.shape[0]))
    A=matrix.copy()
    gpu_transpose[blockspergrid,threadsperblock](A,B)
    out=cp.asarray(B)
    del A,B
    return(out)
    
    
    
    
@cuda.jit
def gpu_transpose(matrix_in,matrix_out):
    '''Finds the location of the non zero elements of a spase matrix and saves it as a list of tuples'''
    row,column=cuda.grid(2)
    if row<matrix_out.shape[0] and column<matrix_out.shape[1]:
            matrix_out[row,column]=matrix_in[column,row]     
        
        
def sign(left,right,blockspergrid=(125,125),threadsperblock=(16,16)):
    C=cp.zeros((right.shape[0],right.shape[1]))
    A=left
    B=right.copy()
    gpu_sign[blockspergrid,threadsperblock](A,B,C)
    out=cp.asarray(C)
    del A,B,C
    return(out)

@cuda.jit
def gpu_sign(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)  
      
    if i < C.shape[0] and j < C.shape[1]:
        if A-B[i,j]<=0:
            C[i, j] = -1
        else:
            C[i,j]=1      

def sparse_rep(matrix):
    A=cp.asnumpy(matrix)
    
    out=[]
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j]!=0:
                out.append((i,j))
    return(out)

def norm(A):
    tot=0
    for i in range(len(A)):
        tot=tot+A[i]*A[i]
    return(tot)             
        
def regulated_norm(A,X,Y,P,weight,gravity,list_non_zero):
    '''Returns the regulated norm'''
    num=len(list_non_zero)
    Z=matmul(X,Y)
    Project=sparser(A)
    Proj_AZ=multiply(Project,sub(A,Z))
    err1=(1/num)*cp.trace(matmul(Proj_AZ,transpose(sub(A,Z))))
    
    
    err2=(weight/A.shape[0])*cp.trace(matmul(X,transpose(X)))\
    +(weight/A.shape[1])*cp.trace(matmul(Y,transpose(Y)))
    
    
    err3=(gravity/(A.shape[0]*A.shape[1]))*cp.trace(matmul(Z,transpose(Z)))

    return(cp.asnumpy(err1),cp.asnumpy(err2),cp.asnumpy(err3),Z) 
    
    
def fact_reg_sparse(learn_rate,steps,rating,weight,gravity,u=None,v=None,choice=0,dim=0,blockspergrid=(125,125),threadsperblock=(16,16)):
    
    '''Matrix factorization of the form R=UV with regulated square norm'''
    R=cp.asarray(rating)

    errsq=[]
    errg=[]
    errreg=[]
    c1=weight/R.shape[0]
    c2=weight/R.shape[1]
    c3=gravity/(R.shape[0]*R.shape[1])
    start_time=time.perf_counter()

    error=[]
    p=sparser(R)
    list_non_zero=sparse_rep(R)
    omega=len(list_non_zero)

    
    if choice==0:
        U=cp.random.normal(0,0.5,(R.shape[0],int(dim)))
        V=cp.random.normal(0,0.5,(int(dim),R.shape[1]))
        for l in range(steps):
            err=regulated_norm(R,U,V,p,weight,gravity,list_non_zero)
            errsq.append(err[0])
            errg.append(err[2])
            errreg.append(err[1])
            print(("Factorising Matrix-------steps:{} sparse square error:{} gravity:{} regularisation:{}").format(l,err[0],err[1],err[2]),end='\r',flush=True)
            X=err[3]
            Y=multiply(p,sub(R,X))
            Uchange=scalar_multiply(-(2/omega),matmul(Y,transpose(V)))\
             +scalar_multiply(2,scalar_multiply(c3,matmul(X,transpose(V))))+scalar_multiply(c1,U)
            #print(scalar_multiply(learn_rate,Uchange))
            U=sub(U,scalar_multiply(learn_rate,Uchange))

            X=err[3]
            Y=multiply(p,sub(R,X))
            Vchange=scalar_multiply(-(2/omega),matmul(transpose(U),Y))\
                +scalar_multiply(2*c3,matmul(transpose(U),X))+scalar_multiply(c2,V)
            V=sub(V,scalar_multiply(learn_rate,Vchange))
            
            
        del Y,Vchange,Uchange
        
        
    elif choice==1:
        V=cp.asarray(v)
        U=cp.random.normal(0,0.5,(R.shape[0],V.shape[0]))
        for l in range(steps):
            err=regulated_norm(R,U,V,p,weight,gravity,list_non_zero)
            errsq.append(err[0])
            errg.append(err[2])
            errreg.append(err[1])
            print(("Factorising Matrix-------steps:{} sparse square error:{} gravity:{} regularisation:{}").format(l,err[0],err[1],err[2]),end='\r',flush=True)
            X=err[3]
            Y=multiply(p,sub(R,X))
            
            

            Uchange=scalar_multiply(-(2/omega),matmul(Y,transpose(V)))\
             +scalar_multiply(2*c3,matmul(X,transpose(V)))+scalar_multiply(c1,U)
            
            
            U=sub(U,scalar_multiply(learn_rate,Uchange))
            
        del Y,Uchange
   
            
    elif choice==2:
        U=cp.asarray(u)
        V=cp.random.normal(0,0.5,(U.shape[1],R.shape[1]))
        for l in range(steps):
            err=regulated_norm(R,U,V,p,weight,gravity,list_non_zero)
            errsq.append(err[0])
            errg.append(err[2])
            errreg.append(err[1])
            print(("Factorising Matrix-------steps:{} sparse square error:{} gravity:{} regularisation:{}").format(l,err[0],err[1],err[2]),end='\r',flush=True)
            X=err[3]
            Y=multiply(p,sub(R,X))

            
            Vchange=scalar_multiply(-(2/omega),matmul(transpose(U),Y))\
                +scalar_multiply(2*c3,matmul(transpose(U),X))+scalar_multiply(c2,V)
            V=sub(V,scalar_multiply(learn_rate,Vchange))
            
        del Y,Vchange
   
            
    err=regulated_norm(R,U,V,p,weight,gravity,list_non_zero)
    errsq.append(err[0])
    errg.append(err[2])
    errreg.append(err[1])    
    stop_time=time.perf_counter()
    u=cp.asnumpy(U)
    v=cp.asnumpy(V)
    del R,U,V,X,list_non_zero,p
    print(("sparse square error:{} gravity:{} regularisation:{}").format(err[0],err[1],err[2]))
    print(("Finished in {} secs").format(stop_time-start_time))
    plt.plot(errsq,label="sparse square error")
    plt.plot(errg,label="gravity",)
    plt.plot(errreg,label="regularisation")
    plt.legend()
    return(err[3],u,v)
    
    
    
    
    
    
    
    
    
#def product_space(features):
#    if len(features)>1:
#        X=product_space(features[1:]).copy()
#        #print(X)
#        Y=features[0]
#        #print(Y)
#        new_space=[]
#        #print(new_space)
#        for y in Y:
#            for x in X:
#                new_space.append([y]+x)
                #print(new_space)
#        return(new_space)
#    else:
#        new_space=[]
#        for x in features[0]:
#            new_space.append([x])
#        return(new_space)
        
        
        
#def feature_matrix_construct(features,columns_index,df):
#    basis_feature=product_space(features)
#    #print(user_feature)
#    feature_matrix=[]
#    for i in range(len(df)):
#        print(i,end="\r")
#        temp=[]
#        for x in basis_feature:
#            check_left=[df.iloc[i,j] for j in columns_index]
#            check_right=[x[k] for k in range(len(features))]
            #print(check_left,check_right)
#            if check_left==check_right:
                #print("yes")
#                temp.append(int(1))
#            else:
#                #print("no")
#                temp.append(int(0))
        #print(temp)
#        feature_matrix.append(temp)
#    feature_matrix=np.array(feature_matrix)
#    return(feature_matrix)


def feature_matrix_construct(features,columns_index,df,drop=False):
    
    temp=[]
    for i in range(len(features)):
        print(str(i)+"...done")
        temp_feature_matrix=np.zeros((len(df),len(features[i])))
        for j in range(temp_feature_matrix.shape[0]):
                for k in range(len(features[i])):
                    if df.iloc[j,columns_index[i]]==features[i][k]:
                        temp_feature_matrix[j,k]=1
       # print(np.delete(temp_feature_matrix,-1,1))
        if drop==False:
            temp.append(temp_feature_matrix)
        else:
            temp.append(np.delete(temp_feature_matrix,-1,1))
        #print(temp)
    feature_matrix=np.concatenate(tuple(temp),axis=1)
    return(feature_matrix)


def recommendation(user_id,rank_matrix,query,length,score_name,itemid_name,userid_name):
    Y=rank_matrix#[rank_matrix["score"]<6]
    Y=Y.sort_values(by=score_name,ascending=False)
    recommendation_new=[]
    for y in Y[itemid_name].values:
        if len(recommendation_new)>length:
                break
        else:
            if y not in query[query[userid_name]==user_id][itemid_name].values:
                recommendation_new.append(y)
    
    return(Y[Y[itemid_name].isin(recommendation_new)])

class glink():

	def __init__(self,link):
		self.link=link
		
		
	def printlink(self):
		return(self.link)
	
	
	
	def __ge__(self,other):
		if self.link[-1]>=other.link[-1]:
			return(True)
		else:
			return(False)
	def __eq__(self,other):
		if self.link[-1]==other.link[-1]:
			return(True)
		else:
			return(False)
	def __le__(self,other):
		if self.link[-1]<=other.link[-1]:
			return(True)
		else:
			return(False)
	def __gt__(self,other):
		if self.link[-1]>other.link[-1]:
			return(True)
		else:
			return(False)
	def __lt__(self,other):
		if self.link[-1]<other.link[-1]:
			return(True)
		else:
			return(False)
	def __float__(self):
		return(float(self.link[-1]))
		
		
		
		
		
		
