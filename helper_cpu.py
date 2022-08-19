import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import time


def sparse_rep(matrix):
    '''Finds the location of the non zero elements of a spase matrix and saves it as a list of tuples'''
    list_non_zero=[]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]!=0:
                list_non_zero.append((i,j))
    return(list_non_zero)

def sparse_dim(matrix):
    '''Finds the no. of non zero elements in a sparse matrix'''
    counter=0
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            if matrix[i,j] !=0:
                counter=counter+1
    return(counter)
    
    
    
def sparser(matrix):
    '''Returns a matrix containing zeros and ones. The ones are placed at location 
    where the input matrix has non zero entries. Zeros are placed at the remaining position'''
    p=np.zeros((matrix.shape[0],matrix.shape[1]))
    list_non_zero=sparse_rep(matrix)
    for k in list_non_zero:
        p[k[0],k[1]]=1
    return(p)

def antisparser(matrix):
    '''Returns a matrix containing zeros and ones. The ones are placed at location 
    where the input matrix has zero entries. Zeros are placed at the remaining position'''
    p=np.full((matrix.shape[0],matrix.shape[1]),1)
    list_non_zero=sparse_rep(matrix)
    for k in list_non_zero:
        p[k[0],k[1]]=0
    return(p)
   
def KL_error(A,B,sparse=True):
    '''Returns the Kullback Liebler error between A and B'''
    error=0
    if sparse==True:
        list_non_zero=sparse_rep(A)
    elif sparse==False:
        list_non_zero=[(i,j) for j in range(A.shape[1]) for i in range(A.shape[0])]

    for i,j in list_non_zero:
        error=error+A[i,j]*math.log(A[i,j]/B[i,j])-A[i,j]+B[i,j]
    return(error/sparse_dim(A))   

def square_norm(A,B,sparse=True):
    '''Return the square distance or l2 distance  between A and B matrices'''
    Z=A-B
    if sparse==True:
        normalize=sparse_dim(A)
        Project=sparser(A)
        Proj_Z=np.multiply(Project,Z)
    elif sparse==False:
        normalize=A.shape[0]*A.shape[1]
        Proj_Z=Z
        
    err=np.trace(np.matmul(Z,np.transpose(Proj_Z)))
    return(err/normalize)

def regulated_norm(A,X,Y,P,weight,gravity):
    '''Returns the regulated norm'''
    num=sparse_dim(A)
    Z=np.matmul(X,Y)
    Project=sparser(A)
    Proj_AZ=np.multiply(Project,A-Z)
    err1=(1/num)*np.trace(np.matmul(Proj_AZ,np.transpose(A-Z)))
    err2=(weight/A.shape[0])*np.trace(np.matmul(X,np.transpose(X)))\
    +(weight/A.shape[1])*np.trace(np.matmul(Y,np.transpose(Y)))
    err3=(gravity/(A.shape[0]*A.shape[1]))*np.trace(np.matmul(Z,np.transpose(Z)))

    return(err1,err2,err3,Z)




def norm(A):
    return(np.trace(np.matmul(A,np.transpose(A))))   
    
    
def fact_sparse_2(learn_rate,steps,R,U=None,V=None,choice=0,dim=0):
    '''Matrix factorization of the form R=UV with square norm'''
    
    error=[]
    start_time=time.perf_counter()
    list_non_zero=sparse_rep(R)
    if choice==0:
        U=np.random.normal(0,0.5,(R.shape[0],int(dim)))
        V=np.random.normal(0,0.5,(int(dim),R.shape[1]))
        for l in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X,sparse=True))

            for i,j in list_non_zero:
                a=(R[i,j]-X[i,j])*V[:,j]
                b=(R[i,j]-X[i,j])*U[i,:]
                U[i,:]=U[i,:]+learn_rate*a
                V[:,j]=V[:,j]+learn_rate*b
                
                
            print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]),end='\r',flush=True)
            
            
            
            
    elif choice==1:
        U=np.random.normal(0,0.5,(R.shape[0],V.shape[0]))
        for l in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X,sparse=True))
            
            for i,j in list_non_zero:
                a=(R[i,j]-X[i,j])*V[:,j]
                U[i,:]=U[i,:]+learn_rate*a
                
                
            print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]),end='\r',flush=True)
            
                
                
                
                
    elif choice==2:
        V=np.random.normal(0,0.5,(U.shape[1],R.shape[1]))
        for l in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X,sparse=True))
            
            for i,j in list_non_zero:
                b=(R[i,j]-X[i,j])*U[i,:]
                V[:,j]=V[:,j]+learn_rate*b
                
                
            print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]),end='\r',flush=True)
            
            
            
            
    stop_time=time.perf_counter()
    print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]))
    print(("Finished in {} secs").format(stop_time-start_time))
    plt.plot(error,label="sparse square error")
    plt.legend()
    return(U,V,np.matmul(U,V))


def fact_reg_sparse(learn_rate,steps,R,weight,gravity,U=None,V=None,choice=0,dim=0):
    
    '''Matrix factorization of the form R=UV with regulated square norm'''

    errsq=[]
    errg=[]
    errreg=[]
    c1=weight/R.shape[0]
    c2=weight/R.shape[1]
    c3=gravity/(R.shape[0]*R.shape[1])
    start_time=time.perf_counter()
    omega=sparse_dim(R)
    p=sparser(R)
    
    
    if choice==0:
        U=np.random.normal(0,0.5,(R.shape[0],int(dim)))
        V=np.random.normal(0,0.5,(int(dim),R.shape[1]))
        for l in range(steps):
            err=regulated_norm(R,U,V,p,weight,gravity)
            errsq.append(err[0])
            errg.append(err[2])
            errreg.append(err[1])
            print(("Factorising Matrix-------steps:{} sparse square error:{} gravity:{} regularisation:{}").format(l,err[0],err[1],err[2]),end='\r',flush=True)
            X=err[3]
            Y=p*(R-X)
            Uchange=-(2/omega)*np.matmul(Y,np.transpose(V))+2*c3*np.matmul(X,np.transpose(V))+c1*U
            Vchange=-(2/omega)*np.matmul(np.transpose(U),Y)+2*c3*np.matmul(np.transpose(U),X)+c2*V
            U=U-learn_rate*Uchange
            V=V-learn_rate*Vchange
            
    elif choice==1:
        U=np.random.normal(0,0.5,(R.shape[0],V.shape[0]))
        for l in range(steps):
            err=regulated_norm(R,U,V,p,weight,gravity)
            errsq.append(err[0])
            errg.append(err[2])
            errreg.append(err[1])
            print(("Factorising Matrix-------steps:{} sparse square error:{} gravity:{} regularisation:{}").format(l,err[0],err[1],err[2]),end='\r',flush=True)
            X=err[3]
            Y=p*(R-X)

            Uchange=-(2/omega)*np.matmul(Y,np.transpose(V))+2*c3*np.matmul(X,np.transpose(V))+c1*U
            U=U-learn_rate*Uchange
            
            
            
    elif choice==2:
        V=np.random.normal(0,0.5,(U.shape[1],R.shape[1]))
        for l in range(steps):
            err=regulated_norm(R,U,V,p,weight,gravity)
            errsq.append(err[0])
            errg.append(err[2])
            errreg.append(err[1])
            print(("Factorising Matrix-------steps:{} sparse square error:{} gravity:{} regularisation:{}").format(l,err[0],err[1],err[2]),end='\r',flush=True)
            X=err[3]
            Y=p*(R-X)

            Vchange=-(2/omega)*np.matmul(np.transpose(U),Y)+2*c3*np.matmul(np.transpose(U),X)+c2*V
            V=V-learn_rate*Vchange
            
    err=regulated_norm(R,U,V,p,weight,gravity)
    errsq.append(err[0])
    errg.append(err[2])
    errreg.append(err[1])    
    stop_time=time.perf_counter()
    #print(("sparse square error:{} gravity:{} regularisation:{}").format(err[0],err[1],err[2]))
    print(("Finished in {} secs").format(stop_time-start_time))
    plt.plot(errsq,label="spase square error")
    plt.plot(errg,label="gravity",)
    plt.plot(errreg,label="regularisation")
    plt.legend()
    return(U,V,err[3])



    
    
    
def nmf_sq(R,steps,dim=0,U=None,V=None,choice=0):
    '''Non negative matrix factorization R=UV for square error'''
    start_time=time.perf_counter()
    error=[]
    p=sparser(R)
    if choice==0:
        U=np.absolute(np.random.normal(1,4,(R.shape[0],int(dim))))
        V=np.absolute(np.random.normal(1,4,(int(dim),R.shape[1])))
        for i in range(steps):
            
            X=np.matmul(U,V)
            error.append(square_norm(R,X))

            
            x=np.matmul(R,np.transpose(V))
            y=np.matmul(p*X,np.transpose(V))
            #z=np.matmul(X,np.transpose(V))        

            U=U+U*(x-y)/y
            
            X=np.matmul(U,V)

            x=np.matmul(np.transpose(U),R)
            y=np.matmul(np.transpose(U),p*X)
            #z=np.matmul(np.transpose(U),X)
            V=V+V*(x-y)/y
            

            print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]),end="\r")
            
            
            
    elif choice==1:
        U=np.absolute(np.random.normal(1,4,(R.shape[0],V.shape[0])))
        for i in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X))

            
            x=np.matmul(R,np.transpose(V))
            y=np.matmul(p*X,np.transpose(V))
            z=np.matmul(X,np.transpose(V))        

            U=U+U*(x-y)/z
            

            print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]),end="\r")
            
            
    elif choice==2:
        V=np.absolute(np.random.normal(1,4,(U.shape[1],R.shape[1])))
        for i in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X))

            
            x=np.matmul(np.transpose(U),R)
            y=np.matmul(np.transpose(U),p*X)
            z=np.matmul(np.transpose(U),X)
            V=V+V*(x-y)/z
            

            print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]),end="\r")
    
    X=np.matmul(U,V)
    error.append(square_norm(R,X))
    stop_time=time.perf_counter()
    print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]))
    print(("Finished in {} secs, error: {}").format(stop_time-start_time,error[-1]))
    plt.plot(error)
    return(U,V,X)   



def nmf_kl(R,steps,dim=0,choice=0,U=None,V=None):
    
    '''Non-negative Matrix factorization of the form R=UV with Kullback Liebler'''
    start_time=time.perf_counter()

    error=[]
    p=sparser(R)
    pnot=antisparser(R)
    unit=p+pnot
    if choice==0:
        U=np.absolute(np.random.normal(1,4,(R.shape[0],int(dim))))
        V=np.absolute(np.random.normal(1,4,(int(dim),R.shape[1])))
        for i in range(steps):
            
            X=np.matmul(U,V)
            
            error.append(KL_error(R,X))

            
            x=np.matmul(V,np.transpose(R/X))
            y=np.matmul(V,np.transpose(p))
            #z=np.matmul(V,np.transpose(unit))
            U=U+(U/np.transpose(y)*np.transpose((x-y)))
            
            X=np.matmul(U,V)

            x=np.matmul(np.transpose(R/X),U)
            y=np.matmul(np.transpose(p),U)
            #z=np.matmul(np.transpose(unit),U)

            V=V+(V/np.transpose(y)*np.transpose((x-y)))
            


            print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]),end="\r",flush=True)
            
            
    if choice==1:
        U=np.absolute(np.random.normal(1,4,(R.shape[0],V.shape[0])))
        for i in range(steps):
            
            X=np.matmul(U,V)
                        
            error.append(KL_error(R,X))
            
            x=np.matmul(V,np.transpose(R/X))
            y=np.matmul(V,np.transpose(p))
            z=np.matmul(V,np.transpose(unit))
            U=U+(U/np.transpose(z)*np.transpose((x-y)))
               
            

            
            print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]),end="\r",flush=True)
                
    elif choice==2:
        V=np.absolute(np.random.normal(1,4,(U.shape[1],R.shape[1])))
        for i in range(steps): 
                 
            X=np.matmul(U,V)
            error.append(KL_error(R,X))

        

            x=np.matmul(np.transpose(R/X),U)
            y=np.matmul(np.transpose(p),U)
            z=np.matmul(np.transpose(unit),U)

            V=V+(V/np.transpose(z)*np.transpose((x-y)))
            

            
            print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]),end="\r",flush=True)
            

    X=np.matmul(U,V)
    error.append(KL_error(R,X))   
    stop_time=time.perf_counter()
    #print(("Factorizing Matrix---------steps: {} error: {}").format(i,error[-1]))
    print(("Finished in {} secs").format(stop_time-start_time),flush=True)
    plt.plot(error,label="Kullback Liebler")
    return(U,V,X)  
    
    
def fact_sparse(learn_rate,steps,R,U=None,V=None,choice=0,dim=0):
    '''Matrix factorization of the form R=UV with square norm'''
    
    error=[]
    start_time=time.perf_counter()
    p=sparser(R)
    if choice==0:
        U=np.random.normal(0,0.5,(R.shape[0],int(dim)))
        V=np.random.normal(0,0.5,(int(dim),R.shape[1]))
        for l in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X,sparse=True))

            Y=R-p*X
            A=np.matmul(Y,np.transpose(V))
            B=np.matmul(np.transpose(U),Y)
            U=U+learn_rate*A

            V=V+learn_rate*B
                
                
            print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]),end='\r',flush=True)
            
            
            
            
    elif choice==1:
        U=np.random.normal(0,0.5,(R.shape[0],V.shape[0]))
        for l in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X,sparse=True))
            
            #for i,j in list_non_zero:
            A=np.matmul((R-p*X),np.transpose(V))
            #B=np.matmul(np.transpose(U),(R-p*X))
            U=U+learn_rate*A
                
                
            print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]),end='\r',flush=True)
            
                
                
                
                
    elif choice==2:
        V=np.random.normal(0,0.5,(U.shape[1],R.shape[1]))
        for l in range(steps):
            X=np.matmul(U,V)
            error.append(square_norm(R,X,sparse=True))
            
            
            B=np.matmul(np.transpose(U),(R-p*X))

            V=V+learn_rate*B
                
                
            print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]),end='\r',flush=True)
            
            
            
            
    stop_time=time.perf_counter()
    print(("Factorising Matrix-------steps:{} error:{}").format(l,error[-1]))
    print(("Finished in {} secs").format(stop_time-start_time))
    plt.plot(error,label="sparse square error")
    plt.legend()
    return(U,V,np.matmul(U,V))
    
    
def vectorizer(data_frame,column_name):
    unique=data_frame[column_name].unique()
    for x in unique:
        app=[]
        for y in data_frame[column_name].values:
            if x==y:
                app.append(1)
            else:
                app.append(0)
        data_frame[x]=app
    data_frame.drop(columns=column_name,inplace=True)
    return(data_frame)
 
 
def cont_to_discrete(data_frame,column_name,step_size,start):
    array=[]
    for i in range(len(data_frame)):
        x=int(data_frame[column_name].values[i]/step_size)
        string=str(x*step_size)+"-"+str((x+1)*step_size)
        array.append(string)
    data_frame[column_name]=array
    return(data_frame)
    
    
def add_date(score_df,date_df,current_date=datetime.timestamp(datetime.today()),score_item="item id",date_item="item id",date_date="release"):
    X=pd.merge(score_df,date_df,how="inner",left_on=score_item,right_on=date_item)
    X["movie_age"]=current_date-X["release"]
    return(X)
    
def add_weight(weight_array,score_df,age_name="movie_age",start=0,step_size=31536000):
    weight=[0 for i in range(len(score_df))]
    
    for i in range(len(weight)):
        x=(score_df[age_name].values[i]-start)/step_size
        if x<0:
            weight[i]=-2
        elif 0<= int(x) <len(weight_array):
            weight[i]=weight_array[int(x)]
        else:
            weight[i]=-1
    score_df["weight"]=weight
    return(score_df)
    
def recommendation(user_id,rank_matrix,query):
    Y=pd.DataFrame(rank_matrix[rank_matrix["weight"]>=0])
    #print(Y)
    Z=pd.DataFrame(rank_matrix[rank_matrix["weight"]==-1])
    #print(Z)
    T=pd.DataFrame(rank_matrix[rank_matrix["weight"]==-2])
    #print(T)
    Y["final_score"]=Y["score"]*Y["weight"]
    Y.sort_values(by="final_score",inplace=True,ascending=False)
    Z.sort_values(by="score",inplace=True,ascending=False)
    T.sort_values(by="score",inplace=True,ascending=False)
    recommendation_new=[]
    for y in Y["item id"].values:
        if len(recommendation_new)>9:
                break
        else:
            if y not in query[query["user id"]==user_id]["item id"].values:
                recommendation_new.append(y)
    
    recommendation_old=[]
    for z in Z["item id"].values:
        if len(recommendation_old)>9:
                break
        else:
            if z not in query[query["user id"]==user_id]["item id"].values: 
                recommendation_old.append(z) 
    recommendation_trailer=[]
    for t in T["item id"].values:
        if len(recommendation_trailer)>9:
                break
        else:
            if t not in query[query["user id"]==user_id]["item id"].values: 
                recommendation_trailer.append(t)
    return(recommendation_new,recommendation_old,recommendation_trailer)
    
    
    
    
def collaborative_filter(rating, dim,learn_rate,user_id,steps):
    learned_rating=fact_sparse(learn_rate=learn_rate,dim=dim,R=rating,steps=steps)[2]
    output=[[int(i+1) for i in range(rating.shape[1])],learned_rating[user_id-1,:]]
    rank=pd.DataFrame(np.transpose(output),columns=["item id","score"])
    return(rank)

def collaborative_filter_nmf(rating,dim,user_id,steps,error_type="sq"):
    if error_type=="sq":
        learned_rating=nmf_sq(dim=dim,R=rating,steps=steps)[2]
    elif error_type=="kl":
        learned_rating=nmf_kl(dim=dim,R=rating,steps=steps)[2]
    output=[[int(i+1) for i in range(rating.shape[1])],learned_rating[user_id-1,:]]
    rank=pd.DataFrame(np.transpose(output),columns=["item id","score"])
    return(rank)

def collaborative_filter_reg(rating, dim,learn_rate,weight,gravity,user_id,steps):
    learned_rating=fact_reg_sparse(learn_rate=learn_rate,dim=dim,R=rating,weight=weight,steps=steps,gravity=gravity)[2]
    output=[[int(i+1) for i in range(rating.shape[1])],learned_rating[user_id-1,:]]
    rank=pd.DataFrame(np.transpose(output),columns=["item id","score"])
    return(rank)
    
    
def feature_filter(rating,feature_matrix,learn_rate,user_id,steps,left=True):
    if left==True:
        learned_rating=fact_sparse(learn_rate=learn_rate,R=rating,steps=steps,choice=2,U=feature_matrix)[2]
    else:
        learned_rating=fact_sparse(learn_rate=learn_rate,R=rating,steps=steps,choice=1,V=np.transpose(feature_matrix))[2]
    output=[[int(i+1) for i in range(rating.shape[1])],learned_rating[user_id-1,:]]
    rank=pd.DataFrame(np.transpose(output),columns=["item id","score"])
    return(rank)
    
def feature_filter_nmf(rating,feature_matrix,user_id,steps,left=True,error_type="sq"):
    if left==True and error_type=="sq":
        learned_rating=nmf_sq(R=rating,steps=steps,choice=2,U=feature_matrix)[2]
    elif left==False and error_type=="sq":
        learned_rating=nmf_sq(R=rating,steps=steps,choice=1,V=np.transpose(feature_matrix))[2]
    elif left==True and error_type=="kl":
        learned_rating=nmf_kl(R=rating,steps=steps,choice=2,U=feature_matrix)[2]
    elif left==False and error_type=="kl":
        learned_rating=nmf_kl(R=rating,steps=steps,choice=1,V=np.transpose(feature_matrix))[2]

    output=[[int(i+1) for i in range(rating.shape[1])],learned_rating[user_id-1,:]]
    rank=pd.DataFrame(np.transpose(output),columns=["item id","score"])
    return(rank)


def feature_filter_reg(rating,feature_matrix,learn_rate,user_id,steps,weight,gravity,left=True):
    if left==True:
        learned_rating=fact_reg_sparse(learn_rate=learn_rate,R=rating,steps=steps,choice=2,U=feature_matrix,weight=weight,gravity=gravity)[2]
    else:
        learned_rating=fact_reg_sparse(learn_rate=learn_rate,R=rating,steps=steps,choice=1,V=np.transpose(feature_matrix),weight=weight,gravity=gravity)[2]
    output=[[int(i+1) for i in range(rating.shape[1])],learned_rating[user_id-1,:]]
    rank=pd.DataFrame(np.transpose(output),columns=["item id","score"])
    return(rank)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

