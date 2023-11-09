import math
import os

import numpy as np
def SampleInputMatrix(nrows,npars,bu,bl,iseed,distname='randomUniform'):
    '''
    Create inputparameter matrix for nrows simualtions,
    for npars with bounds ub and lb (np.array from same size)
    distname gives the initial sampling ditribution (currently one for all parameters)

    returns np.array
    '''
    np.random.seed(iseed)
    x=np.zeros((nrows,npars))
    bu=np.array(bu)
    bl=np.array(bl)
    bound = bu-bl
    #problem={
       # 'num_vars': 4,
     #   'names': ['x1', 'x2', 'x3', 'x4'],
     #   'groups': None,
     #   'bounds':[]
   # }
   # for i in range(npars):
   #     if(Si["mu"][i]==0):
    #        bu[i]=bl[i]+0.1
   #     problem["bounds"].append([bl[i],bu[i]])
   # if(nrows<=(npars+1)):
   #     x = sample(problem, N=1, num_levels=4, optimal_trajectories=None)
   # else:
   #     x=sample(problem, N=(int(nrows/(npars+1))), num_levels=4, optimal_trajectories=None)
    for i in range(nrows):
##        x[i,:]= bl + DistSelector([0.0,1.0,npars],distname='randomUniform')*bound  #only used in full Vhoeys-framework
        x[i,:]= bl + np.random.rand(1,npars)*bound
    return x
def CalculationFunctions_Sph(X):
    'Sphere     [-100,100]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if (dimension <= 1):
            x = X
        else:
            x = X[i]
        res9 = 0
        for i in range(len(x)):
            res9 += x[i] ** 2
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str(res9) + "\n")
        testresultfile.close()
        if (dimension <= 1):
            return res9
        Y.append(res9)
    return Y
def CalculationFunctions_Gri(X):
    'Gri    [-100,100]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if (dimension <= 1):
            x = X
        else:
            x = X[i]
        res71 = 0
        res72 = 1
        for j in range(len(x)):
            res71 += (x[j] ** 2) / 4000
        for j in range(len(x)):
            res72 *= math.cos(x[j] / math.sqrt(j + 1))
        res7 = res71 - res72 + 1
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str(res7) + "\n")
        testresultfile.close()
        if (dimension <= 1):
            return res7
        Y.append(res7)
    return Y
def CalculationFunctions(X):
    'Wei [-0.5,0.5]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if (dimension <= 1):
            x = X
        else:
            x = X[i]
        res51 = 0
        res52 = 0
        for j in range(len(x)):
            for k in range(21):
                res51 += ((0.5) ** k) * math.cos(2 * math.pi * (3 ** k) * (x[j] + 0.5))
                res52 += ((0.5) ** k) * math.cos(2 * math.pi * (3 ** k) * 0.5)
        res5 = res51 - len(x) * res52
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str(res5) + "\n")
        testresultfile.close()
        if (dimension <= 1):
            return res5
        Y.append(res5)
    return Y
def CalculationFunctions_Ack(X):
    'Ackley [-32,32]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if (dimension <= 1):
            x = X
        else:
            x = X[i]
        res11 = 0
        res12 = 0
        for j in range(len(x)):
            res11 += x[j] ** 2
            res12 += math.cos(2 * math.pi * x[j])
        res1=(-20) * math.exp((-0.2) * math.sqrt(res11 / len(x))) - math.exp(res12 / len(x)) + 20 + math.e
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str(res1) + "\n")
        testresultfile.close()
        if(dimension<=1):
            return res1
        Y.append(res1)
    return np.array(Y)

def CalculationFunctions_Ras(X):
    'Rastrigin [-5,5]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if(dimension<=1):
            x=X
        else:
            x = X[i]
        res3 = 0
        for j in range(len(x)):
            res3 += x[j] ** 2 - 10 * math.cos(2 * math.pi * x[j]) + 10
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str(res3) + "\n")
        testresultfile.close()
        if(dimension<=1):
            return res3
        Y.append(res3)
    return np.array(Y)
def Rastrigin(X):
    Y=[]
    for i in range(len(X)):
        x=X[i]
        res3 = 0
        for j in range(len(x)):
            res3 += x[j] ** 2 - 10 * math.cos(2 * math.pi * x[j]) + 10
        Y.append(res3)
    return np.array(Y)
def odd1(X):
    'odd1号函数   [0,1]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if (dimension <= 1):
            x = X
        else:
            x = X[i]
        res = 0
        for j in range(len(x)):
            res += x[j] ** (j+1)
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str() + "\n")
        testresultfile.close()
        if (dimension <= 1):
            return res
        Y.append(res)
    return np.array(Y)

def high_demsion_lines_weight(X):
    'high_demsion_lines_weight 高维加权线性函数 [0,10]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if (dimension <= 1):
            x = X
        else:
            x = X[i]
        res = 0
        for j in range(len(x)):
            w=np.floor(j/3)-1
            res += x[j] * w
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str() + "\n")
        testresultfile.close()
        if (dimension <= 1):
            return res
        Y.append(res)
    return np.array(Y)

def high_demsion_高斯_weight(X):
    'high_demsion_高斯_weight 高维加权线性函数 [0,10]'
    Y = []
    for i in range(len(X)):
        dimension = np.array(X).ndim
        if (dimension <= 1):
            x = X
        else:
            x = X[i]
        res = 0
        for j in range(len(x)):
            mean=np.mean(x)
            s=np.var(x)
            w=j%4-2
            res += w*np.exp(-((x[j]-mean)**2)/(2*s))
        testparamsfile = open(os.getcwd() + os.sep + "testparams.txt", "a+")
        testparamsfile.write(str(x) + "\n")
        testparamsfile.close()
        testresultfile = open(os.getcwd() + os.sep + "testresult.txt", "a+")
        testresultfile.write(str(abs(res)) + "\n")
        testresultfile.close()
        if (dimension <= 1):
            return abs(res)
        Y.append(abs(res))
    return np.array(Y)

################################################################################
##   FUNCTION CALL FROM SCE-ALGORITHM !!
################################################################################

# def EvalObjF(npar,x,testcase=True,testnr=1):
def EvalObjF(npar, x,wq_inst,Y,none_anlnyse,x_romver,testcase=True):
    '''
    The SCE algorithm calls this function which calls the model itself
    (minimalisation of function output or evaluation criterium coming from model)
    and returns the evaluation function to the SCE-algorithm

    If testcase =True, one of the example tests are run
    '''
    #将不具备敏感性参数的数据放入原数组进行计算
    if(len(none_anlnyse)>0):
        x=list(x)
        for i in range(len(none_anlnyse)):
            rest=list(x[:none_anlnyse[i]])
            rest.append(x_romver[i])
            rest.extend(x[none_anlnyse[i]:])
            x=rest
        x=np.array(x)
    filepath=os.getcwd()
    if(not os.path.exists(filepath+os.sep+"param_1.txt")):
        param = open(filepath + os.sep + "param_1.txt","w+")
        param.write(str(list(x))+'\n')
        param.close()
    else:
        param = open(filepath + os.sep + "param_1.txt","r+")
        param_list=param.readlines()
        param.close()
        param = open(filepath + os.sep + "param_1.txt","w+")
        for i in param_list:
            param.write(i)
        param.write(str(list(x))+'\n')
        param.close()

##    print 'testnummer is %d' %testnr

    if testcase==True:
        res=Rastrigin(x)
        if(not os.path.exists(filepath+os.sep+"Y.txt")):
            Ytxt=open(filepath+os.sep+"Y.txt","w+")
            Ytxt.write(str(res)+'\n')
            Ytxt.close()
        else:
            Ytxt = open(filepath + os.sep + "Y.txt", "r+")
            Ytxt_list=Ytxt.readlines()
            Ytxt.close()
            Ytxt = open(filepath + os.sep + "Y.txt", "w+")
            for i in Ytxt_list:
                Ytxt.write(i)
            Ytxt.write(str(res)+'\n')
            Ytxt.close()
        return res
        #return testfunctn(npar,x,wq_inst,Y)
    else:
        # Welk model/welke objfunctie/welke periode/.... users keuze!
        return

