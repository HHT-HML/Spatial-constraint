import numpy as np
import logging
import os
import sys
import time
import pandas as pd
from SALib.analyze import morris
from Sobol_G import *
from SALib.plotting.morris import (
horizontal_bar_plot,
covariance_plot
)
import matplotlib as plt
import random
from SALib.sample.morris import sample
from function import *
# from SCE_python import *
from Sobol_G import *

def DynamicDimonsionFilterSpace(x,total_count):
    x=np.array(x)
    low_range=x[0,:]
    up_range=x[0,:]
    if(total_count>len(x)-1):
        total_count=len(x)-1
    for i in range(1,total_count+1):
        for j in range(len(low_range)):
            if(low_range[j]>x[i,j]):
                low_range[j]=x[i,j]
            if(up_range[j]<x[i,j]):
                up_range[j] = x[i, j]
    return low_range,up_range
def DimensionSensitiveZeroArray(Si):
    removedimension=[]
    sensitivity=np.array(Si["mu_star"])
    if(all_elements_same(sensitivity)):
        return removedimension
    total_sensitivity=np.sum(sensitivity)
    precent_sensitivity=sensitivity/total_sensitivity
    for i in range(len(precent_sensitivity)):
        if(precent_sensitivity[i]==0):
            removedimension.append(i)
    return np.array(removedimension)
def RecoverDemonsion(now_array,delimished_index,value):
    dimension = now_array.ndim
    if(dimension>1):
        now_array_new = np.zeros((len(now_array), len(now_array[0]) + len(delimished_index)))
    for i in range(len(now_array)):
        if(dimension<=1):
            now_array_row = list(now_array)
        else:
            now_array_row = list(now_array[i])
        for j in range(len(delimished_index)):
            now_array_row.insert(delimished_index[j],value[j])
        if(dimension<=1):
            return now_array_row
        else:
            now_array_new[i]=now_array_row

    return now_array_new
def RemoveDemonsion(now_array,delimish_index):
    now_array=np.array(now_array)
    dimension = now_array.ndim
    if(dimension>1):
        now_array_new = np.zeros((len(now_array), len(now_array[0]) - len(delimish_index)))
    for i in range(len(now_array)):
        now_array_row=[]
        if(dimension<=1):
            now_array_row=list(now_array)
        else:
            now_array_row=list(now_array[i])
        for j in range(0,len(delimish_index)):
            now_array_row.pop(delimish_index[len(delimish_index)-1-j])
        if (dimension <= 1):
            return now_array_row
        now_array_new[i]=now_array_row
    return now_array_new
def all_elements_same(lst):
    return all(x == lst[0] for x in lst)
def DynamicDimensionSensitive(Si,delimished_index):
    if(all_elements_same(Si['mu_star'])):
        return delimished_index
    value=np.zeros((len(delimished_index)))
    recoverdimension = RecoverDemonsion(Si['mu_star'],delimished_index,value)
    remove_demonsion_idx=DimensionSensitiveZeroArray(Si)
    return remove_demonsion_idx
def ProblemNew(problem_old,removeindx,up_range,low_range):
    problem_new={
        'num_vars': 20,
        'names': [],
        'groups': None,
        'bounds':[]
    }
    value=np.zeros((len(removeindx)))
    low_range=RecoverDemonsion(low_range,removeindx,value)
    up_range=RecoverDemonsion(up_range,removeindx,value)
    problem_new['num_vars']=problem_old['num_vars']-len(removeindx)
    for i in range(problem_old['num_vars']):
        if i in removeindx:
            continue
        else:
            problem_new['names'].append('x' + str(i + 1))
            problem_new['bounds'].append([low_range[i],up_range[i]])
    return  problem_new


def XNormalize(X,nopt):
    X=np.array(X)
    for i in range(nopt):
        min_value=np.min(X[:,i])
        max_value=np.max(X[:,i])
        X[:,i]=(X[:,i]-min_value)/(max_value-min_value+0.0001)
    return X
def YNormalize(Y):
    # 对结果进行归一化处理
    Y_max = np.max(Y)
    Y_min = np.min(Y)
    Y_normalize = (Y - Y_min) / (Y_max - Y_min+0.0001)
    return Y_normalize


def slimararray(a,b):
    if(len(a)!=len(b)):
        return -1
    for i in range(len(a)):
        if(a[i]==b[i]):
            return i
    return -1
def Dispersion(a,b):
    res=0
    for i in range(len(a)):
        res+=abs(a[i]-b[i])
    return res
def DispersionRank(origin_point,x):
    x=list(x)
    dispersion=[]
    for i in range(len(x)):
        dispersion.append(Dispersion(origin_point,x[i]))
    dispersion=np.array(dispersion)
    idx=np.argsort(dispersion)
    return idx
def ComplexDispersionClassification(x,xf,origin_point,origin_point_res,igs,ngs,npg,nopt,k_flag):
    dispersion=[]
    x=np.array(x)
    for i in range(ngs-1,len(x)):
        dispersion.append(Dispersion(origin_point, x[i, :]))
    dispersion=np.array(dispersion)
    idx=np.argsort(dispersion)
    complexes=[]
    complexes_res=[]
    complexes.append(origin_point)
    complexes_res.append(origin_point_res)
    complexes_idx=[igs]
    i=0
    while(len(complexes)<npg):
        if(k_flag[idx[i]+ngs-1]!=1):
            complexes.append(x[idx[i]+ngs-1,:])
            complexes_res.append(xf[idx[i]+ngs-1])
            k_flag[idx[i]+ngs-1]=1
            complexes_idx.append(idx[i]+ngs-1)
        i+=1
    return complexes,complexes_res,k_flag,complexes_idx
def cceua(s,sf,bl,bu,bl_dynamic,bu_dynamic,corr,icall,best_recover_value,best_recover_value_new,dynamic_dimension_idx,remove_demonsion_idx,iseed):
    s=RemoveDemonsion(s,dynamic_dimension_idx)
    nps,nopt=s.shape
    n = nps
    m = nopt
    alpha = corr[-1]*(random.random()+1)
    beta = abs(alpha)*0.5

    # Assign the best and worst points:
    sb=s[0,:]
    fb=sf[0]
    sw=s[-1,:]
    fw=sf[-1]

    # Compute the centroid of the simplex excluding the worst point:

    ce= np.mean(s[:-1,:],axis=0)

    # Attempt a reflection point
    snew = ce + alpha*(ce-sw)

    # Check if is outside the bounds:
    ibound=0
    s1=snew-bl_dynamic
    idx=(s1<0).nonzero()
    if idx[0].size != 0:
        ibound=1

    s1=bu_dynamic-snew
    idx=(s1<0).nonzero()
    if idx[0].size != 0:
        ibound=2

    snew_recover = RecoverDemonsion(snew, dynamic_dimension_idx, best_recover_value_new)
    if ibound >= 1:
        snew = SampleInputMatrix(1,nopt,bu,bl,iseed,distname='randomUniform')[0]  #checken!!
        snew_recover = RecoverDemonsion(snew, remove_demonsion_idx, best_recover_value)


##    fnew = functn(nopt,snew);
    fnew = CalculationFunctions(snew_recover)
    icall += 1

    # Reflection failed; now attempt a contraction point:
    if fnew > fw:
        snew = sw + beta*(ce-sw)
        snew_recover = RecoverDemonsion(snew, dynamic_dimension_idx, best_recover_value_new)
        fnew = CalculationFunctions(snew_recover)
        icall += 1

    # Both reflection and contraction have failed, attempt a random point;
        if fnew > fw:
            snew = SampleInputMatrix(1,nopt,bu,bl,iseed,distname='randomUniform')[0]  #checken!!
            snew_recover = RecoverDemonsion(snew, dynamic_dimension_idx, best_recover_value_new)
            fnew = CalculationFunctions(snew_recover)
            icall += 1

    # END OF CCE
    return snew_recover,fnew,icall
def EvolutionaryDirection(complexes,complexes_res,nps,nspl,nopt,nopt_new,bl,bu,bl_dynamic,bu_dynamic,icall,best_recover_value,best_recover_value_new,dynamic_dimension_idx,remove_demonsion_idx,iseed):
    # Select simplex by sampling the complex according to a linear
    # probability distribution
    complexes=np.array(complexes)
    complexes_res = np.array(complexes_res)
    idx = np.argsort(complexes_res)
    complexes = complexes[idx, :]
    complexes_res = complexes_res[idx]
    complexes_new=np.zeros((len(complexes),nopt))
    complexes_res_new=np.zeros(len(complexes))
    complexes_new=RecoverDemonsion(complexes,dynamic_dimension_idx,best_recover_value_new)
    complexes_res_new=complexes_res
    for loop in range(nspl):
        demonsion_name = []
        for j in range(nopt_new):
            demonsion_name.append('var' + str(j + 1))
        df = pd.DataFrame(complexes, columns=demonsion_name)
        # 计算样本之间的相关系数矩阵
        corr_matrix = df.T.corr(method='pearson')
        corr_array = np.array(corr_matrix)
        sum_corr = np.zeros((len(corr_array)))
        for i in range(len(corr_array)):
            corr_row=np.zeros((len(corr_array)))
            corr_row=corr_array[i, :]
            corr_row=np.sort(corr_row)[::-1]
            sum_corr[i]=np.sum(abs(corr_row[:nps]))
        #确定单纯形
        idx=np.argsort(sum_corr)
        lcs = np.array([0] * nps)
        simplex_idx = np.argsort(abs(corr_array[idx[0], :]))
        for j in range(nps):
            lcs[j] = simplex_idx[-j]
        lcs.sort()
        # Construct the simplex:
        s = np.zeros((nps, nopt_new))
        s = complexes[lcs, :]
        sf = complexes_res[lcs]
        s=RecoverDemonsion(s,dynamic_dimension_idx,best_recover_value_new)
        sub_corr=corr_array[lcs[0],lcs]
        snew,fnew,icall=cceua(s, sf, bl, bu,bl_dynamic,bu_dynamic,sub_corr, icall,best_recover_value,best_recover_value_new,dynamic_dimension_idx,remove_demonsion_idx, iseed)

        # Replace the worst point in Simplex with the new point:
        s[-1, :] = snew
        sf[-1] = fnew

        # Replace the simplex into the complex;
        complexes_new[lcs, :]=s
        complexes_res_new[lcs] = sf
        complexes[lcs, :] = RemoveDemonsion(s,dynamic_dimension_idx)
        complexes_res[lcs] = sf

        # Sort the complex;
        idx = np.argsort(complexes_res)
        complexes_res = np.sort(complexes_res)
        complexes = complexes[idx, :]
    # End of Inner Loop for Competitive Evolution of Simplexes
    # end of Evolve sub-population igs for nspl steps:

    # Replace the complex back into the population;
    return complexes,complexes_res,complexes_new,complexes_res_new,icall



#暂时不考虑黑区域
def BlackArea(x,ngs):
    x=np.array(x,xf)
    low_range = x[0,:]
    up_range=x[0,:]
    xw = x[-1, :]
    dis_rank = DispersionRank(xw, x[:-1, :])
    k=0
    low_range=np.zeros(len(x[0]))
    for i in range(len(dis_rank)):
        if (dis_rank[i] < 10 or (i<=1 and xf[i]<xf[i-1])):
            k=dis_rank[i]
            return low_range,up_range
        for j in range(len(low_range)):
            if(low_range[j]>x[i,j]):
                low_range=x[i,j]
            if(up_range[j]<x[i,j]):
                up_range[j]=x[i,j]
    return low_range, up_range

def DynamicFilterDemonsion(problem,remove_demonsion_idx,up_range, low_range):
    problem_new = ProblemNew(problem, remove_demonsion_idx, up_range, low_range)
    x = sample(problem, N=4, num_levels=4, optimal_trajectories=None)
    Y = CalculationFunctions(x)
    Y_normalize = YNormalize(Y)
    print(Y_normalize)
    print("--------------------------")
    x_normalize = XNormalize(x, problem["num_vars"])

    Si = morris.analyze(
        problem,
        x_normalize,
        Y_normalize,
        conf_level=0.95,
        print_to_console=True,
        num_levels=4,
        num_resamples=100,
    )
    remove_demonsion_idx_new = DynamicDimensionSensitive(Si, remove_demonsion_idx)
    up_range=RecoverDemonsion(up_range,remove_demonsion_idx,np.zeros(len(remove_demonsion_idx)))
    low_range=RecoverDemonsion(low_range,remove_demonsion_idx,np.zeros(len(remove_demonsion_idx)))
    up_range=RemoveDemonsion(up_range,remove_demonsion_idx_new)
    low_range=RemoveDemonsion(low_range,remove_demonsion_idx_new)
    dynamicparamsfile = open(os.getcwd() + os.sep + "dynamicparams.txt", "a+")
    dynamicresultfile=open(os.getcwd() + os.sep + "dynamicresult.txt", "a+")
    for i in range(len(x)):
        for j in range(problem_new['num_vars']):
            dynamicparamsfile.write(str(x[i,j]) +'\t')
        dynamicparamsfile.write('\n')
        dynamicresultfile.write(str(Y[i])+'\n')
    dynamicparamsfile.close()
    dynamicresultfile.close()


    return x,remove_demonsion_idx_new,up_range,low_range

testresultfile=open(os.getcwd()+os.sep+"testresult.txt","w+")
testparamsfile=open(os.getcwd()+os.sep+"testparams.txt","w+")
testparamsfile.close()
testparamsfile.close()
resultfile = open(os.getcwd() + os.sep + "result_value.txt", "w+")
resultfile.close()
dynamicfile=open(os.getcwd()+os.sep+"dynamicparams.txt","w+")
dynamicfile.close()
dynamicresultfile=open(os.getcwd() + os.sep + "dynamicresult.txt", "w+")
dynamicresultfile.close()
bl = np.array([-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5])
bu = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
x0 = np.array([0,0,0,0,0,0,0,0,0,0])
# or define manually without a parameter file:
problem = {
   'num_vars': 10,
   'names': ['x1', 'x2', 'x3', 'x4','x5','x1', 'x2', 'x3', 'x4','x5'],
   'groups': None,
   'bounds': [[-0.5, 0.5],
             [-0.5, 0.5],
             [-0.5, 0.5],
             [-0.5, 0.5],
            [-0.5, 0.5],
[-0.5, 0.5],
             [-0.5, 0.5],
             [-0.5, 0.5],
             [-0.5, 0.5],
            [-0.5, 0.5],
              ]}
# Files with a 4th column for "group name" will be detected automatically, e.g.
# param_file = '../../src/SALib/test_functions/params/Ishigami_groups.txt'
x = sample(problem, N=problem["num_vars"], num_levels=10, optimal_trajectories=None)
Y=CalculationFunctions(x)
Y_normalize=YNormalize(Y)
print(Y_normalize)
print("--------------------------")
x_normalize=XNormalize(x,problem["num_vars"])

Si = morris.analyze(
    problem,
    x_normalize,
    Y_normalize,
    conf_level=0.95,
    print_to_console=True,
    num_levels=4,
    num_resamples=100,
)
remove_demonsion_idx=DimensionSensitiveZeroArray(Si)
idx=np.argsort(Y)
x=x[idx,:]
Y=np.sort(Y)
up_range=list(x[0,:])
low_range=list(x[0,:])
best_remove_value=[]
for i in range(len(remove_demonsion_idx)):
    best_remove_value.append(x[0,remove_demonsion_idx[i]])
bl_new=RemoveDemonsion(bl,remove_demonsion_idx)
bu_new=RemoveDemonsion(bu,remove_demonsion_idx)
bl_array=list(bl_new)
bu_array=list(bu_new)
x=list(x)
for i in range(1,20):
    for j in range(len(x[i])):
        if(x[i][j]>up_range[j]):
            up_range[j]=x[i][j]
        if(low_range[j]>x[i][j]):
            low_range[j]=x[i][j]
print(up_range)
print(low_range)
temp_i=20
while(slimararray(up_range,low_range)!=-1):
    k=slimararray(up_range,low_range)
    print(temp_i)
    if(up_range[k]<x[temp_i][k]):
        up_range[k] = x[temp_i][ k]
    if(low_range[k]>x[temp_i][k]):
        low_range[k] = x[temp_i][ k]
    temp_i+=1
inital_count=(problem['num_vars']-len(remove_demonsion_idx)+1)*(2*(problem['num_vars']-len(remove_demonsion_idx))+1)
if(inital_count>len(x)):
    print("进入sceua的方法进行采样")
    x_sceua = SampleInputMatrix(inital_count - len(x),
                                problem['num_vars'], np.array(up_range), np.array(low_range), 0,
                                distname='randomUniform')
    Y = list(Y)
    Y.extend(list(CalculationFunctions(x_sceua)))
    x = list(x)
    x.extend(list(x_sceua))

Y=np.array(Y)
Y=np.sort(Y)
x=np.array(x)
idx=np.argsort(Y)
x=x[idx,:]
xf=np.sort(Y)
#sceua参数合集
maxn=10000
kstop=30
pcento=0.01
peps=0.01
iseed= 0
iniflg=0
ngs=problem['num_vars']+1
nopt=problem['num_vars']
npg=2*nopt+1
nps=nopt+1
nspl=npg
npt=npg*ngs
# Record the best and worst points;
bestx=x[0,:]
bestf=xf[0]
worstx=x[-1,:]
worstf=xf[-1]
icall=len(xf)
BESTF=bestf
BESTX=bestx
ICALL=icall
xnstd=np.std(x,axis=0)
bound=np.array(up_range)-np.array(low_range)
bound_total=bu-bl
# Computes the normalized geometric range of the parameters
gnrng=np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bound_total)))

print('The Initial Loop: 0')
print(' BESTF:  %f ' %bestf)
print(' BESTX:  ')
print(bestx)
print(' WORSTF:  %f ' %worstf)
print(' WORSTX: ')
print(worstx)
print('     ')


result_file = open(os.getcwd() + os.sep + "result_value.txt", "a+")
result_file.write(str('The Initial Loop: 0')+str('\n'))
result_file.write(str(' BESTF:  %f ' %bestf)+str("\n"))
result_file.write(str(' BESTX:  ')+str("\n"))
result_file.write(str(bestx)+'\n')
result_file.write(str(' WORSTF:  %f ' %worstf)+str("\n"))
result_file.write(str('  WORSTX: ')+str("\n"))
result_file.write(str(worstx)+'\n')
result_file.write('                    '+'\n')
result_file.close()
# Check for convergency;
if icall >= maxn:
    print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
    print('ON THE MAXIMUM NUMBER OF TRIALS ')
    print(maxn)
    print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
    print(icall)


    print('OF THE INITIAL LOOP!')

if gnrng < peps:
    print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

# Begin evolution loops:
nloop = 0
criter=[]
criter_change=1e+5
dynamic_dimension_idx_total=[]
dynamic_dimension_idx_total.append(remove_demonsion_idx)
while icall<maxn and gnrng>peps and criter_change>pcento:
    #降低维度
    #确定此次进化的空间范围
    x_new=RemoveDemonsion(x,remove_demonsion_idx)
    xf_new = np.zeros((len(x)))
    xf_new=xf
    nopt_new=len(x_new[0])
    low_range_new,up_range_new=DynamicDimonsionFilterSpace(x_new,(nopt_new + 1)*(2 * nopt_new + 1))
    #判断是否需要再次进行动态降维
    are_equal=False
    k_loop=0
    for i in range(len(bl_array)):
        dimension = np.array(bl_array).ndim
        if(dimension<=1):
            if (np.array_equal(low_range_new, bl_array) and np.array_equal(up_range_new, bu_array)):
                are_equal = True
                k_loop=0
                break
        else:
            if (np.array_equal(low_range_new, bl_array[i]) and np.array_equal(up_range_new, bu_array[i])):
                are_equal = True
                k_loop=i
    if(are_equal):
        dynamic_dimension_idx = dynamic_dimension_idx_total[k_loop]
    else:
        # 确定此轮动态降维的维度与范围
        x_new_gen,dynamic_dimension_idx, up_range_new, low_range_new = DynamicFilterDemonsion(problem, remove_demonsion_idx,
                                                                                    up_range_new, low_range_new)
        dynamic_dimension_idx_total.append(dynamic_dimension_idx)
        icall+=len(x_new_gen)
    best_remove_value_new =[]
    for i in range(len(dynamic_dimension_idx)):
        best_remove_value_new.append(x[0, dynamic_dimension_idx[i]])
    nopt_new=len(up_range_new)

    nloop+=1
    # Loop on complexes (sub-populations);
    k_flag = np.full((len(x)), -1)
    x_new=np.array(RemoveDemonsion(x,dynamic_dimension_idx))
    npg_new = 2 * nopt_new + 1
    nps_new = nopt_new + 1
    nspl_new = npg_new
    npt_new = npg_new * (nopt_new+1)
    ngs_new=nopt_new+1
    bl_dynamic=RemoveDemonsion(bl,dynamic_dimension_idx)
    bu_dynamic=RemoveDemonsion(bu,dynamic_dimension_idx)
    for igs in range(ngs_new):
        # Partition the population into complexes (sub-populations);
        k_flag[igs]=1
        complexes,complexes_res, k_flag,complexes_idx = ComplexDispersionClassification(x_new,xf_new,x_new[igs, :],xf_new[igs],igs,ngs_new,npg_new,nopt_new, k_flag)
        complexes_new,complexes_res_new,complexes_total,complexes_res_total,icall=EvolutionaryDirection(complexes, complexes_res, nps_new,nspl_new,nopt,nopt_new, bl_new, bu_new, bl_dynamic,bu_dynamic,icall,best_remove_value,best_remove_value_new,dynamic_dimension_idx,remove_demonsion_idx, iseed)
        for i in range(len(complexes_idx)):
            x_new[complexes_idx[i],:]=complexes_new[i,:]
            xf_new[complexes_idx[i]]=complexes_res_new[i]
            x[complexes_idx[i],:]=complexes_total[i,:]
            xf[complexes_idx[i]]=complexes_res_total[i]


    # Shuffled the complexes;
    idx = np.argsort(xf)
    xf = np.sort(xf)
    x = x[idx, :]
    for i in range(len(remove_demonsion_idx)):
        best_remove_value[i]=x[0, remove_demonsion_idx[i]]

    PX = x
    PF = xf

    # Record the best and worst points;
    bestx = x[0, :]
    bestf = xf[0]
    worstx = x[-1, :]
    worstf = xf[-1]

    BESTX = np.append(BESTX, bestx, axis=0)  # appenden en op einde reshapen!!
    BESTF = np.append(BESTF, bestf)
    ICALL = np.append(ICALL, icall)

    # Compute the standard deviation for each parameter
    xnstd = np.std(x, axis=0)

    # Computes the normalized geometric range of the parameters
    gnrng = np.exp(np.mean(np.log((np.max(x, axis=0) - np.min(x, axis=0)) / bound_total)))

    print('Evolution Loop: %d  - Trial - %d' % (nloop, icall))
    print(' BESTF:  %f ' % bestf)
    print(' BESTX:  ')
    print(bestx)
    print(' WORSTF:  %f ' % worstf)
    print(' WORSTX: ')
    print(worstx)
    print('     ')

    result_file = open(os.getcwd() + os.sep + "result_value.txt", "a+")
    result_file.write(str('Evolution Loop: %d  - Trial - %d' % (nloop, icall)) + '\n')
    result_file.write(str(' BESTF:  %f ' % bestf) + '\n')
    result_file.write(str(' BESTX:  ') + '\n')
    result_file.write(str(bestx))
    result_file.write(str(' WORSTF:  %f ' % worstf) + '\n')
    result_file.write(str(' WORSTX: ') + '\n')
    result_file.write(str(worstx))
    result_file.write(str('                                 ') + '\n')
    result_file.close()
    # Check for convergency;
    if icall >= maxn:
        print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
        print('ON THE MAXIMUM NUMBER OF TRIALS ')
        print(maxn)
        print('HAS BEEN EXCEEDED.')

    if gnrng < peps:
        print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

    criter = np.append(criter, bestf)

    if nloop >= kstop:  # nodig zodat minimum zoveel doorlopen worden
        criter_change = np.abs(criter[nloop - 1] - criter[nloop - kstop]) * 100
        criter_change = criter_change / np.mean(np.abs(criter[nloop - kstop:nloop]))
        if criter_change < pcento:
            print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE THRESHOLD %f' % (kstop, pcento))
            print('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')

# End of the Outer Loops
print('SEARCH WAS STOPPED AT TRIAL NUMBER: %d' %icall)
print('NORMALIZED GEOMETRIC RANGE = %f'  %gnrng)
print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f' %(kstop,criter_change))






