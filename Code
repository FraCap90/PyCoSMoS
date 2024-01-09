import similaritymeasures
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.special import gamma
from math import gamma
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns
import numpy as np
from matplotlib import pyplot
import statsmodels.api as sm
import scipy.stats as stats 
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from IPython.display import HTML, display
#from IPython.display import display
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")
import math
import plotly.subplots as sp
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio

# Define ACS funtion
def acfweibull(lag, scale, shape): return math.exp(-(lag / scale) ** shape) 
def acfparetoII(lag, scale, shape): return(1 + (shape * lag) / scale)**((-1)/shape)
def acfburrXII(lag, scale, shape1, shape2): return((1 + shape2 * (lag / scale)** shape1)** -(1 / shape1 * shape2)) 

def lmoments(x):
    x = np.sort(x)
    n = x.shape[0]
    nn = np.repeat(np.array([n-1]), [n])
    pp = np.linspace(start=0, stop=n-1, num=n)
    p1 = pp/nn
    p2 = p1*(pp - 1)/(nn - 1)
    p3 = p2*(pp - 2)/(nn - 2)
  
    b0 = sum(x)/n
    b1 = sum(p1*x)/n
    b2 = sum(p2*x)/n
    b3 = sum(p3*x)/n
  
    l1 = b0
    l2 = 2*b1 - b0
    l3 = 2*(3*b2 - b0)/(2*b1 - b0) - 3
    l4 = 5*(2*(2*b3 - 3*b2) + b0)/(2*b1 - b0) + 6

    return l1,l2,l3,l4

def ECDF(x):
    st = np.sort(x)
    aux = stats.rankdata(st, method='min')/(len(x)+1)
    out = pd.DataFrame({'p': aux, 'value': st})
    return out

def rMSE(x, y):
      return np.sum((x/y - 1)**2)/len(y) ## ratio MSE
def MSE(x, y):
      return np.sum((x - y)**2)/len(y) ## MSE
def MAE(x, y):
      return np.sum(abs(x - y))/len(y) ## MAE 
    
def qggamma(p, shape1, shape2, scale):
    q = scale*stats.gamma.ppf(p, scale = 1, a = shape1/shape2)**(1/shape2)
    return(q)
def pggamma(q, shape1, shape2, scale):
    p = stats.gamma.cdf((q/scale)**shape2, scale = 1, a = shape1/shape2)
    return(p)
def dggamma(x, scale, shape1, shape2):
    d = (shape2*(x/scale)**(-1 + shape1))/(np.exp(x/scale)**shape2*scale*gamma(shape1/shape2))
    return(d)

def qburrXII(p, shape1, shape2, scale):
    q = scale*(-((1 - (1 - p)**(-(shape1*shape2)))/shape2))**shape1**(-1) 
    return(q)
def pburrXII(q, shape1, shape2, scale):
    p = 1 - (1 + shape2*(q/scale)**shape1)**(-(1/(shape1*shape2))) 
    return(p)
def dburrXII(x, scale, shape1, shape2):
    d =  ((x/scale)**(-1 + shape1)*(1 + shape2*(x/scale)**shape1)**(-1 - 1/(shape1*shape2)))/scale
    return(d)

def qburrIII(p, shape1, shape2, scale):
    q = scale*(shape1*(p**(-1/(shape1*shape2)) - 1))**(-shape2) 
    return(q)
def pburrIII(q, shape1, shape2, scale):
    p = (1 + 1/(shape1*(q/scale)**shape2**(-1)))**(-(shape1*shape2))
    return(p)
def dburrIII(x, scale, shape1, shape2):
    d =  ((x/scale)**(-1 - shape2**(-1))*(1 + 1/(shape1*(x/scale)**shape2**(-1)))**(-1 - shape1*shape2))/scale
    return(d)

def pge4(q, shape1, shape2, scale):
    p = 1- (((np.exp(q/scale)**shape2) -1)**(shape1/shape2)+1)**(-shape2/shape1)
    return(p)
def cdf_ge4(x, shape1, shape2, scale):
    return (1 - (((np.exp(x/scale)**shape2) -1)**(shape1/shape2)+1)**(-shape2/shape1))
def inv_cdf_ge4(p, shape1, shape2, scale, tol=1e-6):
    a = 0
    b = 10
    while (b - a > tol):
        mid = (a + b) / 2
        if cdf_ge4(np.array(mid), shape1, shape2, scale) < p:
            a = mid
        else:
            b = mid
    return (a + b) / 2
def qge4(p, shape1, shape2, scale):
    return inv_cdf_ge4(p, shape1, shape2, scale)
def dge4(x, shape1, shape2, scale):
    eps = 1e-6
    d = (cdf_ge4(x + eps, shape1, shape2, scale) - cdf_ge4(x - eps, shape1, shape2, scale)) / (2 * eps)
    return d

def qlogis(p, loc, scale):
    q = loc + scale * np.log(p / (1 - p))
    return q

def analyzeTS(data,lags, parametric_acf = ('parII','wei', 'burrXII'), marg_distr = ('logistic', 'skewnorm','gamma','ggamma','norm', 'lognorm', 'burrIII','burrXII', 'weibull', 'beta', 'weibull3'), n_points = 10): 
    label= []
    for m in range(1,13,1): label.append( 'Month_'+str(m))
    data.columns = ['Time', 'Value']
    
    # Convert Time into Python Time object
    Times = pd.to_datetime(data["Time"])
    data = data.assign(Time = Times)
    
    # Add month variable
    value = pd.DatetimeIndex(data['Time']).month
    data = data.assign(month = value)
    
    # Assign 0 to NA values
    new_value = data['Value'].fillna(data['Value'].mean())
    data = data.assign(Value = new_value)
    
    months = data.month.unique()
    stratified_data = {}
    p0 = {}
    pars_NonZeroValues = {}
    u_t = {}
    Theoretic = {}
    no0values = {}
    empAcs_Observed = {}
    
    def objective_FitDist(par):
        edf = ECDF(val) 
        aux = len(edf) 
        edf = edf.iloc[range(1, aux, n_points), 0:2]
        F = edf['value']
        Xi = edf['p'] 
        if marg_distr == 'gamma':
            a, loc, scale = par
            Xu = stats.gamma.cdf(edf['value'], a, loc, scale)
        elif marg_distr == 'ggamma':
            shape1, shape2, scale = par
            Xu = pggamma(edf['value'], shape1, shape2, scale)
        elif marg_distr == 'ge4':
            F = np.array(edf['value'])
            temp = np.array(Xi)
            shape1, shape2, scale = par
            Xu = pge4(edf['value'], shape1, shape2, scale)
        elif marg_distr == 'norm':
            loc, scale = par
            Xu = stats.norm.cdf(edf['value'], loc, scale)
        elif marg_distr == 'lognorm':
            s, scale = par
            Xu = stats.lognorm.cdf(edf['value'], s, scale)
        elif marg_distr == 'burrIII':
            shape1, shape2, scale = par
            Xu = pburrIII(edf['value'], shape1, shape2, scale)
        elif marg_distr == 'burrXII':
            shape1, shape2, scale = par
            Xu = pburrXII(edf['value'], shape1, shape2, scale)
        elif marg_distr == 'beta':
            a, b = par
            Xu = stats.beta.cdf(edf['value'], a, b) 
        elif marg_distr == 'skewnorm':
            a, loc, scale = par
            Xu = stats.skewnorm.cdf(edf['value'], a,loc, scale)
        elif marg_distr == 'weibull':
            c,  scale = par #loc,
            Xu = stats.weibull_min.cdf(edf['value'], c,  scale)
        elif marg_distr == 'weibull3':
            c, loc, scale = par 
            Xu = stats.weibull_min.cdf(edf['value'], c, loc, scale)
        elif marg_distr == 'logistic':
            loc, scale = par
            Xu = stats.logistic.cdf(edf['value'], loc, scale)

        out = MAE(Xi, Xu)
        return out

    # a) Stratify the observed time series values on a seasonal basis (stratified_data);
    # b) Estimate the probability of zero (p0) 
    # c) Fit a parametric distribution function to nonzero values and store its parameter estimates (pars_NonZeroValues)
    # d) Transform the observed time series to mixed-Uniform time series (u_t)
    for m in range(1, len(months)+1,1):
        data_by_month = pd.DataFrame(data[data['month'] == m])
        n = len(data_by_month)
        MixedUnif = np.zeros(n)
        stratified_data[m] = data_by_month
        NonZeroValues = data_by_month[data_by_month['Value'] != 0]
        val = NonZeroValues['Value']
        df_filter = data_by_month[data_by_month['Value'] > 0]
        data_array = np.squeeze(np.asarray(data_by_month.iloc[:, [1]]))
        if  marg_distr == 'gamma':
            guess_g=[1,1,1]
            bnds = ((0.001, None), (0, None), (0.001, None))
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            res = minimize(objective_FitDist,guess_g,bounds=bnds,method='nelder-mead')
            pars = res['x'].round(2)
            Theoretical = stats.gamma.pdf(NonZeroValues['Value'], a=pars[0],loc=pars[1], scale=pars[2])
            MixedUnif[data_array > 0] = stats.gamma.cdf(df_filter['Value'], a = pars[0], loc=pars[1], scale=pars[2])
            
        elif marg_distr == 'ge4':
            guess_g=[1,1,1]
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            bnds = ((0.05, None), (0.05, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds,method='nelder-mead')
            pars = res['x'].round(2)
            Theoretical = dge4(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) 
            MixedUnif[data_array > 0] = pge4(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) 
            
        elif marg_distr == 'ggamma':
            guess_g=[1,1,1]
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            bnds = ((0.05, None), (0.05, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds,method='nelder-mead')
            pars = res['x'].round(2)
            Theoretical = dggamma(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) 
            MixedUnif[data_array > 0] = pggamma(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) 

        elif marg_distr == 'weibull':
            guess_g=[1,1]
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            bnds = ((0.05, None), (0.05, None))
            res = minimize(objective_FitDist,guess_g,bounds=bnds,method='nelder-mead')
            pars = res['x'].round(2)
            Theoretical = stats.weibull_min.pdf(NonZeroValues['Value'], c = pars[0], scale = pars[1])#loc = pars[1],
            MixedUnif[data_array > 0] = stats.weibull_min.cdf(df_filter['Value'], c = pars[0], scale = pars[1])#loc = pars[1],
        
        elif marg_distr == 'weibull3':
            guess_g=[2,2,2]
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            bnds = ((0.01, None), (0, None), (0.01, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds,method='nelder-mead')
            pars = res['x'].round(2)
            Theoretical = stats.weibull_min.pdf(NonZeroValues['Value'], c = pars[0], loc = pars[1], scale = pars[2])
            MixedUnif[data_array > 0] = stats.weibull_min.cdf(df_filter['Value'], c = pars[0], loc = pars[1], scale = pars[2])
            
        elif marg_distr == 'skewnorm':
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            guess_g=[1,1,1]
            bnds = ((0.05, None),(0.05, None),(0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds)
            pars = res['x'].round(2)
            Theoretical = stats.skewnorm.pdf(NonZeroValues['Value'], a=pars[0],loc=pars[1], scale = pars[2])
            MixedUnif[data_array > 0] = stats.skewnorm.cdf(NonZeroValues['Value'], a=pars[0],loc=pars[1], scale = pars[2]) 
            
        elif marg_distr == 'burrXII':
            guess_g=[1,1,1]
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            bnds = ((0.05, None), (0.05, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds)
            pars = res['x'].round(2)
            Theoretical = dburrXII(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) #stats.burr12.pdf(NonZeroValues['Value'], c = pars[0], d = pars[1],  loc = 0, scale = pars[3])
            MixedUnif[data_array > 0] = pburrXII(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) #stats.burr12.cdf(df_filter['Value'], c = pars[0], d = pars[1] , loc = 0, scale = pars[3])
            
        elif marg_distr == 'burrIII':
            guess_g=[1,1,1]
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            bnds = ((0.05, None), (0.05, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds)
            pars = res['x'].round(2)
            Theoretical = dburrIII(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) #stats.burr12.pdf(NonZeroValues['Value'], c = pars[0], d = pars[1],  loc = 0, scale = pars[3])
            MixedUnif[data_array > 0] = pburrIII(NonZeroValues['Value'], shape1=pars[0], shape2=pars[1], scale=pars[2]) #stats.burr12.cdf(df_filter['Value'], c = pars[0], d = pars[1] , loc = 0, scale = pars[3])
            
        elif marg_distr == 'beta':
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            guess_g=[1,1]
            bnds = ((0.05, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds)
            pars = res['x'].round(2)
            Theoretical = stats.beta.pdf(NonZeroValues['Value'], a = pars[0], b = pars[1])
            MixedUnif[data_array > 0] = stats.beta.cdf(df_filter['Value'], a = pars[0], b = pars[1])
        
        elif marg_distr == 'lognorm':
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            guess_g=[1,1]
            bnds = ((0.05, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds)
            pars = res['x'].round(2)
            Theoretical = stats.lognorm.pdf(NonZeroValues['Value'], s = pars[0],  scale = pars[1])
            MixedUnif[data_array > 0] = stats.lognorm.cdf(df_filter['Value'], s = pars[0],  scale = pars[1])
            
        elif marg_distr == 'norm':
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            guess_g=[5,5]
            bnds = ((None, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds,method='nelder-mead')
            pars = res['x'].round(2)
            Theoretical = stats.norm.pdf(NonZeroValues['Value'], loc = pars[0], scale = pars[1])
            MixedUnif[data_array > 0] = stats.norm.cdf(df_filter['Value'], loc = pars[0], scale = pars[1])
        
        elif marg_distr == 'logistic':
            NonZeroValues.sort_values(by=['Value'], inplace=True)
            guess_g=[1,1]
            bnds = ((0.05, None), (0.05, None)) 
            res = minimize(objective_FitDist,guess_g,bounds=bnds,method='nelder-mead')
            pars = res['x'].round(2)
            Theoretical = stats.logistic.pdf(NonZeroValues['Value'], loc = pars[0], scale = pars[1])
            MixedUnif[data_array > 0] = stats.logistic.cdf(df_filter['Value'], loc = pars[0], scale = pars[1])
            
        u_t[m] = MixedUnif
        Theoretic[m] = Theoretical    
        no0values[m] = NonZeroValues
        pars_NonZeroValues[m] = pars
        count = (data_by_month[data.columns[1]] == 0).sum()/len(data_by_month)
        p0[m] = count.round(3)

    # Optimization: we find the ACS parameters that minimize the objective function    
    # Define objective function for the Nelder-Mead method (Parametric ACS vs Empirical ACS of the Observed TS)
    def objective(par):
        par_acf = np.zeros((1, lags+1))
        if parametric_acf == 'parII':
            shape, scale = par
            for lag in range(0,lags+1,1):
                par_acf[0,lag] = acfparetoII(lag, shape, scale)
        elif parametric_acf == 'wei':
            shape, scale = par
            for lag in range(0,lags+1,1):
                par_acf[0,lag] = acfweibull(lag, shape, scale)
        elif parametric_acf == 'burrXII':
            scale, shape1, shape2 = par
            for lag in range(0,lags+1,1):
                par_acf[0,lag] = acfburrXII(lag, scale, shape1, shape2)
        return similaritymeasures.mae(par_acf,emp_acf)
    
    mae = np.zeros([len(months),3])
    for m in range(0,len(months),1):    
        emp_acf = sm.tsa.acf(stratified_data[m+1]['Value'], nlags=lags, fft=False)
        empAcs_Observed[m] = emp_acf
        if (parametric_acf == 'wei' or parametric_acf == 'parII'):
            first_guess = [2, 2]
            bnds = ((0.05, None), (0.05, None)) 
            result = minimize(objective, first_guess,bounds=bnds,  method = 'nelder-mead')
            mae[m,0] = result.x[0].round(2)
            mae[m,1] = result.x[1].round(2)
        elif parametric_acf == 'burrXII':
            first_guess = [1, 1, 1]
            bnds = ((0.05, None), (0.05, None), (0.05, None)) 
            result = minimize(objective, first_guess,bounds=bnds,  method = 'nelder-mead')
            mae[m,0] = result.x[0].round(2)
            mae[m,1] = result.x[1].round(2)
            mae[m,2] = result.x[2].round(2)

    # Here, we estimate the parametric ACS of the Observed TS using the optimal parameters (par_acf_opt)
    par_acf_opt = np.zeros((len(months), lags))
    for m in range(0,len(months),1):
        for lag in range(0,lags,1):
            if parametric_acf == 'parII':
                par_acf_opt[m,lag] = acfparetoII(lag, mae[m,0], mae[m,1])
            elif parametric_acf == 'wei':
                par_acf_opt[m,lag] = acfweibull(lag, mae[m,0], mae[m,1])
            elif parametric_acf == 'burrXII':
                par_acf_opt[m,lag] = acfburrXII(lag, mae[m,0], mae[m,1], mae[m,2])
            emp_acf = sm.tsa.acf(stratified_data[m+1]['Value'], nlags=lags, fft=False)
            
    return stratified_data, p0, pars_NonZeroValues, par_acf_opt, u_t, Theoretic, no0values, mae, empAcs_Observed


def ACF_MixUnif_TS(data, res, lags, parametric_acf = ('parII','wei', 'burrXII')):
    u_t = res[4]
    label= []
    for m in range(1,13,1): label.append( 'Month_'+str(m))
    # Rename variables
    data.columns = ['Time', 'Value']
    
    # Convert Time into Python Time object
    Times = pd.to_datetime(data["Time"])
    data = data.assign(Time = Times)
    
    # Assign 0 to NA values
    new_value = data['Value'].fillna(data['Value'].mean())
    data = data.assign(Value = new_value)
    
    # Add month variable
    value = pd.DatetimeIndex(data['Time']).month
    data = data.assign(month = value)
    
    months = data.month.unique()
    emp_u_acf = {}
            
    def objective(par):
        par_acf = np.zeros((1, lags))
        if parametric_acf == 'parII':
            shape, scale = par
            for lag in range(0,lags,1):
                par_acf[0,lag] = acfparetoII(lag, shape, scale)
        elif parametric_acf == 'wei':
            shape, scale = par
            for lag in range(0,lags,1):
                par_acf[0,lag] = acfweibull(lag, shape, scale)
        elif parametric_acf == 'burrXII':
            shape1, shape2, scale = par
            for lag in range(0,lags,1):
                par_acf[0,lag] = acfburrXII(lag, scale, shape1, shape2)
        return similaritymeasures.mae(par_acf,emp_acf)

    mae_u = np.zeros([len(months),3])
    for m in range(0,len(months),1):
        emp_acf = sm.tsa.acf(u_t[m+1], nlags=lags-1, fft=False)
        emp_u_acf[m] = emp_acf
        if (parametric_acf == 'wei' or parametric_acf == 'parII'):
            first_guess = [2, 2]
            bnds = ((0.05, None), (0.05, None)) 
            result = minimize(objective, first_guess,bounds=bnds,  method = 'nelder-mead')
            mae_u[m,0] = result.x[0]
            mae_u[m,1] = result.x[1]
        elif parametric_acf == 'burrXII':
            #mae_u = np.zeros([len(months),3])
            first_guess = [1, 1, 1]
            bnds = ((0.05, None), (0.05, None), (0.05, None)) 
            result = minimize(objective, first_guess,bounds=bnds,  method = 'nelder-mead')
            mae_u[m,0] = result.x[0]
            mae_u[m,1] = result.x[1]
            mae_u[m,2] = result.x[2]

    # a) We estimate the parametric ACS of the Mixed-Uniform TS using the optimal parameters (par_acf_u_opt)
    # b) Plots: Observed Parametric ACS vs Empirical ACS for the Mixed-Uniform TS (for each month)
    par_acf_u_opt = np.zeros((12, lags))
    par_acf_u_opt_month = np.zeros((1, lags))
    
    for m in range(1,len(label)+1,1):
        for lag in range(0,lags,1):
            if parametric_acf == 'parII':
                par_acf_u_opt[m-1,lag] = acfparetoII(lag, mae_u[m-1,0], mae_u[m-1,1]) 
            elif parametric_acf == 'wei':
                par_acf_u_opt[m-1,lag] = acfweibull(lag, mae_u[m-1,0], mae_u[m-1,1])
            elif parametric_acf == 'burrXII':
                par_acf_u_opt[m-1,lag] = acfburrXII(lag, mae_u[m-1,0], mae_u[m-1,1], mae_u[m-1,2])
    
    return par_acf_u_opt    


def SimulatedTS(data, res, res1, ptsACTF, lags, marg_distr = ('gamma','ggamma','norm', 'lognorm', 'beta', 'burrIII', 'weibull', 'weibull3'), show_grid=True):
    label= []
    for m in range(1,13,1): label.append( 'Month_'+str(m))
    p0 = res[1]
    par_acf_opt = res[3]
    pars_NonZeroValues = res[2]
    par_acf_u_opt = res1
    
    # Rename variables
    data.columns = ['Time', 'Value']
    # Assign 0 to NA values
    new_value = data['Value'].fillna(data['Value'].mean())
    data = data.assign(Value = new_value)
    
    # Convert Time into Python Time object
    Times = pd.to_datetime(data["Time"])
    data = data.assign(Time = Times)
    # Add month variable
    value0 = pd.DatetimeIndex(data['Time']).month
    data = data.assign(month = value0)
    # Add year variable
    value1 = pd.DatetimeIndex(data['Time']).year
    data = data.assign(year = value1)
    
    years = data.year.unique()
    months = data.month.unique()
    n = np.zeros((len(years), len(months)))
    rho_z_gaus = {}
    
    # Take p0, b, c1 and c2 from ptsACTF matrix
    p0_grid = np.squeeze(np.asarray(ptsACTF.iloc[:, [0]]))
    b = np.squeeze(np.asarray(ptsACTF.iloc[:, [1]]))
    c1 = np.squeeze(np.asarray(ptsACTF.iloc[:, [2]]))
    c2 = np.squeeze(np.asarray(ptsACTF.iloc[:, [3]]))     
    
    # Inizialize the Gaussian TS
    #np.random.seed(0)                                        
    Ts0 = np.random.normal(0, 1, lags-1)
    
    # Store the size of each month in each year (n)
    for y in range(0,len(years),1):
        for m in range(0,len(months),1):
            df2 = pd.DataFrame(data[(data['month'] == m+1) & (data['year'] == years[y])])
            n[y,m] = len(df2)
    
    # Store the size of each month in each year in an array (dim0)
    dim0 = []
    for l in n: dim0.extend(l)
    # Include in dim0 the first steps (# of lags)
    dim = np.concatenate([[lags-1],dim0])
    
    # We construct the Moving Windows to generate the Gaussian TS: we derive the lower and upper bounds of the windows
    # Upper bounds (dim2), Lower Bounds (dim1) are defined using cumulative sum of the elements in dim
    dim2 = np.array(np.cumsum(dim, axis=None, dtype=None, out=None))
    dim1 = dim2-lags+1
    
    # ind_to_add: Sequence of the Cumulative sum of the numbers of months in Observed TS [0,12,24,36,...]
    ind_to_add = np.arange(0, len(years)*len(months), len(months)).tolist()
    
    # Generate Gaussian TS
    for y in range(0,len(years),1):
        for m in range(1,len(months)+1,1):
            interpolate_p0 = p0[m]
            b_interp = interp1d(p0_grid, b)
            c1_interp = interp1d(p0_grid, c1)
            c2_interp = interp1d(p0_grid, c2)
           
            df2 = pd.DataFrame(data[(data['month'] == m) & (data['year'] == years[y])])
            n = len(df2)
            
            # ACTF computed from Eq. 11
            rho_u = par_acf_u_opt[m-1]
            rho_z = ((((1+b_interp(interpolate_p0)*(rho_u)**c1_interp(interpolate_p0))**c2_interp(interpolate_p0))-1)**c2_interp(interpolate_p0))/((((1+b_interp(interpolate_p0))**c2_interp(interpolate_p0))-1)**c2_interp(interpolate_p0))
            rho_z_gaus[m] = rho_z
            
            #Start to generate Gaussian TS using AR(p)
            #Create matrix P and its inverse
            P = np.zeros((len(rho_z)-1, len(rho_z)-1))
 
            for j in range(len(rho_z)-1):
                for i in range(len(rho_z)-1):
                    ind = abs(i-j)
                    P[j,i] = rho_z[ind] 
            
            Pinv = np.linalg.inv(P)
            rho_z2 = np.delete(rho_z, 0)
            alpha = Pinv.dot(rho_z2)
            mu_eps = 0
            sigma_eps = math.sqrt(1-sum(alpha*rho_z2))
            alpha_flipped = np.flipud(alpha)
            for k in range(n):
                eps = np.random.normal(mu_eps, sigma_eps, 1)
                to_be_add = sum(Ts0[int(dim1[(m-1)+ind_to_add[y]]+k):int(dim2[(m-1)+ind_to_add[y]]+k)]*alpha_flipped)+ eps
                Ts0 = np.concatenate((Ts0, to_be_add))    
                
    GaussianTS = pd.DataFrame(Ts0[lags-1:], columns = ['Gauss_Ts'])
    date = data['Time']
    m = pd.DatetimeIndex(data['Time']).month
    y = pd.DatetimeIndex(data['Time']).year
    
    # Add Time, month and year variables
    GaussianTS = GaussianTS.assign(Time = date)
    GaussianTS = GaussianTS.assign(month = m)
    GaussianTS = GaussianTS.assign(year = y)
    
    # STEP 5: Retrieve original time series using Eq. 13         
    Simulated_Ts = []
    for y in range(0,len(years),1): # 
        for m in range(1,len(months)+1,1):
            temp = GaussianTS[(GaussianTS['month'] == m) & (GaussianTS['year'] == years[y])] 
            zm = temp.iloc[:,[0]]
            n = len(zm)
            # Generate a TS of zeros
            Simulated_TS = np.zeros(n)
            
            # Compute the quantile z_p0_m in Eq. 13
            z_p0_m = stats.norm.ppf(p0[m], loc=0, scale=1)
            z_filt0 = zm[zm['Gauss_Ts'] > z_p0_m]
            z_arr = np.squeeze(np.asarray(zm.iloc[:, [0]]))

            if marg_distr == 'gamma':
                Simulated_TS[z_arr > z_p0_m] = stats.gamma.ppf((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),a = pars_NonZeroValues[m][0], loc = pars_NonZeroValues[m][1], scale=pars_NonZeroValues[m][2])
            elif marg_distr == 'ggamma':
                Simulated_TS[z_arr > z_p0_m] = qggamma((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),shape1 = pars_NonZeroValues[m][0],shape2 = pars_NonZeroValues[m][1], scale = pars_NonZeroValues[m][2])
            elif marg_distr == 'ge4':
                temporary = np.array((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]))
                temporary1 = []
                for i in range(0,len(temporary),1):
                    temporary1.append(qge4(temporary[i],shape1 = pars_NonZeroValues[m][0],shape2 = pars_NonZeroValues[m][1], scale = pars_NonZeroValues[m][2]))
                temporary2 = pd.DataFrame(temporary1)
                Simulated_TS[z_arr > z_p0_m] = temporary2[0]   
                
            elif marg_distr == 'norm':
                Simulated_TS[z_arr > z_p0_m] = stats.norm.ppf((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),loc = pars_NonZeroValues[m][0],scale = pars_NonZeroValues[m][1])
            elif marg_distr == 'lognorm':
                Simulated_TS[z_arr > z_p0_m] = stats.lognorm.ppf((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),s = pars_NonZeroValues[m][0], scale = pars_NonZeroValues[m][1])
            elif marg_distr == 'weibull3':
                Simulated_TS[z_arr > z_p0_m] = stats.weibull_min.ppf((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),c = pars_NonZeroValues[m][0], loc = pars_NonZeroValues[m][1], scale = pars_NonZeroValues[m][2])
            elif marg_distr == 'burrIII':
                Simulated_TS[z_arr > z_p0_m] = qburrIII((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),shape1 = pars_NonZeroValues[m][0],shape2 = pars_NonZeroValues[m][1], scale = pars_NonZeroValues[m][2])
            elif marg_distr == 'beta':
                Simulated_TS[z_arr > z_p0_m] = stats.beta.ppf((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),a = pars_NonZeroValues[m][0], b = pars_NonZeroValues[m][1])
            elif marg_distr == 'burrXII':
                Simulated_TS[z_arr > z_p0_m] = qburrXII((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),shape1 = pars_NonZeroValues[m][0],shape2 = pars_NonZeroValues[m][1], scale = pars_NonZeroValues[m][2])
            elif marg_distr == 'skewnorm':
                Simulated_TS[z_arr > z_p0_m] = stats.skewnorm.ppf((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),a = pars_NonZeroValues[m][0],loc = pars_NonZeroValues[m][1], scale = pars_NonZeroValues[m][2])
            elif marg_distr == 'weibull':
                Simulated_TS[z_arr > z_p0_m] = stats.weibull_min.ppf((stats.norm.cdf(z_filt0['Gauss_Ts'], loc=0, scale=1)- p0[m])/(1-p0[m]),c = pars_NonZeroValues[m][0],  scale = pars_NonZeroValues[m][1])
      
            df_temp = pd.DataFrame(Simulated_TS, columns = ['Value'])
            m_sim = np.repeat(months[m-1], len(temp))
            y_sim = np.repeat(years[y], len(temp))
            
            # Add Time, month and year variables
            df_temp = df_temp.assign(month = m_sim)
            df_temp = df_temp.assign(year = y_sim)
            Simulated_Ts.append(df_temp)

    SimulatedTS = pd.concat(Simulated_Ts,ignore_index=True)
    SimulatedTS = pd.DataFrame(SimulatedTS)
    
    return SimulatedTS, GaussianTS
    
def Report_ObservedTS(res, lags, method = ('dist','acf', 'stat'),marg_distr = ('logistic', 'skewnorm','gamma','ggamma','norm', 'lognorm', 'burrIII','burrXII', 'weibull', 'beta'), parametric_acf = ('parII','wei', 'burrXII')): 
    label= []
    for m in range(1,13,1): label.append( 'Month_'+str(m))
    stratified_data = res[0]
    p0 = res[1]
    pars_NonZeroValues = res[2]
    par_acf_opt =res[3]
    Theoretic = res[5]
    no0values = res[6]
    acs_par = res[7]
    empAcs_Observed = res[8]
    
    results = {}
    
    # Probability density and the Histogram
    if method == 'dist':
        fig, axs = plt.subplots(3,4, figsize=(15, 8), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=0.25)
        fig.text(0.5, 0.04, 'Nonzero values', ha='center', size=15)
        fig.text(0.07, 0.5, 'Probability density', va='center', rotation='vertical', size=15)
        col_patch = mlines.Line2D([], [], color=(152/255,190/255,88/255), marker='_', 
                          markersize=3, label='Fitted')
        fig.legend(handles = [col_patch],edgecolor='white',handlelength=0.8,bbox_to_anchor=(.9, 0.85), 
                   borderaxespad=0,fontsize=12, markerfirst=True, markerscale=3) #loc="center right",
        
        axs = axs.ravel()
        for m in range(1,len(label)+1,1):   
            axs[m-1].plot(no0values[m]['Value'], Theoretic[m], color=(152/255,190/255,88/255), linewidth=3)
            axs[m-1].hist(no0values[m]['Value'], bins=20, density = True, stacked=False, edgecolor='white', linewidth=.5)
            axs[m-1].set_title(label[m-1], size=14)
            axs[m-1].set_xlabel('',size=12)
            axs[m-1].set_ylabel('',size=12)
            axs[m-1].tick_params(axis='both', which='major', labelsize=12.5)
            axs[m-1].spines['right'].set_visible(False)
            axs[m-1].spines['top'].set_visible(False)
        fig.savefig('Distr_ObservedTS.png', bbox_inches='tight')

    # Plots: Observed Parametric ACS vs Empirical ACS of the Observed TS (for each month)  
    elif method == 'acf':
        fig, axs = plt.subplots(3,4, figsize=(15, 8), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=0.25)
        fig.text(0.5, 0.04, 'Lags', ha='center', size=15)
        fig.text(0.08, 0.5, 'Autocorrelation', va='center', rotation='vertical', size=15)
        col1_patch = mlines.Line2D([], [],  marker='o', linestyle='None',
                          markersize=3, label='Empirical')
        col2_patch = mlines.Line2D([], [], color=(168/255, 164/255, 162/255), marker='_', linestyle='None',
                          markersize=3, label='Target')
        fig.legend(bbox_to_anchor=(.9, 0.85), handles = [col1_patch, col2_patch],markerscale=2,handlelength=0.8,
                   loc="center right", borderaxespad=0, edgecolor='white',fontsize=12)
        axs = axs.ravel()

        for m in range(1,len(label)+1,1):
            emp_acf = empAcs_Observed[m-1]
            resu=np.concatenate((emp_acf[:10],emp_acf[14::5]))
            axs[m-1].plot(range(0,len(par_acf_opt[m-1]),1), par_acf_opt[m-1], color=(168/255, 164/255, 162/255),linewidth=1.5)
            axs[m-1].plot(np.where(np.isin(emp_acf, resu))[0],  np.concatenate((emp_acf[:10],emp_acf[14::5])), marker="o", linestyle='')
            axs[m-1].set_title(label[m-1], size=14)
            axs[m-1].set_ylim([-0.05,1.05])
            axs[m-1].set_xlabel('',size=12)
            axs[m-1].set_ylabel('',size=12)
            axs[m-1].tick_params(axis='both', which='major', labelsize=12.5)
            axs[m-1].spines['right'].set_visible(False)
            axs[m-1].spines['top'].set_visible(False)
    
        fig.savefig('Acf_ObservedTS.png', bbox_inches='tight')     
        
    # Summary statistics
    elif method == 'stat':
        for m in range(1,len(label)+1,1):
            lmom = lmoments(stratified_data[m]['Value'])
            lmoment = [round(lmom[0],2), round(lmom[1],2), round(lmom[2],2), round(lmom[3],2)] 
            summary=pd.DataFrame(stratified_data[m]['Value'].describe(percentiles=[.25, .5, .75, .9, .95,.99]).round(2))
            summary = summary.T
            l_mom = pd.DataFrame(lmoment).T
            l_mom.columns = ['l-mom 1', 'l-mom 2', 'l-mom 3', 'l-mom 4'] #, 'month'
            summary.reset_index(drop=True, inplace=True)
            l_mom.reset_index(drop=True, inplace=True)
            result = pd.concat([summary, l_mom], axis=1)
            results[m]=result
        resu = pd.concat(results)
        resu = resu.drop('count', axis=1)
        resu = pd.DataFrame(resu.reset_index(drop=True))
        resu.insert(0, "Month", ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6','Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11',  'Month_12'], True)
        pardist_est = np.transpose(pd.DataFrame(pars_NonZeroValues)).round(2)
        if  marg_distr == 'gamma':
            pardist_est.columns = ['a', 'scale']
        elif marg_distr == 'beta':
            pardist_est.columns = ['a', 'b']
        elif marg_distr == 'norm':
            pardist_est.columns = ['loc', 'scale']
        elif marg_distr == 'lognorm':
            pardist_est.columns = ['s', 'scale']
        elif marg_distr == 'skewnorm':
            pardist_est.columns = ['a','loc', 'scale']
        elif marg_distr == 'logistic':
            pardist_est.columns = ['loc', 'scale']
        elif marg_distr == 'weibull':
            pardist_est.columns = ['c', 'scale']
        elif marg_distr == 'weibull3':
            pardist_est.columns = ['c', 'loc','scale']
        elif marg_distr == 'ggamma' or marg_distr =='ge4' or marg_distr == 'burrIII' or marg_distr =='burrXII':
            pardist_est.columns = ['shape1',  'shape2',  'scale']
        pardist_est.reset_index(drop=True, inplace=True)
        
        if  parametric_acf == 'parII' or parametric_acf == 'wei':
            paracs_est = pd.DataFrame(acs_par).iloc[ :,[0,1]].round(2) 
            paracs_est.columns = ['b',  'c']
        elif parametric_acf == 'burrXII':
            paracs_est = pd.DataFrame(acs_par).iloc[ :,[0,1,2]].round(2) 
            paracs_est.columns = ['shape1',  'shape2',  'scale']

        paracs_est.reset_index(drop=True, inplace=True)
        summary_stats = pd.concat([resu, pardist_est, paracs_est], axis=1)
        
        html_df = HTML(summary_stats.to_html(index=False))

        return html_df
    
    
def Report_SimulatedTS(res, res2, method = ('acf','stat','diff_stat')): 
    label= []
    for m in range(1,13,1): label.append( 'Month_'+str(m))
    styles = [dict(selector="caption",props=[("text-align", "center"),("font-size", "150%"),("color", 'black')])]
    ObservedTS = res[0]
    SimulatedTS = res2[0]
    par_acf_opt = res[3]  

    if method == 'acf':
        fig, axs = plt.subplots(3,4, figsize=(15, 8), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=0.25)
        fig.text(0.5, 0.04, 'Lags', ha='center', size=15)
        fig.text(0.08, 0.5, 'Autocorrelation', va='center', rotation='vertical', size=15)
        col1_patch = mlines.Line2D([], [],  marker='o', linestyle='None',
                          markersize=3, label='Simulated', color=(181/255, 57/255, 34/255))
        col2_patch = mlines.Line2D([], [], marker='_', linestyle='-',markersize=3,
                         label='Target', color=(168/255, 164/255, 162/255))
        fig.legend(bbox_to_anchor=(.9, 0.85), handles = [col1_patch, col2_patch],markerscale=2,handlelength=0.8,
                   loc="center right", borderaxespad=0, edgecolor='white',fontsize=12)
        axs = axs.ravel()

        for m in range(1,len(label)+1,1):
            Simulated_TS_month = SimulatedTS[SimulatedTS['month'] == m]
            acf_Ts_simulated_month = sm.tsa.acf(Simulated_TS_month['Value'], nlags=lags, fft=False)  
            resu=np.concatenate((par_acf_opt[m-1][:10],par_acf_opt[m-1][14::5]))
            resu1=np.concatenate((acf_Ts_simulated_month[:10],acf_Ts_simulated_month[14::5]))
            axs[m-1].plot(np.where(np.isin(par_acf_opt[m-1], resu))[0],  np.concatenate((par_acf_opt[m-1][:10],par_acf_opt[m-1][14::5])),  color=(168/255, 164/255, 162/255),linewidth=2)
            axs[m-1].plot(np.where(np.isin(acf_Ts_simulated_month, resu1))[0],  np.concatenate((acf_Ts_simulated_month[:10],acf_Ts_simulated_month[14::5])),color=(181/255, 57/255, 34/255), marker="o", linestyle='')
            axs[m-1].set_title(label[m-1], size=14)
            axs[m-1].set_ylim([-0.05,1.05])
            axs[m-1].set_xlabel('',size=12)
            axs[m-1].set_ylabel('',size=12)
            axs[m-1].tick_params(axis='both', which='major', labelsize=12.5)
            axs[m-1].spines['right'].set_visible(False)
            axs[m-1].spines['top'].set_visible(False)
        fig.savefig('Acf_SimulatedTS.png', bbox_inches='tight')
        
    elif method == 'stat':
        results = {}
        for m in range(1,13,1):
            stratified_data = pd.DataFrame(SimulatedTS[SimulatedTS['month'] == m])
            lmom = lmoments(stratified_data['Value'])
            lmoment = [round(lmom[0],2), round(lmom[1],2), round(lmom[2],2), round(lmom[3],2)] 
            summary = pd.DataFrame(stratified_data['Value'].describe(percentiles=[.25, .5, .75, .9, .95,.99]).round(2))
            summary = summary.T
            l_mom = pd.DataFrame(lmoment).T
            l_mom.columns = ['l-mom 1', 'l-mom 2', 'l-mom 3', 'l-mom 4'] 
            summary.reset_index(drop=True, inplace=True)
            l_mom.reset_index(drop=True, inplace=True)
            result = pd.concat([summary, l_mom], axis=1)
            results[m]=result
        resu_sim = pd.concat(results)
        resu_sim = resu_sim.drop('count', axis=1)
        resu_sim = pd.DataFrame(resu_sim.reset_index(drop=True))
        resu_sim.insert(0, "Month", ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6','Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11',  'Month_12'], True)

        resu_sim.style.set_caption("Simulated TS: Summary statistics").set_table_styles(styles)
        display(HTML(resu_sim.to_html(index=False)))
            
    elif method == 'diff_stat':
        results = {}
        for m in range(1,13,1):
            stratified_data = pd.DataFrame(SimulatedTS[SimulatedTS['month'] == m])
            lmom = lmoments(stratified_data['Value'])
            lmoment = [round(lmom[0],2), round(lmom[1],2), round(lmom[2],2), round(lmom[3],2)] 
            summary = pd.DataFrame(stratified_data['Value'].describe(percentiles=[.25, .5, .75, .9, .95,.99]).round(2))
            summary = summary.T
            l_mom = pd.DataFrame(lmoment).T
            l_mom.columns = ['l-mom 1', 'l-mom 2', 'l-mom 3', 'l-mom 4'] 
            summary.reset_index(drop=True, inplace=True)
            l_mom.reset_index(drop=True, inplace=True)
            result = pd.concat([summary, l_mom], axis=1)
            results[m]=result
        resu_sim = pd.concat(results)
        resu_sim = resu_sim.drop('count', axis=1)
        resu_sim = pd.DataFrame(resu_sim.reset_index(drop=True))
        resu_sim.insert(0, "Month", ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6','Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11',  'Month_12'], True)
        
        for m in range(1,13,1):
            stratified_data = ObservedTS[m]
            lmom = lmoments(stratified_data['Value'])
            lmoment = [round(lmom[0],2), round(lmom[1],2), round(lmom[2],2), round(lmom[3],2)] 
            summary=pd.DataFrame(stratified_data['Value'].describe(percentiles=[.25, .5, .75, .9, .95,.99]).round(2))
            summary = summary.T
            l_mom = pd.DataFrame(lmoment).T
            l_mom.columns = ['l-mom 1', 'l-mom 2', 'l-mom 3', 'l-mom 4'] 
            summary.reset_index(drop=True, inplace=True)
            l_mom.reset_index(drop=True, inplace=True)
            result = pd.concat([summary, l_mom], axis=1)
            results[m]=result
        resu_observed = pd.concat(results)
        resu_observed = resu_observed.drop('count', axis=1)
        resu_observed = pd.DataFrame(resu_observed.reset_index(drop=True))
        resu_observed.insert(0, "Month", ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6','Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11',  'Month_12'], True)

        table = resu_observed
        colnames = table.columns
        for m in range(1, 15,1):
            table[colnames[m]] = round(resu_sim[colnames[m]]-resu_observed[colnames[m]],2)
        table = pd.DataFrame(data=table)
        display(HTML(table.to_html(index=False)))

    return 

def PyCoSMoS_Plots(data, res2): 
    SimulatedTS = res2[0]
    label= []
    for m in range(1,13,1): label.append( 'Month_'+str(m))
    # Rename variables
    data.columns = ['Time', 'Value']
    new_value = data['Value'].fillna(data['Value'].mean())
    data = data.assign(Value = new_value)
    
    data['Time'] = pd.to_datetime(data['Time'])
    data_temp = data
    data_temp['Time'] = data_temp['Time'].astype(str)
    data_temp['YM'] = pd.to_datetime(data_temp['Time']).dt.to_period('M')
    
    data_temp['YM'] = pd.to_datetime(data_temp['Time']).dt.to_period('M')
    data_temp['YM'] = data_temp['YM'].astype(str)
    data_temp = data_temp.set_index('YM')
    data_temp0 = data_temp.drop('Time', axis=1)
    
    SimulatedTS = pd.DataFrame(SimulatedTS)

    # Plot of the observed Time Series
    data_temp1 = data
    data_temp1 = data_temp1.drop('Value', axis=1)
    value = SimulatedTS['Value']
    data_temp1 = data_temp1.assign(Values = value)
    data_temp1['YM'] = pd.to_datetime(data_temp1['Time']).dt.to_period('M')

    data_temp1['YM'] = data_temp1['YM'].astype(str)
    data_temp1 = data_temp1.set_index('YM')
    data_temp2 = data_temp1.drop('Time', axis=1)    
    
    fig = plt.figure(figsize = (30, 18), layout="constrained")
    spec0 = fig.add_gridspec(ncols=2, nrows=3, width_ratios=[1, 1], height_ratios=[1.6, .7, 2])
    spec01 = spec0[0].subgridspec(ncols=1, nrows=2)
    spec02 = spec0[2].subgridspec(ncols=3, nrows=1)

    ax1 = fig.add_subplot(spec01[0])
    ax2 = fig.add_subplot(spec01[1])
    ax3 = fig.add_subplot(spec02[0])
    ax4 = fig.add_subplot(spec02[1])
    ax5 = fig.add_subplot(spec02[2])

    def y_fmt(x, pos):
        return '{:.0f}'.format(x)

    # Aggiunta del primo grafico alla griglia
    data_temp0.plot(label = 'Observed', linewidth=0.5, ax=ax1) #
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_ylabel("Precipitation (mm)", fontsize = 16) 
    ax1.set_xlabel("Time", fontsize = 16)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.legend().remove()
    ax1.legend(['Observed'], loc='upper right', fontsize=14, handlelength=0, frameon=True,
                       edgecolor='lightgray', facecolor=(227/255, 222/255, 220/255), shadow=True)

    # Aggiunta del primo grafico alla griglia
    data_temp2.plot(label = 'Simulated', color= (182/255, 90/255, 73/255),linewidth=0.5 ,  ax=ax2)#
    ax2.spines['right'].set_visible(False) 
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel("Precipitation (mm)", fontsize = 16)
    ax2.set_xlabel("Time", fontsize = 16)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
    ax2.legend().remove()
    ax2.legend(['Simulated'], loc='upper right', fontsize=14, handlelength=0, frameon=True,
                       edgecolor='lightgray', facecolor= (227/255, 222/255, 220/255), shadow=True)

    #Plot 3
    y = pd.DataFrame(data[data['Value'] != 0])
    yy = pd.DataFrame(SimulatedTS[SimulatedTS['Value'] != 0])

    # Creazione del plot
    sns.kdeplot(y['Value'], alpha=0.4,shade=True, ax=ax3,fill=True, common_norm=False, linewidths=2)
    sns.kdeplot(yy['Value'], color=(182/255, 90/255, 73/255), alpha=0.4,shade=True, ax=ax3)
    
    # Impostazione del titolo e delle etichette degli assi
    ax3.set_ylabel('Density', fontsize = 16)
    ax3.set_xlabel('Nonzero values', fontsize = 16)
    ax3.tick_params(axis='both', labelsize=15)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    #Seasonal components
    df_temp = data
    df_temp['Time'] = pd.to_datetime(df_temp['Time'], errors='coerce')
    df_temp = (df_temp.groupby([pd.Grouper(key='Time', freq='MS')])['Value']
           .sum().reset_index())

    value = pd.DatetimeIndex(df_temp['Time']).month
    df_temp = df_temp.assign(month = value)
    df_temp0 = df_temp.groupby(['month']).mean()

    df_temp1 = pd.DataFrame(SimulatedTS)
    df_temp1 = df_temp1.groupby(['year','month'])["Value"].sum()
    df_temp2 = pd.DataFrame(df_temp1.groupby(['month']).mean())
    labels=[]
    for m in range(1,13,1): labels.append(str(m))
    df_temp2 = df_temp2.assign(Month = labels)
    ax4.plot(labels, df_temp0['Value'], linewidth=2)
    ax4.plot(labels, df_temp2['Value'],linewidth=2, color=(182/255, 90/255, 73/255))
    ax4.set_xlabel('Months',size=16)
    ax4.set_ylabel('Precipitation (mm)',size=16) 
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)

    #Autocorrelation function
    acf_ObservedTs = sm.tsa.acf(data['Value'], nlags=lags, fft=False)
    acf_SimulatedTs = sm.tsa.acf(SimulatedTS['Value'], nlags=lags, fft=False)
    resu=np.concatenate((acf_ObservedTs[:10],acf_ObservedTs[14::5]))
    resu1=np.concatenate((acf_SimulatedTs[:10],acf_SimulatedTs[14::5]))

    ax5.plot(np.where(np.isin(acf_ObservedTs, resu))[0],  np.concatenate((acf_ObservedTs[:10],acf_ObservedTs[14::5])) , 'o')
    ax5.plot(np.where(np.isin(acf_SimulatedTs, resu1))[0], np.concatenate((acf_SimulatedTs[:10],acf_SimulatedTs[14::5])) , color=(182/255, 90/255, 73/255), marker='o', linestyle='')
    ax5.set_xlabel('Lags',size=16)
    ax5.set_ylabel('ACF',size=16)
    ax5.tick_params(axis='both', which='major', labelsize=15)
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.set_ylim([-0.05,1.05])
    fig.savefig('PyCoSMoS_Plots.png', bbox_inches='tight')

    # Visualizzazione del plot
    plt.show()
    
    return fig
    
