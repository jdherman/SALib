from __future__ import division
from __future__ import print_function
import itertools
import time
import numpy as np
from scipy import (stats,special)
import matplotlib.pyplot as plt
from . import common_args
from ..util import read_param_file, ResultDict

def analyze(problem, X, Y, options={}):
    # High-Dimensional Model Representation (HDMR) using B-spline functions  
    # for variance-based global sensitivity analysis (GSA) with correlated   
    # and uncorrelated inputs. This function uses as input a N x d matrix    
    # of N different d-vectors of model inputs (factors/parameters) and a    
    # N x 1 vector of corresponding model outputs and returns to the user    
    # each factor's first, second, and third order sensitivity coefficient   
    # (separated in total, structural and correlative contributions), an     
    # estimate of their 95% confidence intervals (from bootstrap method)     
    # and the coefficients of the significant B-spline basis functions that  
    # govern output, Y (determined by an F-test of the error residuals of    
    # the HDMR model (emulator) with/without a given first, second and/or    
    # third order B-spline). These coefficients define an emulator that can  
    # be used to predict the output, Y, of the original (CPU-intensive)      
    # model for any d-vector of model inputs. For uncorrelated model inputs  
    # (columns of X are independent), the HDMR sensitivity indices reduce    
    # to a single index (= structural contribution), consistent with their   
    # values derived from commonly used variance-based GSA methods.          
   
    # INPUT ARGUMENTS   
    #  problem     Problem definition dictionary
    #   ['names']     parameter names
    #   ['num_vars']  model dimensioniality
    #   ['bounds']    parameter bounds                                                 
    #  X           Nxd matrix: N vectors of d parameters                    
    #  Y           Nx1 vector: single model output for each row of X        
    #  options     (optional) dictionary: Fields specify HDMR variables      
    #   ['graphics']  integer [0,1]: graphical output?             (def: 1)
    #   ['maxorder']  integer [1-3]: HDMR expansion max order      (def: 2)    
    #   ['maxiter']   integer [1-1000]: max iterations backfitting (def: 100)   
    #   ['m']         integer [2-5]: # B-spline intervals          (def: 2)    
    #   ['K']         integer [1-100] # bootstrap iterations       (def: 20)  
    #   ['R']         integer [100-N/2] # bootstrap samples        (def: N/2)        
    #   ['alfa']      real [0.5-1]: confidence interval F-test     (def: 0.95)
    #   ['lambdax']   real [0-10]: regularization term             (def: 0.01) 

    # OUTPUT ARGUMENTS
    #  Si (Sensitivity Indices) Python Dictionary
    #   ['Sa']       Uncorrelated contribution
    #   ['Sa_CI']    Confidence interval of Sa
    #   ['Sb']       Correlated contribution
    #   ['Sb_CI']    Confidence interval of Sb
    #   ['S']        Total contribution of a particular term
    #   ['S_CI']     Confidence interval of S
    #   ['ST']       Total contribution of a particular dimension/parameter
    #   ['ST_CI']    Confidence interval of ST
    #   ['Sa']       Uncorrelated contribution
    #   ['select']   Number of selection (F-Test)


    
    
    # Initial part: Check input arguments and define HDMR variables
    settings = hdmr_setup(X,Y,options)
    init_vars = hdmr_init(X,Y,settings)
    # Sensitivity Analysis Computation with/without bootstraping
    SA,Em,RT,Y_em,idx = hdmr_compute(X,Y,settings,init_vars)
    # Finalize results
    Si = hdmr_finalize(problem,SA,Em,(settings[i] for i in [1,8,3]),(init_vars[i] for i in [0,2]))
    # Print results to console
    hdmr_print(Si,settings[1],settings[-1])
    # Now, print the figures
    if settings[2] == 1:
        hdmr_figures(problem,Si,Em,RT,X,Y,Y_em,idx)

    return Si


def hdmr_compute(X,Y,settings,init_vars):
    N, d, graphics, maxorder, maxiter, m, K, R, alfa, lambdax, print_to_console = settings
    Em, idx, SA, RT, Y_em, Y_id, m1, m2, m3, j1, j2, j3 = init_vars
    printProgressBar(0, K, prefix = 'SALib-HDMR :', suffix = 'Completed', length = 50)    
    for k in range(K):                                      # DYNAMIC PART: Bootstrap for confidence intervals
        tic = time.time()                                   # Start timer   
        Y_id[:,0] = Y[idx[:,k]]                             # Extract the "right" Y values
        SA['V_Y'][k,0] = np.var(Y_id)                       # Compute the variance of the model output
        Em['f0'][k] = np.sum(Y_id[:,0])/R                   # Mean of the output
        Y_res = Y_id - Em['f0'][k]                          # Compute residuals
        Y_em[:,j1], Y_res = hdmr_first_order(Em['B1'][idx[:,k],:,:],Y_res,R,Em['n1'], \
            m1,maxiter,lambdax)                             # 1st order component functions: ind/backfitting
        if ( maxorder > 1 ):
            Y_em[:,j2], Y_res = hdmr_second_order(Em['B2'][idx[:,k],:,:],Y_res,R,Em['n2'], \
                m2,lambdax)                                 # 2nd order component functions: individual 
        if ( maxorder == 3 ):
             Y_em[:,j3] = hdmr_third_order(Em['B3'][idx[:,k],:,:],Y_res,R,Em['n3'], \
                 m3,lambdax)                                # 3rd order component functions: individual
        Em['select'][:,k] = f_test(Y_id,Em['f0'][k],Y_em,R,alfa,m1,m2,m3,Em['n1'],Em['n2'], \
            Em['n3'],Em['n'])                               # Identify significant and insignificant terms
        Em['Y_e'][:,k] = Em['f0'][k] + np.sum(Y_em,axis=1)  # Store the emulated output
        Em['RMSE'][k] = np.sqrt(np.sum(np.square(Y_id - Em['Y_e'][:,k]))/R) # RMSE of emulator
        SA['S'][:,k],SA['Sa'][:,k],SA['Sb'][:,k]= ancova(Y_id,Y_em,SA['V_Y'][k], \
            R,Em['n'])                                      # Compute sensitivity indices
        printProgressBar(k + 1, K, prefix = 'SALib-HDMR :', \
            suffix = 'Completed', length = 50)              # Print progress bar
        RT[0,k] = time.time() - tic                         # Compute CPU time kth emulator
    
    return (SA,Em,RT,Y_em,idx)
 

def hdmr_setup(X,Y,options={}):
    # Get dimensionality of Numpy Array
    Sx = X.shape   
    Sy = Y.shape
    # Now check input-output mismatch
    error_msg = "SALib-HDMR Error: Matrix X contains only a single column: No point to do sensitivity analysis when d = 1. "
    assert Sx[1] != 1, error_msg
    error_msg = "SALib-HDMR Error: Number of samples, N, of N x d matrix X is insufficient. "
    assert Sx[0] >= 300, error_msg
    error_msg = "SALib-HDMR Error: Dimension mismatch. The number of rows, N of Y should match number of rows of X. "
    assert Sx[0] == Sy[0], error_msg
    error_msg = "SALib-HDMR Error: Y should be a N x 1 vector with one simulated output for each N parameter vectors. "
    assert Y.size == Sx[0], error_msg
    N = Sx[0]
    d = Sx[1]
    # Default options will be used if user does not define at least one option
    def_options = {'graphics': 1,'maxorder': 2,'maxiter': 100,'m': 2,'K': 20,'R': Sy[0]//2,'alfa': 0.95,'lambdax': 0.01,'print_to_console': 1}

    # Unpack the options 
    if ('graphics' in options):
        graphics = options['graphics']
        error_msg = "SALib-HDMR ERROR: Field \"graphics\" of options should take on the value of 0 or 1. "
        assert ismember(graphics,(0,1)), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"graphics\" of options not specified. Default: graphics = 1 assumed. ")
        graphics = def_options['graphics']
    
    # Unpack the options 
    if ('print_to_console' in options):
        print_to_console = options['print_to_console']
        error_msg = "SALib-HDMR ERROR: Field \"print_to_console\" of options should take on the value of 0 or 1. "
        assert ismember(graphics,(0,1)), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"print_to_console\" of options not specified. Default: graphics = 1 assumed. ")
        print_to_console = def_options['print_to_console']
    
    if 'maxorder' in options:
        maxorder = options['maxorder']
        error_msg = "SALib-HDMR ERROR: Field \"maxorder\" of options should be an integer with values of 1, 2 or 3. "
        assert ismember(maxorder,(1,2,3)), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"maxorder\" of options not specified. Default: maxorder = 2 used. ")
        maxorder = def_options['maxorder']

    # Important next check for maxorder - as maxorder relates to d
    if (d == 2):
        if (maxorder > 2):
            print("SALib-HDMR WARNING: Field \"maxorder\" of options set to 2 as d = 2 (X has two columns) ")
            maxorder = 2
    
    if 'maxiter' in options:
        maxiter = options['maxiter']
        error_msg = "SALib-HDMR ERROR: Field \"maxiter\" of options should be an integer between 1 to 1000. "
        assert ismember(maxiter,np.arange(1,1001)), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"maxiter\" of options not specified. Default: maxiter = 100 used. ")
        maxiter = def_options['maxiter']
    
    if 'm' in options:
        m = options['m']
        error_msg = "SALib-HDMR ERROR: Field \"m\" of options should be an integer between 1 to 5. "
        assert ismember(m,np.arange(1,6)), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"m\" of options not specified. Default: m = 4 used. ")
        m = def_options['m']
    
    if 'K' in options:
        K = options['K']
        error_msg = "SALib-HDMR ERROR: Field \"K\" of options should be an integer between 1 to 100. "
        assert ismember(K,np.arange(1,101)), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"K\" of options not specified. Default: K = 20 used. ")
        K = def_options['K']

    if 'R' in options:
        R = options['R']
        error_msg = "SALib-HDMR ERROR: Field \"R\" of options should be an integer between 300 and N, number of rows matrix X."
        assert ismember(R,np.arange(300,N+1)), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"R\" of options not specified. Default: R = Y/2 used.")
        R = def_options['R']
    
    if (K == 1) and (R != Sy[0]):
        R = Sy[0]
        print("SALib-HDMR DEFAULT: Field \"R\" of options is set to sample size (N) when K == 1. ")

    if 'alfa' in options:
        alfa = options['alfa']
        error_msg = "SALib-HDMR ERROR: Field \"alfa\" of options should be an integer between 0.5 to 1. "
        assert (alfa > 0.5 and alfa < 1.0), error_msg
    else:
        print("SALib-HDMR DEFAULT: Field \"alfa\" of options not specified. Default: alfa = 0.95 used. ")
        alfa = def_options['alfa']

    if 'lambdax' in options:
        lambdax = options['lambdax']
        error_msg = "SALib-HDMR ERROR: Field \"lambdax\" (regularization term) of options should not be a string but a numerical value "
        assert type(lambdax) != str, error_msg
        error_msg = "SALib-HDMR WARNING: Field \"lambdax\" of options set rather large. Default: lambdax = 0.01"
        assert lambdax < 10, error_msg
        error_msg = "SALib-HDMR ERROR: Field \"lambdax\" (regularization term) of options cannot be smaller than zero. Default: 0.01 "
        assert lambdax > 0, error_msg
    else:
        error_msg = "SALib-HDMR DEFAULT: Field \"lambdax\" of options not specified. Default: lambdax = 0.01 is used "
        print(error_msg)
        lambdax = def_options['lambdax']

    return [N,d,graphics,maxorder,maxiter,m,K,R,alfa,lambdax,print_to_console]
    

def hdmr_init(X,Y,settings):
    N, d, graphics, maxorder, maxiter, m, K, R, alfa, lambdax, ptc = settings
    # Random Seed
    np.random.seed(1+round(100*np.random.rand()))
    # Setup Bootstrap (if K > 1)
    if (K == 1):
        # No Bootstrap
        idx = np.arange(0,N).reshape(N,1)
    else:
        # Now setup the boostrap matrix with selection matrix, idx, for samples
        idx = np.argsort(np.random.rand(N,K),axis=0)[:R]

    # Compute normalized X-values
    X_n = (X - np.tile(X.min(0),(N,1))) / np.tile((X.max(0)) - X.min(0),(N,1))
    # DICT Em: Important variables
    n2 = 0
    n3 = 0
    c2 = np.empty
    c3 = np.empty
    # determine all combinations of parameters (coefficients) for each order
    c1 = np.arange(0,d)
    n1 = d
    # now return based on maxorder used
    if maxorder > 1:
        c2 = np.asarray(list(itertools.combinations(np.arange(0,d),2)))
        n2 = c2.shape[0]
    if maxorder == 3:
        c3 = np.asarray(list(itertools.combinations(np.arange(0,d),3)))
        n3 = c3.shape[0]
    # calulate total number of coefficients
    n = n1 + n2 + n3
    # Initialize m1, m2 and m3 - number of coefficients first, second, third order
    m1 = m + 3; m2 = m1**2; m3 = m1**3
    # STRUCTURE Em: Initialization
    if (maxorder == 1):
        Em = {'nterms': np.zeros(K),'RMSE': np.zeros(K),'m': m,'Y_e': np.zeros((R,K)),'f0': np.zeros(K),
            'c1': c1,'n1': n1,'c2': c2,'n2': n2,'c3': c3,'n3': n3,'n': n,'maxorder': maxorder,'select': np.zeros((n,K)),
            'B1': np.zeros((N,m1,n1))}
    elif (maxorder == 2):
        Em = {'nterms': np.zeros(K),'RMSE': np.zeros(K),'m': m,'Y_e': np.zeros((R,K)),'f0': np.zeros(K),
            'c1': c1,'n1': n1,'c2': c2,'n2': n2,'c3': c3,'n3': n3,'n': n,'maxorder': maxorder,'select': np.zeros((n,K)),
            'B1': np.zeros((N,m1,n1)),'B2': np.zeros((N,m2,n2))}
    elif (maxorder == 3):
        Em = {'nterms': np.zeros(K),'RMSE': np.zeros(K),'m': m,'Y_e': np.zeros((R,K)),'f0': np.zeros(K),
            'c1': c1,'n1': n1,'c2': c2,'n2': n2,'c3': c3,'n3': n3,'n': n,'maxorder': maxorder,'select': np.zeros((n,K)),
            'B1': np.zeros((N,m1,n1)),'B2': np.zeros((N,m2,n2)),'B3': np.zeros((N,m3,n3))}
    # Compute B-Splines
    Em['B1'] = B_spline(X_n,N,d,m)

    # Now compute B values for second order
    if (maxorder > 1):
        beta = np.array(list(itertools.product(range(m1),repeat=2)))
        for k in range(n2):
            for j in range(m2):
                Em['B2'][:,j,k] = np.multiply(Em['B1'][:,beta[j,0],Em['c2'][k,0]], Em['B1'][:,beta[j,1],Em['c2'][k,1]])
    
    # Compute B values for third order
    if (maxorder == 3):
        # Now compute B values for third order
        beta = np.array(list(itertools.product(range(m1),repeat=3)))
        for k in range(n3):
            for j in range(m3):
                Em['B3'][:,j,k] = np.multiply(np.multiply(Em['B1'][:,beta[j,0],Em['c3'][k,0]], Em['B1'][:,beta[j,1],Em['c3'][k,1]]), \
                    Em['B1'][:,beta[j,2],Em['c3'][k,2]])
        
    # DICT SA: Sensitivity analysis and analysis of variance decomposition
    SA = {'S': np.zeros((Em['n'],K)),'Sa': np.zeros((Em['n'],K)),'Sb': np.zeros((Em['n'],K)),'ST': np.zeros((d,K)), 'V_Y': np.zeros((K,1))}
    # Return runtime
    RT = np.zeros((1,K))
    # Initialize emulator matrix
    Y_em = np.zeros((R,Em['n']))
    # Initialize index of first, second and third order terms (columns of Y_em)
    j1 = range(n1); j2 = range(n1,n1+n2); j3 = range(n1+n2,n1+n2+n3)
    # Initialize temporary Y_id for bootstrap
    Y_id = np.zeros((R,1))

    return [Em,idx,SA,RT,Y_em,Y_id,m1,m2,m3,j1,j2,j3]


def ismember(A, B):
    return A in B


def B_spline(X,R,d,m):
    # Initialize B-Spline Matrix
    B = np.zeros((R,m+3,d))
    # Calculate the interval
    h1 = 1/m
    # Now loop over each parameter of X through each interval
    for i in range(d):
        for j in range(m+3):
            k = j-1
            for r in range(R):
                if (X[r,i] > ( k + 1 ) * h1 and X[r,i] <= ( k + 2 ) * h1):
                    B[r,j,i] = ( ( k + 2 ) * h1 - X[r,i] )**3
                if (X[r,i] > k * h1 and X[r,i] <= ( k + 1 ) * h1):
                    B[r,j,i] = ( ( k + 2 ) * h1 - X[r,i] )**3 - 4 * ( ( k + 1 ) * h1 - X[r,i] )**3
                if (X[r,i] > ( k - 1 ) * h1 and X[r,i] <= k * h1):
                    B[r,j,i] = ( ( k + 2 ) * h1 - X[r,i] )**3 - 4 * ( ( k + 1 ) * h1 - X[r,i] )**3 + \
                         6 * ( k * h1 - X[r,i] )**3
                if (X[r,i] > ( k - 2 ) * h1 and X[r,i] <= ( k - 1 ) * h1):
                    B[r,j,i] = ( ( k + 2 ) * h1 - X[r,i] )**3 - 4 * ( ( k + 1 ) * h1 - X[r,i] )**3 + \
                         6 * ( k * h1 - X[r,i] )**3 - 4 * ( ( k - 1 ) * h1 - X[r,i] )**3
    # Multiply B with m^3
    B *= m**3

    return B


def hdmr_first_order(B1, Y_res, R, n1, m1, maxiter, lambdax):
    C1 = np.zeros((m1,n1))       # Initialize coefficients
    Y_i = np.zeros((R,n1))       # Initialize 1st order contributions
    T1 = np.zeros((m1,R,n1))     # Initialize T(emporary) matrix - 1st
    it = 0                       # Initialize iteration counter
    # First order individual estimation
    for j in range(n1):
        # Regularized least squares inversion ( minimize || C1 ||_2 )
        B11 = np.matmul(np.transpose(B1[:,:,j]),B1[:,:,j])
        if (np.linalg.det(B11) == 0):
            C1[:,j] = 0 # Ill-defined
            # print("SALib-HDMR Warning: Matrix B11 is ill-defined. Try to decrease b-spline interval \"m\". Solution, C1, is set to zero.")
        else:
            T1[:,:,j] = np.linalg.solve(np.add(B11,np.multiply(lambdax,np.identity(m1))),np.transpose(B1[:,:,j]))
        C1[:,j] = np.matmul(T1[:,:,j],Y_res).reshape(m1,)
        Y_i[:,j] = np.matmul(B1[:,:,j],C1[:,j])

    # Backfitting Method
    var1b_old = np.sum(np.square(C1),axis=0)
    varmax = 1
    while ( varmax > 1e-3 ) and ( it < maxiter ):
        for j in range(n1):
            Y_r = Y_res
            for z in range(n1):
                if j != z:
                    Y_r = np.subtract(Y_r, np.matmul(B1[:,:,z],C1[:,z]).reshape(R,1))
            C1[:,j] = np.matmul(T1[:,:,j], Y_r).reshape(m1,)
        var1b_new= np.sum(np.square(C1),axis=0)
        varmax = np.max(np.absolute(np.subtract(var1b_new,var1b_old)))
        var1b_old = var1b_new
        it += 1
    # Now compute first-order terms
    for j in range(n1):
        Y_i[:,j] = np.matmul(B1[:,:,j],C1[:,j])
        # Subtract each first order term from residuals
        Y_res = np.subtract(Y_res,Y_i[:,j].reshape(R,1))

    return (Y_i, Y_res)


def hdmr_second_order(B2, Y_res, R, n2, m2, lambdax):
    C2 = np.zeros((m2,n2))       # Initialize coefficients
    Y_ij = np.zeros((R,n2))      # Initialize 1st order contributions
    T2 = np.zeros((m2,R,n2))     # Initialize T(emporary) matrix - 1st
    it = 0                       # Initialize iteration counter
    # First order individual estimation
    for j in range(n2):
        # Regularized least squares inversion ( minimize || C1 ||_2 )
        B22 = np.matmul(np.transpose(B2[:,:,j]),B2[:,:,j])
        if (np.linalg.det(B22) == 0):
            C2[:,j] = 0 # Ill-defined
            # print("SALib-HDMR Warning: Matrix B22 is ill-defined. Try to decrease b-spline interval \"m\". Solution, C2, is set to zero. \n")
        else:
            # CHECK: Numpy Linear Algebra Solver slightly differentiate with MATLAB's Solver
            T2[:,:,j] = np.linalg.solve(np.add(B22,np.multiply(lambdax,np.identity(m2))),np.transpose(B2[:,:,j]))
        C2[:,j] = np.matmul(T2[:,:,j],Y_res).reshape(m2,)
        Y_ij[:,j] = np.matmul(B2[:,:,j],C2[:,j])

    # Now compute first-order terms
    for j in range(n2):
        Y_ij[:,j] = np.matmul(B2[:,:,j],C2[:,j])
        # Subtract each first order term from residuals
        Y_res = np.subtract(Y_res,Y_ij[:,j].reshape(R,1))

    return (Y_ij, Y_res)


def hdmr_third_order(B3, Y_res, R, n3, m3, lambdax):
    C3 = np.zeros((m3,n3))       # Initialize coefficients
    Y_ijk = np.zeros((R,n3))      # Initialize 1st order contributions
    T3 = np.zeros((m3,R,n3))     # Initialize T(emporary) matrix - 1st
    # First order individual estimation
    for j in range(n3):
        # Regularized least squares inversion ( minimize || C1 ||_2 )
        B33 = np.matmul(np.transpose(B3[:,:,j]),B3[:,:,j])
        if (np.linalg.det(B33) == 0):
            C3[:,j] = 0 # Ill-defined
            # print("SALib-HDMR Warning: Matrix B33 is ill-defined. Try to decrease b-spline interval \"m\". Solution, C3, is set to zero. \n")
        else:
            # CHECK: Numpy Linear Algebra Solver slightly differentiate with MATLAB's Solver
            T3[:,:,j] = np.linalg.solve(np.add(B33,np.multiply(lambdax,np.identity(m3))),np.transpose(B3[:,:,j]))
        C3[:,j] = np.matmul(T3[:,:,j],Y_res).reshape(m3,)
        Y_ijk[:,j] = np.matmul(B3[:,:,j],C3[:,j])

    
    return Y_ijk


def f_test(Y, f0, Y_em, R, alfa, m1, m2, m3, n1, n2, n3, n):
    # Model selection using the F test
    # Initialize ind with zeros (all terms insignificant)
    select = np.zeros((n,1))
    # Determine the significant components of the HDMR model via the F-test
    Y_res0 = Y - f0
    SSR0 = np.sum(np.square(Y_res0))
    p0 = 0
    for i in range(n):
        # model with ith term included
        Y_res1 = Y_res0 - Y_em[:,i].reshape(R,1)
        # Number of parameters of proposed model (order dependent)
        if i <= n1:        
            p1 = m1        # 1st order
        elif i > n1 and i <= n1 + n2:
            p1 = m2        # 2nd order
        else:
            p1 = m3        # 3rd order
        # Calculate SSR of Y1
        SSR1 = np.sum(np.square(Y_res1)) 
        # Now calculate the F_stat (F_stat > 0 -> SSR1 < SSR0 )
        F_stat = ( (SSR0 - SSR1)/(p1 - p0) ) / ( SSR1/(R - p1) )
        # Now calculate critical F value
        F_crit = stats.f.ppf(q=alfa,dfn=p1-p0,dfd=R-p1)
        # Now determine whether to accept ith component into model
        if F_stat > F_crit:
            # ith term is significant and should be included in model
            select[i] = 1

    return select.reshape(n,)


def ancova(Y, Y_em, V_Y, R, n):
    # Compute the sum of all Y_em terms
    Y0 = np.sum(Y_em[:,:],axis = 1)
    # Initialize each variable
    S,S_a,S_b = [np.zeros((n,)) for _ in range(3)]

    # Analysis of covariance
    for j in range(n):
        C = np.cov(np.stack((Y_em[:,j],Y.reshape(R,)),axis = 0))              # Covariance matrix of jth term of Y_em and actual Y 
        S[j] = C[0,1]/V_Y                                       # Total sensitivity of jth term         ( = Eq. 19 of Li et al )
        C = np.cov(np.stack((Y_em[:,j],Y0 - Y_em[:,j]),axis = 0)) # Covariance matrix of jth term with emulator Y without jth term
        S_a[j] = C[0,0]/V_Y                                     # Structural contribution of jth term   ( = Eq. 20 of Li et al ) 
        S_b[j] = C[0,1]/V_Y                                     # Correlative contribution of jth term  ( = Eq. 21 of Li et al )

    return (S, S_a, S_b)


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    # Call in a loop to create terminal progress bar
    # Input params:
    #     iteration   - Required  : current iteration (Int)
    #     total       - Required  : total iterations (Int)
    #     prefix      - Optional  : prefix string (Str)
    #     suffix      - Optional  : suffix string (Str)
    #     decimals    - Optional  : positive number of decimals in percent complete (Int)
    #     length      - Optional  : character length of bar (Int)
    #     fill        - Optional  : bar fill character (Str)
    #     printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
    

def hdmr_finalize(problem,SA,Em,settings,init_vars):
    d, alfa, maxorder = settings
    Em, SA = init_vars
    # Create Sensitivity Indices Result Dictionary
    keys = ('Sa', 'Sa_CI', 'Sb', 'Sb_CI', 'S', 'S_CI','select','Sa_sum','Sa_sum_CI','Sb_sum','Sb_sum_CI','S_sum','S_sum_CI')
    Si = ResultDict((k, np.zeros(Em['n'])) for k in keys)
    Si['Term'] = [None]*Em['n']
    Si['ST'] = np.zeros(d)
    Si['ST_CI'] = np.zeros(d)
    # Z score
    z = lambda p : (-1) * np.sqrt(2) * special.erfcinv(p*2)
    # Multiplier for confidence interval
    mult = z(alfa+(1-alfa)/2)
    # Compute the total sensitivity of each parameter/coefficient
    for r in range(Em['n1']):
        if maxorder == 1:
            TS = SA['S'][r,:]
        elif maxorder == 2:
            ij = Em['n1'] + np.where(np.sum(Em['c2'] == r,axis=1) == 1)[0]
            TS = np.sum(SA['S'][np.append(r,ij),:],axis=0)
        elif maxorder == 3:
            ij = Em['n1'] + np.where(np.sum(Em['c2'] == r,axis=1) == 1)[0]
            ijk = Em['n1'] + Em['n2'] + np.where(np.sum(Em['c3'] == r,axis=1) == 1)[0]
            TS = np.sum(SA['S'][np.append(r,np.append(ij,ijk)),:],axis=0)
        Si['ST'][r] = np.mean(TS)
        Si['ST_CI'][r] = mult * np.std(TS)
    ct = 0
    # Fill Expansion Terms
    for i in range(Em['n1']):
        Si['Term'][ct] = problem['names'][i]
        ct += 1
    for i in range(Em['n2']):
        Si['Term'][ct] = '/'.join([problem['names'][Em['c2'][i,0]], problem['names'][Em['c2'][i,1]]])
        ct += 1
    for i in range(Em['n3']):
        Si['Term'][ct] = '/'.join([problem['names'][Em['c3'][i,0]], problem['names'][Em['c3'][i,1]], problem['names'][Em['c3'][i,2]]])
        ct += 1
    # Assign Bootstrap Results to Si Dict
    Si['Sa'] = np.mean(SA['Sa'],axis=1)
    Si['Sb'] = np.mean(SA['Sb'],axis=1)
    Si['S'] = np.mean(SA['S'],axis=1)
    Si['Sa_sum'] = np.mean(np.sum(SA['Sa'],axis=0))
    Si['Sb_sum'] = np.mean(np.sum(SA['Sb'],axis=0))
    Si['S_sum'] = np.mean(np.sum(SA['S'],axis=0))
    # Compute Confidence Interval
    Si['Sa_CI'] = mult * np.std(SA['Sa'],axis=1)
    Si['Sb_CI'] = mult * np.std(SA['Sb'],axis=1)
    Si['S_CI'] = mult * np.std(SA['S'],axis=1)
    Si['Sa_sum_CI'] = mult * np.std(np.sum(SA['Sa']))
    Si['Sb_sum_CI'] = mult * np.std(np.sum(SA['Sb']))
    Si['S_sum_CI'] = mult * np.std(np.sum(SA['S']))
    # F-test # of selection to print out
    Si['select'] = Em['select']

    return Si


def hdmr_print(Si,d,print_to_console):
    if print_to_console == 1:
        print("\n")
        Si['names'] = ['Inputs', 'Sa','Sb','S','ST']
        print('Term    \t      Sa            Sb             S             ST         #select ')
        print('------------------------------------------------------------------------------------')
        format1 = ("%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)    %-3.0f")
        format2 = ("%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)                  %-3.0f")
        for i in range(len(Si['Sa'])):
            if i < d:
                print(format1 %(Si['Term'][i],Si['Sa'][i], Si['Sa_CI'][i],Si['Sb'][i],Si['Sb_CI'][i],Si['S'][i], \
                    Si['S_CI'][i],Si['ST'][i],Si['ST_CI'][i],np.sum(Si['select'][i])))
            else:
                print(format2 %(Si['Term'][i],Si['Sa'][i], Si['Sa_CI'][i],Si['Sb'][i],Si['Sb_CI'][i],Si['S'][i], \
                    Si['S_CI'][i],np.sum(Si['select'][i])))
        print('------------------------------------------------------------------------------------')
        format3 = ("%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)")
        print(format3 %('Sum',Si['Sa_sum'], Si['Sa_sum_CI'],Si['Sb_sum'],Si['Sb_sum_CI'],Si['S_sum'],Si['S_sum_CI']))
    keys = ('Sa_sum','Sb_sum','S_sum','Sa_sum_CI','Sb_sum_CI','S_sum_CI')
    for k in keys:
        Si.pop(k, None)
    
    return Si


def hdmr_figures(problem,Si,Em,RT,X,Y,Y_em,idx):
    # Close all figures
    plt.close('all')
    # Get number of bootstrap from Runtime and sample size N
    K = RT.shape[1]
    N = Y.shape[0]
    # Print emulator performance
    Y_p = np.linspace(np.min(Y), np.max(Y), N, endpoint=True)
    ct =  0
    flag = 1
    while flag:
        fig, ax = plt.subplots(2,5)
        for j, axs in enumerate(ax.flatten()):
            if ct < K:
                axs.plot(Em['Y_e'][:,ct], Y[idx[:,ct]], 'r+')
                axs.plot(Y_p, Y_p, 'darkgray')
                axs.set_xlim(np.min(Y), np.max(Y))
                axs.set_ylim(np.min(Y), np.max(Y))
                axs.axis('tight')
                axs.axis('square')
                t_str = 'Bootstrap Trial of '
                axs.title.set_text(t_str + str(ct+1))
                axs.legend(['Emulator','1:1 Line'],loc='upper left')
                axs.axis('off')
                ct += 1
            if ct >= K:
                axs.axis('off')
                ct += 1
                if ct % 10 % 5 == 0:
                    flag = 0
    
    ct =  0
    flag = 1
    while flag:
        fig, ax = plt.subplots(3,2)
        for j, axs in enumerate(ax.flatten()):
            if ct < problem['num_vars']:
                axs.plot(X[idx[:,-1],ct],Y[idx[:,-1]],'r.')
                axs.plot(X[idx[:,-1],ct],Em['f0'][-1]+Y_em[:,ct],'k.')
                axs.axis('off')
                t_str = 'HDMR Regression Performance of '
                axs.title.set_text(t_str + problem['names'][ct])
                axs.legend(['Model','Regression'],loc='upper left')
                ct += 1
            if ct >= problem['num_vars']:
                axs.axis('off')
                ct += 1
                if ct % 6 % 2 == 0:
                    flag = 0
    
    plt.show()

    return


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file', type=str, required=True,
                        default=None,
                        help='Model input file')
    parser.add_argument('-g', '--graphics', type=int, required=False,
                        default=1,
                        help='1: Prints graphics, 0: Does not print graphics')
    parser.add_argument('-mor', '--maxorder', type=int, required=False,
                        default=2,
                        help='Maximum order of expansion 1-3')
    parser.add_argument('-mit', '--maxiter', type=int, required=False,
                        default=100,
                        help='Maximum iteration number')
    parser.add_argument('-m', '--m-int', type=int, required=False,
                        default=2,
                        help='B-spline interval')
    parser.add_argument('-K', '--K-bootstrap', type=int, required=False,
                        default=20,
                        help='Number of bootstrap')
    parser.add_argument('-R', '--R-subsample', type=int, required=False,
                        default=None,
                        help='Subsample size')
    parser.add_argument('-a', '--alfa', type=float, required=False,
                        default=0.95,
                        help='Confidence interval')
    parser.add_argument('-lambda', '--lambdax', type=float, required=False,
                        default=0.01,
                        help='Regularization constant')
    parser.add_argument('-print', '--print-to-console', type=int, required=False,
                        default=1,
                        help='1: Prints to console, 0: Does not print to console')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    g = args.graphics
    mor = args.maxorder
    mit = args.maxiter
    m = args.m_int
    K = args.K_bootstrap
    R = args.R_subsample
    alfa = args.alfa
    lambdax = args.lambdax
    p = args.print_to_console
    options = options = {'graphics': g,'maxorder': mor,'maxiter': mit,'m': m,'K': K,'R': R,'alfa': alfa,'lambdax': lambdax,'print_to_console': p}
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))
    # Default option is already defined under hdmr_setup function
    analyze(problem, X, Y, options)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)