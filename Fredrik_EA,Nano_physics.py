# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:34:10 2021

@author: fredr
"""
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

def generate_uniform_points(x_N = 600, x_range = 60000):
    X = np.zeros(x_N+2)
    X[-1] = x_range
    X[1:-1] = np.sort(np.random.uniform(0,x_range,x_N))
    deltas = X[1:] - X[:-1]
    return X, deltas

def calculate_translation_matrices(deltas, N):
    K_m = np.array([np.sqrt((N+0.5)**2-m**2) for m in range(1, N+1)])
    P=np.zeros((len(deltas),2*N,2*N), dtype=np.complex_)
    for i in range(len(deltas)):
        diag_entries = np.exp(1j*K_m*deltas[i])
        P[i] = np.diag(np.append(diag_entries, diag_entries)) 
        #Add diagonal entries since p_n has a repetition of the values along the diagonal
    return P

def generate_incoherent_translation_matrices(deltas, N):
    P = np.array([np.identity(2*N) for i in range(len(deltas))])
    return P
    
def generate_impurity_scattering_matrices(alpha, N):
    return sp.expm(1j*alpha*np.ones((2*N,2*N))) #D = np.ones((2*N,2*N))

def generate_incoherent_impurity_scattering_matrices(alpha, N):
        return np.abs(sp.expm(1j*alpha*np.ones((2*N,2*N))))**2


def product_formula(M_1, M_2, N):
    #Extract values from input matrices M_1
    t1 = M_1[:N,:N]; r1 = M_1[N:,:N];t11 = M_1[N:,N:]; r11 = M_1[:N,N:]
    #Extract from M_2
    t2 = M_2[:N,:N]; r2 = M_2[N:,:N];t22 = M_2[N:,N:]; r22 = M_2[:N,N:]
    #Create product matrix
    inverse_factor1 = np.linalg.inv(np.identity(N) - r11@r2) 
    inverse_factor2 = np.linalg.inv(np.identity(N) - r2@r11) 
    product = np.zeros((2*N,2*N),dtype=np.complex_)
    product[:N,:N] = t2@inverse_factor1@t1 #t
    product[N:,:N] = r1 + t11@inverse_factor2@r2@t1 #r
    product[:N,N:] = r22 + t2@inverse_factor1@r11@t22 #r'
    product[N:,N:] = t11@inverse_factor2@t22 #t'
    return product

def calculate_total_matrix(P, S, N):
    cumulative_matrix = P[0]
    for i in range(1, len(P)):
        cumulative_matrix = product_formula(cumulative_matrix, S, N)
        cumulative_matrix = product_formula(cumulative_matrix, P[i], N)
    return cumulative_matrix

def run(alpha, N, coherent):
    #Generate points, uniform distribution
    X, deltas = generate_uniform_points()   
    #Check if coherent of incoherent calculation
    if(coherent):
        #Calculate translation matrices P 
        P = calculate_translation_matrices(deltas, N)
        #Calculate scattering matrices for impurities S
        S = generate_impurity_scattering_matrices(alpha, N)
        exponent = 2 #Need to exponentiate when summing probabilities
    else:
        P = generate_incoherent_translation_matrices(deltas, N)
        S = generate_incoherent_impurity_scattering_matrices(alpha, N)
        exponent = 1 #Do not need to exponentiate when summing probabilities
    #Calculate total scattering matrix M
    M = calculate_total_matrix(P, S, N) 
    #Get transmission and reflection amplitudes from from total scat. matrix
    t = M[:N,:N]
    r = M[N:,:N]
    #Create lists of transm. and refl. probabilities 
    T = np.sum(np.abs(t)**exponent,axis = 0)
    R = np.sum(np.abs(r)**exponent,axis = 0)
    return M,T,R

def run_repeatedly(alpha, N, times, coherent = True):
    total_conductance_list = np.zeros(times)
    total_reflection_list = np.zeros(times)
    for i in range(times):
        _, T, R = run(alpha, N, coherent)
        total_conductance_list[i] = np.sum(T) 
        total_reflection_list[i] = np.sum(R)
    return total_conductance_list, total_reflection_list

def plot_conductances(G_list, times = 200):
    plt.title("Conductance for different impurity-configurations")
    plt.xlabel("Impurity-configuration")
    plt.ylabel("Dimensionless conductance G/G_Q")
    plt.scatter(np.linspace(1,times,times),G_list, color = "g")
    plt.axhline(np.average(G_list),c = "b")
    plt.show()
    return 0
    
def main():
    times = 1
    alpha = 0.035
    N = 30
    coherent = False
    G_list, R_list = run_repeatedly(alpha, N, times, coherent)
    plot_conductances(G_list, times)
    print("Iterations         :", times, \
          "\nAlpha              :", alpha, \
          "\nAverage conductance:", round(np.average(G_list),4), \
          "\nAverage reflection :", round(np.average(R_list),4), \
          "\nAverage G + R      :", round(np.average(G_list)+np.average(R_list),4), \
          "\nVariance           :", round(np.var(G_list),4))
main()