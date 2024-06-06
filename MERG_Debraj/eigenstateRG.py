"""
OVERVIEW
This is a library for wavefunction renormalisation calculations with the URG.
The strategy adopted is as follows:
1. Perform the forward Hamiltonian RG to get the set of couplings along the RG flow.
2. Using the couplings at the IR fixed point, write down a zero-bandwidth Hamiltonian.
3. Diagonalise this Hamiltonian to get the IR ground state.
4. Express this ground state as a superposition of classical states. For eg:, 
   |up,dn> - |dn,up> = |10> - |01>, where {|10>,|01>} are the classical states and 
   {1,-1} are the coefficients.
5. Apply the inverse RG transformations on these classical states to generate a new 
   set of coefficients for each RG step going backwards.
6. These sets of coefficients constitute the wavefunction RG flow. Use these to 
   calculate various measures along the RG flow.

ASSUMPTIONS & CONVENTIONS
1. Fermionic impurity model Hamiltonian.
2. All operators must be composed of c^dag, c, n or 1-n, where each operator acts on a 
   single fermionic fock state.
3. The indexing of sites is as follows:
        d_up    d_dn    k1_up   k1_dn   k2_up   k2_dn   ...
        0       1       2       3       4       5       ...,
   where up,dn indicates spins and k1,k2 need not be momentum space but can be real
   space indices. That will be decided based on whether the provided Hamiltonian is
   in real or k-space.
"""


import itertools
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from tqdm.notebook import tqdm
from multiprocessing.pool import ThreadPool as Pool
from time import time
from operator import itemgetter
from fermionise import *
from random import randint
import json
import os


def init_wavefunction(hamlt, mb_basis, couplings, displayGstate=False):
    """ Generates the initial wavefunction at the fixed point by diagonalising
    the Hamiltonian provided as argument. Expresses the state as a superposition
    of various classical states, returns these states and the associated coefficients.
    No IOMS are taken into account at this point.
    """
    Ek, hop_strength, imp_U, imp_Ed, kondo_J, zerothsite_U = couplings
    eigvals, eigstates = diagonalise(mb_basis, hamlt)
    tolerance = 10
    print (eigvals[:2])
    print ("G-state energy:", eigvals[eigvals == min(eigvals)])

    if kondo_J != 0:
        if sum (np.round(eigvals, tolerance) == min(np.round(eigvals, tolerance))) == 1:
            gstate = eigstates[0]
        else:
            assert False, "Ground state is degenerate! No SU(2)-symmetric ground state exists."
    else :
        gstate = eigstates[0]
        
    if displayGstate:
        print (visualise_state(mb_basis, gstate))

    return gstate
    

def applyInverseTransform(decomposition_old, num_entangled, etaFunc, alpha):
    """ Apply the inverse unitary transformation on a state. The transformation is defined
    as U = 1 + eta + eta^dag.
    """

    # expand the basis by inserting configurations for the IOMS, in preparation of applying 
    # the eta,eta^dag on them. 
    decomposition_old = dict([(key + "1100", val) for key, val in decomposition_old.copy().items()])
    '''
    for Kondo
    # obtain the appropriate eta and eta_dag for this step
    eta_dag, eta = etaFunc(alpha, num_entangled)
    '''
    # obtain the appropriate eta and eta_dag for this step (for SIAM)
    eta_dag, eta = etaFunc(alpha, num_entangled, decomposition_old)
    
    decomposition_new_eta = decomposition_old.copy()
    decomposition_new_etadag = decomposition_old.copy()
    
    with Pool(1) as pool:
        worker_eta = pool.apply_async(applyOperatorOnState, (decomposition_old, eta),
                              kwds={'finalState': decomposition_new_eta, 
                                    'tqdmDesc': "Applying eta, size=" + str(num_entangled)})
        worker_etadag = pool.apply_async(applyOperatorOnState, (decomposition_old, eta_dag),
                              kwds={'finalState': decomposition_new_etadag,
                                    'tqdmDesc': "Applying eta^dag, size=" + str(num_entangled)})
        decomposition_new_eta = worker_eta.get()
        decomposition_new_etadag = worker_etadag.get()
        
    decomposition_new_total = decomposition_new_eta.copy()
    #decomposition_new_total = decomposition_new_etadag.copy()
    decomposition_new_total.update(decomposition_new_etadag)
    
    total_norm = np.linalg.norm(list(decomposition_new_total.values()))

    decomposition_new_total = {k: v / total_norm for k, v in decomposition_new_total.items() if np.abs(v) / total_norm > 1e-5}


    return decomposition_new_total


def getWavefunctionRG(init_couplings, alpha_arr, num_entangled, num_IOMs, hamiltonianFunc, etaFunc, displayGstate=False):
    """ Manager function for obtaining wavefunction RG. 
    1. init_couplings is the set of couplings that are sufficient to construct the IR Hamiltonian
       and hence the ground state.
    2. alpha_arr is the array of Greens function denominators that will be used to construct the 
       eta operators at various steps of the RG. Each element of alpha_arr can itself be an array. 
    3. num_entangled is the number of states in the bath that are intended to be part of the emergent 
       window at the IR.
    4. num_IOMs is the number of states in the bath that we wish to re-entangle eventually.
    5. hamiltonianFunc is a string containing the name of a function with a definition 
       func(mb_basis, num_entangled, init_couplings) that creates a Hamiltonian matrix for the model 
       at the IR. It must be defined on a per-model basis.
    6. etaFunc is a string containing the name of a function with a definition etaFunc(alpha, num_entangled)
       that returns the eta and eta^dag operators given the step-dependent parameters alpha and num_entangled.
       This function must be defined on a per-model basis.
    """
    
    # make sure there are sufficient number of values provided in alpha_arr to re-entangled num_IOMs.
    assert len(alpha_arr) >= num_IOMs, """Number of values of 'alpha' is not enough for the
    requested number of reverse RG steps."""
    
    # convert the string into function objects
    # hamiltonianFunc = eval(hamiltonianFunc)
    # etaFunc = eval(etaFunc)

    # get the basis of all classical states.
    mb_basis = getBasis(2 * (1 + num_entangled))
    
    # obtain the zero-bandwidth Hamiltonian at the IR
    hamlt = hamiltonianFunc(mb_basis, num_entangled, init_couplings)

    # obtain the superposition decomposition of the ground state
    decomposition_init = init_wavefunction(hamlt, mb_basis, init_couplings, displayGstate=displayGstate)
    
    # Initialise empty arrays to store the RG flow of the basis states and 
    # corresponding coefficients at each step of the reverse RG
    decomposition_arr = [decomposition_init]
    
    # loop over the values of alpha and apply the appropriate unitary for each value.
    for i, alpha in tqdm(enumerate(alpha_arr[:num_IOMs]), total=num_IOMs, desc="Applying inverse unitaries", disable=True):

        # obtain the renormalised coefficients and the new set of superposition states by passing the coefficients and states
        # of the previous step, the number of currently entangled states (num_entangled + i), the eta generating function and
        # the value of alpha at the present step
        decomposition_new = applyInverseTransform(decomposition_arr[-1], num_entangled + 2 * i,
                                                  etaFunc, alpha)

        # append new results to full array
        decomposition_arr.append(decomposition_new)

    return decomposition_arr


def computations(decomposition_arr, computables):
    """ Perform various computations by passing the wavefunction RG data.
    The computables argument is a dictionary, of the examplary form
    {"VNE": [0,1], "I2": [[0,1],[2,3]]}. Each key is a quantity to calculate,
    and the list in the value of the key is the set of indices with whom to
    perform the calculation. For eg., the first key asks to calculate the 
    von Neumann entanglement entropy for the set of indices (0,1).
    """

    # initialise a dictionary with the already-provided keys to store the results
    computations = dict.fromkeys(computables.keys())

    # dictionary for mapping computable names to corresponding functions,
    funcNameMaps = {"VNE": getEntanglementEntropy,
                     "I2": getMutualInfo,
                     }

    # loop over all the computables that have been required
    # for each computable, loop over the coefficient RG flow
    # and perform the computation at every RG step.
    for computable, members in computables.items():
        computations[computable] = [funcNameMaps[computable](decomposition, members) for decomposition in tqdm(decomposition_arr, total=len(decomposition_arr), disable=False, desc="Computing {}".format(computable))]
    return computations


def getEtaKondo(alpha, num_entangled):
    """ Defines eta and eta dagger operators for the Kondo model.
    """

    eta_dag = {
            # The first interaction kind is Sdz c^dag qbeta c_kbeta. First two lines are for beta=up,
            # last two lines are for beta=down.
            "n+-":
            [[0.25 * alpha[0], [0, 2 * (num_entangled + 2), 2 * i]] for i in range(1, num_entangled + 1)] + 
            [[-0.25 * alpha[1], [1, 2 * (num_entangled + 2), 2 * i]] for i in range(1, num_entangled + 1)] +
            [[-0.25 * alpha[0], [0, 2 * (num_entangled + 2) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[0.25 * alpha[1], [1, 2 * (num_entangled + 2) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)],
            # The second interaction kind is c^dag_dbetabar c_dbeta c^dag qbeta c_kbetabar. 
            # First line is for beta=up, last line is for beta=down.
            "+-+-":
            [[0.5 * alpha[1], [1, 0, 2 * (num_entangled + 2), 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[0.5 * alpha[0], [0, 1, 2 * (num_entangled + 2) + 1, 2 * i]] for i in range(1, num_entangled + 1)]
            }
    # Simply the hermitian conjugate of each of the lines.
    eta = {"n+-": 
           [[0.25 * alpha[0], [0, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[-0.25 * alpha[1], [1, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] +
           [[-0.25 * alpha[0], [0, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)] + 
           [[0.25 * alpha[1], [1, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)],
           "+-+-": 
           [[0.5 * alpha[0], [0, 1, 2 * i + 1, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[0.5 * alpha[1], [1, 0, 2 * i, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)]
          }
    return eta_dag, eta

def getEta_eSIAM(alpha, num_entangled, gstate):
    """ Defines eta and eta dagger operators for the eSIAM model.
    """
    gstate_keys = gstate.keys()
    eta_dag = {
            # The first interaction kind is Sdz c^dag qbeta c_kbeta. First two lines are for beta=up,
            # last two lines are for beta=down.
            "n+-":
            [[0.25 * alpha[0], [0, 2 * (num_entangled + 2), 2 * i]] for i in range(1, num_entangled + 1)] + 
            [[-0.25 * alpha[0], [1, 2 * (num_entangled + 2), 2 * i]] for i in range(1, num_entangled + 1)] +
            [[-0.25 * alpha[0], [0, 2 * (num_entangled + 2) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[0.25 * alpha[0], [1, 2 * (num_entangled + 2) + 1, 2 * i + 1]] for i in range(1, num_entangled + 1)],
            # The second interaction kind is c^dag_dbetabar c_dbeta c^dag qbeta c_kbetabar. 
            # First line is for beta=up, last line is for beta=down.
            "+-+-":
            [[0.5 * alpha[0], [1, 0, 2 * (num_entangled + 2), 2 * i + 1]] for i in range(1, num_entangled + 1)] + 
            [[0.5 * alpha[0], [0, 1, 2 * (num_entangled + 2) + 1, 2 * i]] for i in range(1, num_entangled + 1)],
            
            # The Third interaction kind is c^dag(-qbeta) c_k'beta c^dag_kbetabar c_qbetabar
            # First line is for beta=up, last line is for beta=down.
            "+-+-":
            [[alpha[3], [2 * (num_entangled + 2), 2 * k1, 2 * k2 + 1, 2 * (num_entangled + 1) + 1]] for k1, k2 in itertools.product(range(1, num_entangled +  1), repeat=2)] +
            [[alpha[3], [2 * (num_entangled + 2) + 1, 2 * k1 + 1, 2 * k2, 2 * (num_entangled + 1)]] for k1, k2 in itertools.product(range(1, num_entangled +  1), repeat=2)],
            # The forth interaction kind is c^dag(-qbeta) c_qbeta c^dag_kbetabar c_k'betabar
            # First line is for beta=up, last line is for beta=down.
            "+-+-":
            [[alpha[3], [2 * (num_entangled + 2), 2 * (num_entangled + 1), 2 * k1 + 1, 2 * k2 + 1]] for k1, k2 in itertools.product(range(1, num_entangled +  1), repeat=2)] +
            [[alpha[3], [2 * (num_entangled + 2) + 1, 2 * (num_entangled + 1) + 1, 2 * k1, 2 * k2]] for k1, k2 in itertools.product(range(1, num_entangled +  1), repeat=2)],
            # The fifth interaction kind is Sdz c^dag (-qbeta) c_qbeta. First two lines are for beta=up,
            # last two lines are for beta=down.
            "n+-":
            [[0.25 * alpha[0], [0, 2 * (num_entangled + 2), 2 * (num_entangled + 1)]]] + 
            [[-0.25 * alpha[0], [1, 2 * (num_entangled + 2), 2 * (num_entangled + 1)]]] +
            [[-0.25 * alpha[0], [0, 2 * (num_entangled + 2) + 1, 2 * (num_entangled + 1) + 1]]] + 
            [[0.25 * alpha[0], [1, 2 * (num_entangled + 2) + 1, 2 * (num_entangled + 1) + 1]]],
            # The sixth interaction kind is c^dag_dbetabar c_dbeta c^dag (-qbeta) c_qbetabar. 
            # First line is for beta=up, last line is for beta=down.
            "+-+-":
            [[0.5 * alpha[0], [1, 0, 2 * (num_entangled + 2), 2 * (num_entangled + 1) + 1]]] + 
            [[0.5 * alpha[0], [0, 1, 2 * (num_entangled + 2) + 1, 2 * (num_entangled + 1)]]]
            }
    # Interaction kind is c^dag qbeta c_dbeta
    for individual_state in gstate_keys:
        impurity_digits = individual_state[:2]
        sumImp = 0
        for digit in impurity_digits:
            sumImp += int(digit)
            if sumImp == 2 or sumImp == 0:
                alpha_5 = alpha[2] + alpha[4]
                # First line for up and the next for down
                eta_dag["+-"] = (
                [[alpha_5, [2 * (num_entangled + 2), 0]]] +
                [[alpha_5, [2 * (num_entangled + 2) + 1, 1]]]
                )
            else :
                alpha_6 = alpha[2] - alpha[1]
                # First line for up and the next for down
                eta_dag["+-"] = (
                [[alpha_6, [2 * (num_entangled + 2), 0]]] +
                [[alpha_6, [2 * (num_entangled + 2) + 1, 1]]]
                )
            
    # Simply the hermitian conjugate of each of the lines.
    eta = {"n+-": 
           [[0.25 * alpha[0], [0, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[-0.25 * alpha[0], [1, 2 * i, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] +
           [[-0.25 * alpha[0], [0, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)] + 
           [[0.25 * alpha[0], [1, 2 * i + 1, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)],
           "+-+-": 
           [[0.5 * alpha[0], [0, 1, 2 * i + 1, 2 * (num_entangled + 1)]] for i in range(1, num_entangled + 1)] + 
           [[0.5 * alpha[0], [1, 0, 2 * i, 2 * (num_entangled + 1) + 1]] for i in range(1, num_entangled + 1)]
          }
    # The interaction kind is c^dag dbeta c_qbeta
            
    for individual_state in gstate_keys:
        impurity_digits = individual_state[:2]
        sumImp = 0
        for digit in impurity_digits:
            sumImp += int(digit)
            alpha_5 = alpha[2] + alpha[4] if sumImp == 2 or sumImp == 0 else alpha[2] - alpha[1]
            eta["+-"] = (
                [[alpha_5, [0, 2 * (num_entangled + 1)]]] +
                [[alpha_5, [1, 2 * (num_entangled + 1) + 1]]]
            )
            
    return eta_dag, eta