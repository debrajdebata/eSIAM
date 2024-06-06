from qutip import *
import numpy as np

####     TAKES A QUTIP STATE AS INPUT AND RETURNS     ####
####     THE NUMBER OF FOCK STATES AND NUMBER OF      ####
####     LATTICE SITES PRESENT WITHIN THE STATE       ####

def get_state_dim(X):
    total_dim = int(np.log2(X.shape[0]))
    total_num_sites = int(total_dim/2) # total number of sites
    assert total_num_sites == total_dim/2
    return total_dim, total_num_sites



####     (REAL SPACE) CONSTRUCTS A QUTIP HAMILTONIAN       ####
####     FOR THE ESIAM MODEL, GIVEN A SET OF PARAMETERS    ####


def get_ham_rspace(t, lattice_dim, U, V, J, Ubath=0, mu=0):
    imp_length = 1
    total_dim = 2*(imp_length + lattice_dim)
    c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
    
    Sz_imp = 0.5 * (c_all[0].dag()*c_all[0] - c_all[1].dag()*c_all[1])
    Sp_imp = c_all[0].dag()*c_all[1]
    Sm_imp = Sp_imp.dag()
    Sz_bath = 0.5 * (c_all[2].dag()*c_all[2] - c_all[3].dag()*c_all[3])
    Sp_bath = c_all[2].dag()*c_all[3]
    Sm_bath = Sp_bath.dag()
    
    H_U = (-U*2) * Sz_imp**2
    H_Ubath = - 0.5 * Ubath * (2*Sz_bath)**2 
    H_V = V * (c_all[0].dag() * c_all[2] + c_all[1].dag() * c_all[3])
    H_J = J * (Sz_imp * Sz_bath + 0.5 * Sp_imp * Sm_bath + 0.5 * Sm_imp * Sp_bath)
    H_t = sum([-t * c_all[i].dag() * c_all[i+2] for i in range(2*imp_length, total_dim-2)])
    H_mu = -mu * sum([c_all[i].dag()*c_all[i] for i in range(total_dim)])

    return H_U + H_J + H_V + H_V.dag() + H_t + (H_t.dag() if H_t != 0 else 0) + H_Ubath + H_mu



####     (MOM. SPACE) CONSTRUCTS A QUTIP HAMILTONIAN       ####
####     FOR THE ESIAM MODEL, GIVEN A SET OF PARAMETERS    ####


def get_ham_kspace(Ek_0, ed, U, V, J, Ubath=0, ignore=0): 
    assert ed == -U/2
    dim = len(Ek_0)
    total_dim = 2 * (dim + 1)
    c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
    Sz_imp = 0.5 * (c_all[0].dag()*c_all[0] - c_all[1].dag()*c_all[1])
    Sp_imp = c_all[0].dag()*c_all[1]
    Sm_imp = Sp_imp.dag()
    c0_up = sum([c_all[i] for i in range(2, total_dim-1-ignore, 2)])
    c0_dn = sum([c_all[i+1] for i in range(2, total_dim-1-ignore, 2)])
    S0_z = 0.5 * (c0_up.dag()*c0_up - c0_dn.dag()*c0_dn)
    S0_plus = c0_up.dag() * c0_dn
    S0_minus = S0_plus.dag()
    H_J = J * (Sz_imp * S0_z + 0.5 * (Sp_imp * S0_minus + Sm_imp * S0_plus))
    H_U = (-U*2) * Sz_imp**2
    H_Ubath = - 0.5 * Ubath * (2*S0_z)**2 
    H_K = sum([Ek_0[i-1]*(c_all[2*i].dag()*c_all[2*i] + c_all[2*i+1].dag()*c_all[2*i+1]) for i in range(1, dim+1)])
    H_V = V * (c_all[0].dag() * c0_up + c_all[1].dag() * c0_dn)

    return H_U + H_Ubath + H_K + H_V + H_V.dag() + H_J



####    GIVEN SET OF PARAMS, RETURNS THE EIGENVALUES     ####
####    AND EIGENVECTORS OF A MODEL HAMILTONIAN, FOR     ####
####    SPECTRAL FUNCTION CALCULATIONS                   ####

def get_spectrum_kspace(num_sites,U,V,J,Ub,omega):
    num_pos = int(num_sites/2)
    num_neg = num_sites - num_pos
    # Ek = np.linspace(omega, omega*(1 + 1e-5), num_sites)
    Ek = np.linspace(-abs(omega), abs(omega),num_sites)
    ed = -U/2
    H = get_ham_kspace(Ek, ed, U, V, J, Ubath=Ub)
    E, X = H.eigenstates()

        
    E = np.round(E, 5)
    return E,X,len(X[E == min(E)])
