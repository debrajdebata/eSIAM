from eigenSolver import *
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
from rgFlowGenerators import *
import numpy as np
from qutip import *


####     GIVEN A STATE/SET OF STATES, COMPUTES            ####
####     THE AVERAGE QFI IN THE STATE, CORRESPONDING      ####
####     TO SOME PREDEFINED OPERATORS                     ####

def get_QFI(Xgs):
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
        
    total_dim, total_num_sites = get_state_dim(Xgs[0])
    
    c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
    Sz_all = [0.5 * (c_all[2*i].dag()*c_all[2*i] - c_all[2*i + 1].dag()*c_all[2*i + 1]) for i in range(total_num_sites)]
    Cz_all = [0.5 * (c_all[2*i].dag()*c_all[2*i] + c_all[2*i + 1].dag()*c_all[2*i + 1] - 1) for i in range(total_num_sites)]
    Sp_all = [c_all[2*i].dag() * c_all[2*i + 1] for i in range(total_num_sites)]
    Cp_all = [c_all[2*i].dag() * c_all[2*i + 1].dag() for i in range(total_num_sites)]

    SpSm = Sp_all[0] * Sp_all[1].dag() + Sp_all[1] * Sp_all[0].dag()
    Sdz = Sz_all[0]
    CpCm_01 = Cp_all[1] * Cp_all[2].dag() + Cp_all[2] * Cp_all[1].dag()
    
    QFI_labels = ("spin-flip", "charge", "spin-d")
    QFI_dict = {}
    for label,op in zip(QFI_labels, (SpSm, CpCm_01, Sdz)):
        assert op == op.dag()
        QFI_dict[label] = sum([4*np.real(np.array(X.dag() * op**2 * X)[0][0]) 
                               - 4*np.real(np.array(X.dag() * op * X)[0][0])**2
                               for X in Xgs])/len(Xgs)
    return QFI_dict



####     GIVEN A STATE/SET OF STATES AND A PAIR OF POSITIONS,    ####
####     COMPUTES THE AVERAGE MUTUAL INFORMATION BETWEEN         ####
####     SITES AT THE PROVIDED POSITIONS                         ####

def get_MI(Xgs, i, j):
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    
    mut_info = 0
    for X in Xgs:
        rho = X * X.dag()
        rho_part = rho.ptrace(i+j)
        mut_info += entropy_mutual(rho_part, list(range(len(i))), list(range(len(i), len(i)+len(j))))/len(Xgs)
    return mut_info


####     GIVEN A STATE/SET OF STATES AND A POSITIONS,      ####
####     COMPUTES THE AVERAGE ENTANGL. ENTROPY             ####
####     AT THE PROVIDED LATTICE SITE                      ####

def get_EE(Xgs, i):
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    ent_entr = 0
    for X in Xgs:
        rho = X * X.dag()
        rho_part = rho.ptrace((2*i, 2*i+1))
        ent_entr += entropy_vn(rho_part) / len(Xgs)
    return ent_entr



#### COMPUTES THE CORRELATION c^dag_i c_j c^dag_k c_l,    ####
#### GIVEN A STATE AND A SET OF INDICES i,j,k,l           ####

def get_2particle_corr(Xgs, guys):
    i,j,k,l = guys  
    if isinstance(Xgs, list):
        total_dim, _ = get_state_dim(Xgs[0])
        c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
        two_p_corr = sum([(X.dag() * c_all[i].dag() * c_all[j] * c_all[k].dag() * c_all[l] * X)[0][0][0] for X in Xgs]) / len(Xgs)
    else:
        total_dim, _ = get_state_dim(Xgs)
        c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
        two_p_corr = (Xgs.dag() * c_all[i].dag() * c_all[j] * c_all[k].dag() * c_all[l] * Xgs)[0][0][0]
    return two_p_corr



#### COMPUTES THE SPIN-FLIP CORRELATION IN A GIVEN    ####
#### STATE BETWEEN THE PROVIDED PAIR OF INDICES       ####

def get_spin_flip_corr(Xgs, pair):
    (i, j) = pair    
    guys_plus_minus = (2*i, 2*i+1, 2*j+1, 2*j)
    guys_minus_plus = (2*i+1, 2*i, 2*j, 2*j+1)

    return 0.5 * (get_2particle_corr(Xgs, guys_plus_minus) + get_2particle_corr(Xgs, guys_minus_plus))



#### COMPUTES THE DENSITY-DENSITY CORRELATION       ####
#### IN THE UP-UP OR DOWN-DOWN STATE, IN THE        ####
#### PROVIDED PAIR OF INDICES                       ####

def get_density_ferro_corr(Xgs, pair, spin):
    (i, j) = pair
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    guys = (2*i, 2*i, 2*j, 2*j) if spin == 1 else (2*i+1, 2*i+1, 2*j+1, 2*j+1)
    return get_2particle_corr(Xgs, guys)
    
    

#### COMPUTES THE DENSITY-DENSITY CORRELATION               ####
#### IN THE UP-DOWN STATE, IN THE PROVIDED PAIR OF INDICES     ####

def get_density_antiferro_corr(Xgs, pair, spin):
    (i, j) = pair
    if not isinstance(Xgs, list):
        Xgs = [Xgs] 
    guys = (2*i, 2*i, 2*j+1, 2*j+1) if spin == 1 else (2*i+1, 2*i+1, 2*j, 2*j)
    return get_2particle_corr(Xgs, guys)
    
    

#### COMPUTES THE ISING CORRELATION           ####
#### BETWEEN THE PROVIDED PAIR OF INDICES     ####

def get_spin_ising_corr(Xgs, pair):
    (i, j) = pair
    if not isinstance(Xgs, list):
        Xgs = [Xgs] 
    return 0.25*(get_density_ferro_corr(Xgs, (i,j), 1) + get_density_ferro_corr(Xgs, (i,j), -1) 
                 - get_density_antiferro_corr(Xgs, (i,j), 1) - get_density_antiferro_corr(Xgs, (i,j), -1))



#### COMPUTES THE ISING CORRELATION           ####
#### BETWEEN THE PROVIDED PAIR OF INDICES     ####

def get_charge_flip_corr(Xgs, pair):
    (i, j) = pair
    if not isinstance(Xgs, list):
        Xgs = [Xgs]  
    guys_plus_minus = (2*i, 2*j, 2*i+1, 2*j+1)
    guys_minus_plus = (2*j, 2*i, 2*j+1, 2*i+1)
    return 0.5 * (get_2particle_corr(Xgs, guys_plus_minus) + get_2particle_corr(Xgs, guys_minus_plus))



#### COMPUTES THE CHARGE ISING CORRELATION    ####
#### BETWEEN THE PROVIDED PAIR OF INDICES     ####

def get_charge_ising_corr(Xgs, pair):
    (i, j) = pair
    if not isinstance(Xgs, list):
        Xgs = [Xgs] 
    return (0.25 * (get_density_ferro_corr(Xgs, (i,j), 1) + get_density_ferro_corr(Xgs, (i,j), -1)
                    + get_density_antiferro_corr(Xgs, (i,j), 1) + get_density_antiferro_corr(Xgs, (i,j), -1)
                    - get_density_ferro_corr(Xgs, (i,i), 1) - get_density_ferro_corr(Xgs, (i,i), -1)
                    - get_density_ferro_corr(Xgs, (j,j), 1) - get_density_ferro_corr(Xgs, (j,j), -1) + 1))




#### COMPUTES THE SPIN-SPIN CORRELATION  S_i . S_j    ####
#### BETWEEN THE PROVIDED PAIR OF INDICES i,j         ####
    
def get_spin_spin_corr(Xgs, pair):
    (i, j) = pair  
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    return get_spin_ising_corr(Xgs, (i, j)) + get_spin_flip_corr(Xgs, (i, j))




#### COMPUTES THE CHARGE-CHARGE CORRELATION  C_i . C_j    ####
#### BETWEEN THE PROVIDED PAIR OF INDICES i,j         ####
    
def get_charge_charge_corr(Xgs, pair):
    (i, j) = pair
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    return get_charge_ising_corr(Xgs, (i, j)) + get_charge_flip_corr(Xgs, (i, j))




#### COMPUTES THE ONE-PARTICLE CORRELATION  c^dag_i c_j    ####
#### BETWEEN THE PROVIDED PAIR OF INDICES i,j              ####
    
def get_1p(Xgs, guys):
    i, j = guys
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    total_dim, _ = get_state_dim(Xgs[0])
    c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
    one_p_corr = sum([np.real(np.array(X.dag() * c_all[2*i].dag() * c_all[2*j] * X)[0][0]) for X in Xgs]) / len(Xgs)
    return one_p_corr



####    RETURNS OVERLAP OF GIVEN STATE       ####
####    WITH THE SPIN SINGLET STATE          ####

def get_singlet_overlap(Xgs):
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    total_dim, total_num_sites = get_state_dim(Xgs[0])
    c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
    S_z = [0.5 * (c_all[i].dag() * c_all[i] - c_all[i+1].dag() * c_all[i+1]) for i in (0, 2)]
    S_plus = sum([c_all[i].dag() * c_all[i+1] for i in (0, 2)])
    S_tot_sq = 0.25 * (S_plus + S_plus.dag())**2 - 0.25 * (S_plus - S_plus.dag())**2 + sum(S_z)**2
    C_z = sum([0.5 * (c_all[i].dag() * c_all[i] + c_all[i+1].dag() * c_all[i+1] - 1) for i in (0, 2)])
    C_plus = sum([c_all[i].dag() * c_all[i+1].dag() for i in (0, 2)])
    C_tot_sq = 0.25 * (C_plus + C_plus.dag())**2 - 0.25 * (C_plus - C_plus.dag())**2 + C_z**2
    singlet_overlap = sum([4 * abs((X.dag() * S_z[0] * S_z[1] * X)[0][0][0]) for X in Xgs])/len(Xgs)
    return singlet_overlap




####    RETURNS OVERLAP OF GIVEN STATE       ####
####    WITH THE CHARGE TRIPLET-ZERO STATE   ####

def get_charge_triplet_overlap(Xgs):
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    total_dim, total_num_sites = get_state_dim(Xgs[0])
    c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
    C_z = [0.5 * (c_all[i].dag() * c_all[i] + c_all[i+1].dag() * c_all[i+1] - 1) for i in (0, 2)]
    C_plus = sum([c_all[i].dag() * c_all[i+1].dag() for i in (0, 2)])
    C_tot_sq = 0.25 * (C_plus + C_plus.dag())**2 - 0.25 * (C_plus - C_plus.dag())**2 + sum(C_z)**2
    charge_triplet_overlap = sum([4 * abs((X.dag() * C_z[0] * C_z[1] * X)[0][0][0]) for X in Xgs])/len(Xgs)
    return charge_triplet_overlap




####    RETURNS OVERLAP OF GIVEN STATE       ####
####    WITH THE LOCAL MOMENT STATE         ####

def get_loc_mom_overlap(Xgs):
    if not isinstance(Xgs, list):
        Xgs = [Xgs]
    total_dim, total_num_sites = get_state_dim(Xgs[0])
    c_all = [tensor([sigmaz()]*(i) + [destroy(2)] + [identity(2)]*(total_dim - i -1)) for i in range(total_dim)]
    S_dz = 0.5 * (c_all[0].dag() * c_all[0] - c_all[1].dag() * c_all[1])
    C_0z = 0.5 * (c_all[2].dag() * c_all[2] + c_all[3].dag() * c_all[3] - 1)
    loc_mom_overlap = sum([2 * abs((X.dag() * S_dz * X)[0][0][0]) for X in Xgs])/len(Xgs)
    return loc_mom_overlap





####    RETURNS A HOST OF MANY-PARTICLE CORRELATIONS             ####
####    COMPUTED IN THE GROUND STATE OF THE FIXED POINT          ####
####    HAMILTONIAN CORRESPONDING TO THE GIVEN BARE COUPLINGS    ####

def get_gstate_correlations(args):
    Ub,non_inter_args = args
    D0, DELTA_D, energyDOS, V0, J0, U_by_Ub = non_inter_args["couplings"]
    t, num_sites = non_inter_args["sys_params"]
    U0 = - U_by_Ub * Ub
        
    E, U, V, J = complete_RG(D0, DELTA_D, energyDOS, U0, V0, J0, Ub)
    V_fp = V[-1] if V[-1] > V[0] else 0
    J_fp = J[-1] if J[-1] > J[0] else 0
    ham = get_ham_rspace(t, num_sites, U[-1], V_fp, J_fp, Ubath=Ub)
    E, X = ham.eigenstates()
        
    Xgs = [Xi for Ei,Xi in zip(E,X) if Ei == min(E)]
    
    singlet_overlap = get_singlet_overlap(Xgs)
    charge_triplet_overlap = get_charge_triplet_overlap(Xgs)
    loc_mom_overlap = get_loc_mom_overlap(Xgs)
    xi_SS = 1 - singlet_overlap**2
    xi_CT = 1 - charge_triplet_overlap**2
    QFI = get_QFI(Xgs)
    spin_flip_corr_di = [-get_spin_flip_corr(Xgs, (0,i)) for i in range(1, num_sites+1)]
    charge_flip_corr_0i = [get_charge_flip_corr(Xgs, (1,i)) for i in range(2,num_sites+1)]
    charge_flip_corr_d0 = get_charge_flip_corr(Xgs, (0,1))
    I_01 = get_MI(Xgs,[2,3],[4,5])
    I_di = [get_MI(Xgs,[0,1],[i, i+1]) for i in range(2,2*num_sites+1,2)]
    doub_occ = [get_density_antiferro_corr(Xgs, (i,i), 1) for i in (0,1)]


    return {
            "SS_weight": np.abs(singlet_overlap),
            "CT_weight": np.abs(charge_triplet_overlap),
            "LM_weight": np.abs(loc_mom_overlap),
            "geo_ent_ss": xi_SS,
            "geo_ent_ct": xi_CT,
            "QFI_spinflip": QFI["spin-flip"],
            "QFI_charge": QFI["charge"],
            "QFI_Sdz": QFI["spin-d"],
            "spinflip_corr_di": spin_flip_corr_di,
            "chargeflip_corr_01": charge_flip_corr_0i,
            "chargeflip_corr_d0": charge_flip_corr_d0,
            "mutinfo_01": I_01,
            "mutinfo_di": I_di,
            "doub_occ": doub_occ,
           }



####    RETURNS A HOST OF MANY-PARTICLE CORRELATIONS COMPUTED    ####
####    IN THE GROUND STATE OF THE FIXED POINT J-Ub              ####
####    HAMILTONIAN CORRESPONDING TO THE GIVEN BARE COUPLINGS    ####

def get_gstate_correlations_J_Ub(args):
    num_sites, t, D0, DELTA_D, energyDOS, J0, Ub = args
    assert num_sites > 1
    if J0 != 0:
        E, J = complete_RG_J_Ub(D0, DELTA_D, energyDOS, J0, Ub)
        J = J[-1]
        if J < J0: J = 0
    else:
        J = 0
    ham = get_ham_rspace(t, num_sites, 0, 0, J, Ubath=Ub)
    E, X = ham.eigenstates()
        
    Xgs_r = X[0]
    spin_flip_corr_di = [get_spin_flip_corr(Xgs_r, (0,i)) for i in range(1, num_sites+1)]
    I_0i = [get_MI(Xgs_r,[2,3],[2*i,2*i+1]) for i in range(2,num_sites+1)]
    I_di = [get_MI(Xgs_r,[0,1],[i, i+1]) for i in range(2,2*num_sites+1,2)]
    I_d_01 = get_MI(Xgs_r,[0,1],[2,3,4,5])
    I_d01 = abs(I_di[0] + I_di[1] - I_d_01)
    spin_ising_corr_di = [get_spin_ising_corr(Xgs_r, (0, j)) for j in range(1,num_sites+1)]
    spin_spin_corr_di = [get_spin_spin_corr(Xgs_r, (0, j)) for j in range(1,num_sites+1)]
    charge_flip_corr_0i = [get_charge_flip_corr(Xgs_r, (1,i)) for i in range(2,num_sites+1)]
    charge_ising_corr_0i = [get_charge_ising_corr(Xgs_r, (1,i)) for i in range(2, num_sites+1)]
    charge_charge_corr_0i = [get_charge_charge_corr(Xgs_r, (1,i)) for i in range(2,num_sites+1)]
    density_corr_00 = get_density_antiferro_corr(Xgs_r, (1,1), 1)
    imp_EE = get_EE(Xgs_r, 0)
    QFI = get_QFI(Xgs_r)
    
    return {"charge_flip_corr_0i": charge_flip_corr_0i,
             "spin_flip_corr_di": spin_flip_corr_di,
             "I_0i": I_0i,
             "I_di": I_di,
             "I_d01": I_d01,
             "spin_spin_corr_di": spin_spin_corr_di,
             "charge_flip_corr_0i": charge_flip_corr_0i,
             "charge_ising_corr_0i": charge_ising_corr_0i,
             "spin_ising_corr_di": spin_ising_corr_di,
             "charge_charge_corr_0i": charge_charge_corr_0i,
             "density_corr_00": density_corr_00,
             "imp_EE": imp_EE,
             "QFI": QFI,
            }


def get_excited_corrs(args):
    u,v,j,Ub,t,num_sites = args
    total_dim = 2 * (1 + num_sites)
    Ham_r = get_ham_rspace(t, num_sites, u, v, j, Ubath=Ub)
    E, X = Ham_r.eigenstates()
    
    E_lm_mixed_guess = -Ub/2 -(u - Ub)/4 - np.sqrt(4*v**2 + (u - Ub)**2/4)/2
    E_gap = np.round(E[1:] - E[:-1], 2)
    X_exc_index = list(E_gap).index(E_gap[E_gap == max(E_gap)][0]) + 1
    X_exc = X[X_exc_index]
    I_di = [get_MI(X_exc,[0,1],[2*i,2*i+1]) for i in range(1,num_sites+1)]
    I_0i = [get_MI(X_exc,[2*i, 2*i+1],[2,3]) for i in range(2, num_sites+1)]
    one_part_0i = [abs(get_1p(X_exc, (1,i))) for i in range(2, num_sites+1)]
    one_part_d0 = abs(get_1p(X_exc, (0,1)))
    spin_flip_corr_0i =  [abs(get_spin_flip_corr(X_exc, (1,i))) for i in range(2, num_sites+1)]
    charge_flip_corr_0i = [abs(get_charge_flip_corr(X_exc, (1,i))) for i in range(2, num_sites+1)]

    return ({"mutinfo_di": I_di,
             "mutinfo_0i": I_0i,
             "1p_0i": one_part_0i,
             "1p_d0": one_part_d0, 
             "spin_flip_0i": spin_flip_corr_0i,
             "charge_flip_0i": charge_flip_corr_0i,
             })


def get_correlations_near_transition(t, D0, DELTA_D, energyDOS, J0_by_D0, Ub_lims, num_points, num_sites):
    J0 = D0 * J0_by_D0
    Ub_range = (-J0/4) * (1 - 10**np.linspace(Ub_lims[0], Ub_lims[1], num_points))
    args = [(num_sites, t, D0, DELTA_D, energyDOS, J0, Ub) for Ub in Ub_range]
    num_threads = 40 if num_sites < 5 else 2
    x_arr = 1/(0.25 + Ub_range/J0)

    results = list(tqdm(Pool(num_threads).imap(get_gstate_correlations_J_Ub, args), total=len(Ub_range)))
    
    charge_flip_corr_0i = [np.array([r["charge_flip_corr_0i"][i] for r in results]) for i in range(num_sites-1)]
    spin_flip_corr_di = [np.array([r["spin_flip_corr_di"][i] for r in results]) for i in range(num_sites)]
    I_0i = [np.array([r["I_0i"][i] for r in results]) for i in range(num_sites-1)]
    I_di = [np.array([r["I_di"][i] for r in results]) for i in range(num_sites)]
    I_d01 = np.array([r["I_d01"] for r in results])
    spin_spin_corr_di = [np.array([r["spin_spin_corr_di"][i] for r in results]) for i in range(num_sites)]
    charge_flip_corr_0i = [np.array([r["charge_flip_corr_0i"][i] for r in results]) for i in range(num_sites-1)]
    charge_ising_corr_0i = [np.array([r["charge_ising_corr_0i"][i] for r in results]) for i in range(num_sites-1)]
    spin_ising_corr_di = [np.array([r["spin_ising_corr_di"][i] for r in results]) for i in range(num_sites)]
    charge_charge_corr_0i = [np.array([r["charge_charge_corr_0i"][i] for r in results]) for i in range(num_sites-1)]
    density_corr_00 = np.array([r["density_corr_00"] for r in results])
    imp_EE = np.array([r["imp_EE"] for r in results])
    QFI_sf = np.array([r["QFI"]["spin-flip"] for r in results])
    QFI_ch = np.array([r["QFI"]["charge"] for r in results])
    QFI_sd = np.array([r["QFI"]["spin-d"] for r in results])
           
    return x_arr,   {"charge_flip_corr_0i": charge_flip_corr_0i,
                     "spin_flip_corr_di": spin_flip_corr_di,
                     "I_0i": I_0i,
                     "I_di": I_di,
                     "I_d01": I_d01,
                     "spin_spin_corr_di": spin_spin_corr_di,
                     "charge_flip_corr_0i": charge_flip_corr_0i,
                     "charge_ising_corr_0i": charge_ising_corr_0i,
                     "spin_ising_corr_di": spin_ising_corr_di,
                     "charge_charge_corr_0i": charge_charge_corr_0i,
                     "density_corr_00": density_corr_00,
                     "imp_EE": imp_EE,
                     "QFI": (QFI_sf, QFI_ch, QFI_sd),
                    }
