from rgFlowGenerators import *
from tqdm import tqdm
from multiprocessing import Pool
from eigenSolver import *


####    CALCULATES THE BROADENING THAT WILL LEAD TO THE CORRECT    ####
####    HEIGHT FOR THE SPECTRAL FUNCTION, AS PER THE FS RULE       ####

def find_FDS_broadening(rg_data, U0, J0, Ubath, target_height, lattice_num, broad_init, eta_delta):
    omega_arr, U_arr, V_arr, J_arr = rg_data
    broad = broad_init
    omega_arr_full, spec_func_norm = get_spec_func_norm(
                rg_data, U0, J0, Ubath, broad, lattice_num, eta_delta, hide_tqdm=True)
    height_diff = target_height - max(spec_func_norm[int(len(spec_func_norm)/2)-1:int(len(spec_func_norm)/2)+2])
    height_diff_tolerance = 1e-3
    if abs(height_diff)/target_height < height_diff_tolerance:
        return broad
    broad_delta = - np.sign(height_diff) * broad_init/4
    with tqdm(total=0) as pbar:
        while abs(broad_delta) > 1e-7:
            pbar.update(1)
            pbar.set_description("U={:.2f},broad={:.5f},inc={:.5f},distance={:.2f}%".format(U0,broad,broad_delta,100*abs(height_diff)/target_height))
            omega_arr_full, spec_func_norm = get_spec_func_norm(
                rg_data, U0, J0, Ubath, broad, lattice_num, eta_delta, hide_tqdm=True)
            height_diff = target_height - max(spec_func_norm[int(len(spec_func_norm)/2)-1:int(len(spec_func_norm)/2)+2])
            if abs(height_diff)/target_height < height_diff_tolerance:
                break
            if height_diff * broad_delta > 0 or broad + broad_delta <= 0:
                broad -= broad_delta
                broad_delta /= 10
            else:
                broad += broad_delta
    if abs(height_diff)/target_height >= height_diff_tolerance:
        broad = -1
    return broad
    

####    GIVEN THE ENTIRE RG FLOW, THIS JOINS THE SPECTRAL FUNCTION    ####
####    FROM EACH RG STEP TO OBTAIN THE FULL SPECTRAL FUNCTION        ####

def get_spec_func_norm(rg_data, U0, J0, Ubath, broad, lattice_num, eta_delta, hide_tqdm=False):
    omega_arr, U_arr, V_arr, J_arr = rg_data
    num_points = len(omega_arr)
    omega_arr_full = 5 * np.linspace(-max(omega_arr), max(omega_arr), 2*num_points - 1)
    omega_arr_half = np.linspace(-max(omega_arr_full), 0, num_points)
    delta_omega = abs(omega_arr_full[-1] - omega_arr_full[-2])
    args = zip([omega_arr_full]*len(omega_arr_half), omega_arr_half, U_arr, V_arr, 0*J_arr, [U0]*len(omega_arr_half),
               [J0]*len(omega_arr_half), [Ubath]*len(omega_arr_half), [broad]*len(omega_arr_half),
               [lattice_num]*len(omega_arr_half), [eta_delta]*len(omega_arr_half))
    spec_func_unnorm = sum(list(tqdm(Pool().imap(spec_func_per_rg_step, args), total=len(omega_arr_half), disable=True)))
    if U0 > 0 and np.any(J_arr > J0):
        args = zip([omega_arr_full]*len(omega_arr_half), omega_arr_half, 0*U_arr, 0*V_arr, J_arr, [U0]*len(omega_arr_half),
                   [J0]*len(omega_arr_half), [Ubath]*len(omega_arr_half), [broad]*len(omega_arr_half),
                   [lattice_num]*len(omega_arr_half), [eta_delta]*len(omega_arr_half))
        spec_func_unnorm += abs(max(omega_arr)/U0)**2 * sum(list(tqdm(Pool().imap(spec_func_per_rg_step, args), total=len(omega_arr_half), disable=True)))
    spec_func_norm = spec_func_unnorm/np.trapz(spec_func_unnorm, dx=delta_omega)
    return omega_arr_full, spec_func_norm


####    OBTAINS THE RG FLOWS IN THE COUPLINGS IN     ####
####    ORDER TO OBTAIN THE SPECTRAL FUNCTION        ####

def get_rg_flow(D0, DELTA_D, energyDOS, U0, V0, J0, Ubath, num_points):
    omega_arr, U_arr, V_arr, J_arr = complete_RG(D0, DELTA_D, energyDOS, U0, V0, J0, Ubath)
    print(V_arr[-1])
    stopPoint = len(V_arr)
    if stopPoint > len(omega_arr):
        stopPoint = len(omega_arr)
    oldIndices = np.linspace(0, max(stopPoint, num_points) - 1, stopPoint)
    newIndices = np.linspace(0, oldIndices[-1], num_points).astype(int)
    omega_arr_new = np.interp(newIndices, oldIndices, omega_arr[:len(oldIndices)])
    U_arr_new = np.interp(newIndices, oldIndices, U_arr[:stopPoint])
    V_arr_new = np.interp(newIndices, oldIndices, V_arr[:stopPoint])
    J_arr_new = np.interp(newIndices, oldIndices, J_arr[:stopPoint])
    return omega_arr_new, U_arr_new, V_arr_new, J_arr_new
        




####    CALCULATES THE SPECTRAL WEIGHTS THAT ARE USED      ####
####    TO COMPUTE THE SPECTRAL FUNCTION                   ####
def get_spectral_weights(num_sites,E,X):
    Xgs = [Xn for En,Xn in zip(E,X) if En == min(E)] 
    c_d = tensor([destroy(2)] + [identity(2)]*(2*(num_sites + 1) - 1))
    O1_up = c_d
    O2_up = O1_up.dag()

    C1_sq_arr = [sum([np.abs(np.real(((Xg.dag()*O1_up*Xn)*(Xn.dag()*O2_up*Xg)))) for Xg in Xgs]) for En, Xn in zip(E, X)]
    C2_sq_arr = [sum([np.abs(np.real(((Xg.dag()*O2_up*Xn)*(Xn.dag()*O1_up*Xg)))) for Xg in Xgs]) for En, Xn in zip(E, X)]
    
    return np.array(C1_sq_arr), np.array(C2_sq_arr)




####    COMPUTES THE SPECTRAL FUNCTION CORRESPONDING     ####
####    TO A PARTICULAR POINT ALONG THE RG FLOW          ####

def spec_func_per_rg_step(args):
    omega_arr_full,omega,U,V,J,U0,J0,Ub,delta,lattice_num,eta_delta = args
    ed = -U/2
    E,X,gs_deg = get_spectrum_kspace(lattice_num,U,V,J,Ub,omega)
    #print(E[0])
    C1_sq_arr, C2_sq_arr = get_spectral_weights(lattice_num,E,X)
    
    postRange = np.abs(omega_arr_full) - U0/4
    eta = delta * np.ones_like(omega_arr_full)
    eta[postRange > 0] += eta_delta * postRange[postRange > 0]
    x1_arr = [omega_arr_full + min(E) - En for En in E]
    x2_arr = [omega_arr_full - min(E) + En for En in E]


    A_omega = sum([(C1_sq_arr[i] * eta / (np.pi * (x1_arr[i]**2 + eta**2)) + C2_sq_arr[i] * eta / (np.pi * (x2_arr[i]**2 + eta**2))) / gs_deg 
                      for i, (En, Xn) in enumerate(zip(E, X)) if En < min(E) + max(omega_arr_full)])
    if np.trapz(A_omega, dx=abs(omega_arr_full[-1] - omega_arr_full[-2])) > 0:
        A_omega *= 1/np.trapz(A_omega, dx=abs(omega_arr_full[-1] - omega_arr_full[-2]))
    return A_omega




####    HIGH-LEVEL MODULE FOR COMPUTING SPECTRAL FUNCTION     ####

def full_spec_func(V0, J0, U0, D0, DELTA_D, energyDOS, Ub, num_points, lattice_num, target_height, broad_guess, eta_delta, use_guess):
    rg_data = get_rg_flow(D0, DELTA_D, energyDOS, U0, V0, J0, Ub, num_points)
    
    if use_guess == True:
        broad = broad_guess
    else:
        broad = broad_guess if (Ub == 0 and U0 == 0) or (J0 > 0 and -Ub/J0 > 0.25) else find_FDS_broadening(rg_data, U0, J0, Ub, 
                                                                                     target_height, lattice_num, broad_guess, eta_delta)
    print(broad)
    if broad == -1: return [], []
    omega_arr_full, spec_func_norm = get_spec_func_norm(rg_data, U0, J0, Ub, broad, lattice_num, eta_delta)
    return omega_arr_full, spec_func_norm
