import numpy as np
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
from rgFlowGenerators import *


def get_fixed_point_type(args):
    D0, DELTA_D, ENERGY_DOS, V_by_J, J0, Ub_by_J, U_by_Ub = args
    V0 = V_by_J * J0
    Ub = -Ub_by_J * J0
    U0 = -U_by_Ub * Ub
    E, U, V, J = complete_RG(D0, DELTA_D, ENERGY_DOS, U0, V0, J0, Ub)
    if U[-1] == 0 and V[-1] == 0 and J[-1] == 0:
        fp_type = 3
    elif J[-1] >= J0 and V[-1] >= V0:
        fp_type = 0
    elif J[-1] >= J0 and V[-1] < V0:
        fp_type = 1
    elif U[-1] > 0 and V[-1] < V0 and J[-1] < J0:
        fp_type = 2
    elif J[-1] < J0 and V[-1] >= V0:
        fp_type = 4
    else:
        fp_type = 5
    return J0/D0, -Ub/J0, fp_type


def get_rc1_analytical(J_by_D_range, U_by_Ub):
    return 
    
def get_phasemap(args):
    D0 = args["D0"]
    DELTA_D = args["DELTA_D"]
    energyDOS = args["energyDOS"]
    num_points = args["num_points"]
    J0_by_D0_lims = args["J0_by_D0_log10_lims"]
    J0_range = (D0) * np.linspace(J0_by_D0_lims[0], J0_by_D0_lims[1], num_points)
    r_lims = args["r_log10_lims"]
    r_range = np.linspace(r_lims[0], r_lims[1], num_points)
    args_arr = [(D0, DELTA_D, energyDOS, args["V_by_J"], J0, r, args["U_by_Ub"])
                for J0, r in product(J0_range, r_range)]
    results = list(tqdm(Pool().imap(get_fixed_point_type, args_arr), total=len(args_arr)))
    phaseData = np.transpose(np.reshape([r[2] for r in results], (num_points, num_points)))
    x_lims = (min(J0_range)/D0, max(J0_range)/D0)
    y_lims = (min(r_range), max(r_range))
    x_range = J0_range / D0
    rc1_analytical = (2 / x_range - 1)/(args["U_by_Ub"] - 2)
    return phaseData, x_lims, y_lims, rc1_analytical, x_range
