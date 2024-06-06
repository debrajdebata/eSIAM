import numpy as np

#### RETURNS THE DENOMINATORS THAT APPEAR IN RG EQUATIONS ####

def getDenominators(omega, D, U, J, Ub):
    d0 = omega - D/2 + Ub/2 - U/2
    d1 = omega - D/2 + Ub/2 + U/2 + J/4
    d2 = omega - D/2 + Ub/2 + J/4
    return np.array([d0, d1, d2])



#### RETURNS THE MODIFIED SET OF COUPLINGS AT A GIVEN RG STEP ####

def rg_flow(omega, D0, DELTA_D, energyDOS, D, U, V, J, signs, Ub):
    denominators = getDenominators(omega, D, U, J, Ub)
    theta = 1 * (signs * denominators > 0)
    deltaU = energyDOS(D0, D) * DELTA_D * (4*V**2 * (theta[1]/denominators[1] - theta[0]/denominators[0]) - J**2 * theta[2] / denominators[2])
    deltaV = -energyDOS(D0, D) * DELTA_D * (3 * J * V / 8) * (theta[1]/denominators[1] + theta[2]/denominators[2])
    deltaJ = -energyDOS(D0, D) * DELTA_D * (J**2 + 4*J*Ub) * theta[2]/denominators[2]

    U = 0 if (U + deltaU) * U <= 0 else U + deltaU
    V = 0 if (V + deltaV) * V <= 0 else V + deltaV
    J = 0 if (J + deltaJ) * J <= 0 else J + deltaJ

    return U, V, J



#### SAME AS THE PREVIOUS FUNCTION rg_flow(), ####
#### BUT FOR THE J-Ub MODEL ####

def rg_J_Ub(omega, D0, DELTA_D, energyDOS, D, J, Ub):
    denominator = getDenominators(omega, D, 0, J, Ub)[2]
    deltaJ = -energyDOS(D0, D) * DELTA_D * (J**2 + 4*J*Ub) * 1/denominator if denominator < 0 else 0
    J = 0 if (J + deltaJ) * J <= 0 else J + deltaJ
    return J



#### TAKES BARE PARAMETER VALUES AND RETURNS THE COMPLETE RG FLOW ####

def complete_RG(D0, DELTA_D, energyDOS, U0, V0, J0, Ub):
    omega = -U0/4
    signs = np.sign(getDenominators(omega, D0, U0, J0, Ub))
    E, U, V, J = [[g0] for g0 in (D0, U0, V0, J0)]
    for D in np.arange(D0, 0, -DELTA_D):
        U_j, V_j, J_j = rg_flow(omega, D0, DELTA_D, energyDOS, D, U[-1], V[-1], J[-1], signs, Ub)
        U.append(U_j)
        V.append(V_j)
        J.append(J_j)
        E.append(D - DELTA_D)
        denominators = getDenominators(omega, E[-1], U[-1], J[-1], Ub)
        if True not in np.equal(np.sign(denominators), signs):
            break
        
    return np.array(E), np.array(U), np.array(V), np.array(J)



#### SAME AS THE PREVIOUS FUNCTION complete_RG(), ####
#### BUT FOR THE J-Ub MODEL ####

def complete_RG_J_Ub(D0, DELTA_D, energyDOS, J0, Ub):
    U0 = -100*Ub
    omega = -U0/4
    denominator = getDenominators(omega, D0, U0, J0, Ub)[2]
    assert denominator <= 0
    J = [J0]
    E = [D0]
    for D in np.arange(D0, 0, -DELTA_D):
        J_j = rg_J_Ub(omega, D0, DELTA_D, energyDOS, D, J[-1], Ub)
        J.append(J_j)
        E.append(D - DELTA_D)
        denominator = getDenominators(omega, E[-1], 0, J[-1], Ub)[2]
        if denominator >= 0:
            break
    return np.array(E), np.array(J)
