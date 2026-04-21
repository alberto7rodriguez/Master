import numpy as np

def run_scf_hhe_plus():
    R_bohr = 1.4632
    Z_He = 2.0
    Z_H = 1.0
    
    S12 = 0.4508
    S_matrix = np.array([[1.0, S12], [S12, 1.0]])

    t11, t22, t12 = 2.1643, 0.7600, 0.1617
    v11_1, v12_1, v22_1 = -4.1398, -1.1029, -1.2652
    v11_2, v12_2, v22_2 = -0.6772, -0.4113, -1.2266

    h11 = t11 + v11_1 + v11_2
    h22 = t22 + v22_1 + v22_2
    h12 = t12 + v12_1 + v12_2

    H_core_matrix = np.array([[h11, h12], [h12, h22]])

    int_2e = {
        (1,1,1,1): 1.3072,
        (2,2,2,2): 0.7746,
        (2,2,1,1): 0.6057,
        (2,1,1,1): 0.4373,
        (2,2,2,1): 0.3118,
        (2,1,2,1): 0.1773
    }

    def get_2e_integral(mu, nu, lam, sig):
        i, j, k, l = mu + 1, nu + 1, lam + 1, sig + 1
        candidates = [
            (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
            (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)
        ]
        for key in candidates:
            if key in int_2e: return int_2e[key]
        return 0.0

    s_vals, U = np.linalg.eigh(S_matrix)
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s_vals))
    S_inv_half = U @ s_inv_sqrt @ U.T

    P_matrix = np.zeros((2, 2))
    
    iteration = 0
    is_converged = False
    convergence_threshold = 1e-6

    print("--- SCF Calculation for HHe+ ---")

    while not is_converged and iteration < 50:
        iteration += 1
        P_old = np.copy(P_matrix)

        G_matrix = np.zeros((2, 2))
        for mu in range(2):
            for nu in range(2):
                for lam in range(2):
                    for sigma in range(2):
                        coulomb = get_2e_integral(mu, nu, lam, sigma)
                        exchange = get_2e_integral(mu, lam, nu, sigma)
                        G_matrix[mu, nu] += P_matrix[lam, sigma] * (coulomb - 0.5 * exchange)

        Fock_matrix = H_core_matrix + G_matrix
        F_prime = S_inv_half.T @ Fock_matrix @ S_inv_half
        E_orbs, C_prime = np.linalg.eigh(F_prime)
        C_matrix = S_inv_half @ C_prime

        C_occ = C_matrix[:, 0].reshape(-1, 1)
        P_matrix = 2.0 * C_occ @ C_occ.T

        rms = np.sqrt(np.sum((P_matrix - P_old)**2) / 4)
        
        if rms < convergence_threshold:
            is_converged = True

    E_elec = 0.5 * np.sum(P_matrix * (H_core_matrix + Fock_matrix))
    E_nuc = (Z_He * Z_H) / R_bohr
    E_total = E_elec + E_nuc

    print(f"Converged in {iteration} iterations.")
    print(f"Total Energy: {E_total:.6f} a.u.")
    print(f"Orbital Energies: {E_orbs}")
    print(f"Coefficients:\n{C_matrix}")

run_scf_hhe_plus()