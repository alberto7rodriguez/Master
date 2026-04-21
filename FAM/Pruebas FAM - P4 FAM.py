import numpy as np
import matplotlib.pyplot as plt

# --- Atomic Units Conversion ---
# The problem gives R in Angstroms, but equations are in atomic units.
# 1 Angstrom = 1.889726 Bohr (atomic unit of length)
angstrom_to_bohr = 1.889726

# --- Problem 4: Define the calculation functions ---

def S_ab(k, R):
    """Calculates the overlap integral S_ab [cite: 55]"""
    kR = k * R
    return np.exp(-kR) * (1 + kR + (kR**2) / 3.0)

def H_aa(k, R):
    """Calculates the Hamiltonian matrix element H_aa [cite: 57]"""
    return (k**2 / 2.0) - k - (1.0 / R) + np.exp(-2 * k * R) * (k + 1.0 / R)

def H_ab(k, R):
    """Calculates the Hamiltonian matrix element H_ab [cite: 58]"""
    s_ab = S_ab(k, R)
    kR = k * R
    
    # [cite: 58] H_ab = -1/2*k^2*S_ab + k(k-2)(1+kR)e^{-kR}
    term1 = -0.5 * k**2 * s_ab
    term2 = k * (k - 2) * (1 + kR) * np.exp(-kR)
    return term1 + term2

def calculate_energies(k, R):
    """
    Calculates the ground state (e1) and first excited state (e2)
    electronic energies for a given k and R.
    """
    # Get the integral values
    s_ab = S_ab(k, R)
    h_aa = H_aa(k, R)
    h_ab = H_ab(k, R)
    
    # Calculate energies
    # Ground state energy [cite: 38]
    epsilon_1 = (h_aa + h_ab) / (1 + s_ab)
    
    # Excited state energy [cite: 40]
    epsilon_2 = (h_aa - h_ab) / (1 - s_ab)
    
    return epsilon_1, epsilon_2

# --- Problem 5: Run the program and find optimal k ---

# Set a fixed internuclear distance (R)
R_fixed_angstrom = 1.0  # Example: 1.0 Angstrom
R_fixed_bohr = R_fixed_angstrom * angstrom_to_bohr

# Create a range of k values to test 
k_values = np.linspace(0.4, 2.0, 200) # 200 points for a smooth curve

# Calculate the energies for each k
E1_values = []
E2_values = []

for k in k_values:
    e1, e2 = calculate_energies(k, R_fixed_bohr)
    E1_values.append(e1)
    E2_values.append(e2)

# Find the optimal k for the ground state (variational principle)
min_energy = np.min(E1_values)
optimal_k_index = np.argmin(E1_values)
optimal_k = k_values[optimal_k_index]

print(f"--- Optimization for R = {R_fixed_angstrom} Å ({R_fixed_bohr:.4f} Bohr) ---")
print(f"Optimal variational parameter k: {optimal_k:.4f}")
print(f"Minimum electronic energy (E_el): {min_energy:.4f} Hartrees")


# --- Plotting the results ---
plt.figure(figsize=(10, 6))
plt.plot(k_values, E1_values, label=r'$\epsilon^1$ (Ground State / Bonding)')
plt.plot(k_values, E2_values, label=r'$\epsilon^2$ (Excited State / Antibonding)')

# Mark the minimum energy
plt.plot(optimal_k, min_energy, 'ro', 
         label=f'Optimum k = {optimal_k:.3f}')

plt.xlabel('Variational Parameter (k)')
plt.ylabel('Electronic Energy (Hartrees)')
plt.title(f'Energy vs. k for $H_2^+$ at R = {R_fixed_angstrom} Å')
plt.legend()
plt.grid(True)
plt.ylim(min_energy - 0.1, min_energy + 0.8) # Adjust y-axis to see both curves
plt.show()