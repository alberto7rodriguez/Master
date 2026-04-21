import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def calculate_S(alpha):
    """
    Calculates the value of S for a given alpha.
    """
    
    # --- 1. Define the Integrand ---
    # The integral is ∫ d³r e⁻ʳ e⁻(αr²)
    # We switch to spherical coordinates: d³r -> 4πr²dr
    def integrand(r, a):
        return 4 * np.pi * r**2 * np.exp(-r) * np.exp(-a * r**2)

    # --- 2. Calculate the Integral ---
    integral_result, integral_error = quad(integrand, 0, np.inf, args=(alpha,))
    
    # --- 3. Calculate the Pre-factors ---
    prefactor_1 = (np.pi)**(-1/2)
    prefactor_2 = (2 * alpha / np.pi)**(3/4)
    
    # --- 4. Combine all parts ---
    S = prefactor_1 * prefactor_2 * integral_result
    
    return S

# --- 5. Set up the Data ---

# Create an array of alpha values from 0.1 to 0.5
alpha_values = np.linspace(0.1, 0.5, 200)

# Calculate the corresponding S value for each alpha
vectorized_S = np.vectorize(calculate_S)
s_values = vectorized_S(alpha_values)

# --- NEW: Find and print the maximum ---
max_s_index = np.argmax(s_values)
max_alpha = alpha_values[max_s_index]
max_s_value = s_values[max_s_index]

print(f"Analysis complete for alpha from {alpha_values[0]:.2f} to {alpha_values[-1]:.2f}:")
print(f"  Maximum S value found: {max_s_value:.6f}")
print(f"  This maximum occurs at alpha = {max_alpha:.6f}")

# --- 6. Generate the Plot ---
plt.figure(figsize=(9, 6))
plt.plot(alpha_values, s_values, label='S(α)', color='blue', linewidth=2)

# --- NEW: Add a marker for the maximum ---
plt.plot(max_alpha, max_s_value, 
         'ro', # 'r' for red, 'o' for circle marker
         markersize=8, 
         label=f'Maximum: S={max_s_value:.4f} at $\\alpha={max_alpha:.4f}$')

plt.xlabel('α (alpha)', fontsize=12)
plt.ylabel('S(α)', fontsize=12)
plt.title('Plot of $S(\\alpha)$ vs. $\\alpha$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("s_vs_alpha_plot_with_max.png")

print("\nPlot saved as 's_vs_alpha_plot_with_max.png'")