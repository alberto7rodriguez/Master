import psi4
import numpy as np
import matplotlib.pyplot as plt

psi4.set_memory('4 GB')
psi4.set_output_file('problem4_scan.dat')

print("--- Problem 4: PES Scan (B3LYP/6-31G) ---")

distances = np.arange(2.02, 3.52, 0.02)
energies = []

print(f"Scanning {len(distances)} points...")

for i, r_val in enumerate(distances):
    geometry_string = f"""
        -1 1
        C
        H 1 1.080878
        H 1 1.080878 2 113.880757
        H 1 1.080878 3 113.880757 2 132.876699
        Br 1 {r_val:.4f} 3 104.571394 2 -113.561651
        Cl 1 2.903066 3 75.475051 2 66.438349
    """
    mol = psi4.geometry(geometry_string)
    psi4.set_options({
        'basis': '6-31G',
        'maxiter': 200,
        'geom_maxiter': 500,
        'opt_type': 'min',
        'g_convergence': 'gau_loose',
        'frozen_distance': '1 5'
    })
    try:
        e = psi4.optimize('b3lyp', molecule=mol)
        energies.append(e)
        if i % 5 == 0: print(f"  R={r_val:.2f} Å -> E={e:.6f}")
    except Exception as e:
        print(f"  Step {i+1} Failed: {e}")
        energies.append(None)

valid_data = [(d, e) for d, e in zip(distances, energies) if e is not None]
d_plot, e_plot = zip(*valid_data)
plt.figure(figsize=(10, 6))
plt.plot(d_plot, e_plot, 'o-', markersize=3)
plt.xlabel('C-Br Bond Distance ($\AA$)')
plt.ylabel('Total Energy (a.u.)')
plt.title('Minimum Energy Path (PES Scan)')
plt.grid(True)
plt.show()
print("\n\t 6-31G B3LYP energy as a function of C-Br distance\n")
for r, e in valid_data:
    print(f"\t{r:5.3f}  {e:20.10f}")

