import psi4
import numpy as np

# 1. Set Output and Memory
psi4.set_memory('4 GB')
psi4.set_output_file('problem3_ts.dat')

# 2. Define Method and Basis
method = 'b3lyp'
basis_set = '6-31G'

print(f"--- Problem 3: Transition State Optimization ({method}/{basis_set}) ---")

# 3. Define Geometry (Specific Guess for TS)
# Note: C1 in the PDF refers to Chlorine (Cl)
# The geometry defines a trigonal bipyramidal structure
ts_guess = psi4.geometry("""
    -1 1
    C
    H 1 1.074525
    H 1 1.074525 2 119.950
    H 1 1.074525 3 119.950 2 -175.516
    Br 1 2.555872 4 88.704 3 -87.759
    Cl 1 2.375383 4 91.296 5 -179.999
""")

# 4. Set Special Optimization Options for TS
psi4.set_options({
    'opt_type': 'ts',           # Search for a Transition State (Saddle Point)
    'full_hess_every': 1,       # Re-calc Hessian every step (Robustness)
    'g_convergence': 'gau_loose' # Often helps TS convergence
})

print(f"\nRunning TS Optimization: {method.upper()}/{basis_set} ...")
print("Note: This calculation is slower because it calculates the Hessian every step.")

# 5. Run Optimization
final_energy = psi4.optimize(f'{method}/{basis_set}', molecule=ts_guess)

# 6. Print Results
print(f"  -> Final TS Energy: {final_energy:.6f} a.u.")

# --- GEOMETRY ANALYSIS ---
geom_ang = ts_guess.geometry().to_array() * psi4.constants.bohr2angstroms

# Calculate Critical Bond Lengths (C-Br and C-Cl)
# Indices: 0=C, 4=Br, 5=Cl
dist_c_br = np.linalg.norm(geom_ang[0] - geom_ang[4])
dist_c_cl = np.linalg.norm(geom_ang[0] - geom_ang[5])

print(f"  -> C...Br Distance: {dist_c_br:.4f} Å")
print(f"  -> C...Cl Distance: {dist_c_cl:.4f} Å")

# Calculate the Cl-C-Br Angle (Should be close to 180)
# Vector C->Cl
v1 = geom_ang[5] - geom_ang[0]
# Vector C->Br
v2 = geom_ang[4] - geom_ang[0]
# Angle calculation
cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.degrees(np.arccos(cosine_angle))

print(f"  -> Cl-C-Br Angle:   {angle:.2f} degrees")

print("\nTS Optimization complete!")