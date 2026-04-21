import psi4
import numpy as np

psi4.set_memory('4 GB')
psi4.set_output_file('problem3_ts.dat')

method = 'b3lyp'
basis_set = '6-31G'

print(f"--- Problem 3: Transition State Optimization ({method}/{basis_set}) ---")
ts_guess = psi4.geometry("""
    -1 1
    C
    H 1 1.074525
    H 1 1.074525 2 119.950
    H 1 1.074525 3 119.950 2 -175.516
    Br 1 2.555872 4 88.704 3 -87.759
    Cl 1 2.375383 4 91.296 5 -179.999
""")

psi4.set_options({
    'opt_type': 'ts',           
    'full_hess_every': 1,       
    'g_convergence': 'gau_loose' 
})

print(f"\nRunning TS Optimization: {method.upper()}/{basis_set} ...")
print("Note: This calculation is slower because it calculates the Hessian every step.")
final_energy = psi4.optimize(f'{method}/{basis_set}', molecule=ts_guess)
print(f"  -> Final TS Energy: {final_energy:.6f} a.u.")
geom_ang = ts_guess.geometry().to_array() * psi4.constants.bohr2angstroms
dist_c_br = np.linalg.norm(geom_ang[0] - geom_ang[4])
dist_c_cl = np.linalg.norm(geom_ang[0] - geom_ang[5])
v1 = geom_ang[5] - geom_ang[0]
v2 = geom_ang[4] - geom_ang[0]
cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.degrees(np.arccos(cosine_angle))
print(f"  -> C...Br Distance: {dist_c_br:.4f} Å")
print(f"  -> C...Cl Distance: {dist_c_cl:.4f} Å")
print(f"  -> Cl-C-Br Angle:   {angle:.2f} degrees")
print("\nTS Optimization complete!")