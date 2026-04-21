import psi4
import numpy as np

psi4.set_memory('4 GB')
psi4.set_output_file('problem1_reactants.dat')
methods_list = ['hf', 'mp2', 'b3lyp']
basis_set = '6-31G'

print(f"--- Problem 1: Reactants Optimization ({basis_set}) ---")

for method in methods_list:
    print(f"\nRunning Optimization: {method.upper()}/{basis_set} ...")
    reactants = psi4.geometry("""
        -1 1
        C
        H 1 1.07
        H 1 1.07 2 109.5
        H 1 1.07 3 109.5 2 120.0
        Br 1 2.60 3 109.5 2 -120.0
        Cl 1 2.90 3 70.0  2 60.0
    """)
    final_energy = psi4.optimize(f'{method}/{basis_set}', molecule=reactants)
    print(f"  -> Final Energy: {final_energy:.6f} a.u.")
    geom_bohr = reactants.geometry().to_array()
    geom_ang = geom_bohr * psi4.constants.bohr2angstroms
    dist_c_br = np.linalg.norm(geom_ang[0] - geom_ang[4])
    dist_c_cl = np.linalg.norm(geom_ang[0] - geom_ang[5])
    print(f"  -> C-Br Bond Length: {dist_c_br:.4f} Å")
    print(f"  -> C...Cl Distance:  {dist_c_cl:.4f} Å")
    print("  -> Final Geometry (Angstroms):")
    print(f"     {'Atom':<4} {'X':>10} {'Y':>10} {'Z':>10}")
    for i in range(reactants.natom()):
        symbol = reactants.symbol(i)
        x = geom_ang[i][0]
        y = geom_ang[i][1]
        z = geom_ang[i][2]
        print(f"     {symbol:<4} {x:10.5f} {y:10.5f} {z:10.5f}")
        
    print("-" * 40)

print("\nOptimizations complete. Data in 'problem1_reactants.dat'.")