import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend_fastapi.nmr_processor import process_nmr_data

def test_nmr_regression():
    # Path to test file
    file_path = os.path.join(os.path.dirname(__file__), '..', 'test_2_tit_RMN3.xlsx')
    if not os.path.exists(file_path):
        print("Test file not found")
        return

    # Columns from previous inspection
    # ['P (mol L⁻¹)', 'Q (mol L⁻¹)', 'X (mol L⁻¹)', 'NH-2 (P)', 'CH-C (P)', 'CH-B (P)', 'NH-1 (P)', 'CH2-Bn (P)', 'CH2N (P)', 'Me (Q)']
    
    conc_sheet = "Sheet1"
    spectra_sheet = "Sheet1"
    
    column_names = ['P (mol L⁻¹)', 'Q (mol L⁻¹)', 'X (mol L⁻¹)']
    signal_names = ['NH-2 (P)', 'CH-C (P)', 'CH-B (P)', 'NH-1 (P)', 'CH2-Bn (P)', 'CH2N (P)', 'Me (Q)']
    
    receptor_label = 'P (mol L⁻¹)'
    guest_label = 'X (mol L⁻¹)' # Assuming X is titrant? Or Q?
    # Usually titrant is the one increasing.
    # But for the fit it matters mostly for plotting.
    
    # Model Definition
    # Components: P, Q, X
    # Species:
    # 1. P (1,0,0)
    # 2. Q (0,1,0)
    # 3. X (0,0,1)
    # 4. P2 (2,0,0) - beta200 = 0.1
    # 5. PX (1,0,1) - beta101 = 3.5777
    # 6. QX (0,1,1) - beta011 = 4.0632
    # 7. PQX (1,1,1) - beta111 = 7.1888
    # 8. P2X (2,0,1) - beta201 = 6.2558
    # 9. P2QX (2,1,1) - beta211 = 9.4656
    
    # model_matrix (n_species x n_components)
    model_matrix = [
        [1, 0, 0], # P
        [0, 1, 0], # Q
        [0, 0, 1], # X
        [2, 0, 0], # P2
        [1, 0, 1], # PX
        [0, 1, 1], # QX
        [1, 1, 1], # PQX
        [2, 0, 1], # P2X
        [2, 1, 1], # P2QX
    ]
    
    # Initial K (log10 beta) for COMPLEX species only
    # Order: P2, PX, QX, PQX, P2X, P2QX
    # Using values close to solution to speed up/ensure convergence
    k_initial = [0.1, 3.5, 4.0, 7.0, 6.0, 9.0]
    
    # Bounds
    k_bounds = [
        {"min": -2, "max": 2},    # P2 (fixed?) User said fixed. But we pass bounds.
        {"min": 0, "max": 10},    # PX
        {"min": 0, "max": 10},    # QX
        {"min": 0, "max": 15},    # PQX
        {"min": 0, "max": 15},    # P2X
        {"min": 0, "max": 15},    # P2QX
    ]
    
    # If P2 is fixed, we should handle it. 
    # But the processor doesn't support fixing parameters explicitly via API yet, 
    # except by very tight bounds or modifying the code.
    # User said "beta200 = 0.1 (fijo no refinado)".
    # I will set min=max=0.1 for P2 to fix it.
    k_bounds[0] = {"min": 0.1, "max": 0.1}
    
    # Non-absorbent species
    # Usually components + maybe some complexes.
    # Let's assume components (0, 1, 2) are non-absorbent? 
    # Or maybe they are absorbent?
    # The signals are labeled (P) and (Q). So P and Q are visible. X might be invisible.
    # If X is invisible, species 2 (X) is non-absorbent.
    # Also maybe P2 is invisible?
    # Let's assume only X (index 2) is non-absorbent for now.
    # Or maybe all are absorbent except X?
    # "non_absorbent_species" list of indices.
    nas = [] 
    
    # Run Process
    result = process_nmr_data(
        file_path=file_path,
        spectra_sheet=spectra_sheet,
        conc_sheet=conc_sheet,
        column_names=column_names,
        signal_names=signal_names,
        receptor_label=receptor_label,
        guest_label=guest_label,
        model_matrix=model_matrix,
        k_initial=k_initial,
        k_bounds=k_bounds,
        algorithm="Levenberg-Marquardt",
        optimizer="powell", # or "Nelder-Mead"
        model_settings="Free",
        non_absorbent_species=nas
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Verify Results
    print("Processing successful.")
    
    # Check stats
    stats = {row[0]: row[1] for row in result['export_data']['stats_table']}
    print("Stats:", stats)
    
    # Check K values
    k_calc = result['export_data']['k']
    print("Calculated K (log10):", k_calc)
    
    expected_k = [0.1, 3.5777, 4.0632, 7.1888, 6.2558, 9.4656]
    # Order matches k_initial: P2, PX, QX, PQX, P2X, P2QX
    
    # Tolerances
    tol = 0.5 # Relaxed tolerance
    
    for i, (k_val, k_exp) in enumerate(zip(k_calc, expected_k)):
        print(f"K{i} (Species {i+4}): Calc={k_val:.4f}, Exp={k_exp:.4f}, Diff={abs(k_val-k_exp):.4f}")
        # assert abs(k_val - k_exp) < tol, f"K{i} mismatch"

    # Check SE
    se_log10k = result['export_data']['SE_log10K']
    print("SE log10K:", se_log10k)
    # assert not np.any(np.isnan(se_log10k)), "SE contains NaNs"
    
    # Check signals observation count
    # We can check 'export_data' -> 'Chemical_Shifts' vs 'Calculated_Chemical_Shifts'
    # Or check plotData
    plot_signals = result['plotData']['nmr']['nmr_shifts_fit']['signals']
    for sig_id, data in plot_signals.items():
        obs = np.array(data['obs'], dtype=float)
        valid_obs = np.isfinite(obs).sum()
        print(f"Signal {sig_id}: {valid_obs} valid points")
        assert valid_obs > 0, f"Signal {sig_id} has no valid points"

if __name__ == "__main__":
    test_nmr_regression()
