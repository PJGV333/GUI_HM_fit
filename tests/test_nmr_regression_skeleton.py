import os
import sys
# import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend_fastapi.nmr_processor import process_nmr_data

def test_nmr_regression():
    # Path to test file
    file_path = os.path.join(os.path.dirname(__file__), '..', 'test_2_tit_RMN3.xlsx')
    assert os.path.exists(file_path), "Test file not found"

    # Parameters based on user description
    spectra_sheet = "Chemical Shift (ppm)" # Need to check actual sheet name
    conc_sheet = "Concentrations (M)"      # Need to check actual sheet name
    
    # Let's verify sheet names first
    xl = pd.ExcelFile(file_path)
    print(f"Sheet names: {xl.sheet_names}")
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    print(f"Columns: {list(df.columns)}")
    return
    
    # Adjust sheet names if needed based on file content
    if "Chemical Shift (ppm)" not in xl.sheet_names:
        # Try to guess or fail
        pass

    # Columns and Signals
    # User said: "no. of spectra = 39", "no. of resonant nuclei = 7"
    # We need to pick the correct columns.
    # Assuming standard format or known columns.
    # Let's read the file to find columns
    df_conc = pd.read_excel(file_path, sheet_name=1) # Assuming 2nd sheet is conc
    df_spec = pd.read_excel(file_path, sheet_name=0) # Assuming 1st sheet is spectra
    
    # Update sheet names from what we found if they match expected patterns
    conc_sheet = xl.sheet_names[1]
    spectra_sheet = xl.sheet_names[0]
    
    column_names = list(df_conc.columns)
    signal_names = list(df_spec.columns)
    
    # Filter signal names to exclude 'Titration' or similar if present
    # User said 7 signals.
    # Check if first column is axis
    if len(signal_names) > 7:
         # Maybe first column is index
         signal_names = signal_names[1:8] # Take 7 columns?
    
    # Setup Model
    # User mentioned betas: 111, 101, 011, 200, 201, 211
    # Model matrix usually: [H, G, L] or similar.
    # Need to know the components.
    # Assuming H and G.
    # Model matrix columns: Species. Rows: Components.
    # Let's assume components are H and G.
    # Species: H, G, HG, H2G, etc?
    # User said: "beta111, beta101, beta011, beta200, beta201, beta211"
    # This implies 3 components? Metal, Ligand, Proton? Or H, G, something else?
    # "test_2_tit_RMN3" suggests Titration.
    
    # Let's try to infer from "beta111" -> M:1, L:1, H:1?
    # If components are M, L, H.
    # But usually in this app we define model matrix manually.
    # Let's use a standard model that fits these betas if possible, or just a dummy model 
    # if we can't reproduce exact physics without more info.
    # BUT, the user gave specific betas to check!
    # β111 = 7.1888
    # β101 = 3.5777
    # β011 = 4.0632
    # β200 = 0.1 (fixed)
    # β201 = 6.2558
    # β211 = 9.4656
    
    # This implies species indices:
    # 111, 101, 011, 200, 201, 211
    # Plus likely 100, 010, 001 (components themselves)
    
    # Let's assume 3 components.
    # Model matrix (n_species x n_components)
    # We need to define the model matrix to pass to the processor.
    # If we don't know the exact model, we might fail to reproduce the exact betas.
    # However, the user said "Chequeo que debe imprimir HM fit for this dataset: no. of spectra = 39..."
    
    # Let's try to read the file first to see headers, maybe they give a clue.
    pass

if __name__ == "__main__":
    test_nmr_regression()
