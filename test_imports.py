#!/usr/bin/env python3
import sys
sys.path.insert(0, '/mnt/HDD_4TB/GUI_HM_fit')

print("Step 1: Testing basic imports...")
try:
    import backend_fastapi
    print("✓ backend_fastapi package imports OK")
except Exception as e:
    print(f"✗ backend_fastapi import failed: {e}")
    sys.exit(1)

print("\nStep 2: Testing spectroscopy_processor...")
try:
    from backend_fastapi import spectroscopy_processor
    print("✓ spectroscopy_processor imports OK")
except Exception as e:
    print(f"✗ spectroscopy_processor import failed: {e}")
    sys.exit(1)

print("\nStep 3: Testing nmr_processor...")
try:
    from backend_fastapi import nmr_processor
    print("✓ nmr_processor imports OK")
except Exception as e:
    print(f"✗ nmr_processor import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Testing main...")
try:
    from backend_fastapi import main
    print("✓ main imports OK")
except Exception as e:
    print(f"✗ main import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ All imports successful! ✓✓✓")
