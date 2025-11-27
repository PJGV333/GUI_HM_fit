"""Simple test to check if backend can start"""
import sys
print("Python path:", sys.path)

try:
    from backend_fastapi import main
    print("✓ main.py imports OK")
except Exception as e:
    print("✗ Error importing main.py:", e)
    import traceback
    traceback.print_exc()

try:
    from backend_fastapi import spectroscopy_processor
    print("✓ spectroscopy_processor.py imports OK")
except Exception as e:
    print("✗ Error importing spectroscopy_processor.py:", e)
    import traceback
    traceback.print_exc()
