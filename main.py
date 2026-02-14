# test_bci_310.py - Run this to verify everything
#!/usr/bin/env python3.10

print("🐍 Python version:", __import__('sys').version)
print("🧠 Testing BCI Digital Twin for Python 3.10...")

# Core imports
import numpy as np
print("✅ NumPy:", np.__version__)

import scipy.signal
print("✅ SciPy:", scipy.__version__)

try:
    import brainflow
    print("✅ BrainFlow:", brainflow.__version__)
except:
    print("❌ BrainFlow missing")

import pylsl
print("✅ PyLSL ready")

# Test digital twin
from digital_twin_bci import DigitalTwinBCI
twin = DigitalTwinBCI()
sample = twin.generate_realistic_eeg()

print("🎯 DIGITAL TWIN EEG SAMPLE:")
print(f"   Channels: {len(sample.eeg)}")
print(f"   C3 (ch2): {sample.eeg[2]:.1f}μV")
print(f"   C4 (ch6): {sample.eeg[6]:.1f}μV")
print(f"   Pz (ch4): {sample.eeg[4]:.1f}μV")
print("🚀 Python 3.10 BCI READY!")