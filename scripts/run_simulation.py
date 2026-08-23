# ============================================================
# RUN SIMULATION — runner script
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from simulation.frame_simulator import simulate_stream

if __name__ == "__main__":
    simulate_stream(n=20, scenario="security")
