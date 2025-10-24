import subprocess
import os
from pathlib import Path

def run_gs2_simulations_Local(dir):

    GS2_EXE = "/home/Felix/Documents/Physics_Work/Project_Codes/GS2_TGLF/gs2/bin/gs2"
    
    run_dir = Path(dir)
    print(run_dir)
    cmd = [
    "mpirun", "-n", "4",
    GS2_EXE,
    run_dir / "gs2.in"
    ]
    # Run and wait for it to finish
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result)

run_gs2_simulations_Local("/home/Felix/Documents/Physics_Work/Project_Codes/Beta_Prime_Scan/gs2_test")