from pyrokinetics import Pyro, PyroScan, template_dir
import os
import pathlib
from typing import Union
import numpy as np
import copy
import subprocess

STEP_CASE = "SPR-045"

def main():
    in_loc = f"/home/Felix/Documents/Physics_Work/Project_Codes/GS2_TGLF/TGLF/STEP_CASES/{STEP_CASE}/input.tglf"
    pyro = Pyro(gk_file=in_loc, gk_code="TGLF")
    pyro.numerics.nky = 1
    pyro.numerics.gamma_exb = 0.0
    pyro.local_species.electron.domega_drho = 0.0

    # Use existing parameter with more realistic ky range
    param_1 = "ky" 
    values_1 = np.arange(0.1, 0.5, 0.1)

    # Add beta parameter with realistic values
    param_2 = "beta"
    values_2 = np.arange(0, 0.20, 0.02)
    
    # Dictionary of param and values
    param_dict = {param_1: values_1, param_2: values_2}

    # Create PyroScan object with more descriptive naming
    pyro_scan_tglf = PyroScan(
        pyro,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
        base_directory="beta_scan_runs"
    )

    # Add function to enforce consistent beta prime
    pyro_scan_tglf.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )
    
    def enforce_beta_prime(pyro):
        pyro.enforce_consistent_beta_prime()
        return pyro    

    # If there are kwargs to function then define here
    param_2_kwargs = {}

    # Add function to pyro
    pyro_scan_tglf.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    try:
        pyro_scan_tglf.write(
            file_name="input.tglf",
            base_directory="parameter_scan_tglf",
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None

    pyro_copy = copy.copy(pyro)
    pyro_copy.gk_input.data["WRITE_WAVEFUNCTION"] = 0
    # Switch to GS2
    pyro_copy.gk_code = "GS2"   


    # Create PyroScan object with more descriptive naming
    pyro_scan_gs2 = PyroScan(
        pyro_copy,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
        base_directory="beta_scan_runs"
    )


    # Add proper parameter mapping for beta
    pyro_scan_gs2.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )

    # Create scan directory and write input files
    try:
        pyro_scan_gs2.write(
            file_name="gs2.in",
            base_directory="parameter_scan_gs2",
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None

    return pyro_scan_tglf, pyro_scan_gs2

TGLF_BINARY_PATH = os.path.expandvars("/users/hmq514/scratch/TGLF/gacode/tglf/src/tglf")
TGLF_PARSE_SCRIPT = os.path.expandvars("$GACODE_ROOT/tglf/bin/tglf_parse.py")




def generate_final_tglf_input(dir):
    parse_script = TGLF_PARSE_SCRIPT
# generate the proper tglf input file
    changed_input = os.path.join(dir, "input.tglf.gen")
    with open(changed_input, "w") as outfile:
        subprocess.run(
            ["python", parse_script], cwd=dir, stdout=outfile, stderr=subprocess.STDOUT
        )



def run_file(dir):
    binary_path = TGLF_BINARY_PATH
    generate_final_tglf_input(dir)

    # ✅ Run the binary and redirect output using Python's stdout
    output_file = os.path.join(dir, "out.tglf.run")
    #print(f"Running TGLF in directory: {dir}")
    with open(output_file, "w") as outfile:
        subprocess.run([binary_path], cwd=dir, stdout=outfile, check=True)



def run_file_full(sim_dir):
    # Make sure the directory exists
    os.makedirs(sim_dir, exist_ok=True)

    parent = os.path.dirname(sim_dir)
    folder = os.path.basename(sim_dir)

    subprocess.run(
        ["tglf", "-e", folder],
        cwd=parent
    )
    print(f"Running simulation in directory: {sim_dir}")

def run_sim(pyro_scan):
    for run_dir in pyro_scan.run_directories:
        run_file_full(run_dir)


import os
import subprocess
from pathlib import Path
import textwrap

def run_gs2_simulations(pyro_scan):

    for run_dir in pyro_scan.run_directories:
        run_dir = Path(run_dir)
        job_name = f"gs2_{run_dir.name}"
        print(run_dir)
        EXE = "/users/hmq514/scratch/gs2/bin/gs2"
        slurm_script = textwrap.dedent(f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}            # Job name
#SBATCH --partition=nodes               # What partition the job should run on
#SBATCH --time=0-32:00:00               # Time limit (DD-HH:MM:SS)
#SBATCH --ntasks=96                      # Number of MPI tasks to request
#SBATCH --cpus-per-task=1               # Number of CPU cores per MPI task
#SBATCH --exclusive                     # tries to take the core exclusively
#SBATCH --account=pet-gspt-2019         # Project account to use
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hmq514@york.ac.uk   # Where to send mail
#SBATCH --output={job_name}-%j.log              # Standard output log
#SBATCH --error={job_name}-%j.err              # Standard error log

# Purge any previously loaded modules
module purge

# Load modules
module load gompi/2022b OpenMPI/4.1.4-GCC-12.2.0 netCDF-Fortran/4.6.0-gompi-2022b FFTW/3.3.10-GCC-12.2.0 OpenBLAS/0.3.21-GCC-12.2.0 Python/3.10.8-GCCcore-12.2.0

# Commands to run. 

export GK_SYSTEM='viking'
export MAKEFLAGS='-IMakefiles'
export HDF5_USE_FILE_LOCKING=FALSE

ulimit -s unlimited

######################## Above is in bashrc anyway

export OMP_NUM_THREADS=1
INPUT_DIR=$(dirname "{run_dir}")  # Extract directory of input file
srun --hint=nomultithread --distribution=block:block -n 96 {EXE} {run_dir}/gs2.in | tee OUTPUT
        """)

        # Write the Slurm script to the run directory
        script_path = run_dir / "jobscript.job"
        script_path.write_text(slurm_script)

        # Submit with sbatch
        print(f"Submitting job for {run_dir}...")
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)

        # Extract job ID from sbatch output
        print(result.stdout.strip())


def load_results(pyro_scan_tglf, pyro_scan_gs2):
    # Load output from tglf
    pyro_scan_tglf.load_gk_output()

    data_tglf = pyro_scan_tglf.gk_output
    growth_rate_tglf = data_tglf['growth_rate']
    print("tglf data")
    print(f"growth rate: {growth_rate_tglf.ky}")
    print(f"growth rate: {growth_rate_tglf.beta}")
    mode_frequency_tglf = data_tglf['mode_frequency']

    growth_rate_tolerance_tglf = data_tglf['growth_rate_tolerance']
    # growth_rate_tglf = growth_rate_tglf.where(growth_rate_tolerance_tglf < 0.1)
    # mode_frequency_tglf = mode_frequency_tglf.where(growth_rate_tolerance_tglf < 0.1)

     # Load output from gs2
    # pyro_scan_gs2.load_gk_output()

    # data_gs2 = pyro_scan_gs2.gk_output
    # growth_rate_gs2 = data_gs2['growth_rate']
    # print("gs2 data")
    # print(f"growth rate: {growth_rate_gs2.ky}")
    # print(f"growth rate: {growth_rate_gs2.beta}")
    # mode_frequency_gs2 = data_gs2['mode_frequency']

    # growth_rate_tolerance_gs2 = data_gs2['growth_rate_tolerance']
    # growth_rate_gs2 = growth_rate_gs2.where(growth_rate_tolerance_gs2 < 0.1)
    # mode_frequency_gs2 = mode_frequency_gs2.where(growth_rate_tolerance_gs2 < 0.1)


    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Plot growth rate and mode frequency vs ky for different beta values

    
    
    fig = plt.figure(figsize=(9, 3*len(growth_rate_tglf.beta)))
    gs = gridspec.GridSpec(len(growth_rate_tglf.beta), 2, hspace=0, wspace=0.3)

    axes = np.empty((len(growth_rate_tglf.beta), 2), dtype=object)

    for i, beta in enumerate(growth_rate_tglf.beta.values):
        # Create subplots
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        axes[i, 0] = ax1
        axes[i, 1] = ax2

        # Plot data
        ax1.plot(growth_rate_tglf.ky, growth_rate_tglf.sel(beta=beta).sel(mode=0), label=rf"tglf_$\beta={beta:.3f}$")
        #ax1.plot(growth_rate_gs2.ky, growth_rate_gs2.sel(beta=beta).sel(mode=0), label=rf"GS2_$\beta={beta:.3f}$")
        ax2.plot(mode_frequency_tglf.ky, mode_frequency_tglf.sel(beta=beta).sel(mode=0), label=rf"tglf_$\beta={beta:.3f}$")
        #ax2.plot(mode_frequency_gs2.ky, mode_frequency_gs2.sel(beta=beta).sel(mode=0), label=rf"GS2_$\beta={beta:.3f}$")

        # Axis labels
        ax1.set_ylabel(r'$\gamma (c_{s}/a)$')
        ax2.set_ylabel(r'$\omega (c_{s}/a)$')

        ax1.grid(True)
        ax2.grid(True)

        # Row label on right-hand side
        ax2.text(
            1.05, 0.5,
            rf"$\beta={beta:.2f}$",
            transform=ax2.transAxes,
            va='center', ha='left',
            fontsize=10
        )
    for i in range(len(growth_rate_tglf.beta) - 1):  # all rows except bottom
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_xlabel("") 
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_xlabel("")
    # Only bottom row gets x-axis labels
    axes[-1, 0].set_xlabel(r"$k_y$")
    axes[-1, 1].set_xlabel(r"$k_y$")

    # Layout and title
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"Plot of Growth rate and Frequency against $k_y$ for different $\beta$ or STEP Case SPR-045",
                fontsize=16, y=0.95)

    # Save everything in ONE file
    plt.savefig(f"Beta_Scans/{STEP_CASE}_all_betas_pairs.png", dpi=300)
    plt.close(fig)




    # Plot growth rate and mode frequency vs beta for different ky values

    fig = plt.figure(figsize=(9, 3*len(growth_rate_tglf.ky)))
    gs = gridspec.GridSpec(len(growth_rate_tglf.ky), 2, hspace=0, wspace=0.3)

    axes = np.empty((len(growth_rate_tglf.ky), 2), dtype=object)

    for i, ky in enumerate(growth_rate_tglf.ky.values):
        # Create subplots run_sim(pyro_scan)
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        axes[i, 0] = ax1
        axes[i, 1] = ax2

        # Plot data
        ax1.plot(growth_rate_tglf.beta, growth_rate_tglf.sel(ky=ky).sel(mode=0), label=rf"tglf_$k_y={ky:.2f}$")
        #ax1.plot(growth_rate_gs2.beta, growth_rate_gs2.sel(ky=ky).sel(mode=0), label=rf"gs2_$k_y={ky:.2f}$")
        ax2.plot(mode_frequency_tglf.beta, mode_frequency_tglf.sel(ky=ky).sel(mode=0), label=rf"tglf_$k_y={ky:.2f}$")
        #ax2.plot(mode_frequency_gs2.beta, mode_frequency_gs2.sel(ky=ky).sel(mode=0), label=rf"gs2_$k_y={ky:.2f}$")

        # Axis labels
        ax1.set_ylabel(r'$\gamma (c_{s}/a)$')
        ax2.set_ylabel(r'$\omega (c_{s}/a)$')

        ax1.grid(True)
        ax2.grid(True)

        # Row label on right-hand side
        ax2.text(
            1.05, 0.5,
            rf"$k_y={ky:.2f}$",
            transform=ax2.transAxes,
            va='center', ha='left',
            fontsize=10
        )
    for i in range(len(growth_rate_tglf.ky) - 1):  # all rows except bottom
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_xlabel("") 
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_xlabel("")
    # Only bottom row gets x-axis labels
    axes[-1, 0].set_xlabel(r"$\beta$")
    axes[-1, 1].set_xlabel(r"$\beta$")

    # Layout and title
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"Plot of Growth rate and Frequency against $\beta$ for different $k_y$ for STEP Case SPR-045",
                fontsize=16, y=0.95)

    # Save everything in ONE file
    plt.savefig(f"Beta_Scans/{STEP_CASE}_all_ky_pairs.png", dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    pyro_scan_tglf,pyro_scan_gs2 = main()
    print(pyro_scan_tglf.run_directories)
    run_sim(pyro_scan_tglf)
    #run_gs2_simulations(pyro_scan_gs2)
    load_results(pyro_scan_tglf,pyro_scan_gs2)