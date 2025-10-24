from pyrokinetics import Pyro, PyroScan, template_dir
import os
import pathlib
from typing import Union
import numpy as np
import copy
import subprocess
from pathlib import Path
from pyrokinetics.diagnostics.gs2_gp import gs2_gp

REPO_ROOT = Path(__file__).resolve().parent   # repo root if this file sits at repo root



STEP_DATA_DIR = "/home/Felix/Documents/Physics_Work/Project_Codes/Beta_Prime_Scan/TGLF_Templates"
STEP_CASE = "r3"

TGLF_BINARY_PATH = os.path.expandvars("/users/hmq514/scratch/TGLF/gacode/tglf/src/tglf")
TGLF_PARSE_SCRIPT = os.path.expandvars("$GACODE_ROOT/tglf/bin/tglf_parse.py")


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d/"


models = [
            "growth_rate_log", "mode_frequency_log",
        ]


def Read_from_gs2():
    in_loc = f"{STEP_DATA_DIR}/{STEP_CASE}/input.tglf"
    pyro = Pyro(gk_file=in_loc, gk_code="TGLF")
    pyro.gk_code = "GS2"

    pyro.numerics.nky = 1
    pyro.numerics.gamma_exb = 0.0
    pyro.local_species.electron.domega_drho = 0.0

    # Use existing parameter with more realistic ky range
    param_1 = "ky" 
    values_1 = np.arange(0.3,0.7, 0.1)

    # Add beta parameter with realistic values
    param_2 = "beta"
    values_2 = np.arange(0.01, 0.2, 0.01)
    
    # Dictionary of param and values
    param_dict = {param_1: values_1, param_2: values_2}

    def enforce_beta_prime(pyro):
        pyro.enforce_consistent_beta_prime()

    # If there are kwargs to function then define here
    param_2_kwargs = {}

    
    pyro.local_species.electron.domega_drho = 0.0

    # Create PyroScan object with more descriptive naming
    pyro_scan_gs2 = PyroScan(
        pyro,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
    )

    # Add proper parameter mapping for beta
    pyro_scan_gs2.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )

    # Add function to gs2
    pyro_scan_gs2.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    try:
        pyro_scan_gs2.write(
            file_name="gs2.in",
            base_directory=REPO_ROOT / "parameter_scan_gs2",
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None
    

    
    pyro_copy = copy.copy(pyro)

    # Switch to TGLF
    pyro_copy.gk_code = "TGLF"

     # Create PyroScan object with more descriptive naming
    pyro_scan_tglf = PyroScan(
        pyro_copy,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
    )

    # Add function to enforce consistent beta prime
    pyro_scan_tglf.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    try:
        pyro_scan_tglf.write(
            file_name="input.tglf",
            base_directory=REPO_ROOT / "parameter_scan_tglf",
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None
    


    pyro_copy_2 = copy.copy(pyro)

     # Switch to TGLF
    pyro_copy_2.gk_code = "TGLF"

    dict = {
        "WIDTH":1.9,
        "WIDTH_MIN":0.495,
        "FILTER":2,
            }
    pyro_copy_2.gk_input.add_flags(dict)
    pyro_copy_2.gk_input.data["nbasis_max"] = 10
    pyro_copy_2.gk_input.data["nbasis_min"] = 2
    pyro_copy_2.gk_input.data["theta_trapped"] = 2


     # Create PyroScan object with more descriptive naming
    pyro_scan_tglf_F = PyroScan(
        pyro_copy_2,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
    )

    # Add function to enforce consistent beta prime
    pyro_scan_tglf_F.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf_F.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    #try:
    pyro_scan_tglf_F.write(
        file_name="input.tglf",
        base_directory=REPO_ROOT / "parameter_scan_tglf_F",
        template_file=None
    )
    # except Exception as e:
    #     print(f"Error writing parameter scan files: {e}")
    #     return None
    

    pyro_copy_3 = copy.copy(pyro)
    pyro_copy_3.gk_code = "TGLF"

    dict = {
        "WIDTH":1,
        "FILTER":2,
        "THETA_TRAPPED":0.57,
        "FIND_WIDTH":"F",
            }
    pyro_copy_3.gk_input.add_flags(dict)
    pyro_copy_3.gk_input.data["nbasis_max"] = 36
    pyro_copy_3.gk_input.data["theta_trapped"] = 0.57


     # Create PyroScan object with more descriptive naming
    pyro_scan_tglf_M = PyroScan(
        pyro_copy_3,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
    )

    # Add function to enforce consistent beta prime
    pyro_scan_tglf_M.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf_M.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    pyro_scan_tglf_M.write(
        file_name="input.tglf",
        base_directory=REPO_ROOT / "parameter_scan_tglf_M",
        template_file=None
    )


    pyro_copy_4 = copy.copy(pyro)
    pyro_copy_4.gk_code = "TGLF"

    dict = {
        "WIDTH":1,
        "FILTER":2,
        "THETA_TRAPPED":0.57,
        "FIND_WIDTH":"F",
            }
    pyro_copy_4.gk_input.add_flags(dict)
    pyro_copy_4.gk_input.data["nbasis_max"] = 20
    pyro_copy_4.gk_input.data["theta_trapped"] = 0.57


     # Create PyroScan object with more descriptive naming
    pyro_scan_tglf_ML = PyroScan(
        pyro_copy_4,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
        file_name="input.tglf",
    )

    # Add function to enforce consistent beta prime
    pyro_scan_tglf_ML.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf_ML.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    pyro_scan_tglf_ML.write(
        file_name="input.tglf",
        base_directory=REPO_ROOT / "parameter_scan_tglf_ML",
        template_file=None
    )

    return pyro_scan_tglf,pyro_scan_tglf_F,pyro_scan_tglf_M,pyro_scan_tglf_ML, pyro_scan_gs2

def Read_from_tglf():
    in_loc = f"{STEP_DATA_DIR}/{STEP_CASE}/input.tglf"
    pyro = Pyro(gk_file=in_loc, gk_code="TGLF")
    pyro.numerics.nky = 1
    pyro.numerics.gamma_exb = 0.0
    pyro.local_species.electron.domega_drho = 0.0

    # Use existing parameter with more realistic ky range
    param_1 = "ky" 
    values_1 = np.arange(0.1, 0.2, 0.1)

    # Add beta parameter with realistic values
    param_2 = "beta"
    values_2 = np.arange(0.01, 0.05, 0.01)
    
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
    )

    # Add function to enforce consistent beta prime
    pyro_scan_tglf.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )
    
    def enforce_beta_prime(pyro):
        pyro.enforce_consistent_beta_prime()

    # If there are kwargs to function then define here
    param_2_kwargs = {}

    # Add function to tglf
    pyro_scan_tglf.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)


    # Create scan directory and write input files
    try:
        pyro_scan_tglf.write(
            file_name="input.tglf",
            base_directory=REPO_ROOT / "parameter_scan_tglf",
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None

    pryo_copy = copy.copy(pyro)

    # Switch to GS2
    pryo_copy.gk_code = "GS2"   

    pryo_copy.local_species.electron.domega_drho = 0.0
    # Create PyroScan object with more descriptive naming
    pyro_scan_gs2 = PyroScan(
        pryo_copy,
        param_dict,
        value_fmt=".4f",  # Increased precision for small beta values
        value_separator="_",
        parameter_separator="_",
    )

    # Add proper parameter mapping for beta
    pyro_scan_gs2.add_parameter_key(
        parameter_key="beta",
        parameter_attr="numerics", 
        parameter_location=["beta"]
    )

    # Add function to gs2
    pyro_scan_gs2.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    try:
        pyro_scan_gs2.write(
            file_name="gs2.in",
            base_directory=REPO_ROOT / "parameter_scan_gs2",
            template_file=None
        )
    except Exception as e:
        print(f"Error writing parameter scan files: {e}")
        return None

    return pyro_scan_tglf, pyro_scan_gs2



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

def run_gs2_simulations_viking(pyro_scan):

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



def run_gs2_simulations_Local(pyro_scan):

    GS2_EXE = "/home/Felix/Documents/Physics_Work/Project_Codes/GS2_TGLF/gs2/bin/gs2"

    for run_dir in pyro_scan.run_directories:
        run_dir = Path(run_dir)
        job_name = f"gs2_{run_dir.name}"
        print(run_dir)
        cmd = [
        "mpirun", "-n", "4",
        GS2_EXE,
        run_dir / "gs2.in"
        ]
        # Run and wait for it to finish
        result = subprocess.run(cmd, capture_output=True, text=True)


def load_results(pyro_scan_tglf,pyro_scan_tglf_F,pyro_scan_tglf_M,pyro_scan_tglf_ML, pyro_scan_gs2):
    # Load output from tglf
    pyro_scan_tglf.load_gk_output()
    data_tglf = pyro_scan_tglf.gk_output
    growth_rate_tglf = data_tglf['growth_rate']
    mode_frequency_tglf = data_tglf['mode_frequency']

    pyro_scan_tglf_F.load_gk_output()
    data_tglf_F = pyro_scan_tglf_F.gk_output
    growth_rate_tglf_F = data_tglf_F['growth_rate']
    mode_frequency_tglf_F = data_tglf_F['mode_frequency']

    pyro_scan_tglf_M.load_gk_output()
    data_tglf_M = pyro_scan_tglf_M.gk_output
    growth_rate_tglf_M = data_tglf_M['growth_rate']
    mode_frequency_tglf_M = data_tglf_M['mode_frequency']

    pyro_scan_tglf_ML.load_gk_output()
    data_tglf_ML = pyro_scan_tglf_ML.gk_output
    growth_rate_tglf_ML = data_tglf_ML['growth_rate']
    mode_frequency_tglf_ML = data_tglf_ML['mode_frequency']


    growth_rate_tolerance_tglf = data_tglf['growth_rate_tolerance']
    # growth_rate_tglf = growth_rate_tglf.where(growth_rate_tolerance_tglf < 0.1)
    # mode_frequency_tglf = mode_frequency_tglf.where(growth_rate_tolerance_tglf < 0.1)

     # Load output from gs2
    pyro_scan_gs2.load_gk_output()

    data_gs2 = pyro_scan_gs2.gk_output
    growth_rate_gs2 = data_gs2['growth_rate']
    mode_frequency_gs2 = data_gs2['mode_frequency']

    growth_rate_tolerance_gs2 = data_gs2['growth_rate_tolerance']
    # growth_rate_gs2 = growth_rate_gs2.where(growth_rate_tolerance_gs2 < 0.1)
    # mode_frequency_gs2 = mode_frequency_gs2.where(growth_rate_tolerance_gs2 < 0.1)

    # data_gs2_gp = gs2_gp(pyro=pyro_scan_tglf, models_path=models_path, models=models)
    # growth_rate_gs2_gp = data_gs2_gp.gk_output["growth_rate_log_M12"]
    # mode_frequency_gs2_gp = data_gs2_gp.gk_output["mode_frequency_log_M12"]



    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Plot growth rate and mode frequency vs ky for different beta values

    
    
    fig = plt.figure(figsize=(15, 3*len(growth_rate_tglf.beta)))
    gs = gridspec.GridSpec(len(growth_rate_tglf.beta), 2, hspace=0, wspace=0.3)

    axes = np.empty((len(growth_rate_tglf.beta), 2), dtype=object)

    for i, beta in enumerate(growth_rate_tglf.beta.values):
        # Create subplots
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        axes[i, 0] = ax1
        axes[i, 1] = ax2

        # Plot data
        ax1.plot(growth_rate_tglf.ky, growth_rate_tglf.sel(beta=beta).sel(mode=0), label=rf"tglf_Defaults",linestyle="dotted")
        ax1.plot(growth_rate_tglf_F.ky, growth_rate_tglf_F.sel(beta=beta).sel(mode=0), label=rf"tglf_STv1",linestyle="dotted")
        ax1.plot(growth_rate_tglf_M.ky, growth_rate_tglf_M.sel(beta=beta).sel(mode=0), label=rf"tglf_STv2",linestyle="dotted")
        #ax1.plot(growth_rate_tglf_ML.ky, growth_rate_tglf_ML.sel(beta=beta).sel(mode=0), label=rf"tglf_ML")
        ax1.plot(growth_rate_gs2.ky, growth_rate_gs2.sel(beta=beta), label=rf"GS2")
        #ax1.plot(growth_rate_gs2_gp.ky, growth_rate_gs2_gp.sel(beta=beta,output="value"), label=rf"GS2_GP")

        ax2.plot(mode_frequency_tglf.ky, mode_frequency_tglf.sel(beta=beta).sel(mode=0), label=rf"tglf_Defaults",linestyle="dotted")
        ax2.plot(mode_frequency_tglf_F.ky, mode_frequency_tglf_F.sel(beta=beta).sel(mode=0), label=rf"tglf_STv1",linestyle="dotted")
        ax2.plot(mode_frequency_tglf_M.ky, mode_frequency_tglf_M.sel(beta=beta).sel(mode=0), label=rf"tglf_STv2",linestyle="dotted")
        #ax2.plot(mode_frequency_tglf_ML.ky, mode_frequency_tglf_ML.sel(beta=beta).sel(mode=0), label=rf"tglf_ML")
        ax2.plot(mode_frequency_gs2.ky, mode_frequency_gs2.sel(beta=beta), label=rf"GS2")
        #ax2.plot(mode_frequency_gs2_gp.ky, mode_frequency_gs2_gp.sel(beta=beta,output="value"), label=rf"GS2_GP")

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
    fig.suptitle(r"Growth rate and Frequency against $k_y$ for different $\beta$ or STEP Case SPR-045",
                fontsize=16, y=0.95)
    

    # Collect all handles and labels from every axis
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

    # Deduplicate by label
    unique = dict(zip(labels, handles))

    # Create one legend for the whole figure
    fig.legend(
        unique.values(),
        unique.keys(),
        loc='upper center',          # position above the subplots
        ncol=4,                      # adjust as needed
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),   # you can tune this
    )

    plt.subplots_adjust(top=0.9, bottom=0.1)  # give room for legend and title


    # Save everything in ONE file
    plt.savefig(f"Beta_Scans/{STEP_CASE}_all_betas_pairs.png", dpi=300)
    plt.close(fig)




    # Plot growth rate and mode frequency vs beta for different ky values

    fig = plt.figure(figsize=(15, 3*len(growth_rate_tglf.ky)))
    gs = gridspec.GridSpec(len(growth_rate_tglf.ky), 2, hspace=0, wspace=0.3)

    axes = np.empty((len(growth_rate_tglf.ky), 2), dtype=object)

    for i, ky in enumerate(growth_rate_tglf.ky.values):
        # Create subplots run_sim(pyro_scan)
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        axes[i, 0] = ax1
        axes[i, 1] = ax2

        # Plot data
        ax1.plot(growth_rate_tglf.beta, growth_rate_tglf.sel(ky=ky).sel(mode=0), label="tglf_Defaults",linestyle="dashed")
        ax1.plot(growth_rate_tglf_F.beta, growth_rate_tglf_F.sel(ky=ky).sel(mode=0), label="tglf_STv1",linestyle="dashed")
        ax1.plot(growth_rate_tglf_M.beta, growth_rate_tglf_M.sel(ky=ky).sel(mode=0), label="tglf_STv2",linestyle="dashed")
        #ax1.plot(growth_rate_tglf_ML.beta, growth_rate_tglf_ML.sel(ky=ky).sel(mode=0), label="tglf_ML")
        ax1.plot(growth_rate_gs2.beta, growth_rate_gs2.sel(ky=ky), label="GS2")
        #ax1.plot(growth_rate_gs2.beta, growth_rate_gs2_gp.sel(ky=ky,output="value"), label="GS2_GP")


        ax2.plot(mode_frequency_tglf.beta, mode_frequency_tglf.sel(ky=ky).sel(mode=0), label="tglf_Defaults",linestyle="dashed")
        ax2.plot(mode_frequency_tglf_F.beta, mode_frequency_tglf_F.sel(ky=ky).sel(mode=0), label="tglf_STv1",linestyle="dashed")
        ax2.plot(mode_frequency_tglf_M.beta, mode_frequency_tglf_M.sel(ky=ky).sel(mode=0), label="tglf_STv2",linestyle="dashed")
        #ax2.plot(mode_frequency_tglf_ML.beta, mode_frequency_tglf_ML.sel(ky=ky).sel(mode=0), label="tglf_ML")
        ax2.plot(mode_frequency_gs2.beta, mode_frequency_gs2.sel(ky=ky), label="GS2")
        #ax2.plot(mode_frequency_gs2.beta, mode_frequency_gs2_gp.sel(ky=ky,output="value"), label="GS2_GP")

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
    gs = gridspec.GridSpec(len(growth_rate_tglf.beta), 2, hspace=0, wspace=0.3)
    # Only bottom row gets x-axis labels
    axes[-1, 0].set_xlabel(r"$\beta$")
    axes[-1, 1].set_xlabel(r"$\beta$")

    # Layout and title
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(r"Growth rate and Frequency against $\beta$ for different $k_y$ for STEP Case SPR-045",
                fontsize=16, y=0.95)
    

    # Collect all handles and labels from every axis
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

    # Deduplicate by label
    unique = dict(zip(labels, handles))

    # Create one legend for the whole figure
    fig.legend(
        unique.values(),
        unique.keys(),
        bbox_to_anchor=(1.5, 0.55),
        fontsize=12,        # ← increases text size
        markerscale=1.5,    # ← makes legend line/marker symbols bigger
    )



    # Save everything in ONE file
    plt.savefig(f"Beta_Scans/{STEP_CASE}_all_ky_pairs.png", dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    pyro_scan_tglf,pyro_scan_tglf_F,pyro_scan_tglf_M,pyro_scan_tglf_ML,pyro_scan_gs2 = Read_from_gs2()
    #print(pyro_scan_tglf.run_directories)
    #run_sim(pyro_scan_tglf)
    #run_sim(pyro_scan_tglf_F)
    #run_sim(pyro_scan_tglf_M)
    run_gs2_simulations_Local(pyro_scan_gs2)
    #run_sim(pyro_scan_tglf_ML)
    load_results(pyro_scan_tglf,pyro_scan_tglf_F,pyro_scan_tglf_M,pyro_scan_tglf_ML, pyro_scan_gs2)