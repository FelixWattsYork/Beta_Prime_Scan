from pyrokinetics import Pyro, PyroScan, template_dir
import os
import pathlib
from typing import Union
import numpy as np
import copy
import subprocess
from pathlib import Path
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import run_simulations
import json
import General_Plots

REPO_ROOT = (
    Path(__file__).resolve().parent.parent
)  # repo root if this file sits at repo root

GYRO_DATA = Path(os.environ["GYRO_DATA_DIR"]).expanduser()

PROJECT_NAME = "Beta_Prime_Scan"


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d/"


models = [
    "growth_rate_log",
    "mode_frequency_log",
]


def Read_from_gs2(step_case):
    in_loc = GYRO_DATA / "GS2" / "Templates" / step_case / "gs2.in"
    base_out_loc = Path("Runs") / PROJECT_NAME / step_case
    pyro = Pyro(gk_file=in_loc, gk_code="GS2")

    pyro.numerics.nky = 1
    pyro.numerics.gamma_exb = 0.0
    pyro.local_species.electron.domega_drho = 0.0

    # Use existing parameter with more realistic ky range
    param_1 = "ky"
    values_1 = np.arange(0.1, 0.4, 0.1) / pyro.norms.pyrokinetics.rhoref

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
        parameter_key="beta", parameter_attr="numerics", parameter_location=["beta"]
    )

    # Add function to gs2
    pyro_scan_gs2.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    print(GYRO_DATA / "GS2" / base_out_loc / "parameter_scan_gs2")

    # Create scan directory and write input files
    pyro_scan_gs2.write(
        file_name="gs2.in",
        base_directory=GYRO_DATA / "GS2" / base_out_loc / "parameter_scan_gs2",
        template_file=None,
    )

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
        parameter_key="beta", parameter_attr="numerics", parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    print(GYRO_DATA / "TGLF" / base_out_loc / "parameter_scan_tglf")
    # Create scan directory and write input files
    pyro_scan_tglf.write(
        file_name="input.tglf",
        base_directory=GYRO_DATA / "TGLF" / base_out_loc / "parameter_scan_tglf",
        template_file=None,
    )

    pyro_copy_2 = copy.copy(pyro)

    # Switch to TGLF
    pyro_copy_2.gk_code = "TGLF"

    dict = {
        "WIDTH": 1.9,
        "WIDTH_MIN": 0.495,
        "FILTER": 2,
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
        parameter_key="beta", parameter_attr="numerics", parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf_F.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    pyro_scan_tglf_F.write(
        file_name="input.tglf",
        base_directory=GYRO_DATA / "TGLF" / base_out_loc / "parameter_scan_tglf_F",
        template_file=None,
    )

    pyro_copy_3 = copy.copy(pyro)
    pyro_copy_3.gk_code = "TGLF"

    dict = {
        "WIDTH": 1,
        "FILTER": 2,
        "THETA_TRAPPED": 0.57,
        "FIND_WIDTH": "F",
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
        parameter_key="beta", parameter_attr="numerics", parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf_M.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    pyro_scan_tglf_M.write(
        file_name="input.tglf",
        base_directory=GYRO_DATA / "TGLF" / base_out_loc / "parameter_scan_tglf_M",
        template_file=None,
    )

    pyro_copy_4 = copy.copy(pyro)
    pyro_copy_4.gk_code = "TGLF"

    dict = {
        "WIDTH": 1,
        "FILTER": 2,
        "THETA_TRAPPED": 0.57,
        "FIND_WIDTH": "F",
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
        parameter_key="beta", parameter_attr="numerics", parameter_location=["beta"]
    )

    # Add function to tglf
    pyro_scan_tglf_ML.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

    # Create scan directory and write input files
    pyro_scan_tglf_ML.write(
        file_name="input.tglf",
        base_directory=GYRO_DATA / "TGLF" / base_out_loc / "parameter_scan_tglf_ML",
        template_file=None,
    )

    return pyro_scan_gs2


def load_gs2_pyroscan(step_case, project, name="gs2"):
    json_path = (
        GYRO_DATA
        / "GS2"
        / "Runs"
        / project
        / step_case
        / f"parameter_scan_{name}"
        / "pyroscan.json"
    )
    in_loc = GYRO_DATA / "GS2" / "Templates" / step_case / "gs2.in"
    pyro_object = Pyro(gk_file=in_loc, gk_code="GS2")
    return PyroScan(pyro=pyro_object, pyroscan_json=json_path)


def load_tglf_pyroscan(step_case, project, name="tglf"):
    json_path = (
        GYRO_DATA
        / "TGLF"
        / "Runs"
        / project
        / step_case
        / f"parameter_scan_{name}"
        / "pyroscan.json"
    )
    in_loc = GYRO_DATA / "GS2" / "Templates" / step_case / "gs2.in"
    pyro_object = Pyro(gk_file=in_loc, gk_code="GS2")
    pyro_object.gk_code = "TGLF"
    return PyroScan(pyro=pyro_object, pyroscan_json=json_path)


if __name__ == "__main__":
    step_case = "n40"

    # Generate the input files
    # Read_from_gs2(step_case)

    gs2_scan_names = ["gs2"]
    tglf_scan_names = ["tglf_F", "tglf_M"]
    gs2_scans = []
    tglf_scans = []
    for name in gs2_scan_names:
        gs2_scans.append(load_gs2_pyroscan(step_case, PROJECT_NAME, name=name))
    for name in tglf_scan_names:
        tglf_scans.append(load_tglf_pyroscan(step_case, PROJECT_NAME, name=name))

    # run simulations
    # for scan in gs2_scans:
    #    run_simulations.gs2_scan(scan)

    plot_location = REPO_ROOT / "Plots" / PROJECT_NAME / step_case

    all_scans = gs2_scans + tglf_scans
    scan_names = gs2_scan_names + tglf_scan_names

    General_Plots.plot_2d(
        all_scans, scan_names, plot_location, Gaussian=True, parameter_1_range=(1, 4)
    )
