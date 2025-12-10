import os
from pathlib import Path
from pyrokinetics.pyroscan import PyroScanGKOutput
from pyrokinetics import Pyro, PyroScan
import sys
import argparse
import difference_metric
import python_scripts.general_plots as general_plots



from python_scripts.run_simulations import tglf_scan

REPO_ROOT = (
    Path(__file__).resolve().parent.parent
)  # repo root if this file sits at repo root


GYRO_DATA = Path(os.environ["GYRO_DATA_DIR"]).expanduser()


args = sys.argv[1:]
work_folder = args[0] if args else "."

parser = argparse.ArgumentParser(description="Case: Which tokamak case to read")
parser.add_argument("case", help="a case to process")
parser.add_argument("project_name", help="the type of scan being performed")


parser.add_argument(
    "-l",
    "--load",
    action="store_true",  # Makes this a boolean flag
    help="Loades the GK Output either from the raw files or from netcdf",
)
parser.add_argument(
    "-rc",
    "--read_netcdf",
    action="store_true",  # Makes this a boolean flag
    help="Reads the GK Output from the netcdf",
)
parser.add_argument(
    "-p",
    "--plot",
    action="store_true",  # Makes this a boolean flag
    help="Plots the results",
)
parser.add_argument(
    "-c",
    "--calculate_difference_metric",
    action="store_true",  # Makes this a boolean flag
    help="Calculates the differences between two scans",
)
parser.add_argument(
    "-wc",
    "--write_netcdf",
    action="store_true",  # Makes this a boolean flag
    help="writes netcdf of GK ouptut",
)
parser.add_argument(
    "-g",
    "--include_Gaussian",
    action="store_true",  # Makes this a boolean flag
    help="Include to add William's Gaussian Regresion model to plots",
)
args = parser.parse_args()


models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d_Up4/"  # This is using the 3000 data point model as opposed to the 100000 data point model for testing purposes


models = [
    "growth_rate_log",
    "mode_frequency_log",
    "kperp2_phi_log",
    "kperp2_apa_log",
    "kperp2_bpar_log",
    "totIonFlux_log",
    "totElecFlux_log",
    "totPartFlux",
    "apa_phi_log",
    "bpar_phi_log",
    "sigmas_log",
]


def load_GS2_pyroscan(step_case, project, name="gs2"):
    run_folder_loc = (
        GYRO_DATA / "GS2" / "Runs" / project / step_case / f"parameter_scan_{name}"
    )
    folder_in_run_folder = [p for p in run_folder_loc.iterdir() if p.is_dir()]
    json_path = run_folder_loc / "pyroscan.json"
    in_loc = folder_in_run_folder[0] / "gs2.in"
    print("Loading GS2 PyroScan from:", in_loc)
    pyro_object = Pyro(gk_file=in_loc, gk_code="GS2")
    GS2_convention = pyro_object.norms.pyrokinetics
    return PyroScan(pyro=pyro_object, pyroscan_json=json_path), GS2_convention


def load_tglf_pyroscan(
    step_case, project, name="tglf"
):  # this doesn't work for some reason??
    run_folder_loc = (
        GYRO_DATA / "TGLF" / "Runs" / project / step_case / f"parameter_scan_{name}"
    )
    folder_in_run_folder = [p for p in run_folder_loc.iterdir() if p.is_dir()]
    json_path = run_folder_loc / "pyroscan.json"
    in_loc = folder_in_run_folder[0] / "input.tglf"
    pyro_object = Pyro(gk_file=in_loc, gk_code="TGLF")
    return PyroScan(pyro=pyro_object, pyroscan_json=json_path)


def load_tglf_pyroscan_gs2(step_case, project, name="tglf"):
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
    tglf_list = ["tglf", "tglf_F", "tglf_M"]

    if args.load:
        gs2_pyroscan, GS2_convention = load_GS2_pyroscan(args.case, args.project_name)
        tglf_pyroscan_dict = {
            tglf: load_tglf_pyroscan_gs2(args.case, args.project_name, name=f"{tglf}")
            for tglf in tglf_list
        }

        if args.write_netcdf:
            gs2_pyroscan.load_gk_output()
            gs2_pyroscan.gk_output.to_netcdf(
                GYRO_DATA
                / "GS2"
                / "Runs"
                / args.project_name
                / args.case
                / "parameter_scan_gs2"
                / "gs2.cdf"
            )
            for tglf_name, tglf_pyroscan in tglf_pyroscan_dict.items():
                tglf_pyroscan.load_gk_output()
                tglf_pyroscan.gk_output.to_netcdf(
                    GYRO_DATA
                    / "TGLF"
                    / "Runs"
                    / args.project_name
                    / args.case
                    / f"parameter_scan_{tglf_name}"
                    / f"{tglf_name}.cdf"
                )
        if args.read_netcdf:
            gs2_pyroscan.gk_output = PyroScanGKOutput.from_netcdf(
                GYRO_DATA
                / "GS2"
                / "Runs"
                / args.project_name
                / args.case
                / "parameter_scan_gs2"
                / "gs2.cdf"
            )
            for tglf_name, tglf_pyroscan in tglf_pyroscan_dict.items():
                tglf_pyroscan.gk_output = PyroScanGKOutput.from_netcdf(
                    GYRO_DATA
                    / "TGLF"
                    / "Runs"
                    / args.project_name
                    / args.case
                    / f"parameter_scan_{tglf_name}"
                    / f"{tglf_name}.cdf"
                )
                tglf_pyroscan.gk_output.to(GS2_convention)
                # Convert all tglf ouptuts to the GS2 normalisation
    if args.include_Gaussian:
        from pyrokinetics.diagnostics.gs2_gp import gs2_gp
        Gaussian_Model = gs2_gp(pyro=gs2_pyroscan, models_path=models_path, models=models)
        Gaussian_Model.evaluate_nonlinear_flux()
    # gs2rate GP
    if args.plot:
        tglf_gk_outputs = []
        tglf_scan_names = []
        for tglf_name, tglf_pyroscan in tglf_pyroscan_dict.items():
            tglf_gk_outputs.append(tglf_pyroscan.gk_output)
            tglf_pyroscan.gk_output.to(GS2_convention)
            tglf_scan_names.append(tglf_name)

        plot_location = REPO_ROOT / "Plots" / args.project_name / args.case
        if args.include_Gaussian:

            general_plots.Ground_Truth_2d(
                gs2_pyroscan.gk_output,
                tglf_gk_outputs,
                tglf_scan_names,
                plot_location,
                Gaussian=True,
                Gaussian_Model=Gaussian_Model.gk_output,
                parameter_1_range=(1, 4),
            )
        else:
            general_plots.Ground_Truth_2d(
                gs2_pyroscan.gk_output,
                tglf_gk_outputs,
                tglf_scan_names,
                plot_location,
                parameter_1_range=(1, 4),
            )
    if args.calculate_difference_metric:
        print("Calculating Difference Metrics:")
        for tglf_name, tglf_pyroscan in tglf_pyroscan_dict.items():
            print(f"Difference Metric for {tglf_name}:")
            dm_basic = difference_metric.basic(
                ground_truth_gr=gs2_pyroscan.gk_output["growth_rate"],
                ground_truth_freq=gs2_pyroscan.gk_output["mode_frequency"],
                alternative_gr=tglf_pyroscan.gk_output["growth_rate"].sel(mode=0),
                alternative_freq=tglf_pyroscan.gk_output["mode_frequency"].sel(mode=0),
            )
            print(f"  Basic: {dm_basic}")
            dm_stabalized = difference_metric.stabalized(
                ground_truth_gr=gs2_pyroscan.gk_output["growth_rate"],
                ground_truth_freq=gs2_pyroscan.gk_output["mode_frequency"],
                alternative_gr=tglf_pyroscan.gk_output["growth_rate"].sel(mode=0),
                alternative_freq=tglf_pyroscan.gk_output["mode_frequency"].sel(mode=0),
            )
            print(f" Stabalised: {dm_stabalized}")
        if args.include_Gaussian:
            print(f"Difference Metric for GP:")
            dm_basic = difference_metric.basic(
                ground_truth_gr=gs2_pyroscan.gk_output["growth_rate"],
                ground_truth_freq=gs2_pyroscan.gk_output["mode_frequency"],
                alternative_gr=Gaussian_Model.gk_output["growth_rate_log"].sel(output="value"),
                alternative_freq=Gaussian_Model.gk_output["mode_frequency_log"].sel(output="value"),
            )
            print(f"  Basic: {dm_basic}")
            dm_stabalized = difference_metric.stabalized(
                ground_truth_gr=gs2_pyroscan.gk_output["growth_rate"],
                ground_truth_freq=gs2_pyroscan.gk_output["mode_frequency"],
                alternative_gr=Gaussian_Model.gk_output["growth_rate_log"].sel(output="value"),
                alternative_freq=Gaussian_Model.gk_output["mode_frequency_log"].sel(output="value"),
            )
            print(f" Stabalised: {dm_stabalized}")
