from typing import Dict, List, Any, Optional
from pyrokinetics import Pyro, PyroScan
import numpy as np
import torch
from pathlib import Path
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d/"


models = [
    "growth_rate_log",
    "mode_frequency_log",
]


def plot_2d(
    runs,
    names,
    plot_location,
    Gaussian=False,
    parameter_1_range=None,
    parameter_2_range=None,
):
    growth_rate_list = []
    mode_freq_list = []

    for run in runs:
        print(run)
        print(names)
        run.load_gk_output()
        data = run.gk_output
        growth_rate_list.append(data["growth_rate"])
        mode_freq_list.append(data["mode_frequency"])

    if Gaussian:
        data_gs2_gp = gs2_gp(pyro=runs[-1], models_path=models_path, models=models)
        Gaussian_growth_rate = data_gs2_gp.gk_output["growth_rate_log_M12"]
        Gaussian_mode_frequency = data_gs2_gp.gk_output["mode_frequency_log_M12"]

    for i in range(0, 2):
        major_cord = list(growth_rate_list[0].coords)[i]
        minor_cord = list(growth_rate_list[0].coords)[1 - i]

        list_to_loop_over = growth_rate_list[0].coords[major_cord]
        if i == 0:
            if parameter_1_range != None:
                list_to_loop_over = growth_rate_list[0].coords[major_cord][
                    slice(*parameter_1_range)
                ]
        elif i == 1:
            if parameter_2_range != None:
                list_to_loop_over = growth_rate_list[0].coords[major_cord][
                    slice(*parameter_2_range)
                ]

        n_rows = len(list_to_loop_over)
        n_cols = 2
        aspect_ratio = 2.0  # width:height ratio per subplot
        width = aspect_ratio * n_cols * 3
        height = n_rows * 2.5
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(len(list_to_loop_over), 2, hspace=0, wspace=0.3)
        axes = np.empty((len(list_to_loop_over), 2), dtype=object)

        Plot_Title = f"Plot of Growth Rate and Frequency against {minor_cord} for fixed {major_cord}"

        for j, coor in enumerate(list_to_loop_over):
            # Create subplots
            ax1 = fig.add_subplot(gs[j, 0])
            ax2 = fig.add_subplot(gs[j, 1])
            axes[j, 0] = ax1
            axes[j, 1] = ax2

            # Plot data
            for growth_rate, mode_frequency, name in zip(
                growth_rate_list, mode_freq_list, names
            ):
                ax1.plot(
                    growth_rate[minor_cord],
                    growth_rate.sel({major_cord: coor}).sel(mode=0),
                    label=rf"{name}",
                )
                ax2.plot(
                    mode_frequency[minor_cord],
                    mode_frequency.sel({major_cord: coor}).sel(mode=0),
                    label=rf"{name}",
                )

            if Gaussian:
                # Extract data
                x1 = Gaussian_growth_rate[minor_cord]
                y1 = Gaussian_growth_rate.sel({major_cord: coor}).sel(output="value")
                y1_max = Gaussian_growth_rate.sel({major_cord: coor}).sel(
                    output="max_value"
                )
                y1_min = Gaussian_growth_rate.sel({major_cord: coor}).sel(
                    output="min_value"
                )

                x2 = Gaussian_mode_frequency[minor_cord]
                y2 = Gaussian_mode_frequency.sel({major_cord: coor}).sel(output="value")
                y2_max = Gaussian_mode_frequency.sel({major_cord: coor}).sel(
                    output="max_value"
                )
                y2_min = Gaussian_mode_frequency.sel({major_cord: coor}).sel(
                    output="min_value"
                )

                # Plot with shaded error region and dotted central line
                # --- Growth rate ---
                ax1.plot(
                    x1, y1, linestyle=":", color="green", label="GS2_GP"
                )  # dotted central line
                ax1.fill_between(
                    x1, y1_min, y1_max, color="green", alpha=0.2
                )  # light shading

                # --- Mode frequency ---
                ax2.plot(
                    x2, y2, linestyle=":", color="green", label="GS2_GP"
                )  # dotted central line
                ax2.fill_between(x2, y2_min, y2_max, color="green", alpha=0.2)

            ax1.grid(True)
            ax2.grid(True)

            # Row label on right-hand side
            ax2.text(
                1.05,
                0.5,
                rf"{major_cord}={coor:.2f}",
                transform=ax2.transAxes,
                va="center",
                ha="left",
                fontsize=10,
            )

        for i in range(
            len(growth_rate_list[0].coords[major_cord]) - 1
        ):  # all rows except bottom
            axes[i, 0].set_xticklabels([])
            axes[i, 0].set_xlabel("")
            axes[i, 1].set_xticklabels([])
            axes[i, 1].set_xlabel("")

        # Only bottom row gets x-axis labels
        axes[-1, 0].set_xlabel(minor_cord)
        axes[-1, 1].set_xlabel(minor_cord)

        # Layout and title
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(Plot_Title)

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
            loc="upper right",  # position above the subplots
            fontsize=16,
        )

        plt.subplots_adjust(top=0.9, bottom=0.1)  # give room for legend and title
        Plot_Name = f"fixed_{major_cord}_from_{list_to_loop_over[0].values}_to_{list_to_loop_over[-1].values}"
        if Gaussian:
            Plot_Name += "_with Gaussian"

        # Save everything in ONE file
        print(f"Saving Figure at location: {plot_location}")
        filename = plot_location / f"{Plot_Name}.png"
        plot_location.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close(fig)
