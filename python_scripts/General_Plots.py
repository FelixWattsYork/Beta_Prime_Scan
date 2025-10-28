from typing import Dict, List, Any, Optional
from pyrokinetics import Pyro,PyroScan
import numpy as np
import torch
from pathlib import Path
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d/"



models = [
            "growth_rate_log", "mode_frequency_log",
        ]



def plot_2d(runs,names,plot_location, Gaussian=False):

        growth_rate_list = []
        mode_freq_list = []
        names = []

        for run in runs:
            run.load_gk_output()
            data = run.gk_output
            growth_rate_list.append(data["growth_rate"])
            mode_freq_list.append(data["mode_frequency"])
        
        if Gaussian:
            data_gs2_gp = gs2_gp(pyro=runs[-1], models_path=models_path, models=models)
            growth_rate_list.append(data_gs2_gp.gk_output["growth_rate_log_M12"])
            mode_freq_list.append(data_gs2_gp.gk_output["mode_frequency_log_M12"])
            names.append("GS2 GP Model")
        
        second_coord_name = list(growth_rate_list[0].coords)[1]


        n_rows = len(growth_rate_list[0].coords[second_coord_name])
        n_cols = 2
        aspect_ratio = 2.0  # width:height ratio per subplot
        width = aspect_ratio * n_cols * 3
        height = n_rows * 2.5
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(len(growth_rate_list[0].coords[second_coord_name]), 2, hspace=0, wspace=0.3)
        axes = np.empty((len(growth_rate_list[0].coords[second_coord_name]), 2), dtype=object)


        for i, second_coor in enumerate(growth_rate_list[0].coords[second_coord_name]):
            # Create subplots
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            axes[i, 0] = ax1
            axes[i, 1] = ax2

            # Plot data
            for growth_rate, mode_frequency, name in zip(growth_rate_list, mode_freq_list, names):
                ax1.plot(growth_rate.ky, growth_rate.sel({second_coord_name: second_coor}).sel(mode=0), label=rf"{name}")
                ax2.plot(mode_frequency.ky, mode_frequency.sel({second_coord_name: second_coor}).sel(mode=0), label=rf"{name}")

            ax1.grid(True)
            ax2.grid(True)

            # Row label on right-hand side
            ax2.text(
                1.05, 0.5,
                rf"{second_coord_name}={second_coor:.2f}",
                transform=ax2.transAxes,
                va='center', ha='left',
                fontsize=10
            )

        for i in range(len(growth_rate_list[0].coords[second_coord_name]) - 1):  # all rows except bottom
            axes[i, 0].set_xticklabels([])
            axes[i, 0].set_xlabel("") 
            axes[i, 1].set_xticklabels([])
            axes[i, 1].set_xlabel("")

        # Only bottom row gets x-axis labels
        axes[-1, 0].set_xlabel(r"$k_y$")
        axes[-1, 1].set_xlabel(r"$k_y$")


        # Layout and title
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(r"Plot")

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
        )

        plt.subplots_adjust(top=0.9, bottom=0.1)  # give room for legend and title


        # Save everything in ONE file
        print(f"Saving Figure at location: {plot_location}")
        plot_location.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_location, dpi=300)
        plt.close(fig)