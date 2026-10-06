## script to plot various results
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 14})
import seaborn as sns


def plot_conformal_coverage_from_csv(csv_path1, csv_path2=None, save_dir=None):
    # --- Data Loading ---
    if csv_path2 is not None:
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        df = pd.concat([df1, df2], ignore_index=True)
    else:    
        df = pd.read_csv(csv_path1)

    # Standardize coverage to decimal (0.9 instead of 90)
    # Adjust this logic if your CSVs are already in decimal format
    if df['coverage'].max() > 2:
        df['coverage'] = df['coverage'] / 100    

    # --- Experiment Parameters ---
    # Manually defined to match your specific research focus [cite: 5, 11]
    methods = ['MLE', 'PLS',
        'Neural Net', 'Neural Net Local', 
        'Quantile Neural Net (Non-Conformal)', 'Asymmetric CQR Neural Net'
    ]
    n_qubits_list = sorted(df['nQubits'].unique())#[6] 
    n_shots_list = sorted(df['nShots'].unique())#[50, 100, 500]

    method_colors = {
        'Neural Net': '#1f77b4',
        'Neural Net Local': '#6baed6',
        'Quantile Neural Net (Non-Conformal)': '#2ca02c',
        'Symmetric CQR Neural Net': '#ff7f0e',
        'Asymmetric CQR Neural Net': '#fdae6b',
        'MLE': '#9467bd',
        'CVX Local': '#c5b0d5',
        'PLS': '#d62728',
        'PLS Local': '#ff9896'
    }

    # -------------------------------
    # LOOP: one figure per nQubits
    # -------------------------------
    for nq in n_qubits_list:
        df_nq = df[df['nQubits'] == nq]

        # We remove the "+ 1" from the column count to give more space to data
        fig, axes = plt.subplots(
            1, len(n_shots_list),
            figsize=(5 * len(n_shots_list), 5), # Increased height slightly for legend
            sharey=True
        )

        axes = np.atleast_1d(axes)

        # -------------------------------
        # LOOP: subplots over nShots
        # -------------------------------
        for ax, n_shots in zip(axes, n_shots_list):
            df_shots = df_nq[df_nq['nShots'] == n_shots]
            ax.set_box_aspect(0.8)
            data = []
            colors = []

            for method in methods:
                vals = df_shots[df_shots['method'] == method]['coverage']
                data.append(vals.values)
                colors.append(method_colors.get(method, 'gray'))
    
            # --- Boxplot ---
            bp = ax.boxplot(
                data,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                medianprops=dict(color='black'),
                meanprops=dict(marker='o', markerfacecolor='yellow', markeredgecolor='black', markersize=3)
            )

            # Color boxes based on method
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            # Title and formatting
            ax.set_title(f"$N$ = {n_shots}", fontsize=14)
            
            # Using numbers 1-6 on x-axis to keep it clean, legend will explain them
            ax.set_xticks(np.arange(1, len(methods) + 1)) 
            
            # Target coverage line for Conformal Prediction [cite: 5, 11]
            ax.axhline(0.9, linestyle='dotted', color='black')
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        # ---- Global Figure Formatting ----
        fig.supylabel("Empirical Coverage", fontsize=14)
        fig.supxlabel("Method", fontsize=14, y=0.14) # Lifted slightly to avoid legend

        # ---- Centered Global Legend ----
        legend_patches = [mpatches.Patch(color=method_colors[m], label=m) for m in methods]
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=3, # Multi-column legend for better fit
            frameon=False,
            fontsize='small'
        )

        # Adjust layout to make room for labels and legend
        plt.tight_layout(rect=[0.02, 0.1, 1, 1])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"coverage_nQubits_{nq}.pdf")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved figure to {save_path}")
        
        plt.show()


def plot_conformal_length_from_csv(csv_path1, csv_path2=None, save_dir=None):
    # --- Data Loading ---
    if csv_path2 is not None:
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        df = pd.concat([df1, df2], ignore_index=True)
    else:    
        df = pd.read_csv(csv_path1)

    # --- Experiment Params ---
    # Standardizing methods and colors to match your existing research style
    methods = [#'MLE', 'PLS', 
               'Neural Net', 'Neural Net Local', 
        'Quantile Neural Net (Non-Conformal)', 'Asymmetric CQR Neural Net'
    ]
    n_qubits_list = sorted(df['nQubits'].unique())#[6] 
    n_shots_list = sorted(df['nShots'].unique()) # [50, 100, 500]
        
    method_colors = {
        'Neural Net': '#1f77b4',
        'Neural Net Local': '#6baed6',
        'Quantile Neural Net (Non-Conformal)': '#2ca02c',
        'Symmetric CQR Neural Net': '#ff7f0e',
        'Asymmetric CQR Neural Net': '#fdae6b',
        'MLE': '#9467bd',
        'CVX Local': '#c5b0d5',
        'PLS': '#d62728',
        'PLS Local': '#ff9896'
    }

    # -------------------------------
    # LOOP: one figure per nQubits
    # -------------------------------
    for nq in n_qubits_list:
        df_nq = df[df['nQubits'] == nq]

        # Use sharey=True to ensure all subplots have the same y-scale for comparison
        fig, axes = plt.subplots(
            1, len(n_shots_list), 
            figsize=(5 * len(n_shots_list), 5), 
            sharey=True
        )

        axes = np.atleast_1d(axes)

        # -------------------------------
        # LOOP: subplots over nShots
        # -------------------------------
        for ax, n_shots in zip(axes, n_shots_list):
            df_shots = df_nq[df_nq['nShots'] == n_shots]
            ax.set_box_aspect(0.8)
            data = []
            colors = []

            for method in methods:
                # Selecting the average interval length for the current method
                vals = df_shots[df_shots['method'] == method]['avg_length']
                data.append(vals.values)
                colors.append(method_colors.get(method, 'gray'))

            # --- Boxplot ---
            bp = ax.boxplot(
                data,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                medianprops=dict(color='black'),
                meanprops=dict(marker='o', markerfacecolor='yellow', markeredgecolor='black', markersize=3)
            )

            # Color assignment
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            ax.set_title(f"$N$ = {n_shots}", fontsize=14)
            ax.set_xticks(np.arange(1, len(methods) + 1))
            
            # Removes x-tick labels to prevent overlapping with the "Method" label below
            # ax.set_xticklabels([]) 
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        # ---- Global Figure Formatting ----
        fig.supylabel("Interval Length", fontsize=14)
        fig.supxlabel("Method", fontsize=14, y=0.14)

        # ---- Global Legend (Centered at Bottom) ----
        legend_patches = [mpatches.Patch(color=method_colors[m], label=m) for m in methods]
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
            fontsize='small'
        )

        # Adjust layout to prevent collisions between legend, labels, and plot area
        plt.tight_layout(rect=[0.02, 0.1, 1, 1])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"length_nQubits_{nq}.pdf")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved length figure to {save_path}")
   
        plt.show()  

if __name__ == "__main__":
    plot_conformal_coverage_from_csv("/Users/aditi/Documents/Fall_25/Quantum/UQ/main/results/conformal_NN/new/conformal_NN_all_n_2.csv",
                                    # "/Users/aditi/Documents/Fall_25/Quantum/UQ/main/results/conformal_classic/conformal_classic_all_methods.csv",
                                     save_dir="/Users/aditi/Documents/Fall_25/Quantum/UQ/main/results/figures/all_methods_marginal"
                                    )
    
    # plot_conformal_length_from_csv("/Users/aditi/Documents/Fall_25/Quantum/UQ/main/results/conformal_NN/new/conformal_NN_all_n_6.csv",
    #                             #    "/Users/aditi/Documents/Fall_25/Quantum/UQ/main/results/conformal_classic/conformal_classic_all_methods.csv",
    #                                save_dir="/Users/aditi/Documents/Fall_25/Quantum/UQ/main/results/figures/all_methods_marginal"
    #                                 )

    

