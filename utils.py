import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import wandb

def layer_importance_plot(layer_influences: List[int], layer_indexes: List[str], save_path: str, block_size: int = 1):
    # --------------------------------------------------------------------------
    # 1) Validate input lists (ensuring they have the same length)
    # --------------------------------------------------------------------------
    if len(layer_influences) != len(layer_indexes):
        raise ValueError("layer_influences and layer_indexes must have the same length.")

    # --------------------------------------------------------------------------
    # 2) Create the plot
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Separate MHSA and MLP layers and assign different colors
    mhsa_influences = [layer_influences[i] for i in range(len(layer_influences)) if i % 2 == 0]  # Even indices for MHSA
    mlp_influences = [layer_influences[i] for i in range(len(layer_influences)) if i % 2 != 0]   # Odd indices for MLP
    mhsa_indexes = [i for i in range(len(layer_influences)) if i % 2 == 0]
    mlp_indexes = [i for i in range(len(layer_influences)) if i % 2 != 0]

    # Plot MHSA and MLP layers with different colors
    plt.scatter(mhsa_indexes, mhsa_influences, color='blue', marker='x', label="MHSA Layers")
    plt.scatter(mlp_indexes, mlp_influences, color='orange', marker='x', label="MLP Layers")

    # Find and highlight the minimum point
    min_index = min(range(len(layer_influences)), key=layer_influences.__getitem__)
    min_x = min_index
    min_y = layer_influences[min_index]
    plt.scatter(min_x, min_y, color='red', marker='x', s=80, label="Minimum", zorder=3)

    # --------------------------------------------------------------------------
    # 3) Set up axes, grid, and labels
    # --------------------------------------------------------------------------
    plt.title(f"Layer Importance Plot", fontsize=14)
    plt.xlabel("Cosine Distance", fontsize=12)
    plt.ylabel("Layer Influence", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    # Display fewer x-ticks to reduce clutter (e.g., show every 2nd or 3rd layer)
    step = 2  # Show every second layer index (you can adjust this if necessary)
    selected_ticks = range(0, len(layer_influences), step)
    selected_labels = [layer_indexes[i] for i in selected_ticks]
    plt.xticks(selected_ticks, selected_labels, rotation=45)

    # Optional: tighten layout so the labels/ticks fit nicely
    plt.tight_layout()

    # Add a legend
    plt.legend()

    # --------------------------------------------------------------------------
    # 4) Save the plot to the specified path
    # --------------------------------------------------------------------------
    save_filepath = os.path.join(save_path, 'layer_importance_plot.png')
    plt.savefig(save_filepath)

    # Log the plot to wandb
    wandb.log({"layer_importance_plot": wandb.Image(save_filepath)})

    # Close the plot to avoid display
    plt.close()
