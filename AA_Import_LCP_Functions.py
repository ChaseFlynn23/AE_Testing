# Functions file

# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
import time
import glob
import re
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Local Compaction Plot Generation Functions
def write_cpptraj_script(name, parmtop, trajectory, protein_length, window, cpptraj_folder, lccdata_folder):
    cpptraj_file = os.path.join(cpptraj_folder, f"{name}_distance_{window}.cpptraj")
    lccdata_file = os.path.join(lccdata_folder, f"{name}.lccdata")
    with open(cpptraj_file, "w") as f:
        f.write("parm ")
        f.write(parmtop)
        f.write("\n")
        f.write("trajin ")
        f.write(trajectory)
        f.write("\n")
        upper_limit = protein_length + 1 - window
        for x in range(1, upper_limit):
            f.write(f"distance :{x} :{x + window} out {lccdata_file}\n")

def setup_folders(cpptraj_folder='cpptraj_files', lccdata_folder='lccdata_files'):
    cpptraj_exists = os.path.isdir(cpptraj_folder) and len(os.listdir(cpptraj_folder)) > 0
    lccdata_exists = os.path.isdir(lccdata_folder) and len(os.listdir(lccdata_folder)) > 0
    
    if cpptraj_exists and lccdata_exists:
        print("cpptraj files and lccdata_files already generated.")
        return cpptraj_folder, lccdata_folder, False
    elif cpptraj_exists:
        print("cpptraj files already generated.")
    elif lccdata_exists:
        print("lccdata files already generated.")
    
    if not cpptraj_exists:
        os.makedirs(cpptraj_folder, exist_ok=True)
    if not lccdata_exists:
        os.makedirs(lccdata_folder, exist_ok=True)
    
    return cpptraj_folder, lccdata_folder, True


def execute_cpptraj_scripts(prmtop_1, nc_1, prmtop_2, nc_2, protein_length, window_range, cpptraj_folder, lccdata_folder):
    cpptraj_folder, lccdata_folder, should_proceed = setup_folders(cpptraj_folder, lccdata_folder)
    if not should_proceed:
        return  # Stop execution if folders exist and have content
    
    for window in window_range:
        write_cpptraj_script("wildtype_" + str(window), prmtop_1, nc_1, protein_length, window, cpptraj_folder, lccdata_folder)
        write_cpptraj_script("myc_091-160_D132-H_" + str(window), prmtop_2, nc_2, protein_length, window, cpptraj_folder, lccdata_folder)


def execute_cpptraj_commands(cpptraj_folder, window_range):
    _, _, should_proceed = setup_folders(cpptraj_folder, 'lccdata_files')  
    if not should_proceed:
        return  # Stop execution if lccdata_files already generated and has content

    tasks = list(window_range) * 2  
    for window in tqdm(tasks, desc="Generating LCCData files"):
        wildtype_file = f"{cpptraj_folder}/wildtype_{window}_distance_{window}.cpptraj"
        myc_file = f"{cpptraj_folder}/myc_091-160_D132-H_{window}_distance_{window}.cpptraj"
        
        wildtype_command = f"cpptraj < {wildtype_file}"
        myc_command = f"cpptraj < {myc_file}"
        
        subprocess.run(wildtype_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(myc_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("LCCData files generated.")

#     
def import_lcc_data(lccdata_folder, prefix):
    """
    Imports LCC data files for a given prefix (wild type or mutant protein) and assigns to a dictionary.

    Parameters:
    - lccdata_folder: The folder where LCC data files are stored.
    - prefix: The prefix used to identify the files for either the wild type or mutant (e.g., 'w' for wild type).

    Returns:
    - A dictionary with window sizes as keys and pandas DataFrames as values.
    """
    files = glob.glob(f'{lccdata_folder}/{prefix}*.lccdata')
    files.sort(key=lambda x: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', x)])

    window_range = list(range(2, 52))
    data_dict = {}
    for window, file in zip(window_range, files):
        data_dict[window] = pd.DataFrame(np.loadtxt(file)).drop(columns=0)
    
    return data_dict

# Create local compaction plots
def LCC_plot_individual(window, wt, mutant, save_folder='Local_Compaction_Plot_Figures'):
    """
    Creates and saves an LCC plot for wt and mutant data for a specific window size.
    Checks if plots already exist to avoid regenerating them.
    """
    # Check if the save folder exists and has files for all window sizes
    if os.path.isdir(save_folder) and len(os.listdir(save_folder)) == 50:
        print("Local compaction plots already generated and saved in 'Local_Compaction_Plot_Figures'")
        return

    plt.figure(figsize=(15, 10))  # Create a new figure for each plot
    ax = plt.gca()  # Get current axis

    wt = wt.to_numpy()
    mutant = mutant.to_numpy()

    frame_number_wt = wt.shape[0]
    frame_number_mutant = mutant.shape[0]

    if frame_number_wt != frame_number_mutant:
        print('Different number of trajectory frames read in for mutant and WT')
        return

    upper_limit = 70 + 1 - window  # max protein length + 1

    for z in range(1, frame_number_wt, 10):
        y = wt[z]
        k = mutant[z]
        y_length = len(y)
        x = np.linspace(1 + window / 2 + 90, upper_limit + window / 2 + 90, y_length)

        ax.plot(x, y, color='blue', alpha=0.002)
        ax.plot(x, k, color='red', alpha=0.002)

    ax.set_xlabel('Amino Acid Sequence Position')
    ax.set_ylabel('Distance ($\AA$)')
    plot_name = 'Sequence Distance Distribution: Window Size ' + str(window) + ' aa'
    ax.set_title(plot_name)

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)
    # Save the plot
    save_path = os.path.join(save_folder, f'LCC_Plot_{window}.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent it from displaying in Jupyter
