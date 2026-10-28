import tkinter as tk
from tkinter import ttk, filedialog, messagebox, PhotoImage 
import tkinter.font as tkFont
import ttkbootstrap as ttk 
from ttkbootstrap.constants import *
from pathlib import Path
import os
import tempfile
import threading
import queue
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from tkinter.simpledialog import askstring
import patsy
import difflib
import multiprocessing
import time 
import re
import mne_connectivity
import config as cfg
import networkx as nx
import graph_metrics as gm
from typing import Dict, List, Sequence, Optional, Tuple 
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import numpy as np, pandas as pd, patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False



import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from fooof import FOOOF
from fooof.bands import Bands

def preprocessing_worker_process(args):
    """
    Run the preprocessing and return the plain data (arrays and info)
    through the queue, so as to avoid pickling and disk problems.
    """
    # 1. Read the arguments
    uploaded_files = args["uploaded_files"]
    low_freq = args["low_freq"]
    high_freq = args["high_freq"]
    notch_freq = args["notch_freq"]
    progress_queue = args["progress_queue"]

    print("\n" + "="*20 + " PREPROCESSING STARTED (IN MEMORY) " + "="*20)
    
    total_subjects = len(uploaded_files)
    if total_subjects == 0:
        progress_queue.put(("status", "No subjects selected."))
        progress_queue.put(("percentage", 100))
        progress_queue.put(("done", [])) # Send an empty list
        return

    # 2. This will hold a list of data packets
    preprocessed_data_packages = []
    failed_subjects = []

    for i, file_path in enumerate(uploaded_files):
        subject_id = os.path.basename(file_path).replace(".vhdr", "")
        
        percentage = int(((i + 1) / total_subjects) * 100)
        status_text = f"Preprocessing {subject_id} ({i + 1}/{total_subjects})..."
        
        progress_queue.put(("status", status_text))
        progress_queue.put(("percentage", percentage))

        try:
            raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
            
            ch_types = {}
            non_eeg_channels = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']
            for ch_name in raw.ch_names:
                ch_types[ch_name] = 'misc' if ch_name in non_eeg_channels else 'eeg'
            raw.set_channel_types(ch_types, verbose='warning')

            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')

            raw.filter(low_freq, high_freq, verbose=False)
            raw.notch_filter(notch_freq, verbose=False)

            # 3. Build a packet with the plain, picklable data
            package = {
                'subject_id': subject_id,
                'data': raw.get_data(),  # Array NumPy
                'info': raw.info         # MNE Info dictionary
            }
            preprocessed_data_packages.append(package)

        except Exception as e:
            print(f"!!! ERROR while preprocessing {subject_id}: {e}")
            failed_subjects.append(subject_id)

    # 4. Send the list of packets to the main process
    progress_queue.put(("status", "Preprocessing completed."))
    progress_queue.put(("percentage", 100))
    progress_queue.put(("done", preprocessed_data_packages))
    progress_queue.put(("failed", failed_subjects))


def ica_plot_worker(ica_dict, raw_dict, plot_type, picks=None):
    """
    Function run in a separate process to handle every MNE-ICA plot.
    """
    try:
        # Rebuild the MNE objects from the temporary file paths
        ica = mne.preprocessing.read_ica(ica_dict)
        raw = mne.io.read_raw_fif(raw_dict, preload=True)
        
        print(f"--- Child process: running plot_type='{plot_type}' ---")

        if plot_type == 'components':
            # This shows the component grid in one or more external windows
            ica.plot_components(inst=raw, show=True)
        
        elif plot_type == 'properties':
            # This shows the detail window for a single component
            ica.plot_properties(raw, picks=picks, show=True)
            
        elif plot_type == 'excluded':
            # This shows only the excluded components
            ica.plot_components(inst=raw, picks=picks, show=True)

        # plt.show() is essential to keep the windows open
        plt.show()

    except Exception as e:
        print(f"!!! ERROR in the ICA plotting worker process: {e}")
    finally:
        # Clean up the temporary files
        os.remove(ica_dict)
        os.remove(raw_dict)

def connectivity_worker_process(args):
    """
    Connectivity worker that processes ONE SINGLE SUBJECT.
    It computes the matrices and the features and returns a complete dictionary.
    """
    subject_id = args['subject_id']
    raw_path = args['raw_path']
    methods_to_run = args['methods_to_run']
    freq_bands = args['freq_bands']
    
    # This function has to be self-contained, hence it is defined here
    def _extract_connectivity_features(matrix, ch_names_eeg):
        out_keys = ["global_mean", "dlpfc_l_mean", "dlpfc_r_mean", "dlpfc_lr_mean", 
                    "mst_leaf_fraction", "mst_diameter", "mst_mean_eccentricity", "mst_mean_weight"]
        features = {k: np.nan for k in out_keys}
        if matrix is None or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(ch_names_eeg):
            return features
        
        A = 0.5 * (np.abs(matrix) + np.abs(matrix).T)
        np.fill_diagonal(A, 0)
        
        if A.shape[0] > 1:
            r, c = np.triu_indices(A.shape[0], k=1)
            features["global_mean"] = float(np.nanmean(A[r, c]))

        roi_defs = {"dlpfc_l": ["F3", "F7", "Fp1"], "dlpfc_r": ["F4", "F8", "Fp2"]}
        idx_map = {ch: i for i, ch in enumerate(ch_names_eeg)}
        
        l_idx = [idx_map[ch] for ch in roi_defs["dlpfc_l"] if ch in idx_map]
        r_idx = [idx_map[ch] for ch in roi_defs["dlpfc_r"] if ch in idx_map]

        if len(l_idx) >= 2: features["dlpfc_l_mean"] = float(np.nanmean(A[np.ix_(l_idx, l_idx)][np.triu_indices(len(l_idx), k=1)]))
        if len(r_idx) >= 2: features["dlpfc_r_mean"] = float(np.nanmean(A[np.ix_(r_idx, r_idx)][np.triu_indices(len(r_idx), k=1)]))
        if l_idx and r_idx: features["dlpfc_lr_mean"] = float(np.nanmean(A[np.ix_(l_idx, r_idx)]))
        
        try:
            D = 1.0 - A
            G = nx.from_numpy_array(D)
            mst = nx.minimum_spanning_tree(G, weight='weight')
            if mst.number_of_nodes() > 0:
                degrees = np.array([d for _, d in mst.degree()])
                features["mst_leaf_fraction"] = float(np.sum(degrees == 1) / mst.number_of_nodes())
                ecc = nx.eccentricity(mst)
                features["mst_diameter"] = float(nx.diameter(mst, e=ecc))
                features["mst_mean_eccentricity"] = float(np.mean(list(ecc.values())))
                if mst.number_of_edges() > 0:
                    weights = np.array([d["weight"] for _, _, d in mst.edges(data=True)])
                    features["mst_mean_weight"] = float(np.mean(weights))
        except Exception:
            pass
        return features

    try:
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        cond_name = "EC" if 'EC' in subject_id.upper() else "EO" if 'EO' in subject_id.upper() else "unknown"
        
        epochs = mne.make_fixed_length_epochs(raw, duration=5.0, overlap=2.5, preload=True, verbose=False)
        epochs.pick_types(eeg=True, verbose=False)
        epochs = mne.preprocessing.compute_current_source_density(epochs, verbose=False)
        
        ch_names_for_features = epochs.ch_names
        subject_results = {'matrices': {}, 'features': {}}

        for band_name, (fmin, fmax) in freq_bands.items():
            con_results_band = mne_connectivity.spectral_connectivity_epochs(
                epochs, method=methods_to_run, mode='multitaper', sfreq=epochs.info['sfreq'],
                fmin=fmin, fmax=fmax, faverage=True, n_jobs=1, verbose=False
            )
            
            for con_result, method_name in zip(con_results_band, methods_to_run):
                matrix = con_result.get_data(output='dense').squeeze()
                matrix_key = f"{method_name}_{band_name}"
                subject_results['matrices'][matrix_key] = matrix
                
                features = _extract_connectivity_features(matrix, ch_names_for_features)
                for feat_name, feat_val in features.items():
                    feature_key = f"{method_name}_{band_name}_{feat_name}_{cond_name}"
                    subject_results['features'][feature_key] = feat_val
        
        return subject_id, subject_results

    except Exception as e:
        print(f"!!! ERROR in the connectivity worker for {subject_id}: {e}")
        return subject_id, None

class EEGAnalysisGUI:
    # --- method correctly indented inside the class ---
    def get_mean_exponent(self, raw, ch_list):
        from fooof import FOOOF
        import numpy as np
        exponents = []
        for ch in ch_list:
            if ch in raw.ch_names:
                psd = raw.copy().pick_channels([ch]).compute_psd(method="welch", fmin=1, fmax=50, n_fft=512)
                psd_ch = psd.get_data()[0]
                freqs_ch = psd.freqs
                min_len = min(len(freqs_ch), len(psd_ch))
                fm_ch = FOOOF(peak_width_limits=[1.5, 8])
                fm_ch.fit(freqs_ch[:min_len], psd_ch[:min_len])
                exponents.append(fm_ch.get_params('aperiodic_params', 'exponent'))
        if len(exponents) == 0:
            return np.nan
        return np.mean(exponents)

    def _process_progress_queue(self):
        """
        Check the progress queue and refresh the GUI.
        Updated to rebuild the Raw objects from the plain data received.
        """
        try:
            while not self.progress_queue.empty():
                msg_type, data = self.progress_queue.get_nowait()
                
                if msg_type == "percentage":
                    self.preprocessing_progress_percent.set(f"{data}%")
                elif msg_type == "status":
                    self.status_var.set(data)
                elif msg_type == "done":
                    self.preprocessing_progress.stop()
                    
                    # --- key change ---
                    # 'data' is now a list of packets [{subject_id, data, info}, ...]
                    data_packages = data
                    self.preprocessed_data = {}  # Reset the container
                    
                    # Rebuild each Raw object from the data received
                    for package in data_packages:
                        subject_id = package['subject_id']
                        try:
                            # mne.io.RawArray builds a Raw object from the plain data
                            self.preprocessed_data[subject_id] = mne.io.RawArray(
                                package['data'], package['info'], verbose=False
                            )
                        except Exception as e:
                            print(f"ERROR: could not rebuild the data for {subject_id}: {e}")
                    
                    print("[GUI] Preprocessed data rebuilt in memory.")
                    # --- end of the change ---

                    subject_list = list(self.preprocessed_data.keys())
                    self.subject_dropdown.configure(values=subject_list)
                    if subject_list:
                        self.selected_subject.set(subject_list[0])
                    
                    # Refresh the other dropdowns that depend on this data as well
                    self.ica_subject_dropdown.configure(values=subject_list)
                    if subject_list:
                        self.selected_ica_subject.set(subject_list[0])

                    return
                elif msg_type == "failed":
                    if data:
                        messagebox.showwarning("Preprocessing Warnings", 
                                            f"Failed to preprocess {len(data)} subject(s):\n" +
                                            "\n".join(data))

        except queue.Empty:
            pass
        
        self.root.after(100, self._process_progress_queue)
    
    def setup_main_window(self):
        """Set up the main window with its title, size and icon."""
        self.root.title("EEG Analysis Pipeline")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=self.colors['background'])
        
        try:
            # Build the right path to the icon file
            # __file__ refers to the .py file being run
            script_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(script_dir, 'icon.png')

            # 1. Set the Tkinter icon (good practice, works on Windows and Linux)
            icon_image = tk.PhotoImage(file=icon_path)
            self.root.iconphoto(False, icon_image)
            
            # 2. --- macOS-specific workaround ---
            # This block forces the icon to show up properly in the Dock
            # and resolves the conflicts with Matplotlib.
            if 'darwin' in self.root.tk.call('tk', 'windowingsystem'):
                # Import the native macOS libraries only when running on macOS
                from AppKit import NSApp, NSImage
                
                # Build a native macOS image from the file path
                ns_image = NSImage.alloc().initWithContentsOfFile_(icon_path)
                
                # Set the application icon at the operating-system level
                NSApp.setApplicationIconImage_(ns_image)
            # --- end of the workaround ---
                
        except Exception as e:
            print(f"Warning: could not load the icon from '{icon_path}'. Error: {e}")

        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def setup_styles(self):
        """Configure ttk styles for modern appearance"""
        self.style = ttk.Style()
        # try:
        #     self.style.theme_use('clam')
        # except tk.TclError:
        #     pass

        # self.style.configure('Custom.TNotebook', background=self.colors['background'], borderwidth=0)
        # self.style.configure('Custom.TNotebook.Tab', background=self.colors['surface'], foreground=self.colors['text_primary'], padding=[20, 10], font=('Arial', 10, 'normal'))
        # self.style.map('Custom.TNotebook.Tab', background=[('selected', self.colors['primary']), ('active', self.colors['accent'])], foreground=[('selected', 'white'), ('active', 'white')])
        # self.style.configure('Card.TFrame', background=self.colors['surface'], relief='flat', borderwidth=1)
        # self.style.configure('Primary.TButton', background=self.colors['primary'], foreground='white', font=('Arial', 10, 'bold'), padding=[20, 10], relief='flat', borderwidth=1)
        # self.style.map('Primary.TButton', background=[('active', self.colors['secondary']), ('pressed', self.colors['text_primary'])], foreground=[('active', 'white'), ('pressed', 'white')])
        # self.style.configure('Secondary.TButton', background=self.colors['accent'], foreground='white', font=('Arial', 9, 'normal'), padding=[15, 8], relief='flat', borderwidth=1)
        # self.style.map('Secondary.TButton', background=[('active', self.colors['success']), ('pressed', self.colors['text_primary'])], foreground=[('active', 'white'), ('pressed', 'white')])
        # self.style.configure('Action.TButton', background=self.colors['success'], foreground='white', font=('Arial', 10, 'bold'), padding=[15, 8], relief='raised', borderwidth=2)
        # self.style.map('Action.TButton', background=[('active', '#A52A2A'), ('pressed', '#8B0000')], foreground=[('active', 'white'), ('pressed', 'white')])
        # self.style.configure('Title.TLabel', background=self.colors['background'], foreground=self.colors['text_primary'], font=('Arial', 16, 'bold'))
        # self.style.configure('Heading.TLabel', background=self.colors['surface'], foreground=self.colors['text_primary'], font=('Arial', 12, 'bold'))
        # self.style.configure('Body.TLabel', background=self.colors['surface'], foreground=self.colors['text_secondary'], font=('Arial', 9, 'normal'))

        # --- new styles added ---
        self.style.configure('Card.TLabelframe', 
                            background=self.colors['surface'], 
                            borderwidth=1, 
                            relief='solid')

        self.style.configure('Card.TLabelframe.Label', 
                            background=self.colors['surface'], 
                            foreground=self.colors['text_primary'],
                            font=('Arial', 11, 'bold'))

        self.style.configure('Param.TLabel', 
                            background=self.colors['surface'], 
                            foreground=self.colors['text_secondary'],
                            font=('Arial', 9, 'normal'))
            
    def setup_variables(self):
        """Initialize application variables"""
        self.param_widgets = {}
        self.uploaded_files, self.subject_ids = [], []
        self.all_uploaded_files = [] 
        self.analysis_mode = tk.StringVar(value="Single Subject")
        self.low_freq, self.high_freq, self.notch_freq = tk.DoubleVar(value=0.5), tk.DoubleVar(value=80.0), tk.DoubleVar(value=50.0)
        self.n_components, self.freq_range_min, self.freq_range_max = tk.IntVar(value=25), tk.IntVar(value=1), tk.IntVar(value=50)
        self.min_peak_height = tk.DoubleVar(value=0.1)
        self.ica_maps_displayed = None  # Track currently shown subject's ICA maps
        self.ica_method, self.aperiodic_mode = tk.StringVar(value="infomax"), tk.StringVar(value="fixed")
        self.raw_data, self.preprocessed_data, self.ica_results, self.spectral_results, self.fooof_results, self.biomarkers_data = {}, {}, {}, {}, {}, {}
        self.connectivity_results = {}
        self.behavioral_data = pd.DataFrame() 
        self.processing_queue, self.is_processing = queue.Queue(), False
        self.excluded_components = {}
        self.preprocessing_progress_percent = tk.StringVar(value="0%")
        self.ica_progress_percent = tk.StringVar(value="0%")
        self.spectral_progress_percent = tk.StringVar(value="0%")
        self.fooof_progress_percent = tk.StringVar(value="0%")
        self.regression_progress_percent = tk.StringVar(value="0%")
        # Variables of the Graph Theory tab
        self.gt_mode_var = tk.StringVar(value="explore")
        self.gt_viz_label_var = tk.StringVar(value="Display")
        
        # Variables of the controls, both batch and explore
        self.gt_subject_var = tk.StringVar()
        self.gt_matrix_key_var = tk.StringVar()
        self.gt_batch_metric_var = tk.StringVar(value='dpli')
        self.gt_batch_band_var = tk.StringVar(value='alpha')
        
        self.gt_neg_var = tk.StringVar(value='abs')
        self.gt_symmetrize_var = tk.StringVar(value='average')
        self.gt_directed_var = tk.BooleanVar(value=False)
        
        self.gt_thresh_method_var = tk.StringVar(value='proportional')
        self.gt_threshold_var = tk.IntVar(value=20) # Used by both the batch and the proportional mode
        self.gt_alpha_var = tk.DoubleVar(value=0.05) # Per backbone
        
        self.gt_seed_var = tk.IntVar(value=42)
        
        self.gt_progress_percent = tk.StringVar(value="0%")


    def create_condition_filter_tab(self):
        """Creates a more intelligent tab to filter files by condition (EC/EO)."""
        filter_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(filter_frame, text="Condition Filter (EC/EO)")

        # --- Top control panel ---
        control_panel = ttk.Frame(filter_frame, style='Card.TFrame')
        control_panel.pack(fill='x', pady=(0, 20))
        
        ttk.Label(control_panel, text="Filter Files by Condition", style='Title.TLabel').pack(side='left', anchor='w')
        
        button_frame = ttk.Frame(control_panel, style='Card.TFrame')
        button_frame.pack(side='right', anchor='e')
        
        ttk.Button(button_frame, text="Select EC Only", style='Primary.TButton', 
                   command=lambda: self.apply_condition_filter("EC")).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Select EO Only", style='Primary.TButton', 
                   command=lambda: self.apply_condition_filter("EO")).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Use All Conditions", style='Secondary.TButton', 
                   command=lambda: self.apply_condition_filter(None)).pack(side='left', padx=5)

        # --- Main two-column layout ---
        main_layout = ttk.Frame(filter_frame, style='Card.TFrame')
        main_layout.pack(fill='both', expand=True)
        main_layout.columnconfigure(0, weight=1)
        main_layout.columnconfigure(1, weight=1)
        main_layout.rowconfigure(0, weight=1)

        # --- Left Column: Available Files for Selected Subjects ---
        left_panel = ttk.Frame(main_layout, style='Card.TFrame')
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        self.available_files_count_var = tk.StringVar(value="Available Files for Selected Subjects (0)")
        ttk.Label(left_panel, textvariable=self.available_files_count_var, style='Heading.TLabel').pack(anchor='w', pady=(0, 5))
        
        self.available_files_listbox = tk.Listbox(left_panel, height=15, bg=self.colors['surface'], fg=self.colors['text_primary'])
        self.available_files_listbox.pack(fill='both', expand=True)

        # --- Right Column: Files to be Processed ---
        right_panel = ttk.Frame(main_layout, style='Card.TFrame')
        right_panel.grid(row=0, column=1, sticky='nsew', padx=(10, 0))

        self.processing_files_count_var = tk.StringVar(value="Files for Processing (0)")
        ttk.Label(right_panel, textvariable=self.processing_files_count_var, style='Heading.TLabel').pack(anchor='w', pady=(0, 5))

        self.processing_files_listbox = tk.Listbox(right_panel, height=15, bg='#eaf3fb', fg=self.colors['text_primary'])
        self.processing_files_listbox.pack(fill='both', expand=True)

        # Bind tab change event to update the lists
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def on_tab_changed(self, event):
        """Called when the user switches tab."""
        try:
            selected_tab_index = self.notebook.index(self.notebook.select())
            
            # Index 2: the NEW 'Session Filter' tab
            # Has to call its own refresh function: update_session_filter_lists
            if selected_tab_index == 2:
                self.update_session_filter_lists()
                
            # Index 3: the OLD 'Condition Filter' tab, now in third position
            # Has to call its own refresh function: update_condition_filter_lists
            elif selected_tab_index == 3:
                self.update_condition_filter_lists()
                
        except tk.TclError:
            # This can happen if the notebook is destroyed. Ignore it quietly.
            pass


    def update_condition_filter_lists(self):
        """
        Fill the 'Available Files' list of the Condition tab with the files
        filtered by the previous tab (Session).
        """
        # Clear the left-hand list (available files)
        self.available_files_listbox.delete(0, tk.END)
        
        # --- key correction ---
        # The file source is 'self.uploaded_files', which has been refreshed
        # by the subject filter and then by the session filter.
        # That variable always holds the most recent state of the selected files.
        available_files_paths = self.uploaded_files
        
        # Fill the left-hand list with the file names
        for fp in sorted(available_files_paths):
            self.available_files_listbox.insert(tk.END, os.path.basename(fp))
        
        # Refresh the counter
        count = len(available_files_paths)
        self.available_files_count_var.set(f"Available Files for Selected Subjects ({count})")
        
        # Apply the 'All' filter by default so the right-hand list is filled too
        # This makes sure the right-hand list does not start out empty.
        self.apply_condition_filter(None)


    def apply_condition_filter(self, condition_tag):
        """
        Filters the available files by condition and updates the final processing list.
        This version correctly uses the full list of available files as the source for every filter operation.
        """
        self.processing_files_listbox.delete(0, tk.END)
        
        # --- key correction ---
        # The source of the files to filter is ALWAYS the left-hand list ('Available Files'),
        # which holds every valid file after the subject filter.
        # The right-hand list, being the result of an earlier filter, is no longer used.
        source_filenames = self.available_files_listbox.get(0, tk.END)
        # -------------------------
        
        if condition_tag:  # If 'EC' or 'EO' was pressed
            # Filter the SOURCE list
            filtered_filenames = [fn for fn in source_filenames if f"_{condition_tag}_" in fn or f"task-{condition_tag}" in fn or f"rest{condition_tag}" in fn]
            status_message = f"Condition filter '{condition_tag}' applied."
        else:  # If 'Use All Conditions' was pressed
            # Use the whole SOURCE list
            filtered_filenames = list(source_filenames)
            status_message = "Condition filter removed. Using all available files."

        # Fill the right-hand list (Files for Processing) with the filter results
        for fn in sorted(filtered_filenames):
            self.processing_files_listbox.insert(tk.END, fn)
        
        count = len(filtered_filenames)
        self.processing_files_count_var.set(f"Files for Processing ({count})")
        
        # --- final and crucial refresh ---
        # Rebuild the full file paths and refresh the main variables
        # that the next stages of the pipeline will use.
        
        # Build a {file_name: full_path} map for an efficient lookup
        filename_to_path_map = {os.path.basename(fp): fp for fp in self.all_uploaded_files}
        
        # Refresh the list of files to process with the full paths
        self.uploaded_files = [filename_to_path_map[fn] for fn in filtered_filenames if fn in filename_to_path_map]
        
        # Keep the subject-ID list in step with the file list just filtered
        self.subject_ids = [os.path.basename(fp).split('_')[0] for fp in self.uploaded_files]
        
        self.status_var.set(f"{status_message} {count} files selected.")



    def create_subject_filter_tab(self):
        """Creates the subject filter tab with an improved layout and English text."""
        filter_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(filter_frame, text="Subject Filter")

        # --- Top control panel ---
        control_panel = ttk.Frame(filter_frame, style='Card.TFrame')
        control_panel.pack(fill='x', pady=(0, 20))
        
        ttk.Label(control_panel, text="Inclusionary Subject Filter", style='Title.TLabel').pack(side='left', anchor='w')
        
        # A frame for the buttons, so they line up to the right
        button_frame = ttk.Frame(control_panel, style='Card.TFrame')
        button_frame.pack(side='right', anchor='e')

        # Button that applies the filter (already present)
        self.subject_filter_button = ttk.Button(button_frame, text="Apply Inclusion Filter", 
                                                style='Primary.TButton', command=self.apply_subject_filter)
        self.subject_filter_button.pack(side='left', padx=5)

        # New button that clears the filter
        self.clear_subject_filter_button = ttk.Button(button_frame, text="Clear Filter / Show All", 
                                                      style='Secondary.TButton', command=self.clear_subject_filter)
        self.clear_subject_filter_button.pack(side='left', padx=5)

        # --- Main two-column layout ---
        main_layout = ttk.Frame(filter_frame, style='Card.TFrame')
        main_layout.pack(fill='both', expand=True)
        main_layout.columnconfigure(0, weight=1)
        main_layout.columnconfigure(1, weight=1)
        main_layout.rowconfigure(0, weight=1)

        # --- Left Column: All Loaded Subjects ---
        left_panel = ttk.Frame(main_layout, style='Card.TFrame')
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        self.all_subjects_count_var = tk.StringVar(value="All Loaded Subjects (0)")
        ttk.Label(left_panel, textvariable=self.all_subjects_count_var, style='Heading.TLabel').pack(anchor='w', pady=(0, 5))
        
        self.all_subjects_listbox = tk.Listbox(left_panel, height=15, bg=self.colors['surface'], fg=self.colors['text_primary'])
        self.all_subjects_listbox.pack(fill='both', expand=True)

        # --- Right Column: Subjects Included in Analysis ---
        right_panel = ttk.Frame(main_layout, style='Card.TFrame')
        right_panel.grid(row=0, column=1, sticky='nsew', padx=(10, 0))

        self.included_subjects_count_var = tk.StringVar(value="Subjects for Analysis (0)")
        ttk.Label(right_panel, textvariable=self.included_subjects_count_var, style='Heading.TLabel').pack(anchor='w', pady=(0, 5))

        self.included_subjects_listbox = tk.Listbox(right_panel, height=15, bg='#eaf3fb', fg=self.colors['text_primary'])
        self.included_subjects_listbox.pack(fill='both', expand=True)


    def update_subject_lists(self):
        """Updates the 'All' and 'Included' subject lists (called after loading files)."""
        self.all_subjects_listbox.delete(0, tk.END)
        self.included_subjects_listbox.delete(0, tk.END)

        all_ids = [os.path.basename(fp).split('.')[0] for fp in self.all_uploaded_files]
        for sid in sorted(list(set(all_ids))):
            self.all_subjects_listbox.insert(tk.END, sid)
        
        self.all_subjects_count_var.set(f"All Loaded Subjects ({len(self.all_subjects_listbox.get(0, tk.END))})")
        self.included_subjects_count_var.set("Subjects for Analysis (0)")
        self.status_var.set("Ready. Apply filter to select subjects for analysis.")

    def apply_subject_filter(self):
        """Filters subjects and updates the right-hand list and main application variables."""
        self.included_subjects_listbox.delete(0, tk.END)
        
        all_subject_ids = self.all_subjects_listbox.get(0, tk.END)
        
        included_ids = [sid for sid in all_subject_ids if sid.startswith(tuple(self.valid_subject_prefixes))]
        
        for sid in sorted(included_ids):
            self.included_subjects_listbox.insert(tk.END, sid)
        
        count = len(included_ids)
        self.included_subjects_count_var.set(f"Subjects for Analysis ({count})")
        
        self.subject_ids = included_ids
        self.uploaded_files = [fp for fp in self.all_uploaded_files if os.path.basename(fp).split('.')[0] in included_ids]
        
        # This is the message box from your screenshot, now in English.
        messagebox.showinfo("Filter Applied", 
                            f"{count} subjects have been selected for analysis.\n"
                            "Subsequent processing will be performed exclusively on this subset.")
        
        self.status_var.set(f"Filter applied. {count} subjects are ready for analysis.")


    def clear_subject_filter(self):
        """
        Clear the subject filter, bringing every loaded subject back into the analysis.
        """
        # Clear the right-hand list
        self.included_subjects_listbox.delete(0, tk.END)
        
        # Take every subject from the left-hand list
        all_subject_ids = self.all_subjects_listbox.get(0, tk.END)
        
        # Fill the right-hand list with every subject
        for sid in sorted(all_subject_ids):
            self.included_subjects_listbox.insert(tk.END, sid)
        
        count = len(all_subject_ids)
        self.included_subjects_count_var.set(f"Subjects for Analysis ({count})")
        
        # --- crucial refresh of the global variables ---
        # Restore the main variables so that ALL files and IDs are included
        self.subject_ids = list(all_subject_ids)
        self.uploaded_files = self.all_uploaded_files.copy() # Use the copy of the master list
        
        self.status_var.set(f"Subject filter cleared. {count} subjects are now included for analysis.")
        messagebox.showinfo("Filter Cleared", 
                            f"The subject filter has been removed.\n"
                            f"All {count} loaded subjects are now included in the analysis.")


    def create_upload_tab(self):
        """
        Build the loading and configuration tab with a two-column layout
        tuned to remove empty space and maximise the area of the 'About' text.
        """
        # Main container of the tab
        upload_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=(20, 10))
        self.notebook.add(upload_frame, text="Data Upload & Config")
        
        # Configure the main grid for a responsive layout
        upload_frame.columnconfigure(0, weight=1)
        upload_frame.columnconfigure(1, weight=1)
        
        # Row 0 (the main columns) no longer expands (weight=0).
        upload_frame.rowconfigure(0, weight=0) 
        # Row 1 (the About section) expands to fill the whole vertical space (weight=1).
        upload_frame.rowconfigure(1, weight=1)

        # --- LEFT COLUMN ---
        left_column = ttk.Frame(upload_frame, style='Card.TFrame')
        left_column.grid(row=0, column=0, sticky="new", padx=(0, 15))
        
        ttk.Label(left_column, text="File Upload", style='Heading.TLabel').pack(fill='x', anchor='w', pady=(0, 5))
        upload_container = ttk.Frame(left_column, style='Card.TFrame', borderwidth=1, relief='solid', padding=15)
        upload_container.pack(fill='x', pady=(0, 20))

        file_buttons_frame = ttk.Frame(upload_container, style='Card.TFrame')
        file_buttons_frame.pack(fill='x', pady=(5, 15))
        ttk.Button(file_buttons_frame, text="Select Files", style='Primary.TButton',
                command=self.select_files).pack(side='left', expand=True, fill='x', padx=(0, 5))
        ttk.Button(file_buttons_frame, text="Select Directory", style='Secondary.TButton',
                command=self.select_directory).pack(side='left', expand=True, fill='x', padx=(5, 0))
        
        self.file_listbox = tk.Listbox(upload_container, height=8, bg=self.colors['surface'],
                                    fg=self.colors['text_primary'], relief='solid', borderwidth=1)
        self.file_listbox.pack(fill='x', expand=True)

        ttk.Label(left_column, text="Configuration", style='Heading.TLabel').pack(fill='x', anchor='w', pady=(10, 5))
        config_container = ttk.Frame(left_column, style='Card.TFrame', borderwidth=1, relief='solid', padding=15)
        config_container.pack(fill='x')

        config_name_frame = ttk.Frame(config_container, style='Card.TFrame')
        config_name_frame.pack(fill='x', pady=5)
        ttk.Label(config_name_frame, text="Config Name:", style='Body.TLabel').pack(side='left')
        self.config_name = tk.StringVar(value="eeg_analysis_config")
        ttk.Entry(config_name_frame, textvariable=self.config_name).pack(side='left', expand=True, fill='x', padx=10)

        config_buttons_frame = ttk.Frame(config_container, style='Card.TFrame')
        config_buttons_frame.pack(fill='x', pady=10)
        ttk.Button(config_buttons_frame, text="Load Config", style='Secondary.TButton',
                command=self.load_configuration).pack(side='left', expand=True, fill='x', padx=(0, 5))
        ttk.Button(config_buttons_frame, text="Save Config", style='Primary.TButton',
                command=self.save_configuration).pack(side='left', expand=True, fill='x', padx=(5, 0))

        # --- RIGHT COLUMN ---
        right_column = ttk.Frame(upload_frame, style='Card.TFrame')
        right_column.grid(row=0, column=1, sticky="new", padx=(15, 0))
        
        ttk.Label(right_column, text="Processing Parameters", style='Heading.TLabel').pack(fill='x', anchor='w', pady=(0, 5))
        params_container = ttk.Frame(right_column, style='Card.TFrame', borderwidth=1, relief='solid', padding=15)
        params_container.pack(fill='x')
        self.create_parameter_inputs(params_container)

        # --- ABOUT SECTION ---
        about_section = ttk.Frame(upload_frame, style='Card.TFrame')
        about_section.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(20, 0))
        about_section.columnconfigure(0, weight=1)
        about_section.rowconfigure(1, weight=1) 

        header = ttk.Frame(about_section, style='Card.TFrame')
        header.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 6))
        header.columnconfigure(0, weight=1)

        title_lbl = ttk.Label(header, text="About this Tool", font=("Helvetica", 13, "bold"))
        title_lbl.grid(row=0, column=0, sticky="w")

        toggle_bar = ttk.Frame(header, style='Card.TFrame')
        toggle_bar.grid(row=0, column=1, sticky="e")

        ttk.Button(toggle_bar, text="Compact", style='Secondary.TButton',
                command=lambda: _set_about("compact")).pack(side="right", padx=(6, 0))
        ttk.Button(toggle_bar, text="Detailed", style='Primary.TButton',
                command=lambda: _set_about("detailed")).pack(side="right")

        content_box = ttk.Frame(about_section, style='Card.TFrame', borderwidth=1, relief='solid')
        content_box.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 0))

        content_inner = ttk.Frame(content_box, style='Card.TFrame', padding=14)
        content_inner.pack(fill="both", expand=True)

        ABOUT_TEXT_COMPACT = (
            "An integrated GUI for reproducible EEG analysis, encompassing preprocessing, feature extraction, connectivity, graph theory, and predictive modeling. The pipeline features:\n"
            "• **Preprocessing:** Band-pass/notch filtering, bad channel management, and ICA-based artifact removal.\n"
            "• **Feature Extraction:** Spectral power calculation (absolute/relative), aperiodic component parameterization (FOOOF), and derivation of key biomarkers (FAA, IAF, TBR).\n"
            "• **Connectivity Analysis:** Computation of functional connectivity matrices using multiple metrics (e.g., dPLI, imCOH) across frequency bands.\n"
            "• **Graph Theory Module:** Calculation of global and nodal graph metrics (e.g., modularity, efficiency, centrality) with support for MST and multi-threshold analysis.\n"
            "• **Statistical & Predictive Modeling:** Inferential analysis via linear/logistic regression and a nested cross-validation framework for robust predictive model evaluation (Elastic-Net, XGBoost) with model interpretation tools (Permutation Importance, SHAP).\n\n"
            "Designed for subject- and session-level merging with behavioral or clinical data to facilitate robust biomarker discovery pipelines.\n\n"
            "Developed by Dr. Marco Manzo¹,² — Supervision: Prof. Emiliano Santarnecchi¹\n"
            "Affiliations: ¹Precision Neuroscience & Neuromodulation Program, Gordon Center for Medical Imaging, MGH/Harvard Medical School, Boston, MA. "
            "²University of Naples Federico II, School of Medicine, Naples, Italy."
        )

        ABOUT_TEXT_DETAILED = (
            "This graphical user interface provides an integrated and reproducible environment for advanced electroencephalography (EEG) analysis, designed to bridge the gap between raw data processing and sophisticated statistical modeling. The pipeline is structured into distinct, yet interconnected, modules that guide the user from initial data cleaning to robust biomarker discovery and predictive analytics.\n\n"
            "The **Preprocessing Module** forms the foundational stage, engineered to enhance the signal-to-noise ratio of the raw EEG data. It implements essential signal processing steps, including customizable band-pass filtering to isolate frequencies of interest and notch filtering (50/60 Hz) to remove power-line interference. The core of artifact mitigation is performed using Independent Component Analysis (ICA), a powerful blind source separation technique that facilitates the identification and removal of stereotyped artifacts (e.g., blinks, saccades, cardiac pulse, muscle activity).\n\n"
            "The **Feature Extraction Module** quantifies key neurophysiological characteristics of the EEG signal. It computes both absolute and relative power across canonical frequency bands and integrates the 'Fitting Oscillations & One Over F' (FOOOF) algorithm to parameterize the aperiodic component of the power spectrum. This decomposition allows for a more accurate characterization of oscillatory peaks, from which the tool automatically derives widely-used biomarkers, including Frontal Alpha Asymmetry (FAA), Individual Alpha Frequency (IAF), and the Theta/Beta Ratio (TBR).\n\n"
            "The **Connectivity Analysis Module** extends the analysis to brain networks by computing functional connectivity. It supports various metrics, such as the debiased Weighted Phase Lag Index (dPLI) and Imaginary Part of Coherency (imCOH), to estimate statistical dependencies between EEG sensor signals across different frequency bands. This provides a foundational dataset of connectivity matrices for each subject.\n\n"
            "The **Graph Theory Module** uses these connectivity matrices to model brain networks as graphs. It allows for the calculation of a wide range of network metrics at both global (e.g., modularity, characteristic path length, global efficiency) and nodal levels (e.g., degree, clustering coefficient). The module includes advanced features like Minimum Spanning Tree (MST) analysis and multi-threshold integration to ensure robust characterization of network topology.\n\n"
            "Finally, the **Modeling Module** offers a dual framework for statistical analysis. For **inferential analysis**, it supports standard linear (OLS) and logistic regression models. For **predictive analysis**, the tool implements a rigorous nested cross-validation pipeline to provide an unbiased estimate of a model's generalization performance. It supports modern machine learning algorithms like Elastic-Net and XGBoost and includes interpretation tools like Permutation Feature Importance and SHAP (SHapley Additive exPlanations) to identify the most influential predictors.\n\n"
            "Developed by Dr. Marco Manzo¹,² — Supervision: Prof. Emiliano Santarnecchi¹\n"
            "Affiliations: ¹Precision Neuroscience & Neuromodulation Program, Gordon Center for Medical Imaging, MGH/Harvard Medical School, Boston, MA. "
            "²University of Naples Federico II, School of Medicine, Naples, Italy."
        )

        about_text_widget = tk.Text(
            content_inner, wrap="word", relief=tk.FLAT,
            bg=self.colors['surface'], fg=self.colors['text_primary'],
            insertbackground=self.colors['surface'],
            highlightthickness=0, borderwidth=0,
            font=('Helvetica', 11),
            # The height is no longer fixed, the widget expands
            spacing1=2, spacing2=4, spacing3=8
        )
        
        bold_font = tkFont.Font(family="Helvetica", size=11, weight="bold")
        about_text_widget.tag_configure("bold", font=bold_font)
        about_text_widget.pack(fill="both", expand=True)

        def insert_with_tags(text_to_insert):
            parts = re.split(r'(\*\*.*?\*\*)', text_to_insert)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    about_text_widget.insert(tk.END, part[2:-2], "bold")
                else:
                    about_text_widget.insert(tk.END, part)

        def _set_about(mode="detailed"):
            about_text_widget.config(state="normal")
            about_text_widget.delete("1.0", tk.END)
            text_to_display = ABOUT_TEXT_DETAILED if mode == "detailed" else ABOUT_TEXT_COMPACT
            insert_with_tags(text_to_display)
            about_text_widget.config(state="disabled")

        _set_about("detailed")


    def create_parameter_inputs(self, parent):
        """
        Build and arrange every widget for entering the processing parameters.
        """
        # --- 1. Filtering section ---
        filter_frame = ttk.LabelFrame(parent, text="Filtering", style='Card.TLabelframe', padding=10)
        filter_frame.pack(fill='x', pady=5, padx=5)
        self.create_param_row(filter_frame, "High-pass (Hz):", self.low_freq, 0.1, 5.0)
        self.create_param_row(filter_frame, "Low-pass (Hz):", self.high_freq, 30.0, 100.0, is_float=False, round_to_int=True)
        notch_frame = ttk.Frame(filter_frame, style='Card.TFrame')
        notch_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(notch_frame, text="Notch (Hz):", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        ttk.Combobox(notch_frame, textvariable=self.notch_freq,
                     values=[50.0, 60.0], state='readonly', width=10).pack(side='left')

        # --- 2. ICA section ---
        ica_frame = ttk.LabelFrame(parent, text="ICA", style='Card.TLabelframe', padding=10)
        ica_frame.pack(fill='x', pady=5, padx=5)
        self.create_param_row(ica_frame, "Components:", self.n_components, 10, 50, is_float=False, round_to_int=True)
        method_frame = ttk.Frame(ica_frame, style='Card.TFrame')
        method_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(method_frame, text="Method:", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        ttk.Combobox(method_frame, textvariable=self.ica_method,
                     values=['infomax', 'fastica', 'picard'], state='readonly', width=10).pack(side='left')

        # --- 3. FOOOF section ---
        fooof_frame = ttk.LabelFrame(parent, text="FOOOF", style='Card.TLabelframe', padding=10)
        fooof_frame.pack(fill='x', pady=5, padx=5)
        
        self.create_freq_range_row(fooof_frame)
        self.create_param_row(fooof_frame, "Min Peak Height:", self.min_peak_height, 0.0, 1.0)
        
        ap_frame = ttk.Frame(fooof_frame, style='Card.TFrame')
        ap_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(ap_frame, text="Aperiodic Mode:", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        ttk.Combobox(ap_frame, textvariable=self.aperiodic_mode,
                     values=['fixed', 'knee'], state='readonly', width=10).pack(side='left')

    def update_graph_theory_display(self, G, metrics, subject_id, matrix_key):
        """
        Refresh the display widgets of the Graph Theory tab with the results supplied.
        """
        # Refresh the label so it says what is being shown
        self.gt_viz_label_var.set(f"Display for: {subject_id}")

        # Refresh the table of global metrics
        for i in self.gt_summary_table.get_children(): self.gt_summary_table.delete(i)
        for key, value in metrics['global'].items():
            self.gt_summary_table.insert('', 'end', values=(key.replace('_', ' ').title(), f"{value:.4f}"))

        # Refresh the table of per-node metrics
        for i in self.gt_node_table.get_children(): self.gt_node_table.delete(i)
        node_cols = list(metrics['node'].columns)
        self.gt_node_table['columns'] = node_cols
        for col in node_cols:
            self.gt_node_table.heading(col, text=col.replace('_', ' ').capitalize())
            self.gt_node_table.column(col, width=80, anchor='center')
        for _, row in metrics['node'].iterrows():
            formatted_values = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else v for v in row]
            self.gt_node_table.insert('', 'end', values=formatted_values)

        # Refresh the matrix plot
        self.gt_fig_matrix.clear()
        ax_m = self.gt_fig_matrix.add_subplot(111)
        im = ax_m.imshow(nx.to_numpy_array(G), cmap='viridis', origin='lower')
        self.gt_fig_matrix.colorbar(im, ax=ax_m)
        ax_m.set_title(f"Thresholded matrix\n({matrix_key})")
        self.gt_canvas_matrix.draw()

        # Refresh the topology plot
        self.gt_fig_graph.clear()
        ax_g = self.gt_fig_graph.add_subplot(111)
        pos = nx.spring_layout(G, seed=self.gt_seed_var.get())
        colors = metrics['node']['community'].values
        node_sizes = [d * 200 + 50 for d in metrics['node']['degree'].values]
        nx.draw(G, pos, ax=ax_g, node_color=colors, node_size=node_sizes, cmap=plt.cm.jet, with_labels=True, font_size=8, font_color='black', edge_color='grey')
        ax_g.set_title("Graph topology (consensus)")
        self.gt_canvas_graph.draw()

    def create_freq_range_row(self, parent):
        """Build the row dedicated to the FOOOF frequency range."""
        freq_range_frame = ttk.Frame(parent, style='Card.TFrame')
        freq_range_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(freq_range_frame, text="Freq Range (Hz):", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        
        min_entry = ttk.Entry(freq_range_frame, width=6, justify='right')
        min_entry.pack(side='left', padx=(0, 5))
        min_scale = ttk.Scale(freq_range_frame, from_=1, to=10, variable=self.freq_range_min, orient='horizontal')
        min_scale.pack(side='left', expand=True, fill='x')
        
        ttk.Label(freq_range_frame, text="to", style='Param.TLabel').pack(side='left', padx=5)
        
        max_entry = ttk.Entry(freq_range_frame, width=6, justify='right')
        max_entry.pack(side='left', padx=(0, 5))
        max_scale = ttk.Scale(freq_range_frame, from_=30, to=100, variable=self.freq_range_max, orient='horizontal')
        max_scale.pack(side='left', expand=True, fill='x')

        def update_min_from_scale(val_str):
            min_entry.delete(0, tk.END)
            min_entry.insert(0, f"{int(float(val_str))}")

        def update_min_from_entry(event=None):
            try: self.freq_range_min.set(int(min_entry.get()))
            except (ValueError, tk.TclError): update_min_from_scale(self.freq_range_min.get())

        def update_max_from_scale(val_str):
            max_entry.delete(0, tk.END)
            max_entry.insert(0, f"{int(float(val_str))}")

        def update_max_from_entry(event=None):
            try: self.freq_range_max.set(int(max_entry.get()))
            except (ValueError, tk.TclError): update_max_from_scale(self.freq_range_max.get())

        self.freq_range_min.trace_add('write', lambda *args: update_min_from_scale(self.freq_range_min.get()))
        min_entry.bind("<Return>", update_min_from_entry)
        min_entry.bind("<FocusOut>", update_min_from_entry)
        update_min_from_scale(self.freq_range_min.get())

        self.freq_range_max.trace_add('write', lambda *args: update_max_from_scale(self.freq_range_max.get()))
        max_entry.bind("<Return>", update_max_from_entry)
        max_entry.bind("<FocusOut>", update_max_from_entry)
        update_max_from_scale(self.freq_range_max.get())



    def create_summary_tab(self):
        """Build the Summary tab, with a table and a boxplot."""
        summary_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(summary_frame, text="Summary")
        
        table_section = self.create_section(summary_frame, "All Subject Biomarkers")
        
        table_container = ttk.Frame(table_section)
        table_container.pack(fill='x', expand=True)
        
        columns = ["Subject", "participant_ID", "Exponent_asymmetry", "Right_exponent", "Left_exponent", "FAA", "TBR", "IAF", "Offset", "Exponent", "R_squared", "Error", "Theta_CF", "Theta_PW", "Theta_BW", "Alpha_CF", "Alpha_PW", "Alpha_BW", "Beta_CF", "Beta_PW", "Beta_BW", "Gamma_CF", "Gamma_PW", "Gamma_BW"]
        self.summary_table = ttk.Treeview(table_container, columns=columns, show='headings', height=8)
        
        vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.summary_table.yview)
        self.summary_table.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(table_container, orient="horizontal", command=self.summary_table.xview)
        self.summary_table.configure(xscrollcommand=hsb.set)
        
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')

        for col in columns:
            self.summary_table.heading(col, text=col)
            width = 120 if col == "Subject" else 85
            self.summary_table.column(col, width=width, minwidth=60, anchor='center')
            
        self.summary_table.pack(side='left', fill='x', expand=True)

        plot_section = self.create_section(summary_frame, "Group Distribution")
        self.summary_fig = Figure(figsize=(5, 3), facecolor=self.colors['surface'])
        self.summary_canvas = FigureCanvasTkAgg(self.summary_fig, plot_section)
        self.summary_canvas.get_tk_widget().pack(fill='both', expand=True)
        ttk.Button(plot_section, text="Refresh Summary", style='Primary.TButton', command=self.update_summary_tab).pack(pady=10)

    def update_regression_results(self, summary_text):
        """
        Show the textual summary of the regression model in the proper tab.
        It is called by _regression_worker once the model has been fitted.
        """
        # 1. Find the right text widget in which to show the results.
        #    Walk down to the widget built in create_regression_results_display.
        #    For simplicity it is kept in an attribute of the class.
        if hasattr(self, 'regression_results_text'):
            # 2. Clear any earlier result
            self.regression_results_text.delete('1.0', tk.END)
            
            # 3. Insert the new model summary
            self.regression_results_text.insert(tk.END, summary_text)
            
            # 4. Refresh the status bar and take the user to the results tab
            self.status_var.set("Regression analysis complete. Results are displayed.")
            self.notebook.select(self.notebook.tabs().index(self.notebook.select()) + 1) # Move to the next tab
        else:
            # Error message for the case where the widget was not created properly
            messagebox.showerror("GUI Error", "Regression results text area not found.")


    def update_summary_tab(self):
        """
        Collect and merge EVERY biomarker and refresh the summary table.
        Handles the data structure of graph_theory_results correctly.
        """
        print("\n" + "="*20 + " SUMMARY TAB REFRESH STARTED " + "="*20)

        # --- 1. Collect every possible feature key dynamically ---
        base_conn_keys = set()
        if hasattr(self, 'connectivity_results'):
            for data in self.connectivity_results.values():
                base_conn_keys.update(data.get('features', {}).keys())

        graph_feature_keys = set()
        if hasattr(self, 'graph_theory_results'):
            # --- change here ---
            # Only the first level is traversed, to obtain the per-subject dictionaries
            for subject_data in self.graph_theory_results.values():
                # subject_data is a dictionary such as {'gt_dpli_alpha_Q': 0.54, ...}
                # It is merged straight in, by its own keys.
                if isinstance(subject_data, dict):
                    graph_feature_keys.update(subject_data.keys())
            # --- end of the change ---

        # --- 2. Define and assemble the final columns ---
        base_columns = [
            "Subject", "participant_ID", "session_ID", "Exponent_asymmetry", "Right_exponent", "Left_exponent",
            "FAA", "TBR", "IAF", "Offset", "Exponent", "R_squared", "Error",
            "Theta_CF", "Theta_PW", "Theta_BW", "Alpha_CF", "Alpha_PW", "Alpha_BW",
            "Beta_CF", "Beta_PW", "Beta_BW", "Gamma_CF", "Gamma_PW", "Gamma_BW"
        ]
        
        connectivity_columns = sorted(list(base_conn_keys))
        graph_columns = sorted(list(graph_feature_keys))
        final_columns = base_columns + connectivity_columns + graph_columns

        print(f"[DEBUG] Base connectivity columns found: {len(connectivity_columns)}")
        print(f"[DEBUG] Graph theory columns found: {len(graph_columns)}")
        print(f"[DEBUG] Total number of final columns: {len(final_columns)}")

        # --- 3. Build the DataFrame row by row ---
        all_rows_data = []
        for filename in self.uploaded_files:
            subject_id_full = os.path.basename(filename).replace(".vhdr", "")
            
            row = {col: np.nan for col in final_columns}
            row['Subject'] = subject_id_full
            row['participant_ID'] = subject_id_full.split('_')[0]
            
            session_tag_part = next((part for part in subject_id_full.split('_') if part.startswith('ses-')), 'ses-1')
            row['session_ID'] = int(re.sub(r'[^0-9]', '', session_tag_part))

            if subject_id_full in self.preprocessed_data:
                raw = self.preprocessed_data[subject_id_full]
                left_ch = ['F7', 'FT7', 'T7', 'TP7', 'P7', 'PO7', 'O1', 'Fp1', 'F3', 'FC5', 'C3', 'CP5', 'P3', 'PO9']
                right_ch = ['F8', 'FT8', 'T8', 'TP8', 'P8', 'PO8', 'O2', 'Fp2', 'F4', 'FC6', 'C4', 'CP6', 'P4', 'PO10']
                left_exp = self.get_mean_exponent(raw, left_ch)
                right_exp = self.get_mean_exponent(raw, right_ch)
                row["Left_exponent"], row["Right_exponent"] = left_exp, right_exp
                if not np.isnan(left_exp) and not np.isnan(right_exp):
                    row["Exponent_asymmetry"] = left_exp - right_exp

            if subject_id_full in self.spectral_results:
                row.update(self.spectral_results[subject_id_full].get('biomarkers', {}))

            if subject_id_full in self.fooof_results and 'Oz' in self.fooof_results.get(subject_id_full, {}):
                fm = self.fooof_results[subject_id_full]['Oz']['fm']
                if hasattr(fm, 'aperiodic_params_'): row["Offset"], row["Exponent"] = fm.aperiodic_params_
                if hasattr(fm, 'r_squared_'): row["R_squared"] = fm.r_squared_
                if hasattr(fm, 'error_'): row["Error"] = fm.error_
                if hasattr(fm, 'peak_params_') and fm.peak_params_ is not None and fm.peak_params_.size > 0:
                    peaks = fm.peak_params_
                    bands = Bands({'theta': [4, 8], 'alpha': [8, 12], 'beta': [13, 30], 'gamma': [30, 50]})
                    for band_name, band_range in bands:
                        band_peaks = peaks[(peaks[:, 0] >= band_range[0]) & (peaks[:, 0] <= band_range[1])]
                        if band_peaks.shape[0] > 0:
                            dominant_peak = band_peaks[np.argmax(band_peaks[:, 1])]
                            prefix = band_name.capitalize()
                            row[f'{prefix}_CF'], row[f'{prefix}_PW'], row[f'{prefix}_BW'] = dominant_peak

            if hasattr(self, 'connectivity_results') and subject_id_full in self.connectivity_results:
                row.update(self.connectivity_results[subject_id_full].get('features', {}))

            # --- change here ---
            # Refresh the row with every graph metric for this subject
            if hasattr(self, 'graph_theory_results') and subject_id_full in self.graph_theory_results:
                row.update(self.graph_theory_results[subject_id_full])
            # --- end of the change ---
            
            all_rows_data.append(row)

        if not all_rows_data:
            print("[DEBUG] No data row was generated. The table will stay empty.")
            return
            
        self.biomarkers_df = pd.DataFrame(all_rows_data, columns=final_columns)
        
        print(f"\n[DEBUG] Final DataFrame built. Shape: {self.biomarkers_df.shape}")

        # --- 4. Refresh the table (Treeview) in the GUI ---
        for i in self.summary_table.get_children():
            self.summary_table.delete(i)
            
        self.summary_table['columns'] = final_columns
        for col in final_columns:
            self.summary_table.heading(col, text=col)
            width = 180 if col == "Subject" else 120 if col == "participant_ID" else 100
            self.summary_table.column(col, width=width, minwidth=70, anchor='w')
            
        for _, row_data in self.biomarkers_df.iterrows():
            vals = [f"{row_data.get(col, ''):.4f}" if isinstance(row_data.get(col), (float, np.floating)) else row_data.get(col, '') for col in final_columns]
            self.summary_table.insert('', 'end', values=vals)

        # --- 5. Refresh the summary plots ---
        self.summary_fig.clear()
        plot_cols = ["FAA", "TBR", "IAF"]
        axes = self.summary_fig.subplots(1, len(plot_cols))
        if len(plot_cols) == 1: axes = [axes]
        self.summary_fig.suptitle('Biomarker Distributions', fontsize=14)
        for i, col in enumerate(plot_cols):
            ax = axes[i]
            plot_data = self.biomarkers_df[col].dropna()
            if not plot_data.empty:
                sns.boxplot(y=plot_data, ax=ax, showfliers=False, color=sns.color_palette("pastel")[i], width=0.5)
                sns.stripplot(y=plot_data, ax=ax, jitter=True, alpha=0.7, color=sns.color_palette("deep")[i])
                ax.set_title(col)
                ax.set_xlabel("")
                ax.set_ylabel("Value")
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                ax.set_title(col)
        self.summary_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.summary_canvas.draw()
        
        self.update_analysis_summary()
        print("="*20 + " SUMMARY TAB REFRESH FINISHED " + "="*20 + "\n")


    def update_session_filter_lists(self):
        """
        Fill the 'Available' list of the Session tab and apply the
        multi-session filter with the default selections.
        """
        self.session_available_files_listbox.delete(0, tk.END)
        
        # The file source is 'self.uploaded_files', refreshed by the subject filter
        available_files = self.uploaded_files
        
        for fp in sorted(available_files):
            self.session_available_files_listbox.insert(tk.END, os.path.basename(fp))
        
        count = len(available_files)
        self.session_available_files_count_var.set(f"Available Files from Subject Filter ({count})")
        
        # Apply the filter with the current selections, all of them active by default
        self.apply_multi_session_filter()


    def _format_evaluation_report(self, result: Dict) -> str:
        """
        Formats the results of a single model evaluation into a readable text block.
        """
        report_lines = [
            f"--- Model Evaluation Report: {result['model_label']} ---",
            f"Formula: {result['formula']}",
            "-" * 60,
            "Performance Metrics:",
            f"  - Calibrated AUC (95% CI): {result['auc_mean']:.4f} ({result['auc_ci'][0]:.4f} - {result['auc_ci'][1]:.4f})",
            f"  - Brier Score:             {result['brier']:.4f}",
            f"  - Subjects in Analysis (N):  {len(result['y_true'])}",
            "-" * 60,
            "\n" # Add a blank line before the next run
        ]
        return "\n".join(report_lines)



    def apply_multi_session_filter(self):
        """
        Filter the files according to the sessions selected.
        If no session is selected, EVERY available file is included.
        """
        # 1. Obtain the list of selected sessions
        selected_sessions = [session_name for session_name, var in self.session_vars.items() if var.get()]

        # Clear the right-hand list
        self.session_processing_files_listbox.delete(0, tk.END)
        
        # Take the files from the left-hand list, the output of the subject filter
        source_filenames = self.session_available_files_listbox.get(0, tk.END)
        
        # --- new logic ---
        if not selected_sessions:
            # If no checkbox is ticked, treat EVERY file as valid
            print("No session filter applied. Every available session is included.")
            filtered_filenames = list(source_filenames)
            status_message = "Session filter inactive. All available sessions included."
        else:
            # If at least one checkbox is ticked, apply the filter as before
            print(f"Applying the filter for the sessions: {selected_sessions}")
            filtered_filenames = [
                fn for fn in source_filenames 
                if any(f"_{session_tag}_" in fn for session_tag in selected_sessions)
            ]
            status_message = f"Session filter applied for: {', '.join(selected_sessions)}."
        # --- end of the new logic ---
        
        # Fill the right-hand list with the results
        for fn in sorted(filtered_filenames):
            self.session_processing_files_listbox.insert(tk.END, fn)
        
        count = len(filtered_filenames)
        self.session_processing_files_count_var.set(f"Files for Next Step ({count})")
        
        # Refresh the main variables of the application for the next stage
        filename_to_path_map = {os.path.basename(fp): fp for fp in self.all_uploaded_files}
        self.uploaded_files = [filename_to_path_map[fn] for fn in filtered_filenames if fn in filename_to_path_map]
        self.subject_ids = [os.path.basename(fp).split('_')[0] for fp in self.uploaded_files]
        
        self.status_var.set(f"{status_message} {count} files selected.")


    def create_preprocessing_tab(self):
        """Create the preprocessing tab"""
        preprocess_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(preprocess_frame, text="Preprocessing")
        control_panel = self.create_section(preprocess_frame, "Preprocessing Control")
        ttk.Button(control_panel, text="Run Preprocessing", style='Action.TButton', command=self.run_preprocessing).pack(pady=10)
        progress_frame = ttk.Frame(control_panel, style='Card.TFrame')
        progress_frame.pack(fill='x', pady=10)

        ttk.Label(progress_frame, text="Progress:", style='Body.TLabel').pack(side='left')

        self.preprocessing_progress = ttk.Progressbar(progress_frame, mode='indeterminate', style='Custom.Horizontal.TProgressbar')
        self.preprocessing_progress.pack(side='left', fill='x', expand=True, padx=(10, 5))

        # Percentage label (make sure self.preprocessing_progress_percent is defined)
        self.preprocessing_progress_percent = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.preprocessing_progress_percent, style='Body.TLabel').pack(side='right')
        viz_section = self.create_section(preprocess_frame, "EEG Data Visualization")
        # --- Subject Selector for Visualization ---
        selector_frame = ttk.Frame(viz_section, style='Card.TFrame')
        selector_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(selector_frame, text="Select Subject:", style='Body.TLabel').pack(side='left', padx=(0, 10))
        self.selected_subject = tk.StringVar()
        self.subject_dropdown = ttk.Combobox(selector_frame, textvariable=self.selected_subject, state='readonly')
        self.subject_dropdown.pack(side='left', padx=(0, 10))
        ttk.Button(selector_frame, text="Show PSD", style='Primary.TButton', command=self._update_single_preprocessing_viz).pack(side='left')
        self.preprocess_fig = Figure(figsize=(12, 6), facecolor=self.colors['surface'])
        self.preprocess_canvas = FigureCanvasTkAgg(self.preprocess_fig, viz_section)
        self.preprocess_canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar_frame = ttk.Frame(viz_section, style='Card.TFrame')
        toolbar_frame.pack(fill='x')
        self.preprocess_toolbar = NavigationToolbar2Tk(self.preprocess_canvas, toolbar_frame)


    def create_ica_tab(self):
        """
        Build the ICA analysis tab with an improved layout, grouped controls
        and an internal display area for the plots.
        """
        ica_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(ica_frame, text="ICA Analysis")

        # --- unified, compact control panel ---
        control_panel = self.create_section(ica_frame, "ICA Control Panel")
        
        # Frame holding the controls, so they line up on one row
        control_bar = ttk.Frame(control_panel)
        control_bar.pack(fill='x', pady=5)

        # Subject selector
        ttk.Label(control_bar, text="Select Subject:").pack(side='left', padx=(0, 5))
        self.selected_ica_subject = tk.StringVar()
        self.ica_subject_dropdown = ttk.Combobox(control_bar, textvariable=self.selected_ica_subject, state='readonly', width=35)
        self.ica_subject_dropdown.pack(side='left', padx=(0, 15))
        self.ica_subject_dropdown.bind("<<ComboboxSelected>>", self.on_ica_subject_change)
        
        # Action buttons
        ttk.Button(control_bar, text="Run ICA Analysis", style='Primary.TButton', command=self.run_ica).pack(side='left', padx=5)
        
        # Separator
        ttk.Separator(control_bar, orient='vertical').pack(side='left', fill='y', padx=10, pady=5)
        
        # Display buttons
        ttk.Button(control_bar, text="Show All Maps", style='Secondary.TButton', command=self.show_ica_maps_grid).pack(side='left', padx=5)
        ttk.Button(control_bar, text="Show Only Excluded", style='Info.TButton', command=self._show_excluded_ica_viz).pack(side='left', padx=5)
        
        # Separator before the most important button
        ttk.Separator(control_bar, orient='vertical').pack(side='left', fill='y', padx=10, pady=5)
        ttk.Button(control_bar, text="Update Component Exclusions", style='Success.TButton', command=self.update_component_exclusions).pack(side='left')

        # Progress bar, below the controls
        progress_frame = ttk.Frame(control_panel)
        progress_frame.pack(fill='x', pady=10, padx=5)
        self.ica_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.ica_progress.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.ica_progress_percent = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.ica_progress_percent, style='Body.TLabel').pack(side='right')

        # --- display and selection area ---
        main_pane = ttk.PanedWindow(ica_frame, orient=tk.VERTICAL)
        main_pane.pack(fill='both', expand=True, pady=10)

        # Upper frame for the plots
        self.ica_viz_frame = ttk.LabelFrame(main_pane, text="ICA Visualization (Click a map to see properties)", style='Card.TLabelframe', padding=5)
        main_pane.add(self.ica_viz_frame, weight=4)

        # Lower frame for the checkboxes
        self.ica_selection_frame = ttk.LabelFrame(main_pane, text="Component Selection", style='Card.TLabelframe', padding=10)
        main_pane.add(self.ica_selection_frame, weight=1)


    def on_ica_subject_change(self, event=None):
        """
        Called when a new subject is selected. Shows the component grid.
        """
        subject_id = self.selected_ica_subject.get()
        if not subject_id:
            return
        self.show_ica_maps_grid()

    

    def select_files(self):
        """Handles the selection of individual BrainVision files."""
        file_paths = filedialog.askopenfilenames(filetypes=[("BrainVision Files", "*.vhdr")])
        if file_paths:
            self.file_listbox.delete(0, tk.END)
            self.uploaded_files.clear()
            self.all_uploaded_files.clear()
            self.subject_ids.clear()

            for fp in file_paths:
                self.all_uploaded_files.append(fp)
                self.uploaded_files.append(fp)
                self.file_listbox.insert(tk.END, os.path.basename(fp))
            
            self.subject_ids = [os.path.basename(fp).split('.')[0] for fp in self.uploaded_files]
            self.status_var.set(f"{len(file_paths)} files successfully loaded. Ready for filtering.")
            self.update_subject_lists() 

    def select_directory(self):
        """Handles the recursive selection of BrainVision files from a directory."""
        dir_path = filedialog.askdirectory()
        if dir_path:
            found_files = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.vhdr'):
                        found_files.append(os.path.join(root, file))
            
            if found_files:
                self.file_listbox.delete(0, tk.END)
                self.uploaded_files.clear()
                self.all_uploaded_files.clear()
                self.subject_ids.clear()

                for fp in found_files:
                    self.all_uploaded_files.append(fp)
                    self.uploaded_files.append(fp)
                    self.file_listbox.insert(tk.END, os.path.basename(fp))
                
                self.subject_ids = [os.path.basename(fp).split('.')[0] for fp in self.uploaded_files]
                self.status_var.set(f"{len(found_files)} .vhdr files found. Ready for filtering.")
                self.update_subject_lists()
            else:
                self.status_var.set("No .vhdr files were found in the selected directory.")

    def load_configuration(self):
        config_file = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.low_freq.set(config.get('low_freq', self.low_freq.get()))
                self.high_freq.set(config.get('high_freq', self.high_freq.get()))
                self.notch_freq.set(config.get('notch_freq', self.notch_freq.get()))
                self.n_components.set(config.get('n_components', self.n_components.get()))
                self.ica_method.set(config.get('ica_method', self.ica_method.get()))
                self.freq_range_min.set(config.get('freq_range_min', self.freq_range_min.get()))
                self.freq_range_max.set(config.get('freq_range_max', self.freq_range_max.get()))
                self.aperiodic_mode.set(config.get('aperiodic_mode', self.aperiodic_mode.get()))
                self.status_var.set("Configuration successfully loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
                self.status_var.set("Error occurred while loading configuration.")

        def save_configuration(self):
            config_data = {
                'low_freq': self.low_freq.get(),
                'high_freq': self.high_freq.get(),
                'notch_freq': self.notch_freq.get(),
                'n_components': self.n_components.get(),
                'ica_method': self.ica_method.get(),
                'freq_range_min': self.freq_range_min.get(),
                'freq_range_max': self.freq_range_max.get(),
                'aperiodic_mode': self.aperiodic_mode.get()
            }
            config_file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")], initialfile=self.config_name.get())
            if config_file:
                try:
                    with open(config_file, 'w') as f:
                        json.dump(config_data, f, indent=4)
                    self.status_var.set("Configuration successfully saved.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save configuration: {e}")
                    self.status_var.set("Error occurred while saving configuration.")

    def __init__(self):
        self.root = ttk.Window(themename="litera")
        self.debug = False
        
        self.colors = {
            'primary': '#0078d4', 'secondary': '#6264a7', 'accent': '#f7630c',
            'success': '#107c10', 'background': '#ffffff', 'surface': '#ffffff',
            'text_primary': '#000000', 'text_secondary': '#666666', 'border': '#cccccc'
        }
        
        self.setup_main_window()
        self.setup_styles()
        self.setup_variables()
        self.progress_queue = multiprocessing.Queue()
        self.create_widgets()
        self.batch_formulas = []
        self.graph_theory_results = {}         
        self.center_window()
        self.evaluation_results = [] # Holds the results of the models
        
        self.valid_subject_prefixes = [
            "sub-87999321", "sub-88000181", "sub-88000313", "sub-88000489", "sub-88000533",
            "sub-88001661", "sub-88002789", "sub-88003869", "sub-88004853", "sub-88005849",
            "sub-88005941", "sub-88005985", "sub-88006161", "sub-88006209", "sub-88006477",
            "sub-88008321", "sub-88008681", "sub-88009901", "sub-88010033", "sub-88010709",
            "sub-88010753", "sub-88010929", "sub-88010981", "sub-88011333", "sub-88011833",
            "sub-88012053", "sub-88012461", "sub-88012817", "sub-88013177", "sub-88013813",
            "sub-88013905", "sub-88017137", "sub-88017409", "sub-88017765", "sub-88019569",
            "sub-88020381", "sub-88020557", "sub-88020737", "sub-88020917", "sub-88021101",
            "sub-88021321", "sub-88022089", "sub-88022133", "sub-88022765", "sub-88023125",
            "sub-88023485", "sub-88023529", "sub-88024697", "sub-88024789", "sub-88025057",
            "sub-88025061", "sub-88025281", "sub-88025685", "sub-88026005", "sub-88026321",
            "sub-88026409", "sub-88026769", "sub-88026857", "sub-88027173", "sub-88028253",
            "sub-88029645", "sub-88029777", "sub-88029789", "sub-88029833", "sub-88030549",
            "sub-88030641", "sub-88032621", "sub-88033201", "sub-88034645", "sub-88035049",
            "sub-88035501", "sub-88035589", "sub-88035677", "sub-88036037", "sub-88036217",
            "sub-88038069", "sub-88039057", "sub-88039417", "sub-88039773", "sub-88041397",
            "sub-88041665", "sub-88042749", "sub-88042837", "sub-88043021", "sub-88043065",
            "sub-88043381", "sub-88043873", "sub-88044501", "sub-88044545", "sub-88044681",
            "sub-88045085", "sub-88045353", "sub-88045713", "sub-88045809", "sub-88046257",
            "sub-88046437", "sub-88046665", "sub-88047245", "sub-88047789", "sub-88048817",
            "sub-88049405", "sub-88049537", "sub-88049857", "sub-88049905", "sub-88050713",
            "sub-88052013", "sub-88052061", "sub-88052329", "sub-88052465", "sub-88052869",
            "sub-88053317", "sub-88053997", "sub-88054313", "sub-88054357", "sub-88054577",
            "sub-88055749", "sub-88056017", "sub-88057913", "sub-88058229", "sub-88058993",
            "sub-88059169"
        ]

    def create_widgets(self):
        """Build and lay out every widget of the GUI."""
        main_container = ttk.Frame(self.root, style='Card.TFrame')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        self.create_status_bar(main_container)
        
        ttk.Label(main_container, text="EEG Analysis Pipeline", style='Title.TLabel').pack(pady=(0, 20))
        
        self.notebook = ttk.Notebook(main_container, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)

        self.create_upload_tab()
        self.create_subject_filter_tab()
        self.create_session_filter_tab() 
        self.create_condition_filter_tab() 
        self.create_preprocessing_tab()
        self.create_ica_tab()
        self.create_spectral_tab()
        self.create_fooof_tab()
        self.create_connectivity_tab()
        self.create_graph_theory_tab() 
        self.create_regression_tab()
        self.create_modeling_tab()
        self.create_summary_tab()
        self.create_results_tab()
        
        

    def create_status_bar(self, parent):
        """Build the status bar at the bottom of the window."""
        status_frame = ttk.Frame(parent, style='Card.TFrame')
        # .pack() with side='bottom' attaches it to the bottom of its parent.
        status_frame.pack(side='bottom', fill='x', pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Body.TLabel')
        self.status_label.pack(side='left')




    def _show_ica_checkboxes_only(self):
        self.ica_maps_displayed = None
        """Display checkboxes for ICA component selection without re-plotting maps"""
        subject_id = self.selected_ica_subject.get()
        if not subject_id or subject_id not in self.ica_results:
            messagebox.showwarning("Warning", "Please select a valid subject")
            return

        labels = self.ica_results[subject_id]['labels']['labels']

        # Clear entire component_frame before redrawing
        for widget in self.component_frame.winfo_children():
            widget.destroy()

        # Optional: add a placeholder message or canvas
        placeholder = ttk.Label(self.component_frame, text=f"ICA maps for {subject_id} already shown above or in previous step.", style='Body.TLabel')
        placeholder.pack(pady=10)

        # New checkbox section
        checkbox_frame = ttk.Frame(self.component_frame, style='Card.TFrame')
        checkbox_frame.pack(fill='x', pady=10)

        ttk.Label(checkbox_frame, text=f"Please select the components to be excluded for subject {subject_id}:", style='Heading.TLabel').pack(pady=5)

        if subject_id not in self.excluded_components:
            self.excluded_components[subject_id] = []

        for i, label in enumerate(labels):
            var = tk.BooleanVar(value=i in self.excluded_components[subject_id])
            chk = ttk.Checkbutton(
                checkbox_frame,
                text=f"Component {i}: {label}",
                variable=var,
                style='Excluded.TCheckbutton' if var.get() else 'TCheckbutton',
                command=lambda idx=i, v=var: self._toggle_component_exclusion(subject_id, idx, v)
            )
            chk.pack(anchor='w', padx=20)


    def show_ica_maps_grid(self, event=None):
        """
        Draw a grid of ICA maps and handle the clicks by computing the position,
        so that the component is identified beyond doubt.
        """
        subject_id = self.selected_ica_subject.get()
        if not subject_id or subject_id not in self.ica_results:
            return

        for widget in self.ica_viz_frame.winfo_children():
            widget.destroy()
        for widget in self.ica_selection_frame.winfo_children():
            widget.destroy()

        try:
            ica = self.ica_results[subject_id]['ica']
            raw = self.preprocessed_data[subject_id]
            
            # --- new click logic ---
            def on_mouse_click(event):
                # Ignore clicks that fall outside a drawing area
                if event.inaxes is None:
                    return
                
                # The axis on which the click landed
                clicked_ax = event.inaxes
                title = clicked_ax.get_title()
                
                if title and title.startswith("ICA"):
                    try:
                        component_idx = int(title.replace("ICA", ""))
                        print(f"--- CLICK DETECTED --- title: '{title}', index: {component_idx}")
                        self.plot_selected_ica_component(component_idx=component_idx)
                    except (ValueError, TypeError):
                        print(f"!!! ERROR: could not read the index from the title '{title}'")
            # --- end of the new logic ---

            figures = ica.plot_components(inst=raw, show=False, picks=None)
            fig = figures[0] if isinstance(figures, list) else figures
            fig.set_size_inches(12, 8)
            
            # Bind the mouse click event to our new function
            fig.canvas.mpl_connect('button_press_event', on_mouse_click)
            
            canvas = FigureCanvasTkAgg(fig, master=self.ica_viz_frame)
            canvas.draw()
            
            toolbar_frame = ttk.Frame(self.ica_viz_frame)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # --- the rest of the function, handling the checkboxes, is unchanged ---
            labels = self.ica_results[subject_id]['labels']['labels']
            if subject_id not in self.excluded_components:
                self.excluded_components[subject_id] = [i for i, label in enumerate(labels) if label not in ['brain', 'other']]

            def reorder_checkboxes(event):
                for widget in self.ica_selection_frame.winfo_children():
                    widget.destroy()
                
                estimated_width = tkFont.Font().measure("00: muscle artifact") + 30 
                num_cols = max(1, event.width // estimated_width)
                
                for i, label in enumerate(labels):
                    row, col = divmod(i, num_cols)
                    var = tk.BooleanVar(value=i in self.excluded_components.get(subject_id, []))
                    chk = ttk.Checkbutton(
                        self.ica_selection_frame,
                        text=f"{i:02d}: {label}",
                        variable=var,
                        command=lambda idx=i, v=var: self._toggle_component_exclusion(subject_id, idx, v)
                    )
                    chk.grid(row=row, column=col, sticky='w', padx=5, pady=1)
            
            self.ica_selection_frame.bind("<Configure>", reorder_checkboxes)
            self.ica_selection_frame.update_idletasks()
            reorder_checkboxes(type('event', (), {'width': self.ica_selection_frame.winfo_width()})())

        except Exception as e:
            messagebox.showerror("Display error", f"Could not display the ICA maps: {e}")


    def _show_excluded_ica_viz(self):
        """
        Show the maps of only those components currently on the exclusion list,
        drawing them directly inside the GUI.
        """
        subject_id = self.selected_ica_subject.get()
        if not subject_id or subject_id not in self.ica_results:
            messagebox.showwarning("Warning", "Select a valid subject.")
            return

        # Clear only the display frame, leaving the checkboxes untouched
        for widget in self.ica_viz_frame.winfo_children():
            widget.destroy()

        excluded_indices = self.excluded_components.get(subject_id, [])
        if not excluded_indices:
            # Add a button to go back even when there is no component
            back_button = ttk.Button(self.ica_viz_frame, text="< Back to All Component Maps", style="Link.TButton", command=self.show_ica_maps_grid)
            back_button.pack(anchor='nw', pady=5, padx=5)
            ttk.Label(self.ica_viz_frame, text="No component was excluded for this subject.").pack(pady=20)
            return

        try:
            ica = self.ica_results[subject_id]['ica']
            raw = self.preprocessed_data[subject_id]
            
            # Add the 'Back' button before drawing the plots
            back_button = ttk.Button(self.ica_viz_frame, text="< Back to All Component Maps", style="Link.TButton", command=self.show_ica_maps_grid)
            back_button.pack(anchor='nw', pady=5, padx=5)

            # Generate the figures of the excluded components
            figures = ica.plot_components(inst=raw, picks=excluded_indices, show=False)
            
            if not isinstance(figures, list):
                figures = [figures]

            for fig in figures:
                fig.set_size_inches(12, 8)
                canvas = FigureCanvasTkAgg(fig, master=self.ica_viz_frame)
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, self.ica_viz_frame)
                toolbar.update()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Display error", f"Could not display the excluded components: {e}")


    def _prepare_xy(self, X_df, y_series):
        """
        Prepare the X and y matrices for the logistic regression.
        - Standardise (z-score) the numeric predictors.
        - Convert the target to binary 0/1.
        """
        # 1. Convert the target to binary 0/1
        y = y_series.copy()
        if pd.api.types.is_categorical_dtype(y) or y.dtype == object:
            y = y.astype(str).str.strip().str.lower().map({
                "nonresponder": 0, "non-responder": 0, "no": 0, "0": 0, "false": 0,
                "responder": 1, "yes": 1, "1": 1, "true": 1
            }).fillna(-1).astype(int) # Fill the unmapped entries and convert
        
        if not y.isin([0, 1]).all():
                raise ValueError("The target column 'Responder' holds values that cannot be converted to 0 or 1.")
        
        # 2. Standardise (z-score) the continuous variables in X
        X = X_df.copy()
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        numeric_cols = X.select_dtypes(include=np.number).columns
        # Do not standardise the intercept, if present
        numeric_cols = [col for col in numeric_cols if col.lower() != 'intercept']
        
        if numeric_cols:
            X.loc[:, numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X, y



    def create_spectral_tab(self):
        """Create the spectral analysis tab"""
        spectral_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(spectral_frame, text="Spectral Analysis")
        
        # --- upper control panel ---
        control_panel = self.create_section(spectral_frame, "Spectral Analysis Control")
        
        # --- start of the change ---
        # Rather than a separate button_frame, the button goes straight into control_panel
        # and the pack options are used to centre it.
        
        # Analysis button, centred
        ttk.Button(control_panel, text="Run Spectral Analysis", style='Primary.TButton', command=self.run_spectral_analysis).pack(pady=10)
        # --- end of the change ---
        
        # Progress bar with a percentage
        progress_frame = ttk.Frame(control_panel, style='Card.TFrame')
        progress_frame.pack(fill='x', pady=10)
        ttk.Label(progress_frame, text="Progress:", style='Body.TLabel').pack(side='left')
        self.spectral_progress = ttk.Progressbar(progress_frame, mode='indeterminate', style='Custom.Horizontal.TProgressbar')
        self.spectral_progress.pack(side='left', fill='x', expand=True, padx=(10, 5))
        self.spectral_progress_percent = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.spectral_progress_percent, style='Body.TLabel').pack(side='right')

        # --- results display section ---
        results_section = self.create_section(spectral_frame, "Biomarkers & PSD")
        
        # --- subject selector for the display ---
        selector_frame = ttk.Frame(results_section, style='Card.TFrame')
        selector_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(selector_frame, text="Select Subject:", style='Body.TLabel').pack(side='left', padx=(0, 10))
        self.selected_spectral_subject = tk.StringVar()
        self.spectral_subject_dropdown = ttk.Combobox(selector_frame, textvariable=self.selected_spectral_subject, state='readonly')
        self.spectral_subject_dropdown.pack(side='left', padx=(0, 10))
        ttk.Button(selector_frame, text="Show Subject Data", style='Secondary.TButton', command=self._update_spectral_viz).pack(side='left')

        # --- area for the plot and the toolbar ---
        self.spectral_fig = Figure(figsize=(12, 8), facecolor=self.colors['surface'])
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_fig, results_section)
        self.spectral_canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar_frame = ttk.Frame(results_section, style='Card.TFrame')
        toolbar_frame.pack(fill='x')
        self.spectral_toolbar = NavigationToolbar2Tk(self.spectral_canvas, toolbar_frame)


        
    def create_fooof_tab(self):
        """Create the FOOOF analysis tab with subject selector."""
        fooof_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(fooof_frame, text="FOOOF Analysis")
        
        # --- control panel ---
        control_panel = self.create_section(fooof_frame, "FOOOF Control")
        ttk.Button(control_panel, text="Run FOOOF Analysis", style='Primary.TButton', command=self.run_fooof_analysis).pack(pady=10)
        # --- progress bar section of create_fooof_tab ---

        # Progress bar with percentage
        progress_frame = ttk.Frame(control_panel, style='Card.TFrame')
        progress_frame.pack(fill='x', pady=10)
        
        ttk.Label(progress_frame, text="Progress:", style='Body.TLabel').pack(side='left')
        # Inside the create_fooof_tab method
        self.fooof_progress = ttk.Progressbar(progress_frame, mode='indeterminate', style='Custom.Horizontal.TProgressbar')

        self.fooof_progress.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        # Percentage label
        ttk.Label(progress_frame, textvariable=self.fooof_progress_percent, style='Body.TLabel').pack(side='right')

        
        # --- results section ---
        results_section = self.create_section(fooof_frame, "FOOOF Results")
        
        # --- subject selector ---
        selector_frame = ttk.Frame(results_section, style='Card.TFrame')
        selector_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(selector_frame, text="Select Subject:", style='Body.TLabel').pack(side='left', padx=(0, 10))
        
        self.selected_fooof_subject = tk.StringVar()
        self.fooof_subject_dropdown = ttk.Combobox(selector_frame, textvariable=self.selected_fooof_subject, state='readonly')
        self.fooof_subject_dropdown.pack(side='left', padx=(0, 10))
        
        # Button that refreshes the plot
        ttk.Button(selector_frame, text="Show Subject FOOOF", style='Secondary.TButton', command=self._update_fooof_viz).pack(side='left')

        # --- container frame for the plot ---
        self.fooof_results_frame = ttk.Frame(results_section, style='Card.TFrame')
        self.fooof_results_frame.pack(fill='both', expand=True)


    
        
    def create_results_tab(self):
        """Create the results and export tab"""
        results_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(results_frame, text="Results & Export")
        control_panel = self.create_section(results_frame, "Export Control")
        export_frame = ttk.Frame(control_panel, style='Card.TFrame')
        export_frame.pack(fill='x', pady=10)
        ttk.Button(export_frame, text="Export All Results", style='Primary.TButton', command=self.export_all_results).pack(side='left', padx=(0, 10))
        ttk.Button(export_frame, text="Generate Report", style='Secondary.TButton', command=self.generate_report).pack(side='left')
        summary_section = self.create_section(results_frame, "Analysis Summary")
        self.create_results_summary(summary_section)


        
    def create_section(self, parent, title):
        """Build a section with a title and a frame"""
        section_frame = ttk.Frame(parent, style='Card.TFrame')
        section_frame.pack(fill='x', pady=10, padx=10)
        
        # Title label, left aligned
        ttk.Label(section_frame, text=title, style='Heading.TLabel').pack(anchor='w', pady=(0, 10))
        
        content_frame = ttk.Frame(section_frame, style='Card.TFrame')
        content_frame.pack(fill='both', expand=True)
        
        return content_frame
        

    def create_param_row(self, container, text, variable, from_, to, is_float=True, round_to_int=False):
        row = ttk.Frame(container, style='Card.TFrame')
        row.pack(fill='x', pady=4, padx=5)

        ttk.Label(row, text=text, width=15, style='Param.TLabel').pack(side='left', anchor='w')
        entry = ttk.Entry(row, width=8, justify='right')
        entry.pack(side='left', padx=(0, 5))

        def on_scale_move(value_str):
            value = float(value_str)
            if round_to_int:
                variable.set(round(value))
            else:
                variable.set(value)

        scale = ttk.Scale(row, from_=from_, to=to, variable=variable, orient='horizontal', command=on_scale_move)
        scale.pack(side='left', expand=True, fill='x')

        def update_from_scale(value_str):
            value = float(value_str)
            formatted_value = f"{value:.2f}" if is_float else f"{int(value)}"
            entry.delete(0, tk.END)
            entry.insert(0, formatted_value)

        def update_from_entry(event=None):
            try:
                value = float(entry.get())
                value = max(from_, min(to, value))
                if round_to_int:
                    variable.set(round(value))
                else:
                    variable.set(value)
            except (ValueError, tk.TclError):
                update_from_scale(variable.get())

        variable.trace_add('write', lambda *args: update_from_scale(variable.get()))
        entry.bind("<FocusOut>", update_from_entry)
        entry.bind("<Return>", update_from_entry)
        update_from_scale(variable.get())

        self.param_widgets[text] = {'entry': entry, 'scale': scale}
        return row

        # --- building the sections and the parameter rows ---
        
        filter_frame = ttk.LabelFrame(parent, text="Filtering", style='Card.TLabelframe', padding=10)
        filter_frame.pack(fill='x', pady=5, padx=5)
        self.create_param_row(filter_frame, "High-pass (Hz):", self.low_freq, 0.1, 5.0)
        self.create_param_row(filter_frame, "Low-pass (Hz):", self.high_freq, 30.0, 100.0, is_float=False, round_to_int=True)

        
        notch_frame = ttk.Frame(filter_frame, style='Card.TFrame')
        notch_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(notch_frame, text="Notch (Hz):", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        ttk.Combobox(notch_frame, textvariable=self.notch_freq, values=[50.0, 60.0], state='readonly', width=10).pack(side='left')

        ica_frame = ttk.LabelFrame(parent, text="ICA", style='Card.TLabelframe', padding=10)
        ica_frame.pack(fill='x', pady=5, padx=5)
        self.create_param_row(ica_frame, "Components:", self.n_components, 10, 50, is_float=False, round_to_int=True)

        
        method_frame = ttk.Frame(ica_frame, style='Card.TFrame')
        method_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(method_frame, text="Method:", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        ttk.Combobox(method_frame, textvariable=self.ica_method, values=['infomax', 'fastica', 'picard'], state='readonly', width=10).pack(side='left')

        fooof_frame = ttk.LabelFrame(parent, text="FOOOF", style='Card.TLabelframe', padding=10)
        fooof_frame.pack(fill='x', pady=5, padx=5)
        
        freq_range_frame = ttk.Frame(fooof_frame, style='Card.TFrame')
        freq_range_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(freq_range_frame, text="Freq Range (Hz):", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        
        min_entry = ttk.Entry(freq_range_frame, width=6, justify='right')
        min_entry.pack(side='left', padx=(0, 5))
        min_scale = ttk.Scale(freq_range_frame, from_=1, to=10, variable=self.freq_range_min, orient='horizontal')
        min_scale.pack(side='left', expand=True, fill='x')
        
        ttk.Label(freq_range_frame, text="to", style='Param.TLabel').pack(side='left', padx=5)
        
        max_entry = ttk.Entry(freq_range_frame, width=6, justify='right')
        max_entry.pack(side='left', padx=(0, 5))
        max_scale = ttk.Scale(freq_range_frame, from_=30, to=100, variable=self.freq_range_max, orient='horizontal')
        max_scale.pack(side='left', expand=True, fill='x')

        # --- start of the previously missing block ---
        # Synchronisation logic of the MINIMUM slider
        def update_min_from_scale(val_str):
            min_entry.delete(0, tk.END)
            min_entry.insert(0, f"{int(float(val_str))}")

        def update_min_from_entry(event=None):
            try:
                value = int(min_entry.get())
                self.freq_range_min.set(value)
            except (ValueError, tk.TclError):
                update_min_from_scale(self.freq_range_min.get())

        self.freq_range_min.trace_add('write', lambda *args: update_min_from_scale(self.freq_range_min.get()))
        min_entry.bind("<Return>", update_min_from_entry)
        min_entry.bind("<FocusOut>", update_min_from_entry)
        update_min_from_scale(self.freq_range_min.get()) # Set the initial value

        # Synchronisation logic of the MAXIMUM slider
        def update_max_from_scale(val_str):
            max_entry.delete(0, tk.END)
            max_entry.insert(0, f"{int(float(val_str))}")

        def update_max_from_entry(event=None):
            try:
                value = int(max_entry.get())
                self.freq_range_max.set(value)
            except (ValueError, tk.TclError):
                update_max_from_scale(self.freq_range_max.get())

        self.freq_range_max.trace_add('write', lambda *args: update_max_from_scale(self.freq_range_max.get()))
        max_entry.bind("<Return>", update_max_from_entry)
        max_entry.bind("<FocusOut>", update_max_from_entry)
        update_max_from_scale(self.freq_range_max.get()) # Set the initial value
        # --- end of the previously missing block ---

        
        ap_frame = ttk.Frame(fooof_frame, style='Card.TFrame')
        ap_frame.pack(fill='x', pady=4, padx=5)
        ttk.Label(ap_frame, text="Aperiodic Mode:", width=15, style='Param.TLabel').pack(side='left', anchor='w')
        ttk.Combobox(ap_frame, textvariable=self.aperiodic_mode, values=['fixed', 'knee'], state='readonly', width=10).pack(side='left')


    def evaluate_formula_with_nested_cv(self, formula: str, model_label: str) -> Optional[Dict]:
        """
        Run the nested CV pipeline on a given formula and return the results.
        """
        try:
            # 1. Prepare the data (logic reused)
            y_col = formula.split('~')[0].strip()
            use_xgb = self.model_use_xgb.get()
            n_outer = self.model_outer_folds.get()
            n_inner = self.model_inner_folds.get()

            beh_df = self.behavioral_data.rename(columns={'sessID': 'session_ID'})
            beh_df['participant_ID'] = beh_df['participant_ID'].astype(str)
            beh_df['session_ID'] = pd.to_numeric(beh_df['session_ID'], errors='coerce')
            eeg_df = self.biomarkers_df.copy()
            eeg_df['participant_ID'] = eeg_df['participant_ID'].astype(str)
            eeg_df['session_ID'] = pd.to_numeric(eeg_df['session_ID'], errors='coerce')
            
            # Add subject_id for the export
            eeg_df['subject_id_export'] = eeg_df['Subject']
            
            df = pd.merge(beh_df, eeg_df, on=['participant_ID', 'session_ID'], how='inner')
            
            if df.empty: raise ValueError("Merging the data produced an empty DataFrame.")
            if y_col not in df.columns: raise ValueError(f"The target column '{y_col}' was not found.")

            # 2. Build the design matrices and obtain the right indices
            y_patsy, X_patsy = patsy.dmatrices(formula, data=df, return_type='dataframe', NA_action='drop')
            
            # Keep the subject IDs and the real data of the rows patsy actually used
            final_subjects = df.loc[y_patsy.index, 'subject_id_export'].values
            y_true = y_patsy.values.ravel()

            # 3. Run the nested CV, reusing the existing function
            self.root.after(0, lambda: self.status_var.set(f"Running nested CV for: {model_label}..."))
            cv_results = self.nested_cv_auc(X_patsy, y_patsy, use_xgb, n_outer, n_inner)
            y_pred_proba = cv_results['oof_cal_pred']

            # 4. Compute the AUC with a bootstrap confidence interval
            auc_mean, (auc_lo, auc_hi), _, _p_one = self.delta_auc_boot(y_true, np.zeros_like(y_true), y_pred_proba)

            # 5. Return the structured dictionary
            return {
                'y_true': y_true,
                'y_pred_proba': y_pred_proba,
                'subject_ids': final_subjects,
                'auc_mean': auc_mean,
                'auc_ci': (auc_lo, auc_hi),
                'brier': cv_results['brier'],
                'model_label': model_label,
                'formula': formula
            }

        except Exception as e:
            messagebox.showerror("Evaluation error", f"Could not evaluate the formula:\n{formula}\n\nErrore: {e}")
            self.root.after(0, lambda: self.status_var.set("Error during the evaluation."))
            import traceback
            traceback.print_exc()
            return None

    def update_evaluation_plots(self):
        """
        Updates the ROC and Calibration plots. The calibration plot now includes
        a histogram of predicted probabilities.
        """
        from sklearn.metrics import roc_curve
        from sklearn.calibration import calibration_curve
        
        # --- 1. ROC plot refresh (unchanged) ---
        self.roc_ax.clear()
        for result in self.evaluation_results:
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
            label = f"{result['model_label']}: AUC = {result['auc_mean']:.3f} ({result['auc_ci'][0]:.3f}-{result['auc_ci'][1]:.3f})"
            self.roc_ax.plot(fpr, tpr, label=label)

        self.roc_ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
        self.roc_ax.set_xlabel('False Positive Rate')
        self.roc_ax.set_ylabel('True Positive Rate')
        self.roc_ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        self.roc_ax.legend(loc='lower right', fontsize='small')
        self.roc_ax.grid(True, linestyle='--', alpha=0.6)
        self.roc_fig.tight_layout()
        self.roc_canvas.draw()

        # --- 2. Calibration plot refresh, with histogram ---
        self.cal_fig.clear()
        
        # Two subplots: one for the curve, one for the histogram.
        # 'sharex=True' ties their x axes; 'gridspec_kw' gives the upper plot more room.
        ax1, ax2 = self.cal_fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        for result in self.evaluation_results:
            y_true = result['y_true']
            y_pred = result['y_pred_proba']
            model_label = result['model_label']
            brier_score = result['brier']

            # Compute the points of the calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
            
            # Draw the calibration curve on the upper axis (ax1)
            brier_label = f"{model_label} (Brier: {brier_score:.3f})"
            ax1.plot(prob_pred, prob_true, marker='o', linestyle='-', label=brier_label)

            # Draw the histogram of predicted probabilities on the lower axis (ax2)
            ax2.hist(y_pred, range=(0, 1), bins=10, label=model_label, histtype="step", lw=2)

        # --- finalise the calibration plot (ax1) ---
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve & Predicted Probability Histogram')
        ax1.legend(loc='upper left', fontsize='small')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # --- finalise the histogram (ax2) ---
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.legend(loc='upper right', fontsize='small')
        ax2.grid(True, linestyle='--', alpha=0.6)

        self.cal_fig.tight_layout()
        self.cal_canvas.draw()


    def clear_evaluation_plots(self):
        """Clears the evaluation results and plots."""
        self.evaluation_results = []
        self.roc_ax.clear()
        self.cal_ax.clear()
        self.roc_ax.text(0.5, 0.5, "Run a model evaluation to see the ROC curve.", ha='center', va='center', style='italic', color='grey')
        self.cal_ax.text(0.5, 0.5, "Run a model evaluation to see the calibration plot.", ha='center', va='center', style='italic', color='grey')
        self.roc_ax.plot([0, 1], [0, 1], 'k--') # Add chance line back
        self.cal_ax.plot([0, 1], [0, 1], 'k--') # Add perfect calibration line back
        self.roc_canvas.draw()
        self.cal_canvas.draw()
        self.status_var.set("Evaluation plots and results have been cleared.")


    def export_evaluation_results(self):
        """
        Exports the evaluation results to two separate CSV files:
        1. A summary file with aggregated metrics per model.
        2. A detailed file with out-of-fold (OOF) predictions per subject.
        """
        if not self.evaluation_results:
            messagebox.showwarning("No Results", "There are no evaluation results to export. Please run a model first.")
            return

        folder_path = filedialog.askdirectory(title="Select a Folder to Save CSV Files")
        if not folder_path:
            return

        try:
            # --- 1. Assemble the DataFrame of out-of-fold predictions per subject ---
            oof_predictions_list = []
            for result in self.evaluation_results:
                # Make sure every array has the same length
                num_subjects = len(result['y_true'])
                for i in range(num_subjects):
                    oof_predictions_list.append({
                        'subject_id': result['subject_ids'][i],
                        'y_true': result['y_true'][i],
                        'y_pred_proba': result['y_pred_proba'][i],
                        'model_formula': result['formula'],
                        'model_label': result['model_label']
                    })
            
            oof_df = pd.DataFrame(oof_predictions_list)

            # --- 2. Assemble the DataFrame of summary metrics ---
            summary_metrics_list = []
            for result in self.evaluation_results:
                summary_metrics_list.append({
                    'model_label': result['model_label'],
                    'model_formula': result['formula'],
                    'AUC_mean': result['auc_mean'],
                    'AUC_CI_low': result['auc_ci'][0],
                    'AUC_CI_high': result['auc_ci'][1],
                    'Brier_score': result['brier']
                })
            
            summary_df = pd.DataFrame(summary_metrics_list)

            # --- 3. Write the two CSV files ---
            oof_path = os.path.join(folder_path, "evaluation_oof_predictions.csv")
            summary_path = os.path.join(folder_path, "evaluation_summary_metrics.csv")
            
            oof_df.to_csv(oof_path, index=False)
            summary_df.to_csv(summary_path, index=False)

            messagebox.showinfo("Export Successful", 
                                f"Two files were saved successfully in:\n{folder_path}\n\n"
                                "1. evaluation_oof_predictions.csv\n"
                                "2. evaluation_summary_metrics.csv")
            self.status_var.set("Evaluation results exported successfully.")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save files:\n{e}")
            import traceback
            traceback.print_exc()


    def run_custom_formula_evaluation(self):
        """Runs the evaluation for the user-entered formula."""
        formula = self.custom_formula_var.get().strip()
        if not formula or '~' not in formula:
            messagebox.showwarning("Invalid Formula", "Please enter a valid Patsy-style formula.")
            return
        
        model_label = f"CustomModel_{len(self.evaluation_results) + 1}"
        
        threading.Thread(
            target=self._run_and_update_evaluation,
            args=(formula, model_label),
            daemon=True
        ).start()


    def _run_and_update_evaluation(self, formula: str, model_label: str):
        """
        Worker that runs the evaluation, and then updates both the plots and the text report in the GUI.
        """
        # Run the model analysis
        result = self.evaluate_formula_with_nested_cv(formula, model_label)
        
        if result:
            # Add the results to the list used for the plots and the export
            self.evaluation_results.append(result)
            
            # Format the text report for this particular result
            report_text = self._format_evaluation_report(result)
            
            # --- GUI refresh, carried out in the main thread ---
            
            # 1. Refresh the plots (ROC and calibration)
            self.root.after(0, self.update_evaluation_plots)
            
            # 2. Insert the new text report at the top of the text widget
            self.root.after(0, lambda: self.modeling_results_text.insert('1.0', report_text))
            
            # 3. Refresh the status bar
            self.root.after(0, lambda: self.status_var.set(f"Evaluation complete for: {model_label}"))


            
    def create_component_selection(self, parent):
        """Create ICA component selection interface"""
        self.component_frame = ttk.Frame(parent, style='Card.TFrame')
        self.component_frame.pack(fill='both', expand=True, pady=10)
        update_frame = ttk.Frame(parent, style='Card.TFrame')
        update_frame.pack(fill='x', pady=5)
        ttk.Button(update_frame, text="Update Component Exclusions", style='Primary.TButton', command=self.update_component_exclusions).pack()
        
    def create_spectral_results_display(self, parent):
        """Create spectral analysis results display"""
        self.spectral_fig = Figure(figsize=(12, 8), facecolor=self.colors['surface'])
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_fig, parent)
        self.spectral_canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar_frame = ttk.Frame(parent, style='Card.TFrame')
        toolbar_frame.pack(fill='x')
        self.spectral_toolbar = NavigationToolbar2Tk(self.spectral_canvas, toolbar_frame)
        
    def create_fooof_results_display(self, parent):
        """Prepare the container for the FOOOF results."""
        # Only the container frame is built here. The canvas and the figure are created or refreshed after the analysis.
        self.fooof_results_frame = ttk.Frame(parent, style='Card.TFrame')
        self.fooof_results_frame.pack(fill='both', expand=True)
        
    def create_regression_results_display(self, parent):
        """Build a text area in which to show the regression results."""
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.regression_results_text = tk.Text(
            text_frame, 
            wrap=tk.WORD,
            height=15, 
            bg=self.colors['surface'], 
            fg=self.colors['text_primary'],
            font=("Courier", 10)
        )
        
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.regression_results_text.yview)
        self.regression_results_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        self.regression_results_text.pack(side='left', fill='both', expand=True)

    def update_analysis_summary(self):
        """Refresh the Analysis Summary section with every result."""
        if not hasattr(self, 'summary_text'):
            return
            
        self.summary_text.delete('1.0', tk.END)
        
        summary_content = "=== EEG ANALYSIS - SUMMARY OF RESULTS ===\n\n"
        
        # Information about the files processed
        summary_content += f"Files processed: {len(self.uploaded_files)}\\n"
        summary_content += f"Subjects analysed: {len(set([os.path.basename(f).split('_')[0] for f in self.uploaded_files]))}\\n\\n"
        
        # Preprocessing results
        if self.preprocessed_data:
            summary_content += f"PREPROCESSING FINISHED:\\n"
            summary_content += f"- Preprocessed data: {len(self.preprocessed_data)} subjects\\n\\n"
        
        # ICA results
        if self.ica_results:
            summary_content += f"ICA FINISHED:\\n"
            summary_content += f"- ICA analysis: {len(self.ica_results)} subjects\\n\\n"
        
        # Spectral results
        if self.spectral_results:
            summary_content += f"SPECTRAL ANALYSIS FINISHED:\\n"
            summary_content += f"- Biomarkers computed: {len(self.spectral_results)} subjects\\n\\n"
        
        # FOOOF results
        if self.fooof_results:
            summary_content += f"FOOOF FINISHED:\\n"
            summary_content += f"- FOOOF analysis: {len(self.fooof_results)} subjects\\n\\n"
        
        # Regression results
        if hasattr(self, 'regression_results_text') and self.regression_results_text.get('1.0', tk.END).strip():
            summary_content += f"REGRESSION FINISHED:\\n"
            summary_content += f"- Formula used: {self.regression_formula.get()}\\n"
            if hasattr(self, 'biomarkers_df'):
                summary_content += f"- Merged data: {len(self.biomarkers_df)} subjects\\n\\n"
        
        # Available biomarkers
        if hasattr(self, 'biomarkers_df') and not self.biomarkers_df.empty:
            summary_content += "AVAILABLE BIOMARKERS:\\n"
            numeric_cols = self.biomarkers_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Subject']:
                    non_null_count = self.biomarkers_df[col].notna().sum()
                    summary_content += f"- {col}: {non_null_count} values\\n"
            summary_content += "\\n"
        
        summary_content += "=== END OF SUMMARY ===\\n"
        
        self.summary_text.insert(tk.END, summary_content)


        
    def create_results_summary(self, parent):
        """Create results summary display"""
        self.summary_text = tk.Text(parent, height=15, bg=self.colors['surface'], fg=self.colors['text_primary'])
        self.summary_text.pack(fill='both', expand=True)

    # --- placeholder functions for the buttons and the logic --- 
    def on_mode_change(self):
        self.status_var.set(f"Analysis mode switched to: {self.analysis_mode.get()}")


    def select_files(self):
        """Handles the selection of individual BrainVision files."""
        file_paths = filedialog.askopenfilenames(filetypes=[("BrainVision Files", "*.vhdr")])
        if file_paths:
            # Clear all lists to ensure a fresh start
            self.file_listbox.delete(0, tk.END)
            self.uploaded_files.clear()
            self.all_uploaded_files.clear()
            self.subject_ids.clear()

            # Populate lists with the new files
            for fp in file_paths:
                self.all_uploaded_files.append(fp)
                self.uploaded_files.append(fp)
                self.file_listbox.insert(tk.END, os.path.basename(fp))
            
            self.subject_ids = [os.path.basename(fp).split('.')[0] for fp in self.uploaded_files]
            self.status_var.set(f"{len(file_paths)} files successfully loaded. Ready for filtering.")
            
            # This is the corrected function call to update the Subject Filter tab
            self.update_subject_lists() 

    def select_directory(self):
        """Handles the recursive selection of BrainVision files from a directory."""
        dir_path = filedialog.askdirectory()
        if dir_path:
            found_files = []
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.vhdr'):
                        found_files.append(os.path.join(root, file))
            
            if found_files:
                # Clear all lists to ensure a fresh start
                self.file_listbox.delete(0, tk.END)
                self.uploaded_files.clear()
                self.all_uploaded_files.clear()
                self.subject_ids.clear()

                # Populate lists with the new files
                for fp in found_files:
                    self.all_uploaded_files.append(fp)
                    self.uploaded_files.append(fp)
                    self.file_listbox.insert(tk.END, os.path.basename(fp))
                
                self.subject_ids = [os.path.basename(fp).split('.')[0] for fp in self.uploaded_files]
                self.status_var.set(f"{len(found_files)} .vhdr files found. Ready for filtering.")
                
                # This is the corrected function call to update the Subject Filter tab
                self.update_subject_lists()
            else:
                self.status_var.set("No .vhdr files were found in the selected directory.")


    def load_configuration(self):
        config_file = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.low_freq.set(config.get('low_freq', self.low_freq.get()))
                self.high_freq.set(config.get('high_freq', self.high_freq.get()))
                self.notch_freq.set(config.get('notch_freq', self.notch_freq.get()))
                self.n_components.set(config.get('n_components', self.n_components.get()))
                self.ica_method.set(config.get('ica_method', self.ica_method.get()))
                self.freq_range_min.set(config.get('freq_range_min', self.freq_range_min.get()))
                self.freq_range_max.set(config.get('freq_range_max', self.freq_range_max.get()))
                self.aperiodic_mode.set(config.get('aperiodic_mode', self.aperiodic_mode.get()))
                self.status_var.set("Configuration successfully loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
                self.status_var.set("Error occurred while loading configuration.")

    def save_configuration(self):
        config_data = {
            'low_freq': self.low_freq.get(),
            'high_freq': self.high_freq.get(),
            'notch_freq': self.notch_freq.get(),
            'n_components': self.n_components.get(),
            'ica_method': self.ica_method.get(),
            'freq_range_min': self.freq_range_min.get(),
            'freq_range_max': self.freq_range_max.get(),
            'aperiodic_mode': self.aperiodic_mode.get()
        }
        config_file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")], initialfile=self.config_name.get())
        if config_file:
            try:
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=4)
                self.status_var.set("Configuration successfully saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
                self.status_var.set("Error occurred while saving configuration.")

    def run_preprocessing(self):
        if not self.uploaded_files:
            messagebox.showwarning("Warning", "Please upload EEG files before running preprocessing.")
            return
        
        self.status_var.set("Starting preprocessing...")
        self.preprocessing_progress.start()
        
        # Scratch directory for the large temporary files; see TDBRAIN_SCRATCH_DIR
        scratch_dir = cfg.SCRATCH_DIR
        try:
            os.makedirs(scratch_dir, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Disk error", f"Scratch directory not usable: {scratch_dir}\n{exc}\n"
                                               f"Set TDBRAIN_SCRATCH_DIR to a writable folder.")
            self.preprocessing_progress.stop()
            return

        process_args = {
            "uploaded_files": self.uploaded_files,
            "low_freq": self.low_freq.get(),
            "high_freq": self.high_freq.get(),
            "notch_freq": self.notch_freq.get(),
            "progress_queue": self.progress_queue,
            "temp_dir_base": scratch_dir  # the worker writes its temporary files here
        }

        self.analysis_process = multiprocessing.Process(
            target=preprocessing_worker_process,
            args=(process_args,)
        )
        self.analysis_process.daemon = True
        self.analysis_process.start()
        
        self._process_progress_queue()
    
        
    def _update_progress(self, progress_bar, percent_var, status_text, percentage):
        """
        Single point of entry for refreshing a progress bar, its percentage
        label and the status bar of the application.
        """
        if percentage < 100:
            percent_var.set(f"{percentage}%")
            self.status_var.set(status_text)
        else:
            # Once the process is complete
            progress_bar.stop()
            percent_var.set("100%")
            self.status_var.set(status_text)


        
    def _preprocessing_worker(self):
        try:
            subjects_to_process = self.uploaded_files
            total_subjects = len(subjects_to_process)
            if total_subjects == 0:
                self.progress_queue.put((self.preprocessing_progress_percent, 100, "No subjects selected."))
                return

            self.preprocessed_data = {}
            failed_subjects = []

            for i, file_path in enumerate(subjects_to_process):
                subject_id = os.path.basename(file_path).replace(".vhdr", "")
                
                percentage = int(((i + 1) / total_subjects) * 100)
                status_text = f"Preprocessing {subject_id} ({i + 1}/{total_subjects})..."
                self.progress_queue.put((self.preprocessing_progress_percent, percentage, status_text))

                try:
                    # The MNE logic is unchanged
                    print(f"--- Preprocessing started for {subject_id} ---")
                    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
                    
                    ch_types = {}
                    non_eeg_channels = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']
                    for ch_name in raw.ch_names:
                        ch_types[ch_name] = 'misc' if ch_name in non_eeg_channels else 'eeg'
                    raw.set_channel_types(ch_types, verbose='warning')

                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage, on_missing='warn')

                    raw.filter(self.low_freq.get(), self.high_freq.get(), verbose=False)
                    raw.notch_filter(self.notch_freq.get(), verbose=False)

                    self.preprocessed_data[subject_id] = raw
                    print(f"--- Fine Preprocessing per {subject_id} ---")

                except Exception as e:
                    print(f"!!! ERROR while preprocessing {subject_id}: {e}")
                    failed_subjects.append(subject_id)

                time.sleep(0.001)

            self.progress_queue.put((self.preprocessing_progress_percent, 100, "Preprocessing completed."))
            self.root.after(0, self.preprocessing_progress.stop)

            subject_list = list(self.preprocessed_data.keys())
            self.root.after(0, lambda: self.subject_dropdown.configure(values=subject_list))
            if subject_list:
                self.root.after(0, lambda: self.selected_subject.set(subject_list[0]))

            if failed_subjects:
                self.root.after(0, lambda: messagebox.showwarning("Preprocessing Warnings", 
                                    f"Failed to preprocess {len(failed_subjects)} subject(s):\n" +
                                    "\n".join(failed_subjects)))

        except Exception as e:
            self.progress_queue.put((self.preprocessing_progress_percent, 0, "Error during preprocessing!"))
            self.root.after(0, self.preprocessing_progress.stop)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {e}"))
        

    def _update_single_preprocessing_viz(self):
        """Show the PSD of the selected subject in the canvas of the preprocessing tab"""
        subject_id = self.selected_subject.get().strip()
        print("DEBUG - Selected subject:", subject_id)
        print("DEBUG - Preprocessed data keys:", list(self.preprocessed_data.keys()))
        if not subject_id or subject_id.strip() not in self.preprocessed_data:
            messagebox.showwarning("Warning", f"Please select a valid subject. ({subject_id})")
            return


        raw = self.preprocessed_data[subject_id]
        self.preprocess_fig.clear()
        ax = self.preprocess_fig.add_subplot(111)

        try:
            raw.plot_psd(ax=ax, show=False, average=True, fmax=self.high_freq.get(), picks='eeg', verbose=False)
            ax.set_title(f"PSD after preprocessing for {subject_id}", fontsize=12, color=self.colors['text_secondary'])
            ax.tick_params(colors=self.colors['text_secondary'])
            for spine in ax.spines.values():
                spine.set_color(self.colors['border'])
            ax.yaxis.label.set_color(self.colors['text_primary'])
            ax.xaxis.label.set_color(self.colors['text_primary'])
        except Exception as e:
            ax.text(0.5, 0.5, f"Error while plotting data for {subject_id}: {e}", ha="center", va="center")

        self.preprocess_fig.tight_layout()
        self.preprocess_canvas.draw()
        print("DEBUG - Selected subject:", subject_id)
        print("DEBUG - Preprocessed data keys:", list(self.preprocessed_data.keys()))



    def _update_single_ica_viz(self):
        """
        Draw the ICA component maps straight into the GUI frame
        rather than opening a new window.
        """
        subject_id = self.selected_ica_subject.get()
        if not subject_id or subject_id not in self.ica_results:
            messagebox.showwarning("Warning", "Select a valid subject with ICA results.")
            return
        if self.ica_maps_displayed == subject_id:
            self.status_var.set(f"The ICA maps for {subject_id} are already on screen.")
            return
        self.ica_maps_displayed = subject_id

        try:

            for widget in self.component_frame.winfo_children():
                widget.destroy()

            ica_data = self.ica_results[subject_id]
            ica = ica_data['ica']
            raw_for_plot = self.preprocessed_data[subject_id]


            figures = ica.plot_components(inst=raw_for_plot, show=False)

            for fig in figures:

                fig.set_size_inches(10, 8) 
                
                canvas = FigureCanvasTkAgg(fig, master=self.component_frame)
                canvas.draw()
                

                toolbar_frame = ttk.Frame(self.component_frame)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
                toolbar_frame.pack(side=tk.TOP, fill=tk.X)

            self.status_var.set(f"Showing the ICA maps for {subject_id}.")

        except Exception as e:
            messagebox.showerror("Display error", f"Could not display the ICA maps: {e}")
            # Put an error message back into the frame
            for widget in self.component_frame.winfo_children():
                widget.destroy()
            ttk.Label(self.component_frame, text=f"Error while displaying the ICA for {subject_id}.").pack()



    def run_ica(self):
        if not self.preprocessed_data:
            messagebox.showwarning("Warning", "Please run the preprocessing step before performing ICA analysis.")
            return
        self.status_var.set("Starting ICA analysis...")
        self.ica_progress.start()
        threading.Thread(target=self._ica_worker, daemon=True).start()

    def _ica_worker(self):
        try:
            n_components = self.n_components.get()
            ica_method = self.ica_method.get()
            print(f"\n{'='*20} ICA ANALYSIS STARTED {'='*20}")
            print(f"ICA parameters: {n_components} components, method '{ica_method}'")

            subjects_to_process = list(self.preprocessed_data.keys())
            total_subjects = len(subjects_to_process)
            if total_subjects == 0:
                self.root.after(0, lambda: messagebox.showwarning("Missing data", "Run the preprocessing first."))
                return

            self.root.after(0, self.ica_progress.start)
            self.ica_results = {}
            failed_subjects = []

            for i, subject_id in enumerate(subjects_to_process):
                self.root.after(0, lambda p=int(((i + 1) / total_subjects) * 100), s=subject_id: (
                    self.ica_progress_percent.set(f"{p}%"),
                    self.status_var.set(f"Running ICA for {s} ({i + 1}/{total_subjects})...")
                ))

                try:
                    raw = self.preprocessed_data[subject_id].copy()
                    
                    raw_for_ica_fit = raw.copy().pick_types(eeg=True, verbose=False)
                    raw_for_ica_fit.filter(l_freq=1., h_freq=100., verbose=False)

                    # --- key change ---
                    # Initialise fit_params as an empty dictionary.
                    # random_state is passed straight to ICA.
                    fit_params = {}
                    if ica_method == 'infomax':
                        fit_params['extended'] = True
                    elif ica_method == 'picard':
                        fit_params['ortho'] = False
                        fit_params['extended'] = True

                    # random_state is passed here, once only.
                    ica = ICA(n_components=n_components, 
                            method=ica_method, 
                            fit_params=fit_params, 
                            random_state=42,  # This guarantees reproducibility
                            verbose=False)
                    # --- end of the change ---
                    
                    ica.fit(raw_for_ica_fit)

                    ic_labels = label_components(raw_for_ica_fit, ica, method='iclabel')

                    self.ica_results[subject_id] = {'ica': ica, 'labels': ic_labels}
                except Exception as e:
                    print(f"!!! ERROR in the ICA for {subject_id}: {e}")
                    failed_subjects.append(subject_id)

            self.root.after(0, self.ica_progress.stop)
            self.root.after(0, lambda: self.status_var.set("ICA analysis completed."))
            
            subject_list = list(self.ica_results.keys())
            self.root.after(0, lambda: self.ica_subject_dropdown.configure(values=subject_list))
            if subject_list:
                self.root.after(0, lambda: self.selected_ica_subject.set(subject_list[0]))
                self.root.after(0, self.on_ica_subject_change)

            if failed_subjects:
                self.root.after(0, lambda: messagebox.showwarning("ICA Warnings", f"Failed to process ICA for {len(failed_subjects)} subject(s):\n" + "\n".join(failed_subjects)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Critical ICA error", f"An error occurred: {e}"))
            self.root.after(0, self.ica_progress.stop)



    def _update_ica_viz(self):
        if not self.ica_results:
            return

        subject_id = list(self.ica_results.keys())[0]
        ica_data = self.ica_results[subject_id]
        ica = ica_data['ica']
        labels = ica_data['labels']['labels'] # Reach the list of labels

        # Clear the previous frame
        for widget in self.component_frame.winfo_children():
            widget.destroy()

        # A Matplotlib figure to hold the component plots
        figures = ica.plot_components(show=False, picks=range(self.n_components.get()), inst=self.preprocessed_data[subject_id])
        
        # MNE's plot_components method, which is exactly right for this purpose.
        # It takes care of drawing the topographic maps.
        # Our empty figure is passed in to tell it where to draw.
        self.ica_fig = figures[0] if isinstance(figures, list) else figures

        # A Tkinter drawing area, with our figure inside it
        canvas = FigureCanvasTkAgg(self.ica_fig, master=self.component_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Matplotlib's navigation toolbar (zoom, pan and so on)
        toolbar = NavigationToolbar2Tk(canvas, self.component_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # A separate frame for the checkboxes, so they do not sit on top of the plots
        checkbox_frame = ttk.Frame(self.component_frame, style='Card.TFrame')
        checkbox_frame.pack(fill='x', pady=10)
        ttk.Label(checkbox_frame, text=f"Select the components to exclude for {subject_id}:", style='Heading.TLabel').pack(pady=5)

        # Initialise the list of components to exclude
        if subject_id not in self.excluded_components:
            self.excluded_components[subject_id] = []
        
        self.style.configure('TCheckbutton', background=self.colors['surface'], foreground=self.colors['text_primary'])
        self.style.configure('Excluded.TCheckbutton', background='#ffe6e6', foreground='red')

        # The checkboxes, as before, but inside the new frame
        for i, label in enumerate(labels):
            var = tk.BooleanVar()
            # If the component was labelled as an artefact, pre-select it for exclusion
            if label not in ['brain', 'other']:
                var.set(True)
                if i not in self.excluded_components[subject_id]:
                    self.excluded_components[subject_id].append(i)
            # Configure the style separately before the loop or at the top

            # Create the checkbutton
            chk = ttk.Checkbutton(
                checkbox_frame,
                text=f"Component {i}: {label}",
                variable=var,
                style='Excluded.TCheckbutton' if var.get() else 'TCheckbutton',
                command=lambda idx=i, v=var: self._toggle_component_exclusion(subject_id, idx, v)
            )

            chk.pack(anchor='w', padx=20)



    def _toggle_component_exclusion(self, subject_id, component_idx, var):
        if var.get():
            if component_idx not in self.excluded_components[subject_id]:
                self.excluded_components[subject_id].append(component_idx)
        else:
            if component_idx in self.excluded_components[subject_id]:
                self.excluded_components[subject_id].remove(component_idx)
        self.status_var.set(f"Components excluded for {subject_id}: {self.excluded_components[subject_id]}")

        # --- start of the added block ---

    def update_component_exclusions(self):
        """
        Apply the ICA component exclusions to the preprocessed data of every subject.
        It is called when the user clicks the 'Update Component Exclusions' button.
        """
        try:
            # Iterate over every subject for which ICA results exist
            for subject_id, ica_data in self.ica_results.items():
                ica = ica_data['ica']
                
                # Take a clean copy of the preprocessed data, before the ICA was applied
                raw = self.preprocessed_data[subject_id].copy()
                
                # Retrieve the list of component indices to exclude for this subject
                exclude_indices = self.excluded_components.get(subject_id, [])
                
                # Apply the ICA, removing the components listed
                ica.apply(raw, exclude=exclude_indices, verbose=False)
                
                # Overwrite the preprocessed data with the clean version, artefacts removed
                self.preprocessed_data[subject_id] = raw
                
            # Refresh the status bar and show a success message
            self.status_var.set("The ICA exclusions were applied successfully.")
            messagebox.showinfo("Success", "The selected ICA components have been removed from the data.")
            
        except Exception as e:
            # On error, show a detailed message
            messagebox.showerror("Error", f"Could not apply the ICA exclusions: {e}")
            self.status_var.set("Error while applying the ICA exclusions.")


    def run_spectral_analysis(self):
        if not self.preprocessed_data:
            messagebox.showwarning("Warning", "Run the preprocessing and the ICA first.")
            return
        self.status_var.set("Starting the spectral analysis...")
        self.spectral_progress.start()
        threading.Thread(target=self._spectral_worker, daemon=True).start()

    
    def _spectral_worker(self):
        try:
            subjects_to_process = self.uploaded_files
            total_subjects = len(subjects_to_process)

            if total_subjects == 0:
                self.root.after(0, lambda: messagebox.showwarning("No Subjects", "No subjects selected for spectral analysis."))
                return

            self.root.after(0, self.spectral_progress.start)
            self.root.after(0, lambda: self.spectral_progress_percent.set("0%"))

            for i, file_path in enumerate(subjects_to_process):
                subject_id = os.path.basename(file_path).replace(".vhdr", "")
                self.status_var.set(f"Running spectral analysis for {subject_id} ({i+1}/{total_subjects})...")

                percentage = int(((i + 1) / total_subjects) * 100)
                self.root.after(0, lambda p=percentage: self.spectral_progress_percent.set(f"{p}%"))

                if subject_id not in self.preprocessed_data:
                    print(f"Warning: No preprocessed data for {subject_id}, skipping spectral analysis.")
                    continue

                raw = self.preprocessed_data[subject_id].copy()
                psd_all = raw.compute_psd(method="welch", fmin=1, fmax=self.high_freq.get(), verbose=False)
                psds_all, freqs_all = psd_all.get_data(), psd_all.freqs

                biomarkers = {}
                f_alpha_band = (8, 12)

                # FAA
                left_ch, right_ch = ['F3'], ['F4']
                if left_ch[0] in raw.ch_names and right_ch[0] in raw.ch_names:
                    psd_left = raw.copy().pick(left_ch).compute_psd(fmin=f_alpha_band[0], fmax=f_alpha_band[1], verbose=False)
                    left_power = np.mean(psd_left.get_data())
                    psd_right = raw.copy().pick(right_ch).compute_psd(fmin=f_alpha_band[0], fmax=f_alpha_band[1], verbose=False)
                    right_power = np.mean(psd_right.get_data())
                    biomarkers['FAA'] = np.log(right_power) - np.log(left_power) if left_power > 0 and right_power > 0 else np.nan
                else:
                    biomarkers['FAA'] = np.nan

                # TBR
                f_theta_band, f_beta_band = (4, 8), (13, 30)
                theta_power = np.mean(psds_all[:, (freqs_all >= f_theta_band[0]) & (freqs_all <= f_theta_band[1])])
                beta_power = np.mean(psds_all[:, (freqs_all >= f_beta_band[0]) & (freqs_all <= f_beta_band[1])])
                biomarkers['TBR'] = theta_power / beta_power if beta_power > 0 else np.nan

                # IAF
                pz_ch = ['Pz']
                if pz_ch[0] in raw.ch_names:
                    psd_pz = raw.copy().pick(pz_ch).compute_psd(fmin=f_alpha_band[0], fmax=f_alpha_band[1], verbose=False)
                    pz_data, pz_freqs = psd_pz.get_data()[0], psd_pz.freqs
                    biomarkers['IAF'] = pz_freqs[np.argmax(pz_data)] if len(pz_data) > 0 else np.nan
                else:
                    biomarkers['IAF'] = np.nan

                self.spectral_results[subject_id] = {
                    'psds': psds_all,
                    'freqs': freqs_all,
                    'biomarkers': biomarkers
                }

            self.root.after(0, lambda: self.status_var.set("Spectral analysis completed."))
            self.root.after(0, lambda: self.spectral_progress_percent.set("100%"))

            subject_list = list(self.spectral_results.keys())
            self.root.after(0, lambda: self.spectral_subject_dropdown.configure(values=subject_list))
            if subject_list:
                self.root.after(0, lambda: self.selected_spectral_subject.set(subject_list[0]))
                self.root.after(0, self._update_spectral_viz)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during spectral analysis: {e}"))
            self.root.after(0, lambda: self.status_var.set("Spectral analysis failed."))
            self.root.after(0, lambda: self.spectral_progress_percent.set("Error"))
        finally:
            self.root.after(0, self.spectral_progress.stop)
    

            
    def _update_spectral_viz(self):
        """Updates the visualization for the spectral results of the selected subject."""
        subject_id = self.selected_spectral_subject.get()

        if not subject_id or subject_id not in self.spectral_results:
            self.spectral_fig.clear()
            ax = self.spectral_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select a subject to visualize the data.", ha='center', va='center')
            self.spectral_canvas.draw()
            return

        results = self.spectral_results[subject_id]
        psds = results['psds']
        freqs = results['freqs']
        biomarkers = results['biomarkers']

        self.spectral_fig.clear()
        ax1 = self.spectral_fig.add_subplot(2, 1, 1)
        psds_mean = np.mean(psds, axis=0)
        ax1.plot(freqs, 10 * np.log10(psds_mean), color=self.colors['primary'])
        ax1.set_title(f"Mean Power Spectral Density (PSD) for {subject_id}", color=self.colors['text_primary'])
        ax1.set_xlabel("Frequency (Hz)", color=self.colors['text_secondary'])
        ax1.set_ylabel("Power/Frequency (dB/Hz)", color=self.colors['text_secondary'])
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.tick_params(colors=self.colors['text_secondary'])
        for spine in ax1.spines.values():
            spine.set_color(self.colors['border'])

        ax2 = self.spectral_fig.add_subplot(2, 1, 2)
        ax2.set_facecolor(self.colors['surface'])
        ax2.axis('off')
        
        biomarker_text = "Biomarkers:\n\n"
        for key, value in biomarkers.items():
            if not np.isnan(value):
                biomarker_text += f"{key}: {value:.2f}\n"
            else:
                biomarker_text += f"{key}: N/A\n"
                
        ax2.text(0.5, 0.6, biomarker_text, ha='center', va='center', fontsize=12, 
                fontdict={'family': 'Arial', 'color': self.colors['text_primary']})

        self.spectral_fig.tight_layout()
        self.spectral_canvas.draw()


    def _update_fooof_viz(self):
        """
        Refresh the FOOOF display by calling the function that builds the tabbed view.
        """
        subject_id = self.selected_fooof_subject.get()
        if not subject_id or subject_id not in self.fooof_results:
            messagebox.showwarning("Warning", "Select a valid subject with FOOOF results.")
            return

        try:
            # Call the function dedicated to building the tabbed view,
            # passing it the frame in which to draw.
            self.create_fooof_tabbed_view(self.fooof_results_frame, subject_id)
            self.status_var.set(f"Showing the FOOOF plots for {subject_id}.")

        except Exception as e:
            messagebox.showerror("Display error", f"Could not display the FOOOF plots: {e}")
            # Clear the frame on error
            for widget in self.fooof_results_frame.winfo_children():
                widget.destroy()
            ttk.Label(self.fooof_results_frame, text=f"Error while displaying FOOOF for {subject_id}.").pack()


    def create_fooof_tabbed_view(self, parent_frame, subject_id):
        """
        Build a tabbed view (Notebook) for the FOOOF results, one plot per channel.
        """
        # Clear the parent frame of any earlier widget
        for widget in parent_frame.winfo_children():
            widget.destroy()

        # The Notebook widget that will hold the tabs
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill='both', expand=True, pady=10, padx=5)

        # Results and colours are taken from 'self'
        fooof_subject_results = self.fooof_results[subject_id]
        colors = self.colors 

        channels = list(fooof_subject_results.keys())

        # Iterate over every channel for which a FOOOF result exists
        for ch_name in channels:
            # A frame for this channel's tab
            tab_frame = ttk.Frame(notebook, padding=10)
            notebook.add(tab_frame, text=ch_name)

            # A Matplotlib figure for this channel ONLY
            fig = Figure(figsize=(10, 5), facecolor=colors['surface'])
            ax = fig.add_subplot(111)
            
            # Take the FOOOF model and draw it on the axis
            fm = fooof_subject_results[ch_name]['fm']
            fm.plot(ax=ax, plot_peaks='shade', peak_kwargs={'color': 'green'}, add_legend=False)

            # Tune the look of the plot
            ax.set_title(f"Channel: {ch_name}", color=colors['text_primary'])
            ax.set_xlabel("Frequency (Hz)", color=colors['text_secondary'])
            ax.set_ylabel("Log(Power)", color=colors['text_secondary'])
            ax.tick_params(colors=colors['text_secondary'])
            for spine in ax.spines.values():
                spine.set_color(colors['border'])
            fig.tight_layout()

            # Put the plot into its Tkinter canvas
            canvas = FigureCanvasTkAgg(fig, master=tab_frame)
            canvas.draw()
            
            # Add the Matplotlib toolbar
            toolbar_frame = ttk.Frame(tab_frame)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # Place the canvas and the toolbar in the tab frame
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X, pady=(5,0))


    def run_fooof_analysis(self):
        if not self.preprocessed_data:
            messagebox.showwarning("Warning", "Please run Preprocessing and ICA first.")
            return
        self.status_var.set("Initiating FOOOF analysis...")
        self.fooof_progress.start()
        threading.Thread(target=self._fooof_worker, daemon=True).start()


    def _fooof_worker(self):
        try:
            # --- DEBUG: print the FOOOF parameters ---
            freq_min = self.freq_range_min.get()
            freq_max = self.freq_range_max.get()
            min_peak_height = self.min_peak_height.get()
            aperiodic_mode = self.aperiodic_mode.get()
            print("\n" + "="*20 + " FOOOF ANALYSIS STARTED " + "="*20)
            print(f"FOOOF parameters selected:")
            print(f"  - Frequency range: [{freq_min}, {freq_max}] Hz")
            print(f"  - Minimum peak height: {min_peak_height}")
            print(f"  - Aperiodic mode: '{aperiodic_mode}'")
            print("="*60 + "\n")
            # --- FINE DEBUG ---

            subjects_to_process = self.uploaded_files
            total_subjects = len(subjects_to_process)

            if total_subjects == 0:
                self.root.after(0, lambda: messagebox.showwarning(
                    "No Subjects", "No subjects selected for analysis. Please apply filters first."))
                return

            self.root.after(0, self.fooof_progress.start)
            self.root.after(0, lambda: self.fooof_progress_percent.set("0%"))

            for i, file_path in enumerate(subjects_to_process):
                subject_id = os.path.basename(file_path).replace(".vhdr", "")
                self.status_var.set(f"Running FOOOF for {subject_id} ({i+1}/{total_subjects})...")

                percentage = int(((i + 1) / total_subjects) * 100)
                self.root.after(0, lambda p=percentage: self.fooof_progress_percent.set(f"{p}%"))

                if subject_id not in self.preprocessed_data:
                    continue

                raw = self.preprocessed_data[subject_id]
                psd = raw.compute_psd(method="welch", fmin=1.0, fmax=50.0, n_fft=2048, verbose=False)
                freqs = psd.freqs
                psds_data = psd.get_data()

                self.fooof_results[subject_id] = {}
                channels_to_analyze = [
                    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                    'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
                ]

                for ch_idx, ch_name in enumerate(raw.ch_names):
                    if ch_name in channels_to_analyze:
                        fm = FOOOF(
                            peak_width_limits=[1.5, 8.0],
                            max_n_peaks=5,
                            min_peak_height=min_peak_height,
                            aperiodic_mode=aperiodic_mode,
                            verbose=False
                        )
                        fm.fit(freqs, psds_data[ch_idx], freq_range=[freq_min, freq_max])
                        self.fooof_results[subject_id][ch_name] = {'fm': fm}

            self.root.after(0, lambda: self.status_var.set("FOOOF analysis completed."))
            self.root.after(0, lambda: self.fooof_progress_percent.set("100%"))

            subject_list = list(self.fooof_results.keys())
            self.root.after(0, lambda: self.fooof_subject_dropdown.configure(values=subject_list))
            if subject_list:
                self.root.after(0, lambda: self.selected_fooof_subject.set(subject_list[0]))
                self.root.after(0, self._update_fooof_viz)
                self.root.after(0, self.update_summary_tab)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during FOOOF analysis: {e}"))
            self.root.after(0, lambda: self.status_var.set("FOOOF analysis failed."))
        finally:
            self.root.after(0, self.fooof_progress.stop)

    
    
    def update_fooof_tab_after_run(self):
        """Helper function to update the FOOOF tab dropdown and visualization after the analysis."""
        subject_list = list(self.fooof_results.keys())
        self.fooof_subject_dropdown.configure(values=subject_list)
        if subject_list:
            self.selected_fooof_subject.set(subject_list[0])
        self._update_fooof_viz()



    def upload_behavioral_data(self):
        """
        Handle the loading of the behavioural data from a CSV or Excel file and
        make sure the key columns end up with the right data types.
        """
        behavioral_file = filedialog.askopenfilename(
            title="Select Behavioral Data File",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")]
        )
        if not behavioral_file:
            return

        try:
            # Load the data from the selected file
            if behavioral_file.endswith('.csv'):
                self.behavioral_data = pd.read_csv(behavioral_file)
            elif behavioral_file.endswith('.xlsx'):
                self.behavioral_data = pd.read_excel(behavioral_file)
            
            print("--- Behavioral data loaded successfully. Starting type conversion... ---")

            # --- DATA TYPE CONVERSION ---
            
            # The columns expected to be numeric in the behavioural file.
            # 'Responder' is the key column to convert, which is what fixed the regression error.
            behavioral_numeric_cols = ['BDI_pre', 'age', 'Responder'] 
            
            for col in behavioral_numeric_cols:
                if col in self.behavioral_data.columns:
                    # Convert the column to numeric. 'coerce' turns invalid values into NaN.
                    self.behavioral_data[col] = pd.to_numeric(
                        self.behavioral_data[col],
                        errors='coerce' 
                    )
                    print(f"DEBUG: Converted '{col}' to numeric. New dtype: {self.behavioral_data[col].dtype}")
                else:
                    print(f"WARNING: Column '{col}' not found in the behavioral file.")
            
            # Convert the 'gender' column to a categorical type, if it exists.
            if 'gender' in self.behavioral_data.columns:
                self.behavioral_data['gender'] = self.behavioral_data['gender'].astype('category')
                print(f"DEBUG: Converted 'gender' to categorical. New dtype: {self.behavioral_data['gender'].dtype}")

            self.status_var.set("Behavioral data loaded and processed.")
            messagebox.showinfo(
                "Success", 
                "Behavioral data loaded.\n"
                "Key columns have been converted to the correct data types.\n"
                "Please ensure 'participant_ID' and 'session_ID' columns exist for merging."
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process behavioral data: {e}")
            self.status_var.set("Error loading behavioral data.")

    def _plot_bootstrap_distributions(self, bootstrap_results_df):
        """
        Show histograms of the bootstrap distributions of the parameters, embedded in Tkinter.
        No seaborn. Robust handling of NaN and inf, with a dynamic layout.
        """
        import numpy as np

        # Quick checks
        if bootstrap_results_df is None:
            return
        if getattr(bootstrap_results_df, "empty", True):
            return

        # Numeric columns only
        numeric_cols = [
            c for c in bootstrap_results_df.columns
            if np.issubdtype(bootstrap_results_df[c].dtype, np.number)
        ]
        
        numeric_cols = sorted(
            numeric_cols,
            key=lambda c: np.nanstd(bootstrap_results_df[c].to_numpy()),
            reverse=True
        )
        if not numeric_cols:
            return

        # Clean the values, dropping inf and NaN
        clean_df = bootstrap_results_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        clean_df = clean_df.dropna(how="all")
        if clean_df.empty:
            return

        # Build the Tk window
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Bootstrap parameter distributions")

        # Layout: at most 3 columns, for legibility
        n = len(numeric_cols)
        ncols = 3 if n >= 3 else n
        ncols = max(ncols, 1)
        nrows = int(np.ceil(n / ncols))

        # Figure sizes scaled to the grid
        fig_w = max(8, 3.2 * ncols)
        fig_h = max(4, 2.8 * nrows)

        # Theme colours, with a fallback
        surface = self.colors.get('surface', 'white') if hasattr(self, 'colors') else 'white'
        text_primary = self.colors.get('text_primary', 'black') if hasattr(self, 'colors') else 'black'
        outline = self.colors.get('outline', '#cccccc') if hasattr(self, 'colors') else '#cccccc'

        fig = Figure(figsize=(fig_w, fig_h), facecolor=surface)

        # Draw the histograms
        axes = []
        for i, col in enumerate(numeric_cols, start=1):
            ax = fig.add_subplot(nrows, ncols, i)
            vals = clean_df[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", color=text_primary)
                ax.set_axis_off()
            else:
                # Robust bin count: at least 10, at most 40
                nbins = int(np.clip(np.sqrt(vals.size) * 2, 10, 40))
                ax.hist(vals, bins=nbins, edgecolor=outline)
                ax.set_title(str(col), color=text_primary)
                ax.tick_params(axis='both', colors=text_primary)
                for spine in ax.spines.values():
                    spine.set_edgecolor(outline)
            axes.append(ax)

        # Hide the extra axes when the grid is larger than needed
        for j in range(len(numeric_cols)+1, nrows*ncols + 1):
            fig.add_subplot(nrows, ncols, j).set_axis_off()

        fig.tight_layout()

        # Canvas Tk + toolbar
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        try:
            toolbar = NavigationToolbar2Tk(canvas, plot_window, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
        except Exception:
            pass

        # Window theme
        try:
            plot_window.configure(bg=surface)
        except Exception:
            pass

    def create_regression_tab(self):
        """
        Build the regression analysis tab, with options for a single model and for a batch.
        """
        regression_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(regression_frame, text="Regression Analysis")
        
        control_panel = self.create_section(regression_frame, "Regression Setup")
        
        # --- formula section, for a single model ---
        formula_frame = ttk.LabelFrame(control_panel, text="1. Regression Formula (for Single Analysis)", style='Card.TLabelframe', padding=10)
        formula_frame.pack(fill='x', pady=5, padx=5)
        self.regression_formula = tk.StringVar(value="Responder ~ age + C(gender) + BDI_pre")
        ttk.Entry(formula_frame, textvariable=self.regression_formula, font=("Courier", 10)).pack(pady=5, padx=5, fill='x')
        
        # --- container for the options and the buttons ---
        options_action_frame = ttk.Frame(control_panel)
        options_action_frame.pack(fill='x', pady=5, padx=5)
        options_action_frame.columnconfigure(0, weight=1)
        options_action_frame.columnconfigure(1, weight=1)
        options_frame = ttk.LabelFrame(options_action_frame, text="2. Analysis Options", style='Card.TLabelframe', padding=10)
        options_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        # Existing options
        self.regression_type = tk.StringVar(value="Logistic")
        ttk.Label(options_frame, text="Model Type:").pack(anchor='w')
        type_buttons_frame = ttk.Frame(options_frame)
        type_buttons_frame.pack(anchor='w', pady=(0, 10))
        ttk.Radiobutton(type_buttons_frame, text="Logistic (0/1 outcome)", variable=self.regression_type, value="Logistic").pack(side='left', padx=5)
        ttk.Radiobutton(type_buttons_frame, text="Linear (OLS)", variable=self.regression_type, value="OLS").pack(side='left', padx=5)

        self.missing_data_handling = tk.StringVar(value="exclude")
        ttk.Label(options_frame, text="Missing Data Handling:").pack(anchor='w')
        missing_buttons_frame = ttk.Frame(options_frame)
        missing_buttons_frame.pack(anchor='w')
        self.impute_radio = ttk.Radiobutton(missing_buttons_frame, text="Impute with Regression", variable=self.missing_data_handling, value="impute")
        self.impute_radio.pack(side='left', padx=5)
        ttk.Radiobutton(missing_buttons_frame, text="Exclude subject (default)", variable=self.missing_data_handling, value="exclude").pack(side='left', padx=5)

        # Bootstrap frame (already present)
        self.bootstrap_lf = ttk.Frame(options_frame)
        self.bootstrap_lf.pack(fill='x', pady=5)
        self.use_bootstrap = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.bootstrap_lf, text="Use Bootstrap Resampling", variable=self.use_bootstrap).pack(side='left', padx=5)
        self.n_bootstraps = tk.IntVar(value=1000)
        ttk.Label(self.bootstrap_lf, text="Samples:").pack(side='left', padx=(10, 2))
        ttk.Entry(self.bootstrap_lf, textvariable=self.n_bootstraps, width=8).pack(side='left')

        # --- start of the block moved into its proper place ---
        ttk.Separator(options_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)

        self.advanced_regression_lf = ttk.Frame(options_frame)
        self.advanced_regression_lf.pack(fill='x', pady=5)

        self.use_nested_cv_regression = tk.BooleanVar(value=False)
        chk_nested_cv = ttk.Checkbutton(
            self.advanced_regression_lf, 
            text="Run with Nested CV & Imputation", 
            variable=self.use_nested_cv_regression,
            command=self.toggle_advanced_regression_options
        )
        chk_nested_cv.pack(side='left', padx=5)

        self.advanced_regression_folds_frame = ttk.Frame(self.advanced_regression_lf)
        # Note: .pack() is not called here, the toggle function handles it

        ttk.Label(self.advanced_regression_folds_frame, text="Outer:").pack(side='left', padx=(10, 2))
        self.regression_outer_folds = tk.IntVar(value=5)
        ttk.Entry(self.advanced_regression_folds_frame, textvariable=self.regression_outer_folds, width=5).pack(side='left')

        ttk.Label(self.advanced_regression_folds_frame, text="Inner:").pack(side='left', padx=(10, 2))
        self.regression_inner_folds = tk.IntVar(value=3)
        ttk.Entry(self.advanced_regression_folds_frame, textvariable=self.regression_inner_folds, width=5).pack(side='left')
        # --- end of the moved block ---

        # --- actions section (the rest of the function is unchanged) ---
        action_frame = ttk.LabelFrame(options_action_frame, text="3. Execute", style='Card.TLabelframe', padding=10)
        action_frame.grid(row=0, column=1, sticky='nsew')
        
        ttk.Button(action_frame, text="Upload Behavioral Data", style='Primary.TButton', command=self.upload_behavioral_data).pack(fill='x', padx=5, pady=5)
        ttk.Button(action_frame, text="Debug: Merge & Print Data", style='Warning.TButton', command=self.debug_merge_data).pack(fill='x', padx=5, pady=5)
        
        # Change the command bound to this button
        ttk.Button(action_frame, text="Run Single Regression", style='Success.TButton', command=self.run_single_regression_dispatcher).pack(fill='x', padx=5, pady=5)
        
        ttk.Separator(action_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Button(action_frame, text="Load Formulas & Run Batch", style='Info.TButton', command=self.run_batch_regression).pack(fill='x', padx=5, pady=5)

        # --- progress bar section ---
        progress_section = self.create_section(regression_frame, "Analysis Progress")
        progress_frame = ttk.Frame(progress_section)
        progress_frame.pack(fill='x', pady=5)
        ttk.Label(progress_frame, text="Progress:", style='Body.TLabel').pack(side='left')
        self.regression_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.regression_progress.pack(side='left', fill='x', expand=True, padx=(10, 5))
        self.regression_progress_percent = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.regression_progress_percent, style='Body.TLabel').pack(side='right')

        # --- results section ---
        results_section = self.create_section(regression_frame, "Regression Output")
        
        self.regression_notebook = ttk.Notebook(results_section)
        self.regression_notebook.pack(fill='both', expand=True)

        batch_tab = ttk.Frame(self.regression_notebook, padding=10)
        self.regression_notebook.add(batch_tab, text="Batch Summary")
        
        cols = ["Formula","N","AUC","Pseudo R²","Log-Likelihood","AIC"]
        self.regression_summary_table = ttk.Treeview(batch_tab, columns=cols, show='headings', height=6)
        for col in cols:
            self.regression_summary_table.heading(col, text=col, command=lambda c=col: self.sort_regression_table(c, False))
            width = 400 if col == "Formula" else 85
            self.regression_summary_table.column(col, width=width, anchor='w')
        vsb = ttk.Scrollbar(batch_tab, orient="vertical", command=self.regression_summary_table.yview)
        self.regression_summary_table.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        self.regression_summary_table.pack(side='left', fill='both', expand=True)

        detailed_tab = ttk.Frame(self.regression_notebook, padding=10)
        self.regression_notebook.add(detailed_tab, text="Detailed Report")
        self.create_regression_results_display(detailed_tab)

        # A final call, to make sure the layout is right at start-up
        self.toggle_advanced_regression_options()

    def toggle_advanced_regression_options(self):
        """Show or hide the cross-validation options in the Regression tab."""
        if self.use_nested_cv_regression.get():
            # When the checkbox is ticked, show the fold entry fields
            self.advanced_regression_folds_frame.pack(side='left', padx=5)
            # Disable the incompatible options (bootstrap and manual imputation)
            self.impute_radio.configure(state='disabled')
            self.bootstrap_lf.pack_forget() # Hide the bootstrap frame
        else:
            # Otherwise hide the fields and restore the standard options
            self.advanced_regression_folds_frame.pack_forget()
            self.impute_radio.configure(state='normal')
            self.bootstrap_lf.pack(fill='x', pady=5) # Show the bootstrap frame again


    def run_single_regression_dispatcher(self):
        """
        Decide which regression analysis to start, according to what the user selected.
        """
        formula = self.regression_formula.get().strip()
        if not formula or '~' not in formula:
            messagebox.showwarning("Invalid formula", "Enter a valid regression formula.")
            return

        # Clear the output area
        self.regression_results_text.delete('1.0', tk.END)

        if self.use_nested_cv_regression.get():
            # The user chose the new mode, with nested CV
            self.status_var.set("Starting the regression with nested CV...")
            # Start the new worker in a thread
            worker_thread = threading.Thread(
                target=self._regression_nested_cv_worker, 
                args=([formula],),  # Pass the formula as a list
                daemon=True
            )
        else:
            # The user wants the standard inference analysis (existing behaviour)
            self.status_var.set("Starting the inference analysis (statsmodels)...")
            # Call the original worker
            worker_thread = threading.Thread(
                target=self._regression_worker, 
                args=([formula],), 
                daemon=True
            )
        
        worker_thread.start()

    def _regression_nested_cv_worker(self, formulas_to_run):
        """
        Worker that runs the regression with nested CV and automatic imputation.
        It uses the scikit-learn pipeline.
        """
        try:
            # 1. Fetch the parameters from the GUI
            n_outer = self.regression_outer_folds.get()
            n_inner = self.regression_inner_folds.get()
            # ElasticNet for now, though a checkbox for XGBoost could be added
            use_xgb = False 

            # 2. Prepare the merged DataFrame (code reused)
            beh_df = self.behavioral_data.rename(columns={'sessID': 'session_ID'})
            beh_df['participant_ID'] = beh_df['participant_ID'].astype(str)
            beh_df['session_ID'] = pd.to_numeric(beh_df['session_ID'], errors='coerce')
            eeg_df = self.biomarkers_df.copy()
            eeg_df['participant_ID'] = eeg_df['participant_ID'].astype(str)
            eeg_df['session_ID'] = pd.to_numeric(eeg_df['session_ID'], errors='coerce')
            df = pd.merge(beh_df, eeg_df, on=['participant_ID', 'session_ID'], how='inner')

            if df.empty:
                raise ValueError("Merging the data produced no result.")

            # 3. Prepare the report
            report_lines = [
                "--- Regression Report (Nested CV & Imputation) ---\n",
                f"{'Model Formula':<65} | {'N':>5} | {'Calibrated AUC':>16} | {'Brier Score':>12}",
                "-" * 105
            ]
            self.root.after(0, lambda: self.regression_results_text.insert('1.0', "\n".join(report_lines) + "\n"))

            # 4. Iterate over each formula, even though there is only one here
            for i, formula in enumerate(formulas_to_run):
                self.root.after(0, lambda i=i, f=formula: self.status_var.set(f"Processing model {i+1}/{len(formulas_to_run)}..."))
                
                try:
                    y_col = formula.split('~')[0].strip()
                    if y_col not in df.columns:
                        raise ValueError(f"Target '{y_col}' not found")

                    y, X = patsy.dmatrices(formula, data=df, return_type='dataframe', NA_action='drop')
                    
                    if len(y) < (n_outer * 2): # Basic check
                        raise ValueError(f"Insufficient data (n={len(y)}) per la CV.")

                    # Run the nested CV, reusing the existing function
                    results = self.nested_cv_auc(X, y, use_xgb, n_outer, n_inner)
                    
                    result_line = (f"{formula:<65.65s} | {len(y):>5} | "
                                f"{results['auc_cal']:>16.4f} | {results['brier']:>12.4f}")

                except Exception as e:
                    print(f"Error while processing the formula '{formula}': {e}")
                    error_str = str(e).split('\n')[0]
                    result_line = f"{formula:<65.65s} | {'ERROR':>5} | {error_str:<20.20s}"

                self.root.after(0, lambda line=result_line: self.regression_results_text.insert(tk.END, line + "\n"))

            self.root.after(0, lambda: self.status_var.set("Regression analysis with nested CV finished."))

        except Exception as e:
            error_message = f"\nCritical error during the analysis:\n\n{type(e).__name__}: {e}"
            self.root.after(0, lambda: self.regression_results_text.insert(tk.END, error_message))
            self.root.after(0, lambda: self.status_var.set("The analysis failed. Check the output."))
            import traceback
            traceback.print_exc()


    def create_modeling_tab(self):
        """Creates the tab for predictive modeling with a flexible evaluation panel."""
        model_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(model_frame, text="Predictive Modeling")

        # --- Main Layout: Two Columns ---
        model_frame.columnconfigure(0, weight=1)  # Controls column
        model_frame.columnconfigure(1, weight=2)  # Results/plots column
        model_frame.rowconfigure(0, weight=1)

        # --- LEFT COLUMN: SETUP, EXECUTION, AND EXPORT ---
        left_column = ttk.Frame(model_frame, style='Card.TFrame')
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # 1. Model & CV Configuration
        config_lf = ttk.LabelFrame(left_column, text="1. Model & CV Configuration", style='Card.TLabelframe', padding=10)
        config_lf.pack(fill='x', pady=5, anchor='n')
        
        self.model_use_xgb = tk.BooleanVar(value=False)
        xgb_check = ttk.Checkbutton(config_lf, text="Use XGBoost (if available)", variable=self.model_use_xgb)
        xgb_check.pack(anchor='w')
        if not HAVE_XGB:
            xgb_check.configure(state='disabled')

        cv_frame = ttk.Frame(config_lf)
        cv_frame.pack(fill='x', pady=5)
        ttk.Label(cv_frame, text="Outer Folds:").pack(side='left')
        self.model_outer_folds = tk.IntVar(value=5)
        ttk.Entry(cv_frame, textvariable=self.model_outer_folds, width=5).pack(side='left', padx=5)
        ttk.Label(cv_frame, text="Inner Folds:").pack(side='left')
        self.model_inner_folds = tk.IntVar(value=3)
        ttk.Entry(cv_frame, textvariable=self.model_inner_folds, width=5).pack(side='left', padx=5)

        # 2. Model Evaluation Panel
        eval_lf = ttk.LabelFrame(left_column, text="2. Model Evaluation", style='Card.TLabelframe', padding=10)
        eval_lf.pack(fill='x', pady=15, anchor='n')

        ttk.Label(eval_lf, text="Enter Patsy Formula:").pack(anchor='w')
        self.custom_formula_var = tk.StringVar(value="Responder ~ age + BDI_pre + C(gender)")
        custom_formula_entry = ttk.Entry(eval_lf, textvariable=self.custom_formula_var, font=("Courier", 10))
        custom_formula_entry.pack(fill='x', pady=(2, 8))
        
        ttk.Button(eval_lf, text="Run Model Evaluation", style='Primary.TButton',
                command=self.run_custom_formula_evaluation).pack(fill='x')

        # 3. Export & Clear
        export_lf = ttk.LabelFrame(left_column, text="3. Results Management", style='Card.TLabelframe', padding=10)
        export_lf.pack(fill='x', pady=15, anchor='n')
        ttk.Button(export_lf, text="Save Evaluation Results", style='Success.TButton',
                command=self.export_evaluation_results).pack(fill='x')
        ttk.Button(export_lf, text="Clear Plots & Results", style='Info.TButton',
                command=self.clear_evaluation_plots).pack(fill='x', pady=5)

        # --- RIGHT COLUMN: RESULTS VISUALIZATION ---
        right_column = ttk.Frame(model_frame, style='Card.TFrame')
        right_column.grid(row=0, column=1, sticky="nsew")
        
        self.results_notebook = ttk.Notebook(right_column)
        self.results_notebook.pack(fill='both', expand=True)

        # Tab 1: ROC Curve
        roc_tab = ttk.Frame(self.results_notebook, padding=5)
        self.results_notebook.add(roc_tab, text="ROC Curve")
        self.roc_fig = Figure(figsize=(6, 5), facecolor=self.colors['surface'])
        self.roc_canvas = FigureCanvasTkAgg(self.roc_fig, master=roc_tab)
        self.roc_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.roc_ax = self.roc_fig.add_subplot(111)

        # Tab 2: Calibration Curve
        cal_tab = ttk.Frame(self.results_notebook, padding=5)
        self.results_notebook.add(cal_tab, text="Calibration Curve")
        self.cal_fig = Figure(figsize=(6, 5), facecolor=self.colors['surface'])
        self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=cal_tab)
        self.cal_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.cal_ax = self.cal_fig.add_subplot(111)

        # Tab 3: Text Report (kept for detailed output or debugging)
        report_tab = ttk.Frame(self.results_notebook, padding=5)
        self.results_notebook.add(report_tab, text="Text Report")
        self.modeling_results_text = tk.Text(report_tab, wrap="none", height=20, font=("Courier", 9), relief='solid', borderwidth=1)
        self.modeling_results_text.pack(fill='both', expand=True)
        
        # Initialize the plots with placeholder text
        self.clear_evaluation_plots()


    def run_modeling_pipeline(self):
        """Validate the inputs and start the modelling pipeline in a separate thread."""
        if self.behavioral_data.empty or self.biomarkers_df.empty:
            messagebox.showerror("Missing data", "Make sure that both the behavioural data and the EEG biomarkers have been loaded and generated.")
            return
        
        base_feats_str = self.model_base_features.get().strip()
        if not base_feats_str:
            messagebox.showwarning("Input required", "Define the baseline feature set.")
            return

        self.modeling_results_text.delete('1.0', tk.END)
        self.modeling_results_text.insert('1.0', "Starting the analysis, please wait.\nThis can take several minutes.\n\n")
        self.status_var.set("Running the predictive modelling...")
        
        threading.Thread(target=self._modeling_worker, daemon=True).start()

    def _modeling_worker(self):
        """The main worker function, which runs the whole nested CV pipeline."""
        try:
            # 1. Fetch the parameters from the GUI and prepare the merged DataFrame
            y_col = self.model_target_var.get()
            base_feats_str = self.model_base_features.get()
            plus_feats_str = self.model_plus_features.get()
            use_xgb = self.model_use_xgb.get()
            n_outer = self.model_outer_folds.get()
            n_inner = self.model_inner_folds.get()

            beh_df = self.behavioral_data.rename(columns={'sessID': 'session_ID'})
            beh_df['participant_ID'] = beh_df['participant_ID'].astype(str)
            beh_df['session_ID'] = pd.to_numeric(beh_df['session_ID'], errors='coerce')
            eeg_df = self.biomarkers_df.copy()
            eeg_df['participant_ID'] = eeg_df['participant_ID'].astype(str)
            eeg_df['session_ID'] = pd.to_numeric(eeg_df['session_ID'], errors='coerce')
            df = pd.merge(beh_df, eeg_df, on=['participant_ID', 'session_ID'], how='inner')
            
            if df.empty: raise ValueError("Merging the data produced an empty DataFrame. Check the participant and session IDs.")
            if y_col not in df.columns: raise ValueError(f"The target column '{y_col}' was not found in the merged data.")

            # 2. Parse the formulas and build the design matrices with patsy
            base_feats_clean = " + ".join(filter(None, re.split(r'[,\s]+', base_feats_str)))
            base_formula = f"{y_col} ~ {base_feats_clean}"
            y_base, X_base = patsy.dmatrices(base_formula, data=df, return_type='dataframe', NA_action='drop')

            # 3. Run the nested CV for the baseline model
            report = ["--- Predictive Modeling Report ---\n"]
            report.append(f"Target variable: {y_col}")
            report.append(f"Model: {'XGBoost' if use_xgb and HAVE_XGB else 'Elastic-Net Logistic Regression'}")
            report.append(f"Validation: {n_outer}-fold Nested CV (Inner folds: {n_inner})\n")

            res_base = self.nested_cv_auc(X_base, y_base, use_xgb, n_outer, n_inner)
            report.append("[Baseline model performance]")
            report.append(f"Feature: {base_formula.split('~')[1].strip()}")
            report.append(f"  - AUC Calibrata: {res_base['auc_cal']:.4f}")
            report.append(f"  - Brier Score:    {res_base['brier']:.4f}\n")

            # 4. Run the nested CV for the advanced model, where applicable
            if plus_feats_str.strip():
                plus_feats_clean = " + ".join(filter(None, re.split(r'[,\s]+', plus_feats_str)))
                combined_formula = f"{base_formula} + {plus_feats_clean}"
                y_plus, X_plus = patsy.dmatrices(combined_formula, data=df, return_type='dataframe', NA_action='drop')
                
                res_plus = self.nested_cv_auc(X_plus, y_plus, use_xgb, n_outer, n_inner)
                report.append("[Advanced model performance (baseline plus extra features)]")
                report.append(f"Feature: {combined_formula.split('~')[1].strip()}")
                report.append(f"  - AUC Calibrata: {res_plus['auc_cal']:.4f}")
                report.append(f"  - Brier Score:    {res_plus['brier']:.4f}\n")

                # 5. Compute the delta AUC
                self.root.after(0, lambda: self.status_var.set("Bootstrapping Delta AUC..."))
                d_mean, (d_lo, d_hi), p_value, _p_one = self.delta_auc_boot(
                    y_plus.values, res_base["oof_cal_pred"], res_plus["oof_cal_pred"]
                )
                report.append("[Model Comparison]")
                report.append(f"  - Delta AUC (Enhanced - Baseline): {d_mean:.4f}")
                report.append(f"  - 95% Confidence Interval:       [{d_lo:.4f}, {d_hi:.4f}]")
                # Add the row holding the p value
                report.append(f"  - p-value (one-sided):           {p_value:.4f}\n")

                conclusion = "gives a statistically significant improvement (p < 0.05)." if p_value < 0.05 else "does NOT give a significant improvement (p >= 0.05)."
                report.append(f"Conclusion: judging by the p value, adding the new features {conclusion}")

            # 6. Refresh the GUI
            final_report = "\n".join(report)
            self.root.after(0, lambda: self.modeling_results_text.delete('1.0', tk.END))
            self.root.after(0, lambda: self.modeling_results_text.insert('1.0', final_report))
            self.root.after(0, lambda: self.status_var.set("Predictive modelling finished."))
            self.root.after(0, lambda: self.run_interp_button.config(state="normal"))

        except Exception as e:
            error_message = f"An error occurred during the modelling process:\n\n{type(e).__name__}: {e}"
            self.root.after(0, lambda: self.modeling_results_text.insert(tk.END, f"\n\n--- ERROR ---\n{error_message}"))
            self.root.after(0, lambda: self.status_var.set("Modelling failed. Check the output for the error."))
            self.root.after(0, lambda: self.run_interp_button.config(state="disabled"))
            import traceback
            traceback.print_exc()

    def run_interpretation_pipeline(self):
        """Validate the inputs and start the interpretation pipeline in a separate thread."""
        self.status_var.set("Starting the interpretation analysis...")
        for widget in self.interp_canvas_frame.winfo_children():
            widget.destroy()
        self.results_notebook.select(1) 
        ttk.Label(self.interp_canvas_frame, text="Generating the plots, please wait.").pack()
        threading.Thread(target=self._interpretation_worker, daemon=True).start()

    def _interpretation_worker(self):
        """Worker that fits the final model and generates the interpretation plots."""
        try:
            y_col = self.model_target_var.get()
            base_feats_str = self.model_base_features.get()
            plus_feats_str = self.model_plus_features.get()
            use_xgb = self.model_use_xgb.get()

            beh_df = self.behavioral_data.rename(columns={'sessID': 'session_ID'})
            beh_df['participant_ID'] = beh_df['participant_ID'].astype(str)
            beh_df['session_ID'] = pd.to_numeric(beh_df['session_ID'], errors='coerce')
            eeg_df = self.biomarkers_df.copy()
            eeg_df['participant_ID'] = eeg_df['participant_ID'].astype(str)
            eeg_df['session_ID'] = pd.to_numeric(eeg_df['session_ID'], errors='coerce')
            df = pd.merge(beh_df, eeg_df, on=['participant_ID', 'session_ID'], how='inner')

            base_feats_clean = " + ".join(filter(None, re.split(r'[,\s]+', base_feats_str)))
            plus_feats_clean = " + ".join(filter(None, re.split(r'[,\s]+', plus_feats_str)))

            formula = f"{y_col} ~ {base_feats_clean} + {plus_feats_clean}" if plus_feats_clean else f"{y_col} ~ {base_feats_clean}"
            
            y_patsy, X_patsy = patsy.dmatrices(formula, data=df, return_type='dataframe', NA_action='drop')
            best_feats = X_patsy.columns.tolist()
            
            self.root.after(0, lambda: self.status_var.set("Fitting the final model on all of the data..."))
            
            numeric_cols_for_pipe = X_patsy.select_dtypes(include=np.number).columns.tolist()
            preprocessor = self.build_preprocessor(numeric_cols_for_pipe)
            
            model = self.make_xgb()[0] if use_xgb and HAVE_XGB else LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000, random_state=0)
            
            final_pipe = Pipeline([("pre", preprocessor), ("clf", model)])
            final_pipe.fit(X_patsy, y_patsy.values.ravel())

            self.root.after(0, lambda: self.status_var.set("Computing the permutation importance..."))
            perm_imp = permutation_importance(final_pipe, X_patsy, y_patsy.values.ravel(), n_repeats=30, scoring="roc_auc", n_jobs=-1, random_state=0)
            perm_df = pd.DataFrame({'feature': best_feats, 'importance_mean': perm_imp.importances_mean, 'importance_std': perm_imp.importances_std}).sort_values('importance_mean', ascending=False)

            shap_values = None
            if use_xgb and HAVE_XGB and HAVE_SHAP:
                try:
                    self.root.after(0, lambda: self.status_var.set("Computing the SHAP values..."))
                    X_transformed = final_pipe.named_steps["pre"].transform(X_patsy)
                    explainer = shap.Explainer(final_pipe.named_steps["clf"])
                    shap_values = explainer(X_transformed)
                except Exception as e:
                    print(f"Could not generate the SHAP values: {e}")
                    self.root.after(0, lambda e=e: self.modeling_results_text.insert(tk.END, f"\n\nSHAP warning: could not generate the SHAP plot. Error: {e}"))

            self.root.after(0, self.plot_interpretation_results, perm_df, shap_values, best_feats, X_patsy)

        except Exception as e:
            error_message = f"An error occurred during the interpretation:\n\n{type(e).__name__}: {e}"
            self.root.after(0, lambda: messagebox.showerror("Interpretation error", error_message))
            self.root.after(0, lambda: self.status_var.set("Interpretation failed."))
            import traceback
            traceback.print_exc()

    def plot_interpretation_results(self, perm_df, shap_values, feature_names, X_df):
        """Build and show the interpretation plots in the GUI."""
        for widget in self.interp_canvas_frame.winfo_children():
            widget.destroy()

        nrows = 2 if shap_values is not None and HAVE_SHAP else 1
        fig = Figure(figsize=(10, 6 * nrows), facecolor=self.colors['surface'])

        ax1 = fig.add_subplot(nrows, 1, 1)
        perm_df_plot = perm_df.head(20)
        ax1.barh(perm_df_plot['feature'], perm_df_plot['importance_mean'], xerr=perm_df_plot['importance_std'], align='center', alpha=0.8, ecolor='grey', capsize=3)
        ax1.invert_yaxis()
        ax1.set_title("Permutation Feature Importance (Top 20)", color=self.colors['text_primary'])
        ax1.set_xlabel("Drop in the ROC AUC score", color=self.colors['text_secondary'])
        ax1.tick_params(axis='x', colors=self.colors['text_secondary'])
        ax1.tick_params(axis='y', colors=self.colors['text_primary'])
        ax1.axvline(0, color='grey', linewidth=0.8, linestyle='--')

        if shap_values is not None and HAVE_SHAP:
            ax2 = fig.add_subplot(nrows, 1, 2)
            shap.summary_plot(shap_values, X_df, feature_names=feature_names, show=False, plot_size=None, ax=ax2)
            ax2.set_title("SHAP Summary Plot", color=self.colors['text_primary'])
            ax2.tick_params(colors=self.colors['text_secondary'])

        fig.tight_layout(pad=3.0)

        canvas = FigureCanvasTkAgg(fig, master=self.interp_canvas_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.interp_canvas_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.status_var.set("Interpretation analysis finished.")

    def nested_cv_auc(self, X_df, y_series, use_xgb=False, n_outer=5, n_inner=3,
                      seed=0, select_features=True, k_grid=None):
        """Nested cross-validation with imputation AND feature selection inside the loop.

        METHODOLOGICAL CORRECTION. In the earlier version the choice of which
        features to include was made by comparing candidate models on the whole
        dataset, hence outside the outer loop. The resulting AUCs were the maxima
        of a search rather than unbiased estimates. Here SelectKBest is part of the
        pipeline, is refitted on every training fold and never sees the test fold.

        Passing select_features=False reproduces the earlier behaviour, which is
        useful only for historical comparison.
        """
        from sklearn.feature_selection import SelectKBest, f_classif

        X, y = X_df, y_series.values.ravel()
        numeric_cols = [col for col in X.columns if X[col].dtype in [np.float64, np.int64] and col != 'Intercept']
        pre = self.build_preprocessor(numeric_cols)
        base, grid = self.make_xgb() if use_xgb and HAVE_XGB else self.make_elastic_net()

        if select_features and X.shape[1] > 3:
            pipe = Pipeline([("pre", pre),
                             ("sel", SelectKBest(f_classif, k=min(8, X.shape[1]))),
                             ("clf", base)])
            if k_grid is None:
                k_grid = sorted({k for k in (3, 5, 8, 12, X.shape[1]) if k <= X.shape[1]})
            grid = dict(grid); grid["sel__k"] = k_grid
        else:
            pipe = Pipeline([("pre", pre), ("clf", base)])

        outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        oof_pred = np.zeros_like(y, dtype=float)
        oof_cal_pred = np.zeros_like(y, dtype=float)
        selected_per_fold = []

        self.root.after(0, lambda: self.status_var.set(f"Running nested CV for {len(X.columns)} feature..."))
        for i, (tr, te) in enumerate(outer.split(X, y)):
            self.root.after(0, lambda i=i: self.status_var.set(f"Processing outer fold {i+1}/{n_outer}..."))
            inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
            gs = GridSearchCV(pipe, grid, cv=inner, n_jobs=-1, scoring="roc_auc", refit=True)
            gs.fit(X.iloc[tr], y[tr])
            oof_pred[te] = gs.best_estimator_.predict_proba(X.iloc[te])[:, 1]
            # sigmoid (Platt) rather than isotonic: with small inner folds isotonic produces
            # probabilities of exactly 0 and 1 and destabilises the calibration slope
            calibrated_clf = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid", cv=inner)
            calibrated_clf.fit(X.iloc[tr], y[tr])
            oof_cal_pred[te] = calibrated_clf.predict_proba(X.iloc[te])[:, 1]
            if "sel" in gs.best_estimator_.named_steps:
                mask = gs.best_estimator_.named_steps["sel"].get_support()
                selected_per_fold.append(list(np.asarray(X.columns)[mask]))

        prevalence = float(np.mean(y))
        brier = brier_score_loss(y, oof_cal_pred)
        return {"auc": roc_auc_score(y, oof_pred),
                "auc_cal": roc_auc_score(y, oof_cal_pred),
                "brier": brier,
                "brier_skill_score": 1.0 - brier / (prevalence * (1 - prevalence)),
                "selected_per_fold": selected_per_fold,
                "oof_pred": oof_pred,
                "oof_cal_pred": oof_cal_pred}


    # ------------------------------------------------------------------
    # Helpers added during revision
    # ------------------------------------------------------------------
    @staticmethod
    def brier_skill_score(y_true, p_pred):
        """Brier skill score against the intercept-only model.

        Negative values mean the model predicts worse than the plain response
        rate of the cohort. This is the diagnostic showing that the baseline
        clinical model does not hold up as a comparator.
        """
        y = np.asarray(y_true, dtype=float)
        prev = y.mean()
        return float(1.0 - brier_score_loss(y, p_pred) / (prev * (1 - prev)))

    @staticmethod
    def bh_fdr(pvalues):
        """Benjamini-Hochberg. To be applied to the WHOLE family of models tested,
        not only to the ones reported, otherwise the denominator is wrong."""
        p = np.asarray(pvalues, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        q = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.minimum(q, 1.0)
        return out

    @staticmethod
    def normalised_ppv(y_true, p_pred, threshold=0.5):
        """nPPV as a percentage: the PPV at the threshold divided by the response rate
        of the cohort. It is the primary metric of the TDBRAIN Challenge, where the
        success criterion is 110%. Careful: if the threshold is chosen on the same
        data, the value is optimistic."""
        y = np.asarray(y_true, dtype=float)
        sel = np.asarray(p_pred) >= threshold
        if sel.sum() < 5:
            return float("nan"), int(sel.sum())
        return float(y[sel].mean() / y.mean() * 100.0), int(sel.sum())

    @staticmethod
    def net_benefit(y_true, p_pred, threshold):
        """Net benefit per la decision curve analysis (Vickers & Elkin, 2006)."""
        y = np.asarray(y_true, dtype=float)
        pred = np.asarray(p_pred) >= threshold
        tp = float(((pred) & (y == 1)).sum()); fp = float(((pred) & (y == 0)).sum())
        n = len(y)
        return tp / n - (fp / n) * (threshold / (1 - threshold))

    def delta_auc_boot(self
, y_true, p_base, p_plus, B=1000, seed=0):
        """
        Compute the bootstrap difference in AUC and its p value.
        """
        rng = np.random.default_rng(seed)
        deltas = []
        y_true = np.asarray(y_true)
        for i in range(B):
            idx = resample(np.arange(len(y_true)), replace=True, stratify=y_true, random_state=int(rng.integers(1e9)))
            if len(np.unique(y_true[idx])) < 2:
                continue
            auc_plus = roc_auc_score(y_true[idx], p_plus[idx])
            auc_base = roc_auc_score(y_true[idx], p_base[idx])
            deltas.append(auc_plus - auc_base)
            
        deltas = np.asarray(deltas)
        delta_mean = float(np.mean(deltas))
        ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])

        # one-sided p (the earlier behaviour) and two-sided p.
        # A one-sided p is legitimate only if the direction was pre-specified:
        # once negative deltas are reported too, the test is two-sided.
        p_one = float(np.mean(deltas <= 0))
        p_two = float(min(1.0, 2 * min(p_one, 1 - p_one)))

        return delta_mean, (float(ci_lower), float(ci_upper)), p_two, p_one
        
    def build_preprocessor(self, numeric_cols):
        """Build a preprocessing pipeline for the numeric data."""
        num_pipeline = Pipeline(steps=[("imputer", IterativeImputer(random_state=0, sample_posterior=True, max_iter=15)), ("scaler", StandardScaler())])
        return ColumnTransformer([("numeric", num_pipeline, numeric_cols)], remainder="passthrough")

    def make_elastic_net(self):
        """Build an elastic-net logistic regression and its hyperparameter grid."""
        lr = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, n_jobs=-1, random_state=0)
        grid = {"clf__C": np.logspace(-3, 2, 8), "clf__l1_ratio": [0.15, 0.3, 0.5, 0.7, 0.9]}
        return lr, grid

    def make_xgb(self):
        """Build an XGBoost model and its hyperparameter grid."""
        xgb = XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=0, n_jobs=-1)
        grid = {"clf__max_depth": [3, 4], "clf__min_child_weight": [1, 3], "clf__subsample": [0.7, 0.9], "clf__colsample_bytree": [0.7, 0.9]}
        return xgb, grid


    def create_connectivity_tab(self):
        """Build the connectivity analysis tab."""
        conn_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(conn_frame, text="Connectivity")

        # --- two-column layout: controls on the left, plots on the right ---
        conn_frame.columnconfigure(0, weight=1)
        conn_frame.columnconfigure(1, weight=2)

        # --- LEFT COLUMN: CONTROLS AND OPTIONS ---
        left_column = ttk.Frame(conn_frame, style='Card.TFrame')
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # 1. Main control panel
        control_lf = ttk.LabelFrame(left_column, text="1. Execution Control", style='Card.TLabelframe', padding=10)
        control_lf.pack(fill='x', pady=5)

        ttk.Button(control_lf, text="Run Connectivity Analysis", style='Primary.TButton', command=self.run_connectivity_analysis).pack(fill='x', pady=5)
        
        # Progress bar
        progress_frame = ttk.Frame(control_lf)
        progress_frame.pack(fill='x', pady=(5, 0))
        self.conn_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.conn_progress.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.conn_progress_percent = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.conn_progress_percent).pack(side='right')

        # 2. Analysis parameters
        params_lf = ttk.LabelFrame(left_column, text="2. Analysis Parameters", style='Card.TLabelframe', padding=10)
        params_lf.pack(fill='x', pady=15)


        self.conn_methods = {
            'dpli': tk.BooleanVar(value=True),      # Corrected from "dwpli"
            'imcoh': tk.BooleanVar(value=True),     # Already correct
            'plv': tk.BooleanVar(value=True),       # Already correct
            'coh': tk.BooleanVar(value=True)        # Corrected from "coherence"
        }
        # The text shown in the GUI is updated too, for consistency
        for name, var in self.conn_methods.items():
            display_text = "dwPLI" if name == 'dpli' else name.upper()
            ttk.Checkbutton(params_lf, text=display_text, variable=var).pack(anchor='w', padx=10)


        # --- RIGHT COLUMN: DISPLAY ---
        right_column = ttk.Frame(conn_frame, style='Card.TFrame')
        right_column.grid(row=0, column=1, sticky="nsew")

        viz_lf = ttk.LabelFrame(right_column, text="Connectivity Visualization", style='Card.TLabelframe', padding=10)
        viz_lf.pack(fill='both', expand=True)

        # Display controls
        viz_controls = ttk.Frame(viz_lf)
        viz_controls.pack(fill='x', pady=(0, 10))
        
        ttk.Label(viz_controls, text="Subject:").pack(side='left')
        self.selected_conn_subject = tk.StringVar()
        self.conn_subject_dropdown = ttk.Combobox(viz_controls, textvariable=self.selected_conn_subject, state='readonly', width=30)
        self.conn_subject_dropdown.pack(side='left', padx=5)

        ttk.Label(viz_controls, text="Band:").pack(side='left')
        self.selected_conn_band = tk.StringVar()
        self.conn_band_dropdown = ttk.Combobox(viz_controls, textvariable=self.selected_conn_band, state='readonly', values=['delta', 'theta', 'alpha', 'beta', 'gamma'])
        self.conn_band_dropdown.pack(side='left', padx=5)
        self.conn_band_dropdown.set('alpha') # Default value

        ttk.Button(viz_controls, text="Show Matrix", style='Secondary.TButton', command=self.update_connectivity_viz).pack(side='left', padx=10)

        # Area for the plot
        self.conn_fig = Figure(figsize=(8, 6), facecolor=self.colors['surface'])
        self.conn_canvas = FigureCanvasTkAgg(self.conn_fig, viz_lf)
        self.conn_canvas.get_tk_widget().pack(fill='both', expand=True)

    def run_connectivity_analysis(self):
        """
        Orchestrate the connectivity analysis through a process pool,
        one subject at a time, so that RAM is used efficiently.
        """
        if not self.preprocessed_data:
            messagebox.showwarning("Missing data", "Run the preprocessing and the ICA first.")
            return

        self.status_var.set("Starting the connectivity analysis...")
        self.conn_progress.start()
        
        # Start the orchestrator in a thread so the GUI is not blocked
        threading.Thread(target=self._connectivity_orchestrator, daemon=True).start()


    def _connectivity_orchestrator(self):
        """
        Manage the process pool for the connectivity analysis.
        It hands out the tasks and collects the complete results, matrices and features.
        """
        scratch_dir = cfg.SCRATCH_DIR
        try:
            os.makedirs(scratch_dir, exist_ok=True)
        except OSError as exc:
            self.root.after(0, lambda: messagebox.showerror(
                "Disk error", f"Scratch directory not usable: {scratch_dir}\n{exc}\n"
                              f"Set TDBRAIN_SCRATCH_DIR to a writable folder."))
            self.root.after(0, self.conn_progress.stop)
            return

        try:
            methods_to_run = [name for name, var in self.conn_methods.items() if var.get()]
            if not methods_to_run:
                self.root.after(0, lambda: messagebox.showwarning("Nothing selected", "Select at least one metric."))
                return

            freq_bands = {'delta': (1.25, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
            
            with tempfile.TemporaryDirectory(dir=scratch_dir) as temp_dir:
                tasks = []
                for subject_id, raw in self.preprocessed_data.items():
                    temp_raw_path = os.path.join(temp_dir, f"{subject_id}-raw.fif")
                    raw.save(temp_raw_path, overwrite=True, verbose=False)
                    
                    task_args = {
                        'subject_id': subject_id,
                        'raw_path': temp_raw_path,
                        'methods_to_run': methods_to_run,
                        'freq_bands': freq_bands
                    }
                    tasks.append(task_args)

                n_processes = max(1, multiprocessing.cpu_count() - 1)
                with multiprocessing.Pool(processes=n_processes) as pool:
                    results = pool.map(connectivity_worker_process, tasks)

                # Collect the results, which are already complete at this point
                self.connectivity_results = {subject_id: result_data for subject_id, result_data in results if result_data}

            self.root.after(0, self._on_connectivity_analysis_complete)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=e: messagebox.showerror("Critical connectivity error", f"An error occurred:\n{err}"))
        finally:
            self.root.after(0, self.conn_progress.stop)

    def _connectivity_worker(self):
        """
        Worker that carries out the connectivity computations.
        A robust version that works in memory but processes one subject at a time,
        to avoid running out of RAM and to avoid disk problems.
        """
        print("\n" + "="*20 + " CONNECTIVITY WORKER STARTED (IN MEMORY, SERIAL) " + "="*20)
        try:
            methods_to_run = [name for name, var in self.conn_methods.items() if var.get()]
            if not methods_to_run:
                self.root.after(0, lambda: messagebox.showwarning("Nothing selected", "Select at least one connectivity metric."))
                return

            freq_bands = {'delta': (1.25, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
            
            subjects_to_process = list(self.preprocessed_data.keys())
            total_subjects = len(subjects_to_process)
            if not subjects_to_process:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No preprocessed data to analyse."))
                return

            self.connectivity_results = {}
            
            for i, subject_id in enumerate(subjects_to_process):
                self.root.after(0, lambda p=int(((i+1)/total_subjects)*100), s=subject_id: (
                    self.conn_progress_percent.set(f"{p}%"),
                    self.status_var.set(f"Connectivity: {s} ({i+1}/{total_subjects})...")
                ))

                raw = self.preprocessed_data[subject_id].copy()
                self.connectivity_results[subject_id] = {'matrices': {}, 'features': {}}
                
                cond_name = "EC" if 'EC' in subject_id.upper() else "EO" if 'EO' in subject_id.upper() else "unknown"
                
                # Longer duration, which fixes the RuntimeWarning and suits the delta band
                epochs = mne.make_fixed_length_epochs(raw, duration=5.0, overlap=2.5, preload=True, verbose=False)
                epochs.pick_types(eeg=True, verbose=False)
                epochs = mne.preprocessing.compute_current_source_density(epochs, verbose=False)
                
                ch_names_for_features = epochs.ch_names

                for band_name, (fmin, fmax) in freq_bands.items():
                    con_results_band = mne_connectivity.spectral_connectivity_epochs(
                        epochs, method=methods_to_run, mode='multitaper', sfreq=epochs.info['sfreq'],
                        fmin=fmin, fmax=fmax, faverage=True, n_jobs=-1, verbose=False
                    )
                    
                    for con_result, method_name in zip(con_results_band, methods_to_run):
                        matrix = con_result.get_data(output='dense').squeeze()
                        matrix_key = f"{method_name}_{band_name}"
                        self.connectivity_results[subject_id]['matrices'][matrix_key] = matrix
                        
                        features = self._extract_connectivity_features(matrix, ch_names_for_features)
                        for feat_name, feat_val in features.items():
                            feature_key = f"{method_name}_{band_name}_{feat_name}_{cond_name}"
                            self.connectivity_results[subject_id]['features'][feature_key] = feat_val

            print("\n" + "="*20 + " CONNECTIVITY WORKER FINISHED " + "="*20)
            self.root.after(0, self._on_connectivity_analysis_complete)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=e: messagebox.showerror("Critical connectivity error", f"An error occurred:\n{err}"))
        finally:
            self.root.after(0, self.conn_progress.stop)


    
    def _extract_connectivity_features(
        self,
        matrix: np.ndarray,
        ch_names_eeg: Sequence[str],
        roi_defs: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, float]:
        """
        Extract features from a connectivity matrix.
        - Supports parameterisable ROIs (default: DLPFC L/R).
        - Robust handling of NaN, shape and symmetry.
        - Network metrics on the MST.

        Parameters
        ----------
        matrix : np.ndarray
            (N x N) connectivity matrix, for instance corr, coh or PLI.
        ch_names_eeg : Sequence[str]
            Channel names, in the order matching the matrix.
        roi_defs : dict, optional
            For instance {"dlpfc_l": ["F3","F7","Fp1"], "dlpfc_r": ["F4","F8","Fp2"]}.

        Returns
        -------
        Dict[str, float]
            Dictionary with fixed keys, holding NaN where a value is unavailable.
        """
        # --- expected keys: the features that will be computed ---
        out_keys = [
            "global_mean",
            "dlpfc_l_mean",
            "dlpfc_r_mean",
            "dlpfc_lr_mean",           # Inter-emisferica L↔R
            "mst_leaf_fraction",
            "mst_diameter",
            "mst_mean_eccentricity",
            "mst_mean_weight",         # mean weight of the MST edges
        ]
        features = {k: np.nan for k in out_keys}

        # --- 1. Validazioni di base ---
        if matrix is None or not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            return features
        n, m = matrix.shape
        if n == 0 or n != m or n != len(ch_names_eeg):
            print(f"WARNING: invalid matrix (shape: {matrix.shape}) or inconsistent with the channels ({len(ch_names_eeg)}).")
            return features

        # --- 2. Clean and prepare the matrix ---
        A = matrix.astype(float, copy=True)
        # Force symmetry and zero the diagonal, for safety
        A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)

        # --- 3. Compute the means, NaN-safe ---
        try:
            # Global mean
            if n > 1:
                r, c = np.triu_indices(n, k=1)
                features["global_mean"] = float(np.nanmean(A[r, c]))

            # ROI definitions
            if roi_defs is None:
                roi_defs = {
                    "dlpfc_l": ["F3", "F7", "Fp1"],
                    "dlpfc_r": ["F4", "F8", "Fp2"],
                }
            
            # channel -> index map, for fast lookups
            idx_map = {ch: i for i, ch in enumerate(ch_names_eeg)}
            
            def _indices_for_roi(names: List[str]) -> List[int]:
                return [idx_map[ch] for ch in names if ch in idx_map]

            # Within-ROI means
            l_idx = _indices_for_roi(roi_defs.get("dlpfc_l", []))
            r_idx = _indices_for_roi(roi_defs.get("dlpfc_r", []))

            if len(l_idx) >= 2:
                sub_l = A[np.ix_(l_idx, l_idx)]
                r_l, c_l = np.triu_indices(len(l_idx), k=1)
                features["dlpfc_l_mean"] = float(np.nanmean(sub_l[r_l, c_l]))

            if len(r_idx) >= 2:
                sub_r = A[np.ix_(r_idx, r_idx)]
                r_r, c_r = np.triu_indices(len(r_idx), k=1)
                features["dlpfc_r_mean"] = float(np.nanmean(sub_r[r_r, c_r]))

            # Media Inter-ROI
            if len(l_idx) >= 1 and len(r_idx) >= 1:
                cross_matrix = A[np.ix_(l_idx, r_idx)]
                features["dlpfc_lr_mean"] = float(np.nanmean(cross_matrix))

        except Exception as e:
            print(f"!!! Error while computing the connectivity means: {e}")

        # --- 4. Graph theory features (MST) ---
        try:
            # Turn weights into distances (1 minus the absolute value)
            D = 1.0 - np.abs(A)
            np.fill_diagonal(D, 0.0) # The distance of a node from itself is 0

            G = nx.from_numpy_array(D)
            mst = nx.minimum_spanning_tree(G, weight="weight")

            if mst.number_of_nodes() > 0:
                # Leaf fraction
                degrees = np.array([d for _, d in mst.degree()])
                features["mst_leaf_fraction"] = float(np.sum(degrees == 1) / mst.number_of_nodes())
                
                # Diameter and eccentricity
                ecc = nx.eccentricity(mst)
                features["mst_diameter"] = float(nx.diameter(mst, e=ecc))
                features["mst_mean_eccentricity"] = float(np.mean(list(ecc.values())))

                # Mean edge weight
                if mst.number_of_edges() > 0:
                    weights = np.array([d["weight"] for _, _, d in mst.edges(data=True)])
                    features["mst_mean_weight"] = float(np.mean(weights))

        except Exception as e:
            print(f"!!! Error while computing the MST features: {e}")

        return features

        
    def create_session_filter_tab(self):
        """
        Build the tab that filters the files by one or more selected sessions.
        """
        session_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=20)
        self.notebook.add(session_frame, text="Session Filter")

        # --- control panel at the top ---
        control_panel = ttk.Frame(session_frame, style='Card.TFrame')
        control_panel.pack(fill='x', pady=(0, 20))

        ttk.Label(control_panel, text="Filter Files by Session", style='Title.TLabel').pack(side='left', anchor='w')

        # Frame holding the filter controls, the checkboxes and the button
        filter_controls_frame = ttk.Frame(control_panel, style='Card.TFrame')
        filter_controls_frame.pack(side='right', anchor='e')

        # Variables backing the checkboxes.
        self.session_vars = {
            "ses-1": tk.BooleanVar(value=True),
            "ses-2": tk.BooleanVar(value=True),
            "ses-3": tk.BooleanVar(value=True)
        }

        # The checkboxes are built dynamically
        for session_name, var in self.session_vars.items():
            chk = ttk.Checkbutton(filter_controls_frame, text=session_name.replace('-', ' ').capitalize(), variable=var)
            chk.pack(side='left', padx=5)

        # A single button that applies the selection
        apply_button = ttk.Button(filter_controls_frame, text="Apply Session Filter", 
                                  style='Primary.TButton', command=self.apply_multi_session_filter)
        apply_button.pack(side='left', padx=(10, 0))

        # --- two-column layout ---
        main_layout = ttk.Frame(session_frame, style='Card.TFrame')
        main_layout.pack(fill='both', expand=True)
        main_layout.columnconfigure(0, weight=1)
        main_layout.columnconfigure(1, weight=1)
        main_layout.rowconfigure(0, weight=1)

        # Left column: available files
        left_panel = ttk.Frame(main_layout, style='Card.TFrame')
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.session_available_files_count_var = tk.StringVar(value="Available Files from Subject Filter (0)")
        ttk.Label(left_panel, textvariable=self.session_available_files_count_var, style='Heading.TLabel').pack(anchor='w', pady=(0, 5))
        self.session_available_files_listbox = tk.Listbox(left_panel, height=15, bg=self.colors['surface'], fg=self.colors['text_primary'])
        self.session_available_files_listbox.pack(fill='both', expand=True)

        # Right column: selected files
        right_panel = ttk.Frame(main_layout, style='Card.TFrame')
        right_panel.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        self.session_processing_files_count_var = tk.StringVar(value="Files for Next Step (0)")
        ttk.Label(right_panel, textvariable=self.session_processing_files_count_var, style='Heading.TLabel').pack(anchor='w', pady=(0, 5))
        self.session_processing_files_listbox = tk.Listbox(right_panel, height=15, bg='#eaf3fb', fg=self.colors['text_primary'])
        self.session_processing_files_listbox.pack(fill='both', expand=True)


    def _on_connectivity_analysis_complete(self):
        """
        Function called once the analysis is over, to refresh the GUI.
        """
        self.status_var.set("Connectivity analysis finished. The results are ready.")
        
        # Fill the dropdown with the FILE NAMES rather than the subject IDs
        file_list = list(self.connectivity_results.keys())
        self.conn_subject_dropdown.configure(values=file_list)
        if file_list:
            self.selected_conn_subject.set(file_list[0])
        self._update_gt_subject_dropdown()
        # Simulate a click on 'Show Matrix' so the first result appears
        self.update_connectivity_viz()
        
        messagebox.showinfo("Success", "The connectivity analysis has finished.")

    def update_connectivity_viz(self):
        """
        Draw the connectivity matrix for the selected subject, band and metric.
        Final version, working against the right data structure.
        """
        # 1. Fetch the parameters from the GUI
        filename = self.selected_conn_subject.get()
        band = self.selected_conn_band.get()
        
        # Take the first metric the user selected, for the display
        method_to_display = next((name for name, var in self.conn_methods.items() if var.get()), None)

        if not all([filename, band, method_to_display]):
            # If something is missing, do nothing
            return

        self.conn_fig.clear()
        ax = self.conn_fig.add_subplot(111)

        # 2. Reach the data directly and correctly
        subject_data = self.connectivity_results.get(filename, {})
        subject_matrices = subject_data.get('matrices', {})
        
        # 3. Build the matrix key and fetch the matrix
        matrix_key = f"{method_to_display}_{band}"
        matrix = subject_matrices.get(matrix_key)

        if matrix is None:
            ax.text(0.5, 0.5, f"No data for:\n{filename}\nMethod: {method_to_display.upper()}\nBand: {band}", 
                    ha='center', va='center', fontsize=10)
            self.conn_canvas.draw()
            return

        # 4. Draw the matrix
        im = ax.imshow(matrix, cmap='viridis', origin='lower', interpolation='nearest')
        self.conn_fig.colorbar(im, ax=ax)
        
        # Take the channel names from the 'epochs' object that was filtered
        ch_names = self.preprocessed_data[filename].copy().pick_types(eeg=True).ch_names
        
        ax.set_xticks(np.arange(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=8)
        
        ax.set_title(f"{method_to_display.upper()} - {band.capitalize()} Band\n({filename})")
        self.conn_fig.tight_layout()
        self.conn_canvas.draw()


    def debug_merge_data(self):
        """
        Perform the data merge only, printing information useful for debugging
        without launching the whole regression analysis.
        """
        try:
            print("\n" + "="*20 + " DATA DEBUG STARTED " + "="*20)
            if self.behavioral_data.empty:
                print("!!! DEBUG ERROR: behavioural data not loaded.")
                messagebox.showwarning("Missing data", "Load the behavioural data first.")
                return
            if not hasattr(self, 'biomarkers_df') or self.biomarkers_df.empty:
                print("!!! DEBUG ERROR: biomarker data not generated.")
                messagebox.showwarning("Missing data", "Generate the biomarkers first (Summary tab -> Refresh Summary).")
                return

            # 1. Prepare the data, with the same logic as the worker
            behavioral_df_prepared = self.behavioral_data.rename(columns={'sessID': 'session_ID'})
            behavioral_df_prepared['participant_ID'] = behavioral_df_prepared['participant_ID'].astype(str)
            behavioral_df_prepared['session_ID'] = pd.to_numeric(behavioral_df_prepared['session_ID'], errors='coerce')
            
            eeg_df_prepared = self.biomarkers_df.copy()
            eeg_df_prepared['participant_ID'] = eeg_df_prepared['participant_ID'].astype(str)
            eeg_df_prepared['session_ID'] = pd.to_numeric(eeg_df_prepared['session_ID'], errors='coerce')

            # 2. Perform the merge
            merged_data = pd.merge(behavioral_df_prepared, eeg_df_prepared, on=['participant_ID', 'session_ID'], how='inner')
            
            if merged_data.empty:
                print("!!! DEBUG ERROR: the merge produced no row. Check the participant and session IDs.")
                messagebox.showerror("Merge error", "No match was found between the EEG and the behavioural data.")
                return

            # 3. Print the debugging information
            print("\n--- INFORMATION ON THE MERGED DATAFRAME ---")
            print(f"Number of rows in the merged DataFrame: {len(merged_data)}")
            print("\nFirst 5 rows of the merged DataFrame:")
            print(merged_data.head().to_string())
            print("\nColumns available in the merged DataFrame:")
            print(merged_data.columns.tolist())
            print("\nData types and non-null counts:")
            merged_data.info()
            
            print("\n" + "="*20 + " DATA DEBUG FINISHED " + "="*20)
            messagebox.showinfo("Debug finished", "The information about the merged DataFrame has been printed to the terminal.")
            
        except Exception as e:
            print(f"!!! ERROR during the debug: {e}")
            messagebox.showerror("Debug error", f"An error occurred during the debug: {e}")


    def run_regression_analysis(self):
        """
        Start a single regression using the formula in the text field.
        """
        formula = self.regression_formula.get().strip()
        if not formula or '~' not in formula:
            messagebox.showwarning("Invalid formula", "Enter a valid formula, for instance 'Y ~ X1 + X2'.")
            return

        self.regression_results_text.delete('1.0', tk.END)
        self.status_var.set("Starting the analysis...")
        self.regression_progress.configure(mode='determinate', maximum=1, value=0)
        self.regression_progress_percent.set("0%")

        th = threading.Thread(target=self._regression_worker, args=([formula],), daemon=True)
        th.start()

    def run_batch_regression(self):
        """
        Load formulas from a text file, one per line, and run the worker in batch.
        """
        if self.behavioral_data.empty:
            messagebox.showwarning("Missing data", "Load the behavioural data first.")
            return
        if not hasattr(self, 'biomarkers_df') or self.biomarkers_df.empty:
            messagebox.showwarning("Missing data", "Generate the biomarkers first (Summary tab -> Refresh Summary).")
            return

        formula_file = filedialog.askopenfilename(
            title="Select the file holding the regression formulas",
            filetypes=[("Text Files", "*.txt")]
        )
        if not formula_file:
            return

        try:
            formulas = []
            with open(formula_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '~' in line:
                        # support for lines wrapped in triple quotes
                        if line.startswith('"""') and line.endswith('"""'):
                            line = line[3:-3]
                        formulas.append(line)
            if not formulas:
                messagebox.showwarning("Empty or invalid file", "The file holds no line with a valid formula, for instance 'Y ~ X1 + X2'.")
                return

            self.status_var.set(f"Loaded {len(formulas)} formulas. Starting the batch analysis...")
            self.regression_summary_table.delete(*self.regression_summary_table.get_children())
            self.regression_results_text.delete('1.0', tk.END)
            self.regression_progress.configure(mode='determinate', maximum=len(formulas), value=0)
            self.regression_progress_percent.set("0%")

            th = threading.Thread(target=self._regression_worker, args=(formulas,), daemon=True)
            th.start()

        except Exception as e:
            messagebox.showerror("Error reading the formulas", f"An error occurred: {e}")


    def _regression_worker(self, formulas_to_run):
        """
        Run the regression with VERBOSE DEBUGGING, tracing every step of the process.
        """
        try:
            # --- DEBUG 1: start of the process and merge of the data ---
            print("\n" + "="*20 + " REGRESSION WORKER STARTED " + "="*20)
            self.root.after(0, lambda: self.status_var.set("Preparing the data for the analysis..."))

            beh = self.behavioral_data.rename(columns={'sessID': 'session_ID'}).copy()
            beh['participant_ID'] = beh['participant_ID'].astype(str)
            beh['session_ID'] = pd.to_numeric(beh['session_ID'], errors='coerce')
            eeg = self.biomarkers_df.copy()
            eeg['participant_ID'] = eeg['participant_ID'].astype(str)
            eeg['session_ID'] = pd.to_numeric(eeg['session_ID'], errors='coerce')
            merged = pd.merge(beh, eeg, on=['participant_ID', 'session_ID'], how='inner')

            if merged.empty:
                raise ValueError("No match was found between the EEG and the behavioural data.")
            print(f"[DEBUG] Data merged successfully. Number of rows: {len(merged)}")
            # --- FINE DEBUG 1 ---

            detailed_report_text = [f"=== REGRESSION REPORT ===", f"Number of models: {len(formulas_to_run)}",
                                    "NA handling: listwise (patsy drop)", f"Model type: {self.regression_type.get().upper()}"]
            self.all_rows = []
            use_bootstrap = self.use_bootstrap.get()
            n_boot = int(self.n_bootstraps.get()) if use_bootstrap else 0
            is_logistic_global = self.regression_type.get().lower().startswith("log")
            
            print(f"[DEBUG] Analysis options: bootstrap={'yes' if use_bootstrap else 'no'}, model={'logistic' if is_logistic_global else 'linear'}")

            for i, formula in enumerate(formulas_to_run, start=1):
                print("\n" + "-"*15 + f" Model analysis {i}/{len(formulas_to_run)} " + "-"*15)
                print(f"[DEBUG] Formula: {formula}")
                self.root.after(0, lambda i=i, n=len(formulas_to_run), f=formula: (
                    self.regression_progress.configure(value=i-1),
                    self.regression_progress_percent.set(f"{int(((i-1)/n)*100)}%"),
                    self.status_var.set(f"Model analysis {i}/{n}: {f}")
                ))

                try:
                    y_patsy, X_patsy = patsy.dmatrices(formula, merged, return_type='dataframe', NA_action='drop')
                    print(f"[DEBUG] Matrices built by patsy. Shapes: y={y_patsy.shape}, X={X_patsy.shape}")
                    if len(y_patsy) < 10:
                        raise ValueError(f"Insufficient data (n={len(y_patsy)}) after dropping the NAs.")
                    
                    X_vif = X_patsy.loc[:, X_patsy.columns != 'Intercept']
                    if X_vif.shape[1] >= 2:
                        vif_vals = [(c, variance_inflation_factor(X_vif.values, k)) for k, c in enumerate(X_vif.columns)]
                        detailed_report_text.append(f"\n[VIF MODEL {i}] (warning only)\n " + ", ".join([f"{c}: {v:.2f}" for c, v in vif_vals]))

                    if use_bootstrap:
                        print("[DEBUG] Starting the BOOTSTRAP analysis...")
                        params_list = []
                        for _ in range(n_boot):
                            try:
                                sample_idx = np.random.choice(y_patsy.index, size=len(y_patsy), replace=True)
                                X_b, y_b = self._prepare_xy(X_patsy.loc[sample_idx], y_patsy.loc[sample_idx].iloc[:, 0])
                                if is_logistic_global:
                                    m = sm.GLM(y_b, X_b, family=sm.families.Binomial()).fit_regularized(alpha=1.0, L1_wt=0.0)
                                else:
                                    m = sm.OLS(y_b, X_b).fit()
                                params_list.append(m.params)
                            except Exception:
                                continue
                        if not params_list: raise RuntimeError("Bootstrap failed: no valid sample was generated.")
                        
                        print(f"[DEBUG] Bootstrap finished with {len(params_list)} valid samples.")
                        boot_df = pd.DataFrame(params_list)
                        self.root.after(0, self._plot_bootstrap_distributions, boot_df)
                        summary_stats = boot_df.agg(['mean', 'std', lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)]).T
                        summary_stats.columns = ['mean', 'std', 'ci_lower', 'ci_upper']
                        detailed_report_text.append(f"\n==== MODEL {i} (BOOTSTRAP {len(params_list)} (BOOTSTRAP {formula}\n{summary_stats.to_string(float_format='%.4f')}")
                        self.all_rows.append({"Formula": formula, "N": len(y_patsy), "AUC": "N/A", "Pseudo R²": "N/A", "Log-Likelihood": "N/A", "AIC": "N/A"})

                    else: # Single fit
                        print("[DEBUG] Starting the SINGLE FIT...")
                        X, y = self._prepare_xy(X_patsy, y_patsy.iloc[:, 0])
                        if is_logistic_global:
                            model = sm.Logit(y, X).fit(disp=0)
                            auc = roc_auc_score(y, model.predict(X))
                            print(f"[DEBUG] Logit fit finished. AUC: {auc:.4f}, Pseudo R²: {model.prsquared:.4f}")
                            detailed_report_text.append(f"\n==== MODEL {i} (LOGIT) ====\nFormula: {formula}\n{model.summary()}")
                            self.all_rows.append({"Formula": formula, "N": int(model.nobs), "AUC": f"{auc:.4f}", 
                                                  "Pseudo R²": f"{model.prsquared:.4f}", "Log-Likelihood": f"{model.llf:.2f}", "AIC": f"{model.aic:.2f}"})
                        else: # OLS
                            model = sm.OLS(y, X).fit()
                            print(f"[DEBUG] OLS fit finished. adjusted R2: {model.rsquared_adj:.4f}")
                            detailed_report_text.append(f"\n==== MODEL {i} (OLS) ====\nFormula: {formula}\n{model.summary()}")
                            self.all_rows.append({"Formula": formula, "N": int(model.nobs), "AUC": "N/A", "Pseudo R²": f"{model.rsquared_adj:.4f}", 
                                                  "Log-Likelihood": f"{model.llf:.2f}", "AIC": f"{model.aic:.2f}"})

                except (np.linalg.LinAlgError, PerfectSeparationError) as e:
                    error_type = "Perfect separation / singular matrix"
                    print(f"[DEBUG] ERROR CAUGHT for model {i}: {error_type}. The model will be skipped.")
                    detailed_report_text.append(f"\n!! MODEL {i}: {error_type}.\nFormula: {formula}\nError: {e}\nSuggestion: this model is not estimable with the data at hand. Try dropping a variable or use the bootstrap.")
                    continue
                except Exception as e:
                    print(f"[DEBUG] UNKNOWN ERROR CAUGHT for model {i}: {e}. The model will be skipped.")
                    detailed_report_text.append(f"\n!! MODEL {i}: unknown error during the fit.\nFormula: {formula}\nError: {e}")
                    continue

            print("\n" + "="*20 + " ANALYSIS FINISHED " + "="*20)
            print("[DEBUG] Every model has been processed. Starting the GUI refresh...")
            self.root.after(0, lambda: (self.regression_progress.configure(value=len(formulas_to_run)), self.regression_progress_percent.set("100%")))
            if self.all_rows:
                def _fill_table(rows=self.all_rows):
                    self.regression_summary_table.delete(*self.regression_summary_table.get_children())
                    for r in rows:
                        values = [r.get(col, 'N/A') for col in self.regression_summary_table['columns']]
                        self.regression_summary_table.insert('', 'end', values=values)
                    self.sort_regression_table("AUC", True)
                    self.regression_notebook.select(0)
                    print("[DEBUG] 'Batch Summary' table refreshed.")
                self.root.after(0, _fill_table)
            
            text_out = "\n".join(map(str, detailed_report_text))
            self.root.after(0, lambda: (self.regression_results_text.delete('1.0', tk.END), self.regression_results_text.insert('1.0', text_out)))
            if not self.all_rows and len(formulas_to_run) > 0: self.root.after(0, self.regression_notebook.select, 1)
            self.root.after(0, self.status_var.set, "Regression analysis finished.")
            print("[DEBUG] GUI refreshed. Process complete.")

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Critical error", f"Error during the analysis: {e}")
            self.root.after(0, self.status_var.set, f"Analysis failed: {e}")
            print(f"[DEBUG] CRITICAL ERROR in the worker: {e}")
        finally:
            self.root.after(0, self.regression_progress.stop)

        
    def sort_regression_table(self, col, reverse):
        """
        Sort the regression results table by the selected column.
        Numbers and strings are both handled safely.
        """
        try:
            # Take every item (row) in the table
            children = list(self.regression_summary_table.get_children(''))
            if not children:
                return

            # Extract the data of the selected column for each row
            # The tuple holds (value_to_sort_by, row_id)
            data = [(self.regression_summary_table.set(cid, col), cid) for cid in children]

            # Columns that should be treated as numbers
            numeric_cols = {"N", "AUC", "Pseudo R²", "Log-Likelihood", "AIC"}

            # Helper that converts to float and swallows errors
            def try_float(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    # If it is not a number, send it to the end of the list
                    return float('-inf') if reverse else float('inf')

            # Sort the data
            if col in numeric_cols:
                # Numeric sort
                data.sort(key=lambda t: try_float(t[0]), reverse=reverse)
            else:
                # Alphabetical sort, for the 'Formula' column
                data.sort(key=lambda t: str(t[0]).lower(), reverse=reverse)

            # Reorder the rows of the table accordingly
            for idx, (_, cid) in enumerate(data):
                self.regression_summary_table.move(cid, '', idx)

            # Flip the sort direction for the next click on the same column
            self.regression_summary_table.heading(
                col, command=lambda: self.sort_regression_table(col, not reverse)
            )
        except Exception as e:
            print(f"Error while sorting the table: {e}")
            # Restore the sort command on error
            self.regression_summary_table.heading(
                col, command=lambda: self.sort_regression_table(col, False)
            )


    
    def create_fooof_tabbed_view(self, parent_frame, subject_id):
        """
        Build a tabbed view (Notebook) for the FOOOF results, one plot per channel.
        """
        # Clear the parent frame of any earlier widget
        for widget in parent_frame.winfo_children():
            widget.destroy()

        # The Notebook widget that will hold the tabs
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill='both', expand=True, pady=10, padx=5)

        # Results and colours are taken from 'self'
        fooof_subject_results = self.fooof_results[subject_id]
        colors = self.colors 

        channels = list(fooof_subject_results.keys())

        # Iterate over every channel for which a FOOOF result exists
        for ch_name in channels:
            # A frame for this channel's tab
            tab_frame = ttk.Frame(notebook, padding=10)
            notebook.add(tab_frame, text=ch_name)

            # A Matplotlib figure for this channel ONLY
            fig = Figure(figsize=(10, 5), facecolor=colors['surface'])
            ax = fig.add_subplot(111)
            
            # Take the FOOOF model and draw it on the axis
            fm = fooof_subject_results[ch_name]['fm']
            fm.plot(ax=ax, plot_peaks='shade', peak_kwargs={'color': 'green'}, add_legend=False)

            # Tune the look of the plot
            ax.set_title(f"Channel: {ch_name}", color=colors['text_primary'])
            ax.set_xlabel("Frequency (Hz)", color=colors['text_secondary'])
            ax.set_ylabel("Log(Power)", color=colors['text_secondary'])
            ax.tick_params(colors=colors['text_secondary'])
            for spine in ax.spines.values():
                spine.set_color(colors['border'])
            fig.tight_layout()

            # Put the plot into its Tkinter canvas
            canvas = FigureCanvasTkAgg(fig, master=tab_frame)
            canvas.draw()
            
            # Add the Matplotlib toolbar
            toolbar_frame = ttk.Frame(tab_frame)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # Place the canvas and the toolbar in the tab frame
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X, pady=(5,0))


    def _update_regression_viz(self):
        """
        Visualizes the regression results with detailed information displayed in the tab.
        """
        if not hasattr(self, 'regression_results') or not self.regression_results:
            return

        # Clear the existing figure
        self.regression_fig.clear()
        
        # Create a layout with subplots for different information
        gs = self.regression_fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        
        # 1. Model Summary (top-left)
        ax1 = self.regression_fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        ax1.set_title('Model Summary', fontweight='bold', fontsize=12)
        
        model_summary = self.regression_results.get('model_summary', 'No model summary available')
        # Extract key information from the summary
        summary_lines = model_summary.split('\n')
        key_info = []
        for line in summary_lines:
            if any(keyword in line for keyword in ['Pseudo R-squ', 'Log-Likelihood', 'AIC', 'BIC']):
                key_info.append(line.strip())
        
        summary_text = '\n'.join(key_info[:8])  # Show first 8 relevant lines
        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        # 2. VIF Data (top-right)
        ax2 = self.regression_fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        ax2.set_title('Variance Inflation Factor (VIF)', fontweight='bold', fontsize=12)
        
        vif_text = self.regression_results.get('vif_data', 'No VIF data available')
        ax2.text(0.05, 0.95, vif_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        # 3. Performance Metrics (bottom-left)
        ax3 = self.regression_fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        ax3.set_title('Performance Metrics', fontweight='bold', fontsize=12)
        
        metrics = self.regression_results.get('metrics', {})
        metrics_text = ""
        for key, value in metrics.items():
            if key != 'Confusion_Matrix':  # Handle confusion matrix separately
                if isinstance(value, float):
                    metrics_text += f"{key}: {value:.4f}\n"
                else:
                    metrics_text += f"{key}: {value}\n"
        
        # Add confusion matrix if available
        if 'Confusion_Matrix' in metrics:
            cm = metrics['Confusion_Matrix']
            metrics_text += f"\nConfusion Matrix:\n{cm}"
        
        ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        # 4. Regression Plot (bottom-right)
        ax4 = self.regression_fig.add_subplot(gs[1, 1])
        
        try:
            formula = self.regression_results.get('formula', '')
            data = self.regression_results.get('data')
            
            if formula and data is not None:
                y_var, x_vars_str = formula.split('~')
                y_var = y_var.strip()
                x_vars = [v.strip() for v in x_vars_str.split('+')]
                
                if len(x_vars) == 1:
                    x_var = x_vars[0]
                    if x_var in data.columns and y_var in data.columns:
                        import seaborn as sns
                        sns.regplot(x=x_var, y=y_var, data=data, logistic=True, ci=None,
                                   scatter_kws={'color': 'blue', 'alpha': 0.5},
                                   line_kws={'color': 'red'}, ax=ax4)
                        ax4.set_title(f"Logistic Regression: {y_var} ~ {x_var}")
                        ax4.set_xlabel(x_var)
                        ax4.set_ylabel(y_var)
                    else:
                        ax4.text(0.5, 0.5, f"Variables not found in data:\n{x_var}, {y_var}", 
                                ha='center', va='center')
                else:
                    ax4.text(0.5, 0.5, "Plotting available only for\nsingle predictor models", 
                            ha='center', va='center')
            else:
                ax4.text(0.5, 0.5, "No data available for plotting", ha='center', va='center')
        except Exception as e:
            ax4.text(0.5, 0.5, f"Plotting error:\n{str(e)}", ha='center', va='center')

        self.regression_fig.tight_layout()
        self.regression_canvas.draw()
        
        # Also show a summary popup (keep the existing behavior)
        metrics = self.regression_results.get('metrics', {})
        summary_text = f"Formula: {self.regression_results.get('formula', 'N/A')}\n\n"
        
        for key, value in metrics.items():
            if key != 'Confusion_Matrix':
                if isinstance(value, float):
                    summary_text += f"{key}: {value:.4f}\n"
                else:
                    summary_text += f"{key}: {value}\n"
        
        messagebox.showinfo("Regression Analysis Complete", summary_text)

    def _prepare_graph(self, matrix: np.ndarray, threshold_perc: int, weight_handling: str, symmetrize: str) -> Optional[nx.Graph]:
        """Prepare the graph from a matrix, applying the threshold and the cleaning."""
        A = matrix.copy()

        # 1. Handling of negative weights
        if weight_handling == 'abs':
            A = np.abs(A)
        elif weight_handling == 'zero_neg':
            A[A < 0] = 0

        # 2. Simmetrizzazione
        if symmetrize == 'average':
            A = 0.5 * (A + A.T)
        elif symmetrize == 'max':
            A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0)

        # 3. Proportional thresholding
        if threshold_perc is not None and 0 < threshold_perc < 100:
            threshold_val = np.percentile(A[np.triu_indices_from(A, k=1)], 100 - threshold_perc)
            A[A < threshold_val] = 0
        
        # 4. Build the graph and clean it
        G = nx.from_numpy_array(A)
        isolated_nodes = [node for node, degree in G.degree() if degree == 0]
        G.remove_nodes_from(isolated_nodes)
        
        if G.number_of_nodes() == 0:
            messagebox.showwarning("Empty graph", "No connected node is left after thresholding. Cannot continue.")
            return None
            
        return G

    def _compute_graph_metrics(self, G: nx.Graph, ch_names: List[str]) -> Dict:
        """Compute every global and per-node metric for a given graph."""
        if G is None: return {}

        # Global metrics
        density = nx.density(G)
        
        # Modularity, through bctpy where available and networkx otherwise
        try:
            import bct
            # bct.modularity_louvain_und() returns (community assignment, Q)
            ci, q = bct.modularity_louvain_und(nx.to_numpy_array(G))
            n_modules = len(np.unique(ci))
        except ImportError:
            print("WARNING: `bctpy` not found. Falling back on `nx.community.louvain_communities`.")
            communities = nx.community.louvain_communities(G, weight='weight', seed=42)
            q = nx.community.modularity(G, communities)
            n_modules = len(communities)
            # Map the communities onto an array of per-node labels
            ci_map = {node: i for i, com in enumerate(communities) for node in com}
            ci = np.array([ci_map.get(n, -1) for n in G.nodes()])

        # Clustering and efficiency
        C_local = nx.clustering(G, weight='weight')
        C_global = np.mean(list(C_local.values()))
        E_global = nx.global_efficiency(G)
        
        # Characteristic path length, on the largest connected component only
        if nx.is_connected(G):
            L_char = nx.average_shortest_path_length(G, weight='weight')
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            sub_G = G.subgraph(largest_cc)
            L_char = nx.average_shortest_path_length(sub_G, weight='weight')

        # Per-node metrics
        node_metrics = pd.DataFrame({
            'channel': list(G.nodes()),
            'degree': [d for n, d in G.degree(weight='weight')],
            'clustering': list(C_local.values()),
            'community': ci,
        })
        
        # Map the original channel names
        node_map = {i: name for i, name in enumerate(ch_names)}
        node_metrics['channel'] = node_metrics['channel'].map(node_map)

        return {
            "global": {
                "Q": q, "n_modules": n_modules, "C_global": C_global,
                "E_global": E_global, "L_char": L_char, "density": density
            },
            "node": node_metrics
        }




    def _build_explore_controls(self, parent_frame):
        """Build the widgets of the exploratory analysis mode."""
        data_lf = ttk.LabelFrame(parent_frame, text="1. Select data", style='Card.TLabelframe', padding=10)
        data_lf.pack(fill='x', pady=5, anchor='n')
        ttk.Label(data_lf, text="Subject:").pack(anchor='w')
        self.gt_subject_dropdown = ttk.Combobox(data_lf, textvariable=self.gt_subject_var, state='readonly')
        self.gt_subject_dropdown.pack(fill='x', pady=(0, 5))
        self.gt_subject_var.trace_add('write', self._update_gt_matrix_dropdown)

        ttk.Label(data_lf, text="Matrix (metric and band):").pack(anchor='w')
        self.gt_matrix_dropdown = ttk.Combobox(data_lf, textvariable=self.gt_matrix_key_var, state='readonly')
        self.gt_matrix_dropdown.pack(fill='x')

        params_lf = ttk.LabelFrame(parent_frame, text="2. Graph parameters", style='Card.TLabelframe', padding=10)
        params_lf.pack(fill='x', pady=10, anchor='n')
        
        ttk.Label(params_lf, text="Negative weight handling:").pack(anchor='w')
        ttk.Combobox(params_lf, textvariable=self.gt_neg_var, values=['abs', 'zero_neg', 'keep'], state='readonly').pack(fill='x', pady=(0, 5))
        
        ttk.Checkbutton(params_lf, text="Directed graph", variable=self.gt_directed_var).pack(anchor='w')
        
        ttk.Label(params_lf, text="Thresholding method:").pack(anchor='w', pady=(5,0))
        ttk.OptionMenu(params_lf, self.gt_thresh_method_var, 'proportional', *['proportional', 'backbone', 'none'], command=self._on_thresh_method_change).pack(fill='x', pady=(0, 5))
        
        self.gt_thresh_param_frame = ttk.Frame(params_lf)
        self.gt_thresh_param_frame.pack(fill='x')
        self.gt_prop_thresh_lf = ttk.LabelFrame(self.gt_thresh_param_frame, text="Proportional threshold (%)", padding=5)
        # NOTE: the same variable as the batch mode is used here, the two modes being mutually exclusive
        ttk.Scale(self.gt_prop_thresh_lf, from_=0, to=100, variable=self.gt_threshold_var, orient='horizontal').pack(fill='x')
        
        self.gt_backbone_lf = ttk.LabelFrame(self.gt_thresh_param_frame, text="Alpha (p-value)", padding=5)
        ttk.Scale(self.gt_backbone_lf, from_=0.01, to=0.5, variable=self.gt_alpha_var, orient='horizontal').pack(fill='x')
        self._on_thresh_method_change()
        
        ttk.Label(params_lf, text="Random Seed:").pack(anchor='w', pady=(5,0))
        ttk.Entry(params_lf, textvariable=self.gt_seed_var, width=10).pack(fill='x')

        exec_lf = ttk.LabelFrame(parent_frame, text="3. Analysis", style='Card.TLabelframe', padding=10)
        exec_lf.pack(fill='x', pady=10, anchor='n')
        ttk.Button(exec_lf, text="Compute single graph", style='Primary.TButton', command=self.run_single_graph_analysis).pack(fill='x')
        ttk.Button(exec_lf, text="Compute multi-threshold", style='Secondary.TButton', command=self.run_multi_threshold_graph_analysis).pack(fill='x', pady=5)
        ttk.Button(exec_lf, text="Compute MST", style='Info.TButton', command=self.run_mst_analysis).pack(fill='x')


    def _on_gt_mode_change(self):
        """Show or hide the control panels according to the selected mode."""
        # Hide both frames before showing the right one
        self.gt_batch_frame.pack_forget()
        self.gt_explore_frame.pack_forget()
        
        mode = self.gt_mode_var.get()
        if mode == "batch":
            self.gt_batch_frame.pack(fill='x', expand=True)
        else: # explore
            self.gt_explore_frame.pack(fill='x', expand=True)

    def _graph_theory_worker(self):
        """
        Extended worker for the batch analysis. It computes the metrics for every
        combination and then forces a refresh of the Summary tab.
        """
        try:
            metrics_to_run = ['dpli', 'imcoh', 'plv', 'coh']
            bands_to_run = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            
            print("\n" + "="*20 + " BATCH GRAPH THEORY ANALYSIS STARTED (FULL) " + "="*20)
            
            prepare_params = {
                "threshold_perc": self.gt_threshold_var.get(),
                "weight_handling": self.gt_neg_var.get(),
                "symmetrize": self.gt_symmetrize_var.get()
            }
            seed = self.gt_seed_var.get()

            self.graph_theory_results = {}
            subjects_to_process = list(self.connectivity_results.keys())
            total_subjects = len(subjects_to_process)
            
            if total_subjects == 0:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No connectivity result was found."))
                return

            for i, subject_id in enumerate(subjects_to_process):
                self.root.after(0, lambda p=int(((i + 1) / total_subjects) * 100), s=subject_id: (
                    self.gt_progress.configure(value=p),
                    self.gt_progress_percent.set(f"{p}%"),
                    self.status_var.set(f"Graph Theory: {s} ({i+1}/{total_subjects})...")
                ))

                self.graph_theory_results[subject_id] = {}

                for metric in metrics_to_run:
                    for band in bands_to_run:
                        matrix_key = f"{metric}_{band}"
                        try:
                            matrix = self.connectivity_results[subject_id]['matrices'][matrix_key]
                            ch_names = self.preprocessed_data[subject_id].copy().pick_types(eeg=True, verbose=False).ch_names
                        except KeyError:
                            continue

                        G = self._prepare_graph(matrix, **prepare_params)
                        if G is None: continue
                        
                        metrics = self._compute_graph_metrics(G, ch_names)
                        if not metrics: continue
                        
                        for g_key, g_val in metrics['global'].items():
                            final_key = f"gt_{metric}_{band}_{g_key}"
                            self.graph_theory_results[subject_id][final_key] = g_val

                if i == total_subjects - 1 and 'G' in locals() and 'metrics' in locals():
                    self.root.after(0, self.update_graph_theory_display, G, metrics, subject_id, f"{metrics_to_run[-1]}_{bands_to_run[-1]}")

            self.root.after(0, lambda: self.status_var.set("Graph theory analysis finished."))
            self.root.after(0, lambda: messagebox.showinfo("Finished", "The graph theory analysis is complete."))
            
            # --- crucial call, at the end ---
            self.root.after(0, self.update_summary_tab)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=e: messagebox.showerror("Critical graph error", f"Error during the batch analysis:\n{err}"))
        finally:
            self.root.after(0, lambda: self.gt_progress.configure(value=0))
            self.root.after(0, lambda: self.gt_progress_percent.set("0%"))


    def run_graph_analysis(self):
        """
        Launch the batch graph theory worker in a separate thread.
        This method is bound to the 'Run Batch Analysis' button.
        """
        if not self.connectivity_results:
            messagebox.showerror("Missing data", "Run the connectivity analysis first.")
            return
            
        self.status_var.set("Starting the batch graph theory analysis...")
        # Set the progress bar to run from 0 to 100
        self.gt_progress.configure(maximum=100, value=0) 
        self.gt_progress_percent.set("0%")
        
        # Start the worker in a thread so the GUI is not blocked
        threading.Thread(target=self._graph_theory_worker, daemon=True).start()


    

    def _update_gt_subject_dropdown(self):
        """Fill the subject dropdown of the Graph Theory tab."""
        if hasattr(self, 'connectivity_results'):
            subjects = list(self.connectivity_results.keys())
            self.gt_subject_dropdown.configure(values=subjects)
            if subjects:
                self.gt_subject_var.set(subjects[0])

    def _update_gt_matrix_dropdown(self, *args):
        """Fill the dropdown of matrices available for the selected subject."""
        subject_id = self.gt_subject_var.get()
        if subject_id and hasattr(self, 'connectivity_results'):
            matrices = list(self.connectivity_results.get(subject_id, {}).get('matrices', {}).keys())
            self.gt_matrix_dropdown.configure(values=matrices)
            if matrices:
                self.gt_matrix_key_var.set(matrices[0])

    
    def export_graph_results(self):
        """Export the results of the last graph analysis that was run."""
        if not hasattr(self, 'last_graph_metrics') or not self.last_graph_metrics:
            messagebox.showerror("Error", "No graph result to export. Run an analysis first.")
            return

        output_dir = filedialog.askdirectory(title="Select the folder for the export")
        if not output_dir:
            return

        # Collect the current parameters
        params = {
            "subject": self.gt_subject_var.get(),
            "matrix_key": self.gt_matrix_key_var.get(),
            "threshold_method": self.gt_thresh_method_var.get(),
            "threshold_value": self.gt_threshold_var.get() if self.gt_thresh_method_var.get() == 'proportional' else self.gt_alpha_var.get(),
            "negative_weights": self.gt_neg_var.get(),
            "is_directed": self.gt_directed_var.get(),
            "seed": self.gt_seed_var.get()
        }

        success = gm.export_graph_analysis_results(
            output_dir=output_dir,
            subject_id=self.gt_subject_var.get(),
            params=params,
            global_metrics=self.last_graph_metrics['global'],
            node_metrics=self.last_graph_metrics['node']
        )

        if success:
            messagebox.showinfo("Success", f"Results exported to:\n{output_dir}")
        else:
            messagebox.showerror("Error", "The export failed. Check the terminal for details.")
    def open_nbs_window(self):
        """Opens a new window to configure and run Network-Based Statistics (NBS) analysis."""
        if not self.connectivity_results:
            messagebox.showerror("Data Missing", "Please run the connectivity analysis first.")
            return

        nbs_window = tk.Toplevel(self.root)
        nbs_window.title("Network-Based Statistics (NBS)")
        nbs_window.geometry("800x600")

        # --- Tkinter variables for this window ---
        group1_var = tk.StringVar()
        group2_var = tk.StringVar()
        matrix_key_var = tk.StringVar()
        t_thresh_var = tk.DoubleVar(value=3.0) # A common default
        perms_var = tk.IntVar(value=1000)
        stat_var = tk.StringVar(value="extent")
        sided_var = tk.StringVar(value="two-sided")

        # --- Main window layout ---
        main_frame = ttk.Frame(nbs_window, padding=15)
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # --- Configuration Panel ---
        config_lf = ttk.LabelFrame(main_frame, text="Analysis Configuration", padding=10)
        config_lf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))

        ttk.Label(config_lf, text="Metric & Band to Compare:").pack(anchor='w')
        first_subject = next(iter(self.connectivity_results))
        matrix_keys = list(self.connectivity_results[first_subject]['matrices'].keys())
        ttk.Combobox(config_lf, textvariable=matrix_key_var, values=matrix_keys, state='readonly').pack(fill='x', pady=(0, 10))
        if matrix_keys:
            matrix_key_var.set(matrix_keys[0])

        # --- Group Definition ---
        all_subjects = list(self.connectivity_results.keys())
        
        group1_lf = ttk.LabelFrame(main_frame, text="Group 1", padding=10)
        group1_lf.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        group1_listbox = tk.Listbox(group1_lf, selectmode='multiple', exportselection=False)
        group1_listbox.pack(fill='both', expand=True)
        for sub in all_subjects: group1_listbox.insert('end', sub)

        group2_lf = ttk.LabelFrame(main_frame, text="Group 2", padding=10)
        group2_lf.grid(row=1, column=1, sticky="nsew")
        group2_listbox = tk.Listbox(group2_lf, selectmode='multiple', exportselection=False)
        group2_listbox.pack(fill='both', expand=True)
        for sub in all_subjects: group2_listbox.insert('end', sub)

        # --- NBS Parameters and Execution Button ---
        params_lf = ttk.LabelFrame(main_frame, text="NBS Parameters", padding=10)
        params_lf.grid(row=2, column=0, columnspan=2, sticky="ew", pady=15)
        
        ttk.Label(params_lf, text="T-statistic Threshold:").grid(row=0, column=0, sticky='w')
        ttk.Entry(params_lf, textvariable=t_thresh_var, width=10).grid(row=0, column=1, sticky='w')
        
        ttk.Label(params_lf, text="Permutations:").grid(row=1, column=0, sticky='w')
        ttk.Entry(params_lf, textvariable=perms_var, width=10).grid(row=1, column=1, sticky='w')

        ttk.Label(params_lf, text="Statistic:").grid(row=0, column=2, sticky='w', padx=(20, 5))
        ttk.Combobox(params_lf, textvariable=stat_var, values=['extent', 'intensity'], state='readonly', width=10).grid(row=0, column=3, sticky='w')

        ttk.Label(params_lf, text="Test Type:").grid(row=1, column=2, sticky='w', padx=(20, 5))
        ttk.Combobox(params_lf, textvariable=sided_var, values=['two-sided', 'one-sided'], state='readonly', width=10).grid(row=1, column=3, sticky='w')

        def start_analysis():
            g1_indices = group1_listbox.curselection()
            g2_indices = group2_listbox.curselection()

            if not g1_indices or not g2_indices:
                messagebox.showerror("Error", "Select at least one subject for each group.", parent=nbs_window)
                return

            group1_subjects = [all_subjects[i] for i in g1_indices]
            group2_subjects = [all_subjects[i] for i in g2_indices]
            
            # Check that the groups do not overlap
            if set(group1_subjects) & set(group2_subjects):
                messagebox.showerror("Error", "The groups cannot hold the same subjects.", parent=nbs_window)
                return

            key = matrix_key_var.get()
            group1_matrices = [self.connectivity_results[s]['matrices'][key] for s in group1_subjects]
            group2_matrices = [self.connectivity_results[s]['matrices'][key] for s in group2_subjects]

            params = {
                "t_threshold": t_thresh_var.get(),
                "n_permutations": perms_var.get(),
                "component_stat": stat_var.get(),
                "two_sided": sided_var.get() == 'two-sided',
                "use_abs_t": sided_var.get() == 'two-sided' # |t| per test a due code
            }
            
            # Run in a thread so the GUI is not blocked
            threading.Thread(
                target=self._run_nbs_analysis_and_show_results,
                args=(group1_matrices, group2_matrices, params),
                daemon=True
            ).start()
            
            self.status_var.set("Running NBS analysis...")
            nbs_window.destroy() # Close the configuration window

        ttk.Button(main_frame, text="Run NBS analysis", style='Success.TButton', command=start_analysis).grid(row=3, column=0, columnspan=2, pady=10)


    def _run_nbs_analysis_and_show_results(self, g1, g2, params):
        """Worker that runs NBS and then calls the function that shows the results."""
        try:
            results = gm.run_nbs(
                group1_matrices=g1,
                group2_matrices=g2,
                t_threshold=params['t_threshold'],
                n_permutations=params['n_permutations'],
                component_stat=params['component_stat'],
                two_sided=params['two_sided'],
                use_abs_t=params['use_abs_t']
            )
            self.root.after(0, self.display_nbs_results, results)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "NBS error", f"An error occurred during the NBS analysis:\n{e}")
            self.status_var.set("NBS analysis failed.")

    def create_graph_theory_tab(self):
        """
        Creates the tab for Graph Theory analysis with an academic English interface.
        """
        gt_frame = ttk.Frame(self.notebook, style='Card.TFrame', padding=10)
        self.notebook.add(gt_frame, text="Graph Theory")

        # --- Main 2-column layout: Controls on the left, Results on the right ---
        gt_frame.columnconfigure(0, weight=1)  # Controls column (narrow)
        gt_frame.columnconfigure(1, weight=3)  # Results column (wide)
        gt_frame.rowconfigure(1, weight=1)     # Content row expands

        # --- 1. MODE SELECTOR (top, spans both columns) ---
        mode_frame = ttk.Frame(gt_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        
        ttk.Label(mode_frame, text="Analysis Mode:", style='Heading.TLabel').pack(side='left', padx=(0, 15))
        ttk.Radiobutton(mode_frame, text="Exploratory Analysis (Single Subject)", variable=self.gt_mode_var, value="explore", command=self._on_gt_mode_change).pack(side='left', padx=5)
        ttk.Radiobutton(mode_frame, text="Batch Analysis", variable=self.gt_mode_var, value="batch", command=self._on_gt_mode_change).pack(side='left', padx=5)

        # --- 2. CONTAINER FRAMES FOR BOTH MODES (in the controls column) ---
        self.gt_controls_frame = ttk.Frame(gt_frame)
        self.gt_controls_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 15))
        
        self.gt_batch_frame = ttk.Frame(self.gt_controls_frame)
        self.gt_explore_frame = ttk.Frame(self.gt_controls_frame)
        
        # --- 3. BUILD THE INNER WIDGETS FOR EACH MODE ---
        self._build_batch_controls(self.gt_batch_frame)
        self._build_explore_controls(self.gt_explore_frame)

        # --- 4. VISUALIZATION AREA (right column) ---
        viz_col = ttk.Frame(gt_frame)
        viz_col.grid(row=1, column=1, sticky="nsew")
        viz_col.rowconfigure(0, weight=1)
        viz_col.columnconfigure(0, weight=2) # Plots
        viz_col.columnconfigure(1, weight=1) # Metrics

        # Notebook for plots
        graph_notebook_frame = ttk.Frame(viz_col)
        graph_notebook_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.gt_notebook = ttk.Notebook(graph_notebook_frame)
        self.gt_notebook.pack(fill='both', expand=True)
        
        tabs = {'Graph Topology': 'graph', 'Thresholded Matrix': 'matrix', 'Metrics Integration': 'integration'}
        for name, key in tabs.items():
            tab = ttk.Frame(self.gt_notebook)
            self.gt_notebook.add(tab, text=name)
            fig = Figure()
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            setattr(self, f'gt_fig_{key}', fig)
            setattr(self, f'gt_canvas_{key}', canvas)

        # Frame for metrics tables
        metrics_frame = ttk.Frame(viz_col)
        metrics_frame.grid(row=0, column=1, sticky='nsew')
        metrics_frame.rowconfigure(0, weight=1)
        metrics_frame.rowconfigure(1, weight=2)
        metrics_frame.columnconfigure(0, weight=1)

        global_lf = ttk.LabelFrame(metrics_frame, text="Global Metrics", style='Card.TLabelframe', padding=10)
        global_lf.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        self.gt_summary_table = ttk.Treeview(global_lf, columns=("Metric", "Value"), show='headings', height=6)
        self.gt_summary_table.heading("Metric", text="Metric")
        self.gt_summary_table.heading("Value", text="Value")
        self.gt_summary_table.column("Metric", width=120)
        self.gt_summary_table.column("Value", width=80, anchor='e')
        self.gt_summary_table.pack(fill='both', expand=True)

        node_lf = ttk.LabelFrame(metrics_frame, text="Nodal Metrics", style='Card.TLabelframe', padding=10)
        node_lf.grid(row=1, column=0, sticky='nsew')
        self.gt_node_table = ttk.Treeview(node_lf, show='headings', height=10)
        vsb = ttk.Scrollbar(node_lf, orient="vertical", command=self.gt_node_table.yview)
        self.gt_node_table.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        self.gt_node_table.pack(fill='both', expand=True)

        # --- 5. Activate the default mode ---
        self._on_gt_mode_change()

    def _build_batch_controls(self, parent_frame):
        """Builds the widgets for the batch analysis mode."""
        params_lf = ttk.LabelFrame(parent_frame, text="1. Batch Analysis Parameters", style='Card.TLabelframe', padding=10)
        params_lf.pack(fill='x', pady=5, anchor='n')
        
        ttk.Label(params_lf, text="Connectivity Metric:").pack(anchor='w')
        ttk.Combobox(params_lf, textvariable=self.gt_batch_metric_var, values=['dpli', 'imcoh', 'plv', 'coh'], state='readonly').pack(fill='x', pady=(0, 5))

        ttk.Label(params_lf, text="Frequency Band:").pack(anchor='w')
        ttk.Combobox(params_lf, textvariable=self.gt_batch_band_var, values=['delta', 'theta', 'alpha', 'beta', 'gamma'], state='readonly').pack(fill='x', pady=(0, 10))

        ttk.Label(params_lf, text="Proportional Threshold (%):").pack(anchor='w')
        ttk.Scale(params_lf, from_=0, to=100, variable=self.gt_threshold_var, orient='horizontal').pack(fill='x')

        exec_lf = ttk.LabelFrame(parent_frame, text="2. Execution", style='Card.TLabelframe', padding=10)
        exec_lf.pack(fill='x', pady=15, anchor='n')
        ttk.Button(exec_lf, text="Run Batch Analysis", style='Primary.TButton', command=self.run_graph_analysis).pack(fill='x')

        progress_frame = ttk.Frame(exec_lf)
        progress_frame.pack(fill='x', pady=(10, 0))
        self.gt_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.gt_progress.pack(fill='x', expand=True)
        ttk.Label(progress_frame, textvariable=self.gt_progress_percent).pack()

    def _build_explore_controls(self, parent_frame):
        """Builds the widgets for the exploratory analysis mode."""
        data_lf = ttk.LabelFrame(parent_frame, text="1. Data Selection", style='Card.TLabelframe', padding=10)
        data_lf.pack(fill='x', pady=5, anchor='n')
        ttk.Label(data_lf, text="Subject:").pack(anchor='w')
        self.gt_subject_dropdown = ttk.Combobox(data_lf, textvariable=self.gt_subject_var, state='readonly')
        self.gt_subject_dropdown.pack(fill='x', pady=(0, 5))
        self.gt_subject_var.trace_add('write', self._update_gt_matrix_dropdown)

        ttk.Label(data_lf, text="Matrix (Metric & Band):").pack(anchor='w')
        self.gt_matrix_dropdown = ttk.Combobox(data_lf, textvariable=self.gt_matrix_key_var, state='readonly')
        self.gt_matrix_dropdown.pack(fill='x')

        params_lf = ttk.LabelFrame(parent_frame, text="2. Graph Parameters", style='Card.TLabelframe', padding=10)
        params_lf.pack(fill='x', pady=10, anchor='n')
        
        ttk.Label(params_lf, text="Negative Weight Handling:").pack(anchor='w')
        ttk.Combobox(params_lf, textvariable=self.gt_neg_var, values=['abs', 'zero_neg', 'keep'], state='readonly').pack(fill='x', pady=(0, 5))
        
        ttk.Checkbutton(params_lf, text="Directed Graph", variable=self.gt_directed_var).pack(anchor='w')
        
        ttk.Label(params_lf, text="Thresholding Method:").pack(anchor='w', pady=(5,0))
        ttk.OptionMenu(params_lf, self.gt_thresh_method_var, 'proportional', *['proportional', 'backbone', 'none'], command=self._on_thresh_method_change).pack(fill='x', pady=(0, 5))
        
        self.gt_thresh_param_frame = ttk.Frame(params_lf)
        self.gt_thresh_param_frame.pack(fill='x')
        self.gt_prop_thresh_lf = ttk.LabelFrame(self.gt_thresh_param_frame, text="Proportional Threshold (%)", padding=5)
        ttk.Scale(self.gt_prop_thresh_lf, from_=0, to=100, variable=self.gt_threshold_var, orient='horizontal').pack(fill='x')
        
        self.gt_backbone_lf = ttk.LabelFrame(self.gt_thresh_param_frame, text="Alpha (p-value)", padding=5)
        ttk.Scale(self.gt_backbone_lf, from_=0.01, to=0.5, variable=self.gt_alpha_var, orient='horizontal').pack(fill='x')
        self._on_thresh_method_change()
        
        ttk.Label(params_lf, text="Random Seed:").pack(anchor='w', pady=(5,0))
        ttk.Entry(params_lf, textvariable=self.gt_seed_var, width=10).pack(fill='x')

        exec_lf = ttk.LabelFrame(parent_frame, text="3. Analysis", style='Card.TLabelframe', padding=10)
        exec_lf.pack(fill='x', pady=10, anchor='n')
        ttk.Button(exec_lf, text="Compute Single Graph", style='Primary.TButton', command=self.run_single_graph_analysis).pack(fill='x')
        ttk.Button(exec_lf, text="Compute Multi-Threshold", style='Secondary.TButton', command=self.run_multi_threshold_graph_analysis).pack(fill='x', pady=5)
        ttk.Button(exec_lf, text="Compute MST", style='Info.TButton', command=self.run_mst_analysis).pack(fill='x')

    def update_graph_theory_display(self, G, metrics, subject_id, matrix_key):
        """
        Updates the display widgets in the Graph Theory tab with the provided results.
        """
        # Update the label to show what we are viewing
        self.gt_viz_label_var.set(f"Displaying results for: {subject_id}")

        # Update the global metrics table
        for i in self.gt_summary_table.get_children(): self.gt_summary_table.delete(i)
        for key, value in metrics['global'].items():
            self.gt_summary_table.insert('', 'end', values=(key.replace('_', ' ').title(), f"{value:.4f}"))

        # Update the nodal metrics table
        for i in self.gt_node_table.get_children(): self.gt_node_table.delete(i)
        node_cols = list(metrics['node'].columns)
        self.gt_node_table['columns'] = node_cols
        for col in node_cols:
            self.gt_node_table.heading(col, text=col.replace('_', ' ').capitalize())
            self.gt_node_table.column(col, width=80, anchor='center')
        for _, row in metrics['node'].iterrows():
            formatted_values = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else v for v in row]
            self.gt_node_table.insert('', 'end', values=formatted_values)

        # Update the matrix plot
        self.gt_fig_matrix.clear()
        ax_m = self.gt_fig_matrix.add_subplot(111)
        im = ax_m.imshow(nx.to_numpy_array(G), cmap='viridis', origin='lower')
        self.gt_fig_matrix.colorbar(im, ax=ax_m)
        ax_m.set_title(f"Thresholded Matrix\n({matrix_key})")
        self.gt_canvas_matrix.draw()

        # Update the graph topology plot
        self.gt_fig_graph.clear()
        ax_g = self.gt_fig_graph.add_subplot(111)
        pos = nx.spring_layout(G, seed=self.gt_seed_var.get())
        colors = metrics['node']['community'].values
        node_sizes = [d * 200 + 50 for d in metrics['node']['degree'].values]
        nx.draw(G, pos, ax=ax_g, node_color=colors, node_size=node_sizes, cmap=plt.cm.jet, with_labels=True, font_size=8, font_color='black', edge_color='grey')
        ax_g.set_title("Graph Topology")
        self.gt_canvas_graph.draw()

    def run_mst_analysis(self):
        """Performs Minimum Spanning Tree analysis and updates the GUI."""
        subject_id = self.gt_subject_var.get()
        matrix_key = self.gt_matrix_key_var.get()
        
        if not all([subject_id, matrix_key]):
            messagebox.showerror("Error", "Please select a valid subject and matrix.")
            return

        self.status_var.set("Computing Minimum Spanning Tree...")
        self.root.update_idletasks()

        matrix = self.connectivity_results[subject_id]['matrices'][matrix_key]
        
        # Assuming gm.compute_mst_metrics exists and works as intended
        mst_metrics = gm.compute_mst_metrics(matrix)
        if not mst_metrics:
            self.status_var.set("MST analysis failed.")
            return

        # Update the global metrics table
        for i in self.gt_summary_table.get_children(): self.gt_summary_table.delete(i)
        self.gt_summary_table.insert('', 'end', values=("--- MST Metrics ---", ""))
        for key, value in mst_metrics.items():
            self.gt_summary_table.insert('', 'end', values=(key.replace('_', ' ').title(), f"{value:.4f}"))

        # Visualize the MST in the graph panel
        self.gt_fig_graph.clear()
        ax_g = self.gt_fig_graph.add_subplot(111)
        
        distance_matrix = 1.0 - np.abs(matrix)
        G_full = nx.from_numpy_array(distance_matrix)
        mst = nx.minimum_spanning_tree(G_full, weight='weight')
        
        pos = nx.spring_layout(mst, seed=self.gt_seed_var.get())
        degrees = [d for _, d in mst.degree()]
        node_sizes = [d * 100 + 50 for d in degrees]
        
        nx.draw(mst, pos, ax=ax_g, node_color='skyblue', node_size=node_sizes, with_labels=True, font_size=8)
        ax_g.set_title(f"Minimum Spanning Tree (MST)\n({subject_id} - {matrix_key})")
        self.gt_canvas_graph.draw()
        self.gt_notebook.select(0)

        self.status_var.set("MST analysis complete.")

    def run_multi_threshold_graph_analysis(self):
        """Runs the analysis across multiple thresholds and displays integrated results."""
        subject_id = self.gt_subject_var.get()
        matrix_key = self.gt_matrix_key_var.get()
        
        if not all([subject_id, matrix_key]):
            messagebox.showerror("Error", "Please select a valid subject and matrix.")
            return

        self.status_var.set("Running multi-threshold analysis... (this may take a moment)")
        self.root.update_idletasks()

        matrix = self.connectivity_results[subject_id]['matrices'][matrix_key]
        ch_names = self.preprocessed_data[subject_id].copy().pick_types(eeg=True).ch_names

        # Assuming gm.run_multi_threshold_analysis exists
        results = gm.run_multi_threshold_analysis(
            matrix, ch_names, threshold_range=(10, 40), n_steps=16, seed=self.gt_seed_var.get()
        )

        if not results:
            self.status_var.set("Multi-threshold analysis failed.")
            return

        # Update the global metrics table with integrated values
        for i in self.gt_summary_table.get_children(): self.gt_summary_table.delete(i)
        self.gt_summary_table.insert('', 'end', values=("--- AUC Integration ---", ""))
        for key, value in results['integrated'].items():
            self.gt_summary_table.insert('', 'end', values=(key.replace('_', ' ').title(), f"{value:.4f}"))

        # Create the ribbon plot
        self.gt_fig_integration.clear()
        ax = self.gt_fig_integration.add_subplot(111)
        
        df_plot = results['steps'][['C_global', 'L_char', 'E_global', 'Q']]
        df_plot_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
        
        for col in df_plot_norm.columns:
            ax.plot(results['thresholds'], df_plot_norm[col], label=col, marker='o', linestyle='-')

        ax.set_title("Normalized Metrics vs. Network Density")
        ax.set_xlabel("Network Density (%)")
        ax.set_ylabel("Normalized Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        self.gt_canvas_integration.draw()
        
        self.gt_notebook.select(2) # Select the integration tab
        self.status_var.set("Multi-threshold analysis complete.")


    def display_nbs_results(self, results):
        """Creates a window to display the results of the NBS analysis."""
        if not results:
            self.status_var.set("NBS analysis yielded no results.")
            return

        self.status_var.set(f"NBS complete. p-value: {results['p_value']:.4f}")

        results_window = tk.Toplevel(self.root)
        results_window.title("Network-Based Statistics (NBS) Results")
        
        text_area = tk.Text(results_window, wrap='word', height=20, width=80)
        text_area.pack(padx=10, pady=10, fill='both', expand=True)

        report = "--- NBS Analysis Results ---\n\n"
        report += f"Corrected p-value: {results['p_value']:.5f}\n"
        report += f"Max component size (real): {results['max_comp_size_real']}\n"
        report += f"Statistic used: {results.get('component_stat', 'N/A')}\n"
        report += f"T-statistic threshold used: {results.get('t_threshold', 'N/A'):.3f}\n\n"
        
        if results['components']:
            report += f"Found {len(results['components'])} significant connected components:\n"
            for i, comp in enumerate(results['components']):
                report += f"  - Component {i+1}: {len(comp)} nodes\n"
        else:
            report += "No significant components found with the specified parameters.\n"
            
        text_area.insert('1.0', report)
        text_area.config(state='disabled')



    def _on_thresh_method_change(self, *args):
        """Show the parameter controls relevant to the chosen thresholding method."""
        method = self.gt_thresh_method_var.get()
        if method == 'proportional':
            self.gt_backbone_lf.pack_forget()
            self.gt_prop_thresh_lf.pack(fill='x', expand=True)
        elif method == 'backbone':
            self.gt_prop_thresh_lf.pack_forget()
            self.gt_backbone_lf.pack(fill='x', expand=True)


    def _update_gt_subject_dropdown(self):
        """Fill the subject dropdown of the Graph Theory tab."""
        if hasattr(self, 'connectivity_results'):
            subjects = list(self.connectivity_results.keys())
            self.gt_subject_dropdown.configure(values=subjects)
            if subjects:
                self.gt_subject_var.set(subjects[0])

    def _update_gt_matrix_dropdown(self, *args):
        """Fill the dropdown of matrices available for the selected subject."""
        subject_id = self.gt_subject_var.get()
        if subject_id and hasattr(self, 'connectivity_results'):
            matrices = list(self.connectivity_results.get(subject_id, {}).get('matrices', {}).keys())
            self.gt_matrix_dropdown.configure(values=matrices)
            if matrices:
                self.gt_matrix_key_var.set(matrices[0])


    def run_batch_graph_analysis(self):
        """Launch the worker for the batch graph theory analysis."""
        if not self.connectivity_results:
            messagebox.showerror("Missing data", "Run the connectivity analysis first.")
            return
        self.status_var.set("Starting the batch graph theory analysis...")
        self.gt_progress.configure(maximum=100, value=0)
        self.gt_progress_percent.set("0%")
        threading.Thread(target=self._graph_theory_worker, daemon=True).start()


    def run_single_graph_analysis(self):
        """
        Run the graph theory analysis for the single subject and matrix selected
        in exploratory mode, then refresh the GUI.
        """
        # 1. Fetch the parameters from the GUI (exploratory mode)
        subject_id = self.gt_subject_var.get()
        matrix_key = self.gt_matrix_key_var.get()
        
        if not all([subject_id, matrix_key]):
            messagebox.showerror("Error", "Select a valid subject and a valid connectivity matrix.")
            return

        try:
            matrix = self.connectivity_results[subject_id]['matrices'][matrix_key]
            ch_names = self.preprocessed_data[subject_id].copy().pick_types(eeg=True).ch_names
        except KeyError:
            messagebox.showerror("Data error", f"Required data not found for {subject_id}. Run the earlier analyses first.")
            return

        # 2. Collect the parameters specific to this analysis
        thresh_method = self.gt_thresh_method_var.get()
        params = {
            "subject_id": subject_id,
            "matrix_key": matrix_key,
            "weight_handling": self.gt_neg_var.get(),
            "threshold_method": thresh_method,
            "threshold_value": self.gt_threshold_var.get() if thresh_method == 'proportional' else (self.gt_alpha_var.get() if thresh_method == 'backbone' else 'N/A'),
            "seed": self.gt_seed_var.get(),
            "is_directed": self.gt_directed_var.get()
        }
        
        self.status_var.set(f"Analysis running for {subject_id}...")
        self.root.update_idletasks()

        # 3. Prepare the base graph, always unthresholded, for methods such as the backbone
        G_base = gm.prepare_graph(
            matrix=matrix,
            threshold_perc=None,
            weight_handling=params['weight_handling'],
            directed=params['is_directed']
        )
        if G_base is None: return

        # 4. Apply the chosen thresholding method to obtain the final graph
        if thresh_method == 'proportional':
            G = gm.prepare_graph(matrix, threshold_perc=params['threshold_value'], weight_handling=params['weight_handling'], directed=params['is_directed'])
        elif thresh_method == 'backbone':
            if params['is_directed']:
                messagebox.showwarning("Warning", "The disparity filter is not defined for directed graphs. Analysis stopped.")
                return
            G = gm.disparity_filter(G_base, alpha=params['threshold_value'])
        else: # 'none'
            G = G_base

        if G is None or G.number_of_nodes() == 0:
            messagebox.showwarning("Empty graph", "The thresholding method produced an empty graph.")
            self.status_var.set("Pronto.")
            return

        # 5. Compute the metrics on the final graph
        metrics = gm.compute_graph_metrics(G, ch_names, seed=params['seed'])
        if not metrics: 
            self.status_var.set("Metric computation failed.")
            return

        # 6. Keep the results for the Summary tab (optional but advisable)
        result_key = f"{matrix_key}_{thresh_method}{params['threshold_value']}"
        if subject_id not in self.graph_theory_results:
            self.graph_theory_results[subject_id] = {}
        self.graph_theory_results[subject_id][result_key] = {f"gt_{k}": v for k, v in metrics['global'].items()}
        
        # 7. Refresh the GUI with the new results
        self.update_graph_theory_display(G, metrics, subject_id, matrix_key)

        # 8. Export the results of this single analysis
        export_dir = os.path.join(os.path.dirname(__file__), "graph_results")
        export_success = gm.export_graph_analysis_results(
            output_dir=export_dir,
            subject_id=f"{subject_id}_{result_key}",
            params=params,  
            global_metrics=metrics['global'],
            node_metrics=metrics['node']
        )
        
        if export_success:
            self.status_var.set("Analysis and export completed successfully.")
        else:
            self.status_var.set("The analysis finished but the export failed.")

    def export_all_results(self):
        """Export every result to CSV files."""
        if not hasattr(self, 'biomarkers_df') or self.biomarkers_df.empty:
            messagebox.showerror("Error", "Nothing to export. Run the analyses first.")
            return
        
        export_dir = filedialog.askdirectory(title="Select the folder for the export")
        if not export_dir:
            return
        
        try:
            # Export the biomarkers
            biomarkers_file = os.path.join(export_dir, "eeg_biomarkers.csv")
            self.biomarkers_df.to_csv(biomarkers_file, index=False)
            
            # Export the regression results where available
            if hasattr(self, 'regression_results_text'):
                regression_content = self.regression_results_text.get('1.0', tk.END)
                if regression_content.strip():
                    regression_file = os.path.join(export_dir, "regression_results.txt")
                    with open(regression_file, 'w') as f:
                        f.write(regression_content)
            
            messagebox.showinfo("Export finished", 
                            f"Results exported to:\\n{export_dir}")
            self.status_var.set("The export completed successfully.")
            
        except Exception as e:
            messagebox.showerror("Export error", f"Error during the export: {e}")


    def generate_report(self):
        """Generate a complete report of the analysis."""
        if not hasattr(self, 'biomarkers_df') or self.biomarkers_df.empty:
            messagebox.showerror("Error", "No data for the report. Run the analyses first.")
            return
        
        report_file = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save the report as..."
        )
        
        if not report_file:
            return
        
        try:
            with open(report_file, 'w') as f:
                f.write("EEG ANALYSIS REPORT\n")
                f.write("=" * 50 + "\\n\\n")
                
                # General information
                f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Files processed: {len(self.uploaded_files)}\\n")
                f.write(f"Subjects analysed: {len(set([os.path.basename(fp).split('_')[0] for fp in self.uploaded_files]))}\\n\\n")
                
                # Parameters used
                f.write("PARAMETERS USED:\\n")
                f.write(f"- High-pass filter: {self.low_freq.get()} Hz\\n")
                f.write(f"- Low-pass filter: {self.high_freq.get()} Hz\\n")
                f.write(f"- Notch filter: {self.notch_freq.get()} Hz\\n")
                f.write(f"- ICA components: {self.n_components.get()}\\n")
                f.write(f"- ICA method: {self.ica_method.get()}\\n\\n")
                
                # Biomarker statistics
                if hasattr(self, 'biomarkers_df'):
                    f.write("BIOMARKER STATISTICS:\\n")
                    numeric_cols = self.biomarkers_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if col not in ['Subject']:
                            data = self.biomarkers_df[col].dropna()
                            if len(data) > 0:
                                f.write(f"- {col}: Media={data.mean():.3f}, SD={data.std():.3f}, N={len(data)}\\n")
                    f.write("\\n")
                
                # Regression results
                if hasattr(self, 'regression_results_text'):
                    regression_content = self.regression_results_text.get('1.0', tk.END).strip()
                    if regression_content:
                        f.write("REGRESSION RESULTS:\n")
                        f.write(regression_content)
                        f.write("\\n\\n")
                
                f.write("=" * 50 + "\\n")
                f.write("Fine report\\n")
            
            messagebox.showinfo("Report generated", f"Report saved to:\\n{report_file}")
            self.status_var.set("The report was generated successfully.")
            
        except Exception as e:
            messagebox.showerror("Report error", f"Error while generating the report: {e}")


    def create_results_summary(self, parent):
        """Create results summary display"""
        self.summary_text = tk.Text(parent, height=15, bg=self.colors['surface'], fg=self.colors['text_primary'])
        self.summary_text.pack(fill='both', expand=True)

if __name__ == "__main__":
    # No window is created here any more.
    # It is created directly inside the class.
    multiprocessing.freeze_support() 
    app = EEGAnalysisGUI()
    app.root.mainloop()
# --- end of the EEGAnalysisGUI class ---



