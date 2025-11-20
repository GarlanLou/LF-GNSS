# gnss_dataset.py
# Module for processing and loading GNSS datasets

import torch
import numpy as np
import pandas as pd
import json
import pymap3d as p3d
from io import StringIO
from torch.utils.data import Dataset
from gnss_foundation.rinex import rnxdec
import gnss_util as util
from gnss_engine import stdpos

# Use CUDA if available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_gt_data(file_path):
    """Read and clean ground truth data from a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find where the actual data starts
    start_idx = -1
    for i, line in enumerate(lines):
        if "UTC Offset:" in line:
            start_idx = i
            break

    if start_idx == -1:
        raise ValueError("Error: 'UTC Offset:' not found in the file.")

    # Data usually starts 4 lines after 'UTC Offset:'
    data_start_idx = start_idx + 4
    data_lines = lines[data_start_idx:]

    # Read the data using pandas
    data_str = ''.join(data_lines)
    gt = pd.read_csv(StringIO(data_str), header=None, sep=' +', engine='python')
    gt = gt.apply(pd.to_numeric, errors='coerce')

    # Remove any non-numeric (NaN) values
    if gt.isnull().values.any():
        print("Warning: Non-numeric values found, converting to NaN.")
        gt = gt.dropna()
    
    # Add leap second correction
    gt[0] = gt[0] + 18
    return gt


class GNSSDataset(Dataset):
    """Class to load and process GNSS data"""
    def __init__(self, config, is_train=True, norm_params=None):
        print(f"-----------Setting up GNSSDataset with config: {config}-----------")
        with open(config) as f:
            self.conf = json.load(f)
        print("Configuration loaded.")
        
        self.is_train = is_train  # True if this is training data
        self.norm_params = norm_params  # Stats for normalizing validation data
        
        # Load ground truth data if it exists
        self.gt_data = []
        if self.conf.get("gt", None):
            print("Loading ground truth data...")
            self.gt = read_gt_data(self.conf['gt'])
            print(f"Ground truth data loaded. Size: {len(self.gt)} rows")
        else:
            self.gt = None
            print("No ground truth data provided.")
        
        # Read observation and navigation files
        print("Setting up RINEX decoder...")
        self.decoder = rnxdec()
        self.sigs = util.convert_signals_to_rSigRnx(self.conf['sigs'])
        self.decoder.setSignals(self.sigs)
        self.decoder.autoSubstituteSignals()
        nsys = len(self.decoder.sig_tab)
        print("Reading observation and navigation data...")
        self.obs, self.nav, self.sta = util.gnss_reader(decoder=self.decoder, obs_files=self.conf['obs'], eph_file=self.conf['eph'])
        print(f"Data reading done. Observations: {len(self.obs)}, Signals: {nsys}")
        
        self.nav.nsys = nsys
        self.std = stdpos(self.nav, self.decoder.pos, 'test_stdpos.log')
        self.obss = []
        self.in_data = []

        # Stats for normalizing data
        self.snr_stats = {'mean': 0, 'std': 1}
        self.elev_stats = {'mean': 0, 'std': 1}
        self.resd_stats = {'mean': 0, 'std': 1}
        
        print("Processing observation times...")
        processed_epochs = 0
        valid_epochs = 0

        # Go through each observation time
        for obs_epoch in self.obs:
            t = float(obs_epoch.t.time) + float(obs_epoch.t.sec)
            # Only use data within the specified time range
            if t >= self.conf['start_time'] and (self.conf['end_time'] == -1 or t <= self.conf['end_time']) and abs(t - round(t)) < 0.1:
                try:
                    # Try to get a rough position estimate
                    ret = self.std.coarse_pos(obs_epoch, mode="extraction")
                except Exception as e:
                    ret = {"status": False, "msg": f"Error during coarse pos: {str(e)}"}
                
                processed_epochs += 1
                if processed_epochs % 1000 == 0:
                    print(f"Processed {processed_epochs} epochs, {valid_epochs} valid...")

                # If we got a good position estimate
                if ret['status']:
                    # Match with ground truth data if available
                    if self.gt is not None:
                        closest_index = (self.gt[0] - t).abs().argmin()
                        closest_diff = abs(self.gt.loc[closest_index, 0] - t)
                        # Skip if the times don't match well
                        if closest_diff > 0.1:
                            continue
                        else:                      
                            gt_row = self.gt.loc[closest_index]
                            # Get true position (lat, lon, height)
                            gt_llh = [(gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600),
                                    (gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600),
                                    gt_row[9]]
                            # Convert estimated position to lat/lon/height
                            ret_llh = p3d.ecef2geodetic(*ret['pos'][:3])
                            # Calculate error in East/North/Up directions
                            enu_error = p3d.geodetic2enu(ret_llh[0], ret_llh[1], ret_llh[2],
                                                        gt_llh[0], gt_llh[1], gt_llh[2])
                            # Skip if error is too big
                            if np.linalg.norm(enu_error) > 1000:
                                continue

                            # Save ground truth and observation data
                            self.gt_data.append([
                                gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600,
                                gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600,
                                gt_row[9]
                            ])
                            self.obss.append(obs_epoch)
                            # Get signal data (strength, angles, residuals)
                            SNR = np.array(ret['data']['SNR'], dtype=np.float64)
                            azel = np.array(ret['data']['azel'], dtype=np.float64)
                            azel = np.degrees(azel)
                            resd = np.array(ret['data']['residual'], dtype=np.float64)
                            
                            # Put all input data together
                            epoch_data = np.hstack([
                                SNR.reshape(-1, 1),
                                azel[:, 0].reshape(-1, 1),
                                azel[:, 1].reshape(-1, 1),
                                resd.reshape(-1, 1),
                            ])
                            # Convert to tensor and save
                            self.in_data.append(torch.tensor(epoch_data, dtype=torch.double))
                            valid_epochs += 1
        
        print(f"Done processing epochs. Total: {processed_epochs}, Valid: {valid_epochs}")

        # Set up data normalization
        if self.is_train:
            # For training data: calculate normalization stats
            print("Calculating normalization stats for training data...")
            self.compute_normalization_params_from_in_data()
            # Save stats for validation data to use
            self.norm_params = {
                'snr_stats': self.snr_stats,
                'azimuth_stats': self.azimuth_stats,
                'elev_stats': self.elev_stats,
                'resd_stats': self.resd_stats
            }
        else:
            # For validation data: use stats from training data
            if self.norm_params is not None:
                print("Using training data normalization stats for validation data")
                self.snr_stats = self.norm_params['snr_stats']
                self.azimuth_stats = self.norm_params['azimuth_stats']
                self.elev_stats = self.norm_params['elev_stats']
                self.resd_stats = self.norm_params['resd_stats']
            else:
                print("Warning: No normalization stats provided, using defaults")
                self.compute_normalization_params_from_in_data()
        
        print("Normalizing input data...")
        self.normalize_inputs()
        print("Normalization done.")

        # Build final dataset
        if self.gt is not None:
            self.data = list(zip(self.obss, self.gt_data, self.in_data))
        else:
            self.data = list(zip(self.obss, [None] * len(self.obss), self.in_data))
        
        print(f"GNSSDataset setup complete. Final dataset size: {len(self.data)}")
    
    def compute_normalization_params_from_in_data(self):
        """
        Calculate mean and standard deviation for all features.
        Used to normalize the data.
        You can design your own normalization strategy here.
        """
        # Combine all data to get overall stats
        all_snr = np.concatenate([d[:, 0].numpy() for d in self.in_data])
        all_azimuth = np.concatenate([d[:, 1].numpy() for d in self.in_data])
        all_elev = np.concatenate([d[:, 2].numpy() for d in self.in_data])
        all_resd = np.concatenate([d[:, 3].numpy() for d in self.in_data])
        
        # Calculate mean and standard deviation
        self.snr_stats = {'mean': all_snr.mean(), 'std': all_snr.std()}
        self.azimuth_stats = {'mean': all_azimuth.mean(), 'std': all_azimuth.std()}
        self.elev_stats = {'mean': all_elev.mean(), 'std': all_elev.std()}
        self.resd_stats = {'mean': all_resd.mean(), 'std': all_resd.std()}

    def normalize_inputs(self):
        """
        Normalize input data.
        """
        for i, epoch_data in enumerate(self.in_data):
            self.in_data[i][:, 0] = (epoch_data[:, 0] - self.snr_stats['mean']) / self.snr_stats['std']
            self.in_data[i][:, 1] = (epoch_data[:, 1] - self.azimuth_stats['mean']) / self.azimuth_stats['std']
            self.in_data[i][:, 2] = (epoch_data[:, 2] - self.elev_stats['mean']) / self.elev_stats['std']
            self.in_data[i][:, 3] = (epoch_data[:, 3] - self.resd_stats['mean']) / self.resd_stats['std']
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        o, gt_row, in_data = self.data[idx]
        if gt_row is None:
            gt_row = [0.0, 0.0, 0.0]
        return in_data, torch.tensor(gt_row, dtype=torch.double), o

def collate_fn(batch):
    """
    Function to handle batches of different sequence lengths.
    Pads sequences so they're all the same length.
    """
    inputs, labels, obss = zip(*batch)

    # Find the longest sequence
    max_len = max([input.size(0) for input in inputs])
    
    padded_inputs = []
    masks = []
    
    # Pad each input sequence
    for input in inputs:
        pad_size = max_len - input.size(0)
        padded_input = torch.cat([input, torch.zeros(pad_size, input.size(1), dtype=torch.double)], dim=0)
        mask = torch.cat([torch.ones(input.size(0), dtype=torch.double), torch.zeros(pad_size, dtype=torch.double)])
        
        padded_inputs.append(padded_input)
        masks.append(mask)
    
    # Convert lists to tensors
    padded_inputs = torch.stack(padded_inputs)
    masks = torch.stack(masks)
    labels = torch.stack(labels)
    
    # Return padded inputs, masks, and original observations
    return padded_inputs, labels, masks, obss