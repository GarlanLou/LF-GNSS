# gnss_util.py
# GNSS utility functions for coordinate transformation and data processing

from gnss_foundation.gnss_type import Nav
from gnss_foundation.gnss_type import rSigRnx,sys2str
import torch
import numpy as np
import pymap3d as p3d
from gnss_foundation.gnss_type import gtime_t
import tqdm
import os

# Earth ellipsoid parameters (WGS84)
a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2-f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def geodetic2ecef_torch(lat, lon, h):
    """
    Convert geodetic coordinates (lat, lon, h) to ECEF coordinates, supporting PyTorch.
    Input and output preserve gradients.
    """
    lamb = torch.deg2rad(lat)
    phi = torch.deg2rad(lon)
    s = torch.sin(lamb)
    N = a / torch.sqrt(1 - e_sq * s * s)

    x = (h + N) * torch.cos(lamb) * torch.cos(phi)
    y = (h + N) * torch.cos(lamb) * torch.sin(phi)
    z = (h + (1 - e_sq) * N) * torch.sin(lamb)
    
    return torch.stack([x, y, z])

def ecef2enu_torch(x, y, z, lat0, lon0, h0):
    """
    Convert ECEF coordinates (x, y, z) to ENU coordinates with reference point (lat0, lon0, h0), supporting PyTorch.
    """
    ref_ecef = geodetic2ecef_torch(lat0, lon0, h0)
    ref_x, ref_y, ref_z = ref_ecef

    lamb = torch.deg2rad(lat0)
    phi = torch.deg2rad(lon0)
    s = torch.sin(lamb)
    c = torch.cos(lamb)

    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    
    xd = x - ref_x
    yd = y - ref_y
    zd = z - ref_z
    
    t = -cos_phi * xd - sin_phi * yd
    
    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = t * s + c * zd
    zUp = c * cos_phi * xd + c * sin_phi * yd + s * zd
    
    return torch.stack([xEast, yNorth, zUp])

def ecef_to_enu_torch(pred_ecef, gt_llh):
    """
    Calculate ENU error between two ECEF coordinates using PyTorch, preserving gradients.
    """
    lat_ref, lon_ref, h_ref = gt_llh
    enu_error = ecef2enu_torch(pred_ecef[0], pred_ecef[1], pred_ecef[2], lat_ref, lon_ref, h_ref)

    return enu_error

def geodetic2ecef(lat, lon, h):
    """
    Convert geodetic coordinates (lat, lon, h) to ECEF coordinates using NumPy.
    """
    lamb = np.radians(lat)
    phi = np.radians(lon)
    s = np.sin(lamb)
    N = a / np.sqrt(1 - e_sq * s * s)

    x = (h + N) * np.cos(lamb) * np.cos(phi)
    y = (h + N) * np.cos(lamb) * np.sin(phi)
    z = (h + (1 - e_sq) * N) * np.sin(lamb)
    
    return np.array([x, y, z])

def ecef2enu(x, y, z, lat0, lon0, h0):
    """
    Convert ECEF coordinates (x, y, z) to ENU coordinates with reference point (lat0, lon0, h0) using NumPy.
    """
    ref_ecef = geodetic2ecef(lat0, lon0, h0)
    ref_x, ref_y, ref_z = ref_ecef

    lamb = np.radians(lat0)
    phi = np.radians(lon0)
    s = np.sin(lamb)
    c = np.cos(lamb)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    xd = x - ref_x
    yd = y - ref_y
    zd = z - ref_z
    
    t = -cos_phi * xd - sin_phi * yd
    
    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = t * s + c * zd
    zUp = c * cos_phi * xd + c * sin_phi * yd + s * zd
    
    return np.array([xEast, yNorth, zUp])

def convert_signals_to_rSigRnx(sigs):
    """
    Convert a list of signal frequency points to a list of rSigRnx objects.

    Parameters
    ----------
    sigs : list of str
        List of signal frequency point strings, e.g., ["GC1C", "GL1C"].

    Returns
    -------
    converted_sigs : list of rSigRnx
        List of converted rSigRnx objects.
    """
    converted_sigs = [rSigRnx(sig) for sig in sigs]
    return converted_sigs

def gnss_reader(decoder, obs_files, eph_file):
    """
    Read observation and navigation data files, and return parsed observation data, navigation data, and station information.
    
    Parameters:
    - decoder (rnxdec): RINEX decoder instance
    - obs_files (str or list): RINEX observation file path or path list
    - eph_file (str): RINEX navigation file path
    
    Returns:
    - obs_data (list): Observation data list
    - nav_data (Nav): Navigation data object
    - sta_data (dict): Station related information
    """
    obs_data = []
    sta_data = {'pos': None, 'ecc': None, 'ant': None, 'rcv': None}

    files_to_process = obs_files if isinstance(obs_files, list) else [obs_files]
    
    total_files = len(files_to_process)
    
    for file_idx, obs_file in enumerate(files_to_process):
        file_name = os.path.basename(obs_file)
        print(f"Processing file {file_idx+1}/{total_files}: {file_name}")
        
        if decoder.decode_obsh(obs_file) >= 0:
            epoch_count = 0
            
            # 简单进度条，不更新描述
            pbar = tqdm.tqdm(unit="epochs", leave=False)
                
            while True:
                obs_epoch = decoder.decode_obs()
                if obs_epoch is None:
                    break
                obs_data.append(obs_epoch)
                epoch_count += 1
                pbar.update(1)
            
            pbar.close()
            print(f"Finished reading {file_name}: {epoch_count} epochs processed")
            
            sta_data.update({'pos': decoder.pos, 'ecc': decoder.ecc, 'ant': decoder.ant, 'rcv': decoder.rcv})
        else:
            print(f"Warning: Failed to decode observation file {obs_file}")

    nav_data = Nav(nf=1)
    
    if eph_file:
        eph_file_name = os.path.basename(eph_file)
        print(f"Decoding navigation file: {eph_file_name}")
        nav_data = decoder.decode_nav(eph_file, nav_data)

    print(f"Total epochs read: {len(obs_data)}")
    return obs_data, nav_data, sta_data


def display_signals(rnx):
    """
    Print information about available signals and selected signals.

    Parameters
    ----------
    rnx : object
        Object containing signal information, must include sig_map and sig_tab attributes.
    """
    print("Available signals")
    for sys, sigs in rnx.sig_map.items():
        txt = "{:7s} {}".format(
            sys2str(sys),
            ' '.join([sig.str() for sig in sigs.values()])
        )
        print(txt)

    print("\nSelected signals")
    for sys, tmp in rnx.sig_tab.items():
        txt = "{:7s} ".format(sys2str(sys))
        for _, sigs in tmp.items():
            txt += "{} ".format(' '.join([sig.str() for sig in sigs]))
        print(txt)

def validate_model(net, dataset, loader, device=DEVICE):
    """
    Validate model performance and calculate accuracy metrics.
    """
    net.eval()
    val_pred_pos = []
    val_gt_pos = []
    val_errors = []
    enu_errors_with_time = []
    pos_with_time = []

    with torch.no_grad():
        dataset.std.nav.t = gtime_t(0, 0.0)
        dataset.std.nav.x = np.zeros(dataset.nav.nx)
        dataset.std.initialize_covariance()
        
        for inputs, labels, masks, obss in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            val_R, val_bias = net(inputs, masks)

            for i, input_data in enumerate(inputs):
                o = obss[i]
                gt_row = labels[i].cpu().numpy()

                try:
                    valid_mask = masks[i].bool()
                    val_valid_R = val_R[i][valid_mask]
                    val_valid_bias = val_bias[i][valid_mask]
                    result = dataset.std.process(obs=o, Net_R=val_valid_R, bias=val_valid_bias)

                    if not result['status']:
                        print(f"Error during observation processing: {result['msg']}")
                        continue

                    pred_ecef = result['pos'][:3].cpu().numpy()
                    pos = p3d.ecef2geodetic(*pred_ecef)
                    val_pred_pos.append(pos)
                    val_gt_pos.append([gt_row[0], gt_row[1], gt_row[2]])

                    enu_error = p3d.geodetic2enu(*val_pred_pos[-1], *val_gt_pos[-1])
                    timestamp = o.t.time + o.t.sec
                    val_errors.append(enu_error)
                    enu_errors_with_time.append([timestamp, *enu_error])
                    pos_with_time.append([timestamp, *pos])
                except Exception as e:
                    print(f"Exception in validation: {e}")
                    continue

    val_errors = np.array(val_errors)
    if val_errors.size > 0:
        val_3d_rmse = np.sqrt(np.mean(np.linalg.norm(val_errors, axis=1) ** 2))
        val_2d_rmse = np.sqrt(np.mean(np.linalg.norm(val_errors[:, :2], axis=1) ** 2))
        val_east_rmse = np.sqrt(np.mean(val_errors[:, 0] ** 2))
        val_north_rmse = np.sqrt(np.mean(val_errors[:, 1] ** 2))
        val_up_rmse = np.sqrt(np.mean(val_errors[:, 2] ** 2))
    else:
        val_3d_rmse = float('inf')
        val_2d_rmse = float('inf')
        val_east_rmse = float('inf')
        val_north_rmse = float('inf')
        val_up_rmse = float('inf')
        print("Warning: No valid samples during validation.")

    return {
        "3d_rmse": val_3d_rmse,
        "2d_rmse": val_2d_rmse,
        "east_rmse": val_east_rmse,
        "north_rmse": val_north_rmse,
        "up_rmse": val_up_rmse,
        "enu_errors_with_time": enu_errors_with_time,
        "pos_with_time": pos_with_time
    }

def test_model(net, dataset, loader, device, print_progress=True):
    """
    Test model performance with detailed output.
    """
    net.eval()
    val_pred_pos = []
    val_gt_pos = []
    val_errors = []
    enu_errors_with_time = []
    pos_with_time = []

    with torch.no_grad():
        dataset.std.nav.t = gtime_t(0, 0.0)
        dataset.std.nav.x = np.zeros(dataset.nav.nx)
        dataset.std.initialize_covariance()
        
        total_batches = len(loader)
        batch_count = 0
        
        for inputs, labels, masks, obss in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            val_R, val_bias = net(inputs, masks)

            for i, input_data in enumerate(inputs):
                o = obss[i]
                gt_row = labels[i].cpu().numpy()

                try:
                    valid_mask = masks[i].bool()
                    val_valid_R = val_R[i][valid_mask]
                    val_valid_bias = val_bias[i][valid_mask]
                    result = dataset.std.process(obs=o, Net_R=val_valid_R, bias=val_valid_bias)

                    if not result['status']:
                        if print_progress:
                            print(f"Error during observation processing: {result['msg']}")
                        continue

                    pred_ecef = result['pos'][:3].cpu().numpy()
                    pos = p3d.ecef2geodetic(*pred_ecef)
                    val_pred_pos.append(pos)
                    val_gt_pos.append([gt_row[0], gt_row[1], gt_row[2]])

                    enu_error = p3d.geodetic2enu(*val_pred_pos[-1], *val_gt_pos[-1])
                    timestamp = o.t.time + o.t.sec
                    val_errors.append(enu_error)
                    enu_errors_with_time.append([timestamp, *enu_error])
                    pos_with_time.append([timestamp, *pos])
                    
                    if print_progress:
                        error_3d = np.linalg.norm(enu_error)
                        error_2d = np.linalg.norm(enu_error[:2])
                        print(f"Time {timestamp:.1f} - "
                              f"3D: {error_3d:.2f}m, "
                              f"2D: {error_2d:.2f}m, "
                              f"E: {enu_error[0]:.2f}m, "
                              f"N: {enu_error[1]:.2f}m, "
                              f"U: {enu_error[2]:.2f}m")
                except Exception as e:
                    if print_progress:
                        print(f"Exception in validation: {e}")
                    continue

    val_errors = np.array(val_errors)
    if val_errors.size > 0:
        val_3d_rmse = np.sqrt(np.mean(np.linalg.norm(val_errors, axis=1) ** 2))
        val_2d_rmse = np.sqrt(np.mean(np.linalg.norm(val_errors[:, :2], axis=1) ** 2))
        val_east_rmse = np.sqrt(np.mean(val_errors[:, 0] ** 2))
        val_north_rmse = np.sqrt(np.mean(val_errors[:, 1] ** 2))
        val_up_rmse = np.sqrt(np.mean(val_errors[:, 2] ** 2))

    else:
        val_3d_rmse = float('inf')
        val_2d_rmse = float('inf')
        val_east_rmse = float('inf')
        val_north_rmse = float('inf')
        val_up_rmse = float('inf')
        if print_progress:
            print("Warning: No valid samples during validation.")

    return {
        "3d_rmse": val_3d_rmse,
        "2d_rmse": val_2d_rmse,
        "east_rmse": val_east_rmse,
        "north_rmse": val_north_rmse,
        "up_rmse": val_up_rmse,
        "enu_errors_with_time": enu_errors_with_time,
        "pos_with_time": pos_with_time
    }