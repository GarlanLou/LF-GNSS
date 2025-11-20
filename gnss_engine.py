# gnss_engine.py
# GNSS positioning engine with machine learning enhancements
# Based on cssrlib by Hirokawa
# Added many new function designs and made extensive adaptations for PyTorch.

import numpy as np
from gnss_foundation.gnss_type import rCST, ecef2pos, geodist, satazel, \
    tropmodel, tropmapf, sat2prn, uGNSS, uTropoModel, uIonoModel, \
    timediff, time2gpst, dops, uTYP, Obs, time2str, sat2id
from gnss_foundation.ephemeris import satposs_enhanced
from math import sin, cos
import torch

def ionKlobuchar(t, pos, az, el, ion=None):
    """ Klobuchar ionosphere delay model """
    psi = 0.0137/(el/np.pi+0.11)-0.022
    phi = pos[0]/np.pi+psi*cos(az)
    phi = np.max((-0.416, np.min((0.416, phi))))
    lam = pos[1]/np.pi+psi*sin(az)/cos(phi*np.pi)
    phi += 0.064*cos((lam-1.617)*np.pi)
    _, tow = time2gpst(t)
    tt = 43200.0*lam+tow
    tt -= tt//86400*86400
    sf = 1.0+16.0*(0.53-el/np.pi)**3

    h = [1, phi, phi**2, phi**3]
    amp = max(h@ion[0, :], 0)
    per = max(h@ion[1, :], 72000.0)
    x = 2.0*np.pi*(tt-50400.0)/per
    if np.abs(x) < 1.57:
        v = 5e-9+amp*(1.0+x*x*(-0.5+x*x/24.0))
    else:
        v = 5e-9
    diono = rCST.CLIGHT*sf*v
    return diono

def ionmodel(t, pos, az, el, nav=None, model=uIonoModel.KLOBUCHAR, cs=None):
    """ Ionosphere delay estimation """
    if model == uIonoModel.KLOBUCHAR:
        diono = ionKlobuchar(t, pos, az, el, nav.ion)
    return diono

class stdpos():
    """ Standard positioning class for GNSS """

    def ICB(self, s=0):
        """ Return index of clock bias (s=0) or clock drift (s=1) """
        return 3 + s if self.nav.pmode == 0 else 6 + s
    def __init__(self, nav, pos0=np.zeros(3), logfile=None, trop_opt=0,
                 iono_opt=0, phw_opt=0, csmooth=False, rmode=0):
        """ Initialize positioning engine """
        self.nav = nav
        self.monlevel = 0
        self.nav.csmooth = csmooth
        self.nav.rmode = rmode
        self.cs_cnt = {}
        self.Lp_ = {}
        self.Ps_ = {}
        self.cs_t0 = {}

        # Models and options
        self.nav.trpModel = uTropoModel.SAAST
        self.ionoModel = uIonoModel.KLOBUCHAR
        self.nav.trop_opt = trop_opt
        self.nav.iono_opt = iono_opt
        self.nav.excl_sat = []
        self.nav.elmin = np.deg2rad(15.0)
        self.nav.cnr_min = 20.0


        # System and state configuration
        self.nav.nsys = 4  # GPS, GLONASS, GALILEO, BEIDOU
        self.nav.na = 3 + (3 if self.nav.pmode > 0 else 0) + 2 * self.nav.nsys
        self.nav.nq = self.nav.na
        self.nav.nx = self.nav.na

        # Initialize state vector
        self.nav.x = np.zeros(self.nav.nx)
        self.nav.err = [0, 0.000, 0.003]
        self.nav.eratio = np.ones(self.nav.nf) * 100

        self.nav.sig_p0 = 100.0   # [m]
        self.nav.sig_v0 = 1.0     # [m/s]

        self.nav.sig_cb0 = 100.0  # [m]
        self.nav.sig_cd0 = 1.0    # [m/s]

        # Process noise
        if self.nav.pmode == 0:
            self.nav.sig_qp = 1.0/np.sqrt(1)
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)
            self.nav.sig_qv = 1.0/np.sqrt(1)
        self.nav.sig_qcb = 0.1
        self.nav.sig_qcd = 0.01


        # Initialize covariance and process noise
        self.initialize_covariance()
        self.initialize_process_noise()
        
        # Fixed solution storage
        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))
        self.nav.el = np.zeros(uGNSS.MAXSAT)

        # Initial position
        self.nav.x[0:3] = 0.0
        if self.nav.pmode >= 1:
            self.nav.x[3:6] = 0.0

        # Logging
        self.nav.fout = None
        if logfile is None:
            self.nav.monlevel = 0
        else:
            self.nav.fout = open(logfile, 'w')
    def initialize_covariance(self):
        """ Initialize covariance matrix """
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        
        # Position and velocity uncertainties
        dP[0:3] = self.nav.sig_p0 ** 2
        if self.nav.pmode >= 1:
            dP[3:6] = self.nav.sig_v0 ** 2
            
        # Clock bias and drift uncertainties
        clk_bias_start = 6 if self.nav.pmode > 0 else 3
        clk_drift_start = clk_bias_start + self.nav.nsys
        dP[clk_bias_start:clk_drift_start] = self.nav.sig_cb0 ** 2
        dP[clk_drift_start:clk_drift_start + self.nav.nsys] = self.nav.sig_cd0 ** 2

    def initialize_process_noise(self):
        """ Initialize process noise matrix """
        self.nav.q = np.zeros(self.nav.nq)
        
        # Position and velocity process noise
        self.nav.q[0:3] = self.nav.sig_qp ** 2
        if self.nav.pmode >= 1:
            self.nav.q[3:6] = self.nav.sig_qv ** 2
            
        # Clock bias and drift process noise
        clk_bias_start = 6 if self.nav.pmode > 0 else 3
        clk_drift_start = clk_bias_start + self.nav.nsys
        self.nav.q[clk_bias_start:clk_drift_start] = self.nav.sig_qcb ** 2
        self.nav.q[clk_drift_start:clk_drift_start + self.nav.nsys] = self.nav.sig_qcd ** 2
    def initialize_position_least_squares(self, obs, rs, dts, maxiter = 5, tol=1e-4, cs = None):
        """ Least squares position initialization """
        if len(obs.sat) <= 4:
            return False, None, None, None, None

        # Extract pseudorange and SNR
        pseudorange = obs.P[:, 0] if obs.P.ndim == 2 else obs.P
        snr = obs.S[:, 0] if obs.S.ndim == 2 else obs.S

        # Filter valid satellites
        valid_sats = [i for i, pr in enumerate(pseudorange) if pr > 0 and snr[i] > 0]
        if len(valid_sats) <= 4:
            return False, None, None, None, None

        valid_sats = np.array(valid_sats)
        valid_rs = rs[valid_sats]
        valid_dts = dts[valid_sats]
        valid_P = pseudorange[valid_sats]
        SNR = snr[valid_sats]
        valid_sats = obs.sat[valid_sats]
        
        # System mapping
        sys_list = list(obs.sig.keys())
        num_systems = len(sys_list)
        sys_map = {sys: i for i, sys in enumerate(sys_list)}

        # Initialize state
        rr = np.zeros(3)
        dtr = np.zeros(num_systems)
        el = np.zeros(len(valid_P))
        az = np.zeros(len(valid_P))
        e = np.zeros((len(valid_P), 3))

        _c = rCST.CLIGHT
        dp = np.ones(3 + num_systems) * 100
        iter_count = 0
        
        # Initial iteration
        while np.linalg.norm(dp[:3]) > tol and iter_count < maxiter:
            H = []
            v = []
            azimuth_angles = []
            elevation_angles = []

            for i, sat in enumerate(valid_sats):
                sat = valid_sats[i]
                sys, _ = sat2prn(sat)
                sys_idx = sys_map[sys]

                r, e[i, :] = geodist(valid_rs[i], rr)
                corrected_dtr = dtr[sys_idx]
                res = valid_P[i] - (r + corrected_dtr - _c * valid_dts[i])
                v.append(res)

                H_row = [-e[i, 0], -e[i, 1], -e[i, 2]]
                for j in range(num_systems):
                    H_row.append(1.0 if j == sys_idx else 0.0)
                H.append(H_row)

            H = np.array(H)
            v = np.array(v)
            dp = np.linalg.lstsq(H, v, rcond=None)[0]

            rr += dp[:3]
            dtr += dp[3:]
            iter_count += 1

        # Second iteration with troposphere and ionosphere
        dp = np.ones(3 + num_systems) * 100
        iter_count = 0
        
        while np.linalg.norm(dp[:3]) > tol and iter_count < maxiter:
            pos = ecef2pos(rr)

            H = []
            v = []
            azimuth_angles = []
            elevation_angles = []

            for i, sat in enumerate(valid_sats):
                sat = valid_sats[i]
                sys, _ = sat2prn(sat)
                sys_idx = sys_map[sys]
                
                r, e[i, :] = geodist(valid_rs[i], rr)
                az[i], el[i] = satazel(pos, e[i])
                azimuth_angles.append(az[i])
                elevation_angles.append(el[i])

                if self.nav.trop_opt == 0:
                    try:
                        trop_hs, trop_wet, _ = tropmodel(obs.t, pos, model=self.nav.trpModel)
                        mapfh, mapfw = tropmapf(obs.t, pos, el[i], model=self.nav.trpModel)
                    except Exception as e:
                        trop_hs, trop_wet = 0.0, 0.0
                        mapfh, mapfw = 0.0, 0.0
                    trop = mapfh * trop_hs + mapfw * trop_wet
                else:
                    trop = 0.0

                if self.nav.iono_opt == 0:
                    try:
                        iono = ionmodel(obs.t, pos, az[i], el[i], self.nav, model=self.ionoModel, cs=cs)
                    except Exception as e:
                        iono = 0.0
                else:
                    iono = 0.0

                corrected_dtr = dtr[sys_idx]
                res = valid_P[i] - (r + corrected_dtr - _c * valid_dts[i] + trop + iono)
                v.append(res)

                H_row = [-e[i, 0], -e[i, 1], -e[i, 2]]
                for j in range(num_systems):
                    H_row.append(1.0 if j == sys_idx else 0.0)
                H.append(H_row)

            H = np.array(H)
            v = np.array(v)
            dp = np.linalg.lstsq(H, v, rcond=None)[0]

            rr += dp[:3]
            dtr += dp[3:]
            iter_count += 1

        # Check convergence
        if np.linalg.norm(dp[:3]) > tol:
            return False, None, None, None, None
        
        if np.sqrt((v * v).sum()) > 1000:
            return False, None, None, None, None

        # Return final state
        pos = np.zeros(6 + num_systems)
        pos[:3] = rr
        pos[6:6 + num_systems] = dtr

        return True, pos, np.array(azimuth_angles), np.array(elevation_angles), v

    def qcedit(self,obs,rs,vs,dts,svh, rr=None):
        """ Quality control and editing of observations """
        tt = timediff(obs.t, self.nav.t)
        if rr is None:
            rr_ = self.nav.x[0:3].copy()
            if self.nav.pmode > 0:
                rr_ += self.nav.x[3:6]*tt
        else:
            rr_ = rr

        pos = ecef2pos(rr_)
        ns = len(obs.sat)
        valid_sat = []
        valid_indices = []

        for i in range(ns):
            sat_i = obs.sat[i]
            sys_i, _ = sat2prn(sat_i)
            sat_valid = True

            if sat_i in self.nav.excl_sat:
                if self.nav.monlevel > 0:
                    self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit - satellite excluded\n")
                sat_valid = False

            if np.isnan(rs[i, :]).any() or np.isnan(dts[i]):
                if self.nav.monlevel > 0:
                    self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit - invalid eph\n")
                sat_valid = False

            if svh[i] > 0:
                if self.nav.monlevel > 0:
                    self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit - satellite unhealthy\n")
                sat_valid = False

            if not sat_valid:
                continue

            sigsPR = obs.sig[sys_i][uTYP.C]
            sigsCP = obs.sig[sys_i][uTYP.L]
            sigsCN = obs.sig[sys_i][uTYP.S]

            for f in range(self.nav.nf):
                if obs.P[i, f] == 0.0:
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit {sigsPR[f].str()} - invalid PR obs\n")
                    sat_valid = False

                cnr_min = self.nav.cnr_min_gpy if sigsCN[f].isGPS_PY() else self.nav.cnr_min
                if obs.S[i, f] < cnr_min:
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            f"{time2str(obs.t)}  {sat2id(sat_i)} - edit {sigsCN[f].str()} - low C/N0 {obs.S[i, f]:4.1f} dB-Hz\n")
                    sat_valid = False

            if not sat_valid:
                continue

            valid_sat.append(sat_i)

        valid_indices = [i for i, sat in enumerate(obs.sat) if sat in valid_sat]
        obs.P = obs.P[valid_indices, :]
        obs.L = obs.L[valid_indices, :]
        obs.S = obs.S[valid_indices, :]
        obs.D = obs.D[valid_indices, :]
        obs.lli = obs.lli[valid_indices, :]
        obs.sat = obs.sat[valid_indices]
        rs = rs[valid_indices, :]
        vs = vs[valid_indices, :]
        dts = dts[valid_indices]
        svh = svh[valid_indices]

        return np.array(valid_sat, dtype=int), obs, rs, vs, dts, svh

    def coarse_quality_check(self, obs, rs, vs, dts, svh):
        """ Coarse quality check for satellites """
        ns = len(obs.sat)
        valid_sat = []
        valid_indices = []

        for i in range(ns):
            sat_i = obs.sat[i]
            sys_i, _ = sat2prn(sat_i)
            sat_valid = True

            if sat_i in self.nav.excl_sat:
                if self.nav.monlevel > 0:
                    self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit - satellite excluded\n")
                sat_valid = False

            if np.isnan(rs[i, :]).any() or np.isnan(dts[i]):
                if self.nav.monlevel > 0:
                    self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit - invalid eph\n")
                sat_valid = False

            if svh[i] > 0:
                if self.nav.monlevel > 0:
                    self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit - satellite unhealthy\n")
                sat_valid = False

            if not sat_valid:
                continue

            sigsPR = obs.sig[sys_i][uTYP.C]
            sigsCP = obs.sig[sys_i][uTYP.L]
            sigsCN = obs.sig[sys_i][uTYP.S]

            for f in range(self.nav.nf):
                if obs.P[i, f] == 0.0:
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(f"{time2str(obs.t)}  {sat2id(sat_i)} - edit {sigsPR[f].str()} - invalid PR obs\n")
                    sat_valid = False

                cnr_min = self.nav.cnr_min_gpy if sigsCN[f].isGPS_PY() else self.nav.cnr_min
                if obs.S[i, f] < cnr_min:
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            f"{time2str(obs.t)}  {sat2id(sat_i)} - edit {sigsCN[f].str()} - low C/N0 {obs.S[i, f]:4.1f} dB-Hz\n")
                    sat_valid = False

            if not sat_valid:
                continue

            valid_sat.append(sat_i)

        valid_indices = [i for i, sat in enumerate(obs.sat) if sat in valid_sat]
        obs.P = obs.P[valid_indices, :]
        obs.L = obs.L[valid_indices, :]
        obs.S = obs.S[valid_indices, :]
        obs.D = obs.D[valid_indices, :]
        obs.lli = obs.lli[valid_indices, :]
        obs.sat = obs.sat[valid_indices]
        rs = rs[valid_indices, :]
        vs = vs[valid_indices, :]
        dts = dts[valid_indices]
        svh = svh[valid_indices]

        return np.array(valid_sat, dtype=int), obs, rs, vs, dts, svh

    def coarse_pos(self, obs, cs=None, orb=None, bsx=None, obsb=None, mode="initialize"):
        """
        Coarse positioning for feature extraction
        """
        if mode not in {"initialize", "extraction"}:
            raise ValueError(f"Invalid mode: {mode}")

        if len(obs.sat) == 0:
            return {"status": False, "pos": None, "msg": "No satellites in observation", "data": {}}

        rs, vs, dts, svh, nsat = satposs_enhanced(obs, self.nav, cs=cs, orb=orb)
        sat_ed, obs, rs, vs, dts, svh = self.coarse_quality_check(obs, rs, vs, dts, svh)

        if nsat < 4:
            return {"status": False, "pos": None, "msg": "Too few satellites (< 4)", "data": {}}

        success, pos, az, el, v = self.initialize_position_least_squares(obs, rs, dts)

        if not success:
            return {"status": False, "pos": None, "msg": "Coarse positioning failed", "data": {}}

        if len(obs.sat) < 4:
            return {"status": False, "pos": None, "msg": "Too few satellites (< 4)", "data": {}}
        
        if mode == "initialize":
            self.nav.t = obs.t
            self.nav.x[0:3] = pos[0:3]
            self.nav.x[self.ICB(0):self.ICB(0)+self.nav.nsys] = pos[self.ICB(0):self.ICB(0)+self.nav.nsys]
            return sat_ed, obs, rs, vs, dts, svh, nsat, pos

        elif mode == "extraction":
            self.nav.x[0:3] = pos[0:3]
            self.nav.x[self.ICB(0):self.ICB(0)+self.nav.nsys] = pos[self.ICB(0):self.ICB(0)+self.nav.nsys]
            return {
                "status": True,
                "pos": pos[:3],
                "msg": "success",
                "data": {
                    "residual": v.tolist(),
                    "azel": np.column_stack((az, el)).tolist(),
                    "dts": dts.tolist(),
                    "sats": obs.sat.tolist(),
                    "SNR": obs.S.tolist()
                }
            }
    def csmooth(self, obs: Obs, sat, Pm, Lm, ns=100, dt_th=1, cs_th=10):
        """ Hatch filter for carrier smoothing """
        sys, _ = sat2prn(sat)

        if Pm == 0.0 or Lm == 0.0:
            self.cs_cnt[sat] = 1
            return Pm

        if sat not in self.cs_cnt or timediff(obs.t, self.cs_t0[sat]) > dt_th:
            self.cs_cnt[sat] = 1

        if self.cs_cnt[sat] == 1:
            self.Ps_[sat] = Pm
        else:
            Pp = self.Ps_[sat] + (Lm - self.Lp_[sat])
            if abs(Pm-Pp) < cs_th:
                alp = 1/self.cs_cnt[sat]
                self.Ps_[sat] = alp*Pm + (1-alp)*Pp
            else:
                if self.monlevel > 0:
                    print("cycle slip detected, cs reset.")
                self.cs_cnt[sat] = 1
                self.Ps_[sat] = Pm
        self.cs_cnt[sat] = min(self.cs_cnt[sat]+1, ns)
        self.Lp_[sat] = Lm
        self.cs_t0[sat] = obs.t
        return self.Ps_[sat]

    def varerr(self, nav, el, f):
        """ Measurement variation """
        s_el = max(np.sin(el), 0.1*rCST.D2R)
        fact = nav.eratio[f]
        a = fact*nav.err[1]
        b = fact*nav.err[2]
        v_sig = a**2+(b/s_el)**2
        return v_sig
    def varerr_torch(self, nav, f, weight=None):
        """ Measurement variation with learned weights """
        device = weight.device if weight is not None else torch.device('cpu')
        fact = nav.eratio[f]
        a = fact * nav.err[1]
        b = fact * nav.err[2]
        if weight is not None:
            b = b / torch.sqrt(weight)
        base_variance = a**2 + b**2
        v_sig = base_variance.clone().detach().to(device).requires_grad_(True)
        return v_sig
    
    def udstate(self, obs):
        """ Time propagation of states """
        tt = timediff(obs.t, self.nav.t)
        
        if tt == 0.0:
            return 0 

        nx = self.nav.nx
        Phi = np.eye(nx)

        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6] * tt
            Phi[0:3, 3:6] = np.eye(3) * tt

        sys_list = list(obs.sig.keys())
        num_systems = len(sys_list)
        sys_map = {sys: i for i, sys in enumerate(sys_list)}

        clk_bias_start = self.ICB(0)
        clk_drift_start = clk_bias_start + num_systems

        for sys, i in sys_map.items():
            clk_bias_idx = clk_bias_start + i
            clk_drift_idx = clk_drift_start + i
            self.nav.x[clk_bias_idx] += self.nav.x[clk_drift_idx] * tt
            Phi[clk_bias_idx, clk_drift_idx] = tt

        self.nav.P[0:nx, 0:nx] = Phi @ self.nav.P[0:nx, 0:nx] @ Phi.T
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        dP[0:6] += self.nav.q[0:6] * tt

        q_clk_bias = self.nav.q[6]
        q_clk_drift = self.nav.q[7]

        for i in range(num_systems):
            clk_bias_idx = clk_bias_start + i
            clk_drift_idx = clk_drift_start + i
            dP[clk_bias_idx] += q_clk_bias * tt
            dP[clk_drift_idx] += q_clk_drift * tt

        return 0

    def udstate_torch(self, obs):
        """ Time propagation of states (Torch version) """
        tt = timediff(obs.t, self.nav.t)
        sys = []
        for sat_i in obs.sat:
            sys_i, _ = sat2prn(sat_i)
            sys.append(sys_i)

        nx = self.nav.nx
        Phi = torch.eye(nx, device=self.nav.P.device, dtype=self.nav.P.dtype)

        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6] * tt
            self.nav.x[6] += self.nav.x[7] * tt
            Phi[0:3, 3:6] = torch.eye(3, device=self.nav.P.device, dtype=self.nav.P.dtype) * tt
            Phi[6, 7] = tt

        self.nav.P[0:nx, 0:nx] = Phi @ self.nav.P[0:nx, 0:nx] @ Phi.T
        return 0
    
    def zdres(self, obs, cs, bsx, rs, vs, dts, x, rtype=1):
        """ Non-differential residual """
        _c = rCST.CLIGHT
        nf = self.nav.nf
        n = len(obs.sat)
        rr = x[0:3]
        sys_list = list(obs.sig.keys())
        sys_map = {sys: i for i, sys in enumerate(sys_list)}
        y = np.zeros((n, nf))
        el = np.zeros(n)
        az = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = rr.copy()
        pos = ecef2pos(rr_)

        for i in range(n):
            sat = obs.sat[i]
            sys, _ = sat2prn(sat)
            r, e[i, :] = geodist(rs[i, :], rr_)
            az[i], el[i] = satazel(pos, e[i, :])

            if self.nav.trop_opt == 0:
                try:
                    trop_hs, trop_wet, _ = tropmodel(obs.t, pos, model=self.nav.trpModel)
                    mapfh, mapfw = tropmapf(obs.t, pos, el[i], model=self.nav.trpModel)
                except Exception as e:
                    trop_hs, trop_wet = 0.0, 0.0
                    mapfh, mapfw = 0.0, 0.0
                trop = mapfh * trop_hs + mapfw * trop_wet
            else:
                trop = 0.0

            if self.nav.iono_opt == 0:
                try:
                    iono = ionmodel(obs.t, pos, az[i], el[i], self.nav, model=self.ionoModel, cs=cs)
                except Exception as e:
                    iono = 0.0
            else:
                iono = 0.0

            sys_idx = sys_map[sys]
            dtr = x[self.ICB(0) + sys_idx]
            r += dtr - _c * dts[i]

            sigsCP = obs.sig[sys][uTYP.L]
            if sys == uGNSS.GLO:
                lam = np.array([s.wavelength(self.nav.glo_ch[sat]) for s in sigsCP])
            else:
                lam = np.array([s.wavelength() for s in sigsCP])

            if self.nav.rmode == 0:
                PR = obs.P[i, 0]
                CP = lam[0] * obs.L[i, 0]
            else:
                iono = 0
                if self.nav.rmode == 1:
                    gam = (rCST.FREQ_G1 / rCST.FREQ_G2) ** 2
                if self.nav.rmode == 2:
                    gam = (rCST.FREQ_S1 / rCST.FREQ_S5) ** 2
                PR = (obs.P[i, 1] - gam * obs.P[i, 0]) / (1 - gam)
                CP = (lam[1] * obs.L[i, 1] - gam * lam[0] * obs.L[i, 0]) / (1 - gam)

            if self.nav.csmooth:
                PR = self.csmooth(obs, sat, PR, CP)

            y[i, 0] = PR - (r + trop + iono)

        return y, e, az, el

    def sdres(self, obs, x, y, e, sat, el):
        """
        SD phase/code residuals for multi-GNSS
        """
        nf = self.nav.nf if self.nav.rmode == 0 else 1
        ns = len(el)
        nc = len(obs.sig.keys())
        nb = np.zeros(nc * nf, dtype=int)
        Rj = np.zeros(ns * nf)
        nv = 0
        b = 0
        H = np.zeros((ns * nf, self.nav.nx))
        v = np.zeros(ns * nf)
        sys_list = list(obs.sig.keys())
        sys_map = {sys: i for i, sys in enumerate(sys_list)}

        for i in range(ns):
            sat_id = sat[i]
            sys, _ = sat2prn(sat_id)
            sys_idx = sys_map[sys]
            v[nv] = y[i]
            H[nv, 0:3] = -e[i, :]
            H[nv, self.ICB(0) + sys_idx] = 1.0
            Rj[i] = self.varerr(self.nav, el[i], 0)
            nb[b] += 1
            nv += 1

        b += 1
        v = np.resize(v, nv)
        H = np.resize(H, (nv, self.nav.nx))
        R = self.ddcov(nb, b, Rj, nv)

        return v, H, R

    def zdres_torch(self, obs, cs, bsx, rs, vs, dts, x, rtype=1, bias=None):
        """Non-differential residual + learning"""
        _c = rCST.CLIGHT
        nf = self.nav.nf
        n = len(obs.P)
        rr = x[0:3]
        dtr = x[self.ICB()]
        y = np.zeros((n, nf))
        el = np.zeros(n)
        az = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = rr.copy()
        sys_list = list(obs.sig.keys())
        sys_map = {sys: i for i, sys in enumerate(sys_list)}
        pos = ecef2pos(rr_)
        y = torch.zeros((n, nf), dtype=torch.double, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(n):
            sat = obs.sat[i]
            sys, _ = sat2prn(sat)
            r, e[i, :] = geodist(rs[i, :], rr_)
            az[i], el[i] = satazel(pos, e[i, :])

            if self.nav.trop_opt == 0:
                try:
                    trop_hs, trop_wet, _ = tropmodel(obs.t, pos, model=self.nav.trpModel)
                    mapfh, mapfw = tropmapf(obs.t, pos, el[i], model=self.nav.trpModel)
                except Exception as e:
                    trop_hs, trop_wet = 0.0, 0.0
                    mapfh, mapfw = 0.0, 0.0
                trop = mapfh * trop_hs + mapfw * trop_wet
            else:
                trop = 0.0

            if self.nav.iono_opt == 0:
                try:
                    iono = ionmodel(obs.t, pos, az[i], el[i], self.nav, model=self.ionoModel, cs=cs)
                except Exception as e:
                    iono = 0.0
            else:
                iono = 0.0

            sys_idx = sys_map[sys]
            dtr = x[self.ICB(0) + sys_idx]
            r += dtr - _c * dts[i]

            sigsCP = obs.sig[sys][uTYP.L]
            if sys == uGNSS.GLO:
                lam = np.array([s.wavelength(self.nav.glo_ch[sat]) for s in sigsCP])
            else:
                lam = np.array([s.wavelength() for s in sigsCP])

            if self.nav.rmode == 0:
                PR = obs.P[i, 0]
                CP = lam[0] * obs.L[i, 0]
            else:
                iono = 0
                if self.nav.rmode == 1:
                    gam = (rCST.FREQ_G1 / rCST.FREQ_G2) ** 2
                if self.nav.rmode == 2:
                    gam = (rCST.FREQ_S1 / rCST.FREQ_S5) ** 2
                PR = (obs.P[i, 1] - gam * obs.P[i, 0]) / (1 - gam)
                CP = (lam[1] * obs.L[i, 1] - gam * lam[0] * obs.L[i, 0]) / (1 - gam)

            if self.nav.csmooth:
                PR = self.csmooth(obs, sat, PR, CP)

            correction = bias[i] if bias is not None else torch.tensor(0.0, dtype=torch.double, device=PR.device)
            PR = torch.tensor(PR, dtype=torch.double, device=correction.device) if not isinstance(PR, torch.Tensor) else PR
            trop = torch.tensor(trop, dtype=torch.double, device=correction.device) if not isinstance(trop, torch.Tensor) else trop
            iono = torch.tensor(iono, dtype=torch.double, device=correction.device) if not isinstance(iono, torch.Tensor) else iono
            r = torch.tensor(r, dtype=torch.double, device=correction.device) if not isinstance(r, torch.Tensor) else r
            y[i, 0] = (PR + correction) - (r + trop + iono)

        return y, e, az, el

    def sdres_torch(self, obs, x, y, e, sat, R_diag):
        """
        SD phase/code residuals with learned R diagonal
        """
        nf = self.nav.nf if self.nav.rmode == 0 else 1
        ns = len(obs.P)
        nc = len(obs.sig.keys())
        device = y.device
        nb = torch.zeros(nc * nf, dtype=torch.int, device=device)
        H = torch.zeros((ns * nf, self.nav.nx), dtype=torch.double, device=device)
        v = torch.zeros(ns * nf, dtype=torch.double, device=device)
        if isinstance(e, np.ndarray):
            e = torch.tensor(e, dtype=torch.double, device=device)

        sys_list = list(obs.sig.keys())
        sys_map = {sys: i for i, sys in enumerate(sys_list)}
        nv = 0
        b = 0

        for i in range(ns):
            sat_id = sat[i]
            sys, _ = sat2prn(sat_id)
            sys_idx = sys_map[sys]
            v[nv] = y[i]
            H[nv, 0:3] = -e[i, :]
            H[nv, self.ICB(0) + sys_idx] = 1.0
            nb[b] += 1
            nv += 1

        b += 1
        v = v[:nv]
        H = H[:nv, :]
        R = torch.diag(R_diag[:nv])

        return v, H, R
    def ddcov(self, nb, n, Rj, nv):
        """ DD measurement error covariance """
        R = np.zeros((nv, nv))
        k = 0
        for b in range(n):
            for j in range(nb[b]):
                R[k+j, k+j] = Rj[k+j]
            k += nb[b]
        return R

    def ddcov_torch(self, nb, n, Rj, nv):
        """ DD measurement error covariance with gradient flow """
        device = Rj.device
        R = torch.zeros((nv, nv), dtype=torch.double, device=device)
        k = 0
        for b in range(n):
            for j in range(nb[b]):
                R[k + j, k + j] = Rj[k + j]
            k += nb[b]
        return R

    def kfupdate(self, x, P, H, v, R):
        """
        Kalman filter measurement update
        """
        PHt = P @ H.T
        S = H @ PHt + R
        K = PHt @ np.linalg.inv(S)
        x += K @ v
        IKH = np.eye(P.shape[0]) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T  
        v_new = v - H @ (K @ v)
        return x, P, S, v_new
    def kfupdate_torch(self, x, P, H, v, R):
        """
        Kalman filter measurement update using PyTorch
        """
        device = x.device
        H = H.to(device)
        v = v.to(device)
        R = R.to(device)
        P = P.to(device)
        PHt = P @ H.T
        S = H @ PHt + R
        S_inv = torch.linalg.inv(S)
        K = PHt @ S_inv
        x = x + K @ v
        IKH = torch.eye(P.shape[0], device=device) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T
        return x, P, S
    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None, Net_R=None,bias=None):
        """
        Standalone positioning with learned bias and R
        """
        if  self.nav.t.time == 0:
            self.nav.t = obs.t

        if np.all(self.nav.x == 0):
            sat_ed, obs, rs, vs, dts, svh, nsat, pos = self.coarse_pos(obs, mode="initialize")
        else:
            rs, vs, dts, svh, nsat = satposs_enhanced(obs, self.nav, cs=cs, orb=orb)
            sat_ed,obs,rs,vs,dts,svh = self.qcedit(obs,rs,vs,dts,svh)

        if len(obs.sat) < 4:
            return {"status": False, "pos": None, "msg": "Too few satellites (< 4)", "data": {}}

        if obsb is None:
            ns = len(obs.sat)
            y = np.zeros((ns, self.nav.nf))
            e = np.zeros((ns, 3))
            obs_ = obs

        self.udstate(obs_)
        xp = self.nav.x.copy()
        Pp = self.nav.P.copy()
        y, e, az, el = self.zdres_torch(obs, cs, bsx, rs, vs, dts, xp, bias = bias)
        sat = obs.sat
        SNR = obs.S
        self.nav.sat = sat
        self.nav.el[sat-1] = el
        self.nav.y = y
        ns = len(sat)
        ny = y.shape[0]
        if ny < 4:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 1
            return {"status": False, "pos": None, "msg": "Not enough satellites after editing", "data": {}}

        v, H, R = self.sdres_torch(obs, xp, y, e, sat, R_diag = Net_R)
        device = v.device
        xp = torch.tensor(xp, dtype=torch.double, device=device, requires_grad=True)
        Pp = torch.tensor(Pp, dtype=torch.double, device=device, requires_grad=True)
        xp, Pp, _ = self.kfupdate_torch(xp, Pp, H, v, R)
        self.nav.x = xp.detach().cpu().numpy()
        self.nav.P = Pp.detach().cpu().numpy()
        self.nav.smode = 1 if cs is None else 2
        self.nav.t = obs.t
        self.dop = dops(az, el)
        self.nsat = len(el)

        return {
            "status": True,
            "pos": xp,
            "msg": "success"
        }