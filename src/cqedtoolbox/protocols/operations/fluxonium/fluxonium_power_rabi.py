import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from qcui_analysis.fitfuncs.resonators import HangerResponseBruno
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.record import record_as, independent, dependent

from qcui_measurement.protocols.base import ProtocolOperation, OperationStatus
from qcui_measurement.protocols.operations.res_spec_vs_flux import _fluxonium_basis, _readout_frequencies
from qcui_measurement.protocols.parameters import (
    Repetition, StartFlux, EndFlux, FluxSteps, NumGainSteps, GainPulseDuration,
    QubitGain, QubitFrequency, ReadoutFrequency, ECParam, ELParam, EJParam, ZeroFluxCurrent
)

logger = logging.getLogger(__name__)
Q_E = 1.602176634e-19
HBAR = 1.054571817e-34
K_B = 1.380649e-23
H = 6.62607015e-34


def _rabi_coefficient(EC, EL, EJ, flux_ext, eta=1, grid=2000, n_levels=8, flux_max=12*np.pi):
    """Compute |n01| and k = (2e*eta/ħ) * |n01|  so that  θ = k * V0 * t  (radians)."""

    E, nQ = _fluxonium_basis(EC, EL, EJ, flux_ext, n_levels, grid, flux_max)
    n01 = nQ[0, 1]
    k = (2 * Q_E * eta / HBAR) * np.abs(n01)
    return E, k


def _solve_for_amplitude(theta_rad, t_nsec, EC, EL, EJ, flux_ext):
    """
    V = θ / (k * t). Solve theoretical rabi power amplitude (V) for a needed gain (rad) and pulse duration (ns).
    The maximum gain used later is 3pi, since we usually sweep gain over 0 to 3pi rad for best accuracy.
    """

    E, k_rad_per_Vs = _rabi_coefficient(EC, EL, EJ, flux_ext)
    V_eff = theta_rad / (k_rad_per_Vs * t_nsec * 1e-9)
    return V_eff


def _pe_rabi_after_pulse(EC, EL, EJ, flux_ext, f_drive_GHz, V_qubit_volt, t_nsec, T_kelvin):
    """Return total excited-state probability (gain) after a rabi pulse, including thermal population at temperature T."""

    E, k_rad_per_Vs = _rabi_coefficient(EC, EL, EJ, flux_ext)
    E_ge = H * (E[1]-E[0]) * 1e9
    beta = E_ge / (K_B * T_kelvin)
    p_e_th = 1.0 / (1.0 + np.exp(beta))
    p_g_th = 1.0 - p_e_th

    omega_R_ns = k_rad_per_Vs * V_qubit_volt * 1e-9
    delta = 2*np.pi * (f_drive_GHz - E[1] + E[0])
    omega_eff = np.sqrt(omega_R_ns**2 + delta**2)
    P_e_given_g = (omega_R_ns**2) / (omega_eff**2) * np.sin(0.5 * omega_eff * t_nsec)**2
    P_e_after = p_g_th * P_e_given_g + p_e_th
    return P_e_after


def _find_state_centers(sig_2d):
    """Find |0> and |1> centers using clustering."""

    sig_flat = sig_2d.reshape(-1)
    X = np.column_stack([sig_flat.real, sig_flat.imag])
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    z0 = centers[0, 0] + 1j * centers[0, 1]
    z1 = centers[1, 0] + 1j * centers[1, 1]
    return z0, z1


def _project_to_prob(signals_1d, z0, z1):
    """Using this axis gives better SNR for isotropic noise (most common for fluxonium) than fitting with real, imaginary or magnitude axis."""

    signals_1d = np.asarray(signals_1d)
    axis = z1 - z0
    t = np.real((signals_1d - z0) * np.conj(axis)) / (np.abs(axis) ** 2)
    return np.clip(t, 0, 1)


def _fit_sin2_and_quality(voltage_1d, p):
    """
    Fit the probability projection p(V) = B + A * sin^2(k V + phi), then return V_pi = pi / (2k).
    Assumptions:
      - The scan span contains at most ~4 peaks of p(V), best for 2 to 3 peaks.
    Returns V_pi in same units as voltage_1d.
    """

    v = np.asarray(voltage_1d, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(v) & np.isfinite(p)
    v = v[m]
    p = p[m]
    order = np.argsort(v)
    v = v[order]
    p = p[order]
    span = float(v[-1] - v[0])

    def sin2(V, A, k, phi, B):
        return B + A * (np.sin(k * V + phi) ** 2)

    pmin = float(np.min(p))
    pmax = float(np.max(p))
    A0 = float(np.clip(pmax - pmin, 1e-3, 1.0))
    B0 = float(np.clip(np.median(p), 0.0, 1.0))
    n = v.size
    w = max(5, n // 25)
    if w % 2 == 0:
        w += 1
    ps = np.convolve(p, np.ones(w) / w, mode="same")
    prom = 0.05 * float(np.ptp(ps)) if np.ptp(ps) > 0 else 0.0
    peaks, _ = find_peaks(ps, prominence=prom)

    if peaks.size >= 2:
        peaks = np.sort(peaks)
        vp = v[peaks[:4]]
        dV = float(np.median(np.diff(vp))) if vp.size >= 3 else float(vp[1] - vp[0])
        k_peak = (np.pi / dV) if (np.isfinite(dV) and dV > 0) else None
    else:
        k_peak = None
    k_max = (4.2 * np.pi) / span
    k_min = (0.4 * np.pi) / span
    k_base = k_peak if (k_peak is not None and np.isfinite(k_peak)) else (2.0 * np.pi / span)
    k_base = float(np.clip(k_base, k_min, k_max))

    bounds_lo = (0.0, k_min, -np.pi, 0.0)
    bounds_hi = (1.0, k_max,  np.pi, 1.0)
    phi_guesses = (0.0, np.pi / 4, np.pi / 2, -np.pi / 4)
    k_scales = (1.0, 0.8, 1.2, 0.6, 1.4)

    best = None
    for ph0 in phi_guesses:
        for s in k_scales:
            k0 = float(np.clip(k_base * s, k_min, k_max))
            p0 = (A0, k0, ph0, B0)
            try:
                popt, _ = curve_fit(sin2, v, p, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=40000)
                resid = p - sin2(v, *popt)
                sse = float(np.sum(resid * resid))
                if (best is None) or (sse < best[0]):
                    best = (sse, popt)
            except Exception:
                continue

    A_fit, k_fit, phi_fit, B_fit = best[1]
    v_pi = np.pi / (2.0 * k_fit)
    v_pi = float(np.clip(v_pi, span / 100.0, 2.0 * span))
    fit_dict = {
        "A": float(A_fit),
        "k": float(k_fit),
        "phi": float(phi_fit),
        "B": float(B_fit),
    }
    return v_pi, fit_dict


class FluxoniumPowerRabi(ProtocolOperation):

    SNR_THRESHOLD = 2.0
    # True fake data params (may change to any)
    _SIM_Q_I = 8000.0
    _SIM_Q_E_MAG = 5000.0
    _SIM_THETA = 0.5
    _SIM_PHASE_OFF = 0.1
    _SIM_PHASE_SLOPE = 2e-9  # rad/Hz
    _SIM_TX_SLOPE = 5e-9  # /Hz
    _SIM_A = 1.0
    _SIM_EC = 1.0  # followings are params for fluxonium
    _SIM_EL = 0.5
    _SIM_EJ = 5.3
    _SIM_FR = 4.07
    _SIM_G = 0.067
    _SIM_KELVIN = 0.05  # qubit temperature for fake data
    _SIM_NOISE_SIGMA = 0.02
    _SIM_EARTH_FLUX = 0.0

    def __init__(self, params):
        super().__init__()
    
        self._register_inputs(
            repetitions=Repetition(params),
            start_flux=StartFlux(params),
            end_flux=EndFlux(params),
            flux_steps=FluxSteps(params),
            f0=ReadoutFrequency(params),
            rabi_duration=GainPulseDuration(params),
            rabi_steps=NumGainSteps(params),
            f_rabi=QubitFrequency(params),
            EC=ECParam(params),
            EL=ELParam(params),
            EJ=EJParam(params),
            Earth_flux=ZeroFluxCurrent(params)
        )
        self._register_outputs(
            pi_power_vs_flux=QubitGain(params)
        )

        self.condition = f"Success if every trace has SNR ≥ {self.SNR_THRESHOLD}"
        self.independents = {"voltages": [], "flux": []}
        self.dependents = {"signal": []}
        self.data_loc = None
        self.fr_g = []
        self.fr_e = []
        self.pg = []
        self.snr = []
        self.fit_results = []


    def _measure_dummy(self) -> Path:
        """
        Generate fake double-hanger res_spec-vs-flux data.
        Uses theoretical fr_g(flux) and fr_e(flux) to build a synthetic
        3D S21[reps, flux, freq] map, with additive complex Gaussian noise.
        The interface matches _measure_quick: same inputs, same ddh5 layout, and measure the faked qubit with given scanned ranges.
        """

        logger.info("Starting dummy resonator spectroscopy vs flux (simulated fake data)")
        n_rep = self.repetitions()
        n_flux = self.flux_steps()
        n_Volts = self.rabi_steps()
        freq = self.f0()
        f01 = self.f_rabi()
        start_flux = self.start_flux()
        end_flux = self.end_flux()
        flux_vals = np.linspace(start_flux, end_flux, n_flux)
        Veff_vec = np.empty(n_flux, dtype=float)
        for i_flux in range(n_flux):
            Veff_vec[i_flux] = _solve_for_amplitude(3*np.pi, self.rabi_duration(), self.EC(), self.EL(),
                self.EJ(), flux_vals[i_flux] + self.Earth_flux())
        V_grid = (Veff_vec[:, None]) * np.linspace(0.0, 1.0, n_Volts)[None, :]
        fr_g_vs_flux=[]
        fr_e_vs_flux=[]
        for flux_ext in flux_vals:
            fr_g, fr_e = _readout_frequencies(self._SIM_EC, self._SIM_EL, self._SIM_EJ, flux_ext+self._SIM_EARTH_FLUX, self._SIM_FR, self._SIM_G, q_levels=10, N_phot=5)
            fr_g_vs_flux.append(fr_g)
            fr_e_vs_flux.append(fr_e)
        fr_g_vs_flux = np.asarray(fr_g_vs_flux)
        fr_e_vs_flux = np.asarray(fr_e_vs_flux)
        p_e_grid = np.empty((n_flux, n_Volts), dtype=float)
        for i_flux in range(n_flux):
            for i_v in range(n_Volts):
                p_e_grid[i_flux, i_v] = _pe_rabi_after_pulse(self._SIM_EC, self._SIM_EL, self._SIM_EJ, flux_vals[i_flux] + self._SIM_EARTH_FLUX, f01[i_flux],
                    float(V_grid[i_flux, i_v]), self.rabi_duration(), self._SIM_KELVIN)
        s21_g_vec = np.array([HangerResponseBruno.model(coordinates=freq[i], A=self._SIM_A, f_0=fr_g_vs_flux[i], Q_i=self._SIM_Q_I, Q_e_mag=self._SIM_Q_E_MAG,
            theta=self._SIM_THETA, phase_offset=self._SIM_PHASE_OFF, phase_slope=self._SIM_PHASE_SLOPE, transmission_slope=self._SIM_TX_SLOPE) for i in range(n_flux)], dtype=np.complex128)
        s21_e_vec = np.array([HangerResponseBruno.model(coordinates=freq[i], A=self._SIM_A, f_0=fr_e_vs_flux[i], Q_i=self._SIM_Q_I, Q_e_mag=self._SIM_Q_E_MAG,
            theta=self._SIM_THETA, phase_offset=self._SIM_PHASE_OFF, phase_slope=self._SIM_PHASE_SLOPE, transmission_slope=self._SIM_TX_SLOPE) for i in range(n_flux)], dtype=np.complex128)
        s21_g_grid = np.broadcast_to(s21_g_vec[:, None], (n_flux, n_Volts))
        s21_e_grid = np.broadcast_to(s21_e_vec[:, None], (n_flux, n_Volts))
        def _dummy_hanger_signal():
            x_grid = np.random.rand(n_flux, n_Volts)
            choose_e = x_grid < p_e_grid
            noise = self._SIM_NOISE_SIGMA * (np.random.randn(n_flux, n_Volts) + 1j*np.random.randn(n_flux, n_Volts)) / np.sqrt(2.0)
            signals = np.where(choose_e, s21_e_grid, s21_g_grid) + noise
            return signals
        
        flux_grid = np.broadcast_to(flux_vals[:, None], (n_flux, n_Volts))
        sweep = (
            sweep_parameter("rep", range(n_rep))
            @ record_as(lambda:flux_grid, independent("current"))
            @ record_as(lambda: V_grid, independent("rabi_power"))
            @ record_as(_dummy_hanger_signal, dependent("signal"))
        )
        loc, data = run_and_save_sweep(sweep, './data', 'my_data')
        logger.info(f"Dummy measurement complete, data saved to {loc}")
        return loc


    def _load_data_dummy(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)
        self.independents["voltages"] = np.asarray(data["rabi_power"]["values"])
        self.independents["flux"] = np.asarray(data["current"]["values"])
        self.dependents["signal"] = np.asarray(data["signal"]["values"])


    def analyze(self):
        """
        Analyze Rabi-amplitude sweep and extract π-pulse amplitude vs flux.
        Method:
        1) Average over repetitions to get one complex mean point per (flux, V).
        2) Estimate the two readout state centers (z0, z1) using clustering.
        3) Project mean IQ points onto the z0→z1 axis to obtain a proxy probability p(V) (This method gives the best snr for isotropic noise).
        4) Extract Vπ from fitting the probability feature p(V) with sine square.
        """
         
        signals = self.dependents["signal"]
        voltage = self.independents["voltages"][0,:,:]
        S = np.mean(signals, axis=0)
        n_flux, n_V = S.shape
        V_grid = voltage
        pi_voltages = np.full(n_flux, np.nan)
        snr_flux = np.full(n_flux, np.nan)
        fit_results = [None] * n_flux

        for flux_idx in range(n_flux):
            sig_1d = S[flux_idx]
            sig_2d = signals[:,flux_idx,:]
            z0, z1 = _find_state_centers(sig_2d)
            p = _project_to_prob(sig_1d, z0, z1)
            pi_voltages[flux_idx], fit_results[flux_idx] = _fit_sin2_and_quality(V_grid[flux_idx], p)
            
            axis = z1 - z0
            sig_flat = sig_2d.reshape(-1)
            X = np.column_stack([sig_flat.real, sig_flat.imag])
            km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(X)
            labels = km.labels_
            den = np.abs(axis)**2
            t = np.real((sig_flat - z0) * np.conj(axis)) / den
            t0, t1 = t[labels == 0], t[labels == 1]
            sigma = np.sqrt(0.5*(np.std(t0)**2 + np.std(t1)**2))
            snr_flux[flux_idx] = (1.0 / (4.0 * sigma)) if sigma > 0 else np.nan

        self.pi_power_vs_flux = pi_voltages
        self.snr = snr_flux
        self.fit_results = np.asarray(fit_results)

        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ds.add(
                pi_power_vs_flux=np.asarray(self.pi_power_vs_flux),
                snr=np.asarray(self.snr),
                snr_threshold=float(self.SNR_THRESHOLD),
                fit_results=self.fit_results,
            )
            fig, ax = plt.subplots()
            ax.plot(self.independents["flux"][0,:,0], self.pi_power_vs_flux, "o-")
            ax.set_xlabel("Flux (rad)")
            ax.set_ylabel("Pi pulse amplitude (V)")
            ax.set_title("Extracted π pulse amplitude vs flux")
            ds.add_figure(f"{self.name}_pi_vs_flux", fig=fig)
            image_path = ds._new_file_path(ds.savefolders[1], f"{self.name}_pi_vs_flux", suffix="png")
            self.figure_paths.append(image_path)


    def evaluate(self) -> OperationStatus:
        """
        Final evaluation: determine quality of extracted π-pulse amplitude vs flux.
        Criteria:
        1) Edge avoidance (and finite): Vπ not at scan edges.
        2) Readout quality: SNR at each flux point must be good often enough.
        3) Smoothness: Vπ(Φ) should be reasonably smooth on the good points.
        """

        if not len(self.pi_power_vs_flux):
            self.report_output = ["Empty π-pulse result."]
            return OperationStatus.FAILURE

        vmin = np.nanmin(self.independents["voltages"][0, :, :], axis=1)
        vmax = np.nanmax(self.independents["voltages"][0, :, :], axis=1)
        scan_span = vmax - vmin

        GOOD_FRAC_MIN = 0.80 # minimum fraction of flux points that must have good Vπ extraction (not edge/finite and good SNR) for overall success
        EDGE_FRAC = 0.05  # minimum distance from edges as fraction of scan span for a point to be considered edge-good
        JUMP_FRAC = 0.40
        JUMP_RATE_MAX = 0.20

        edge_bad = (
            (self.pi_power_vs_flux <= vmin + EDGE_FRAC * scan_span)
            | (self.pi_power_vs_flux >= vmax - EDGE_FRAC * scan_span)
            | ~np.isfinite(self.pi_power_vs_flux)
        )

        snr_bad = (
            ~np.isfinite(self.snr)
            | (self.snr < self.SNR_THRESHOLD)
        )

        good = (~edge_bad) & (~snr_bad)
        good_frac = float(np.mean(good))

        pi_g = self.pi_power_vs_flux[good]
        if pi_g.size >= 3:
            d = np.abs(np.diff(pi_g))
            scale = np.maximum(np.abs(pi_g[:-1]), np.abs(pi_g[1:]))
            jump_rate = np.mean(d > JUMP_FRAC * scale)
            smooth_pass = jump_rate <= JUMP_RATE_MAX
        else:
            jump_rate = np.nan
            smooth_pass = True
        
        finite_frac = float(np.mean(np.isfinite(self.pi_power_vs_flux)))
        edge_good_frac = float(np.mean(~edge_bad))
        snr_good_frac = float(np.mean(~snr_bad))
        snr_med = float(np.nanmedian(self.snr)) if np.any(np.isfinite(self.snr)) else np.nan

        self.report_output = [(
            "## Rabi π-pulse extraction\n"
            f"Flux points: {len(self.pi_power_vs_flux)}\n"
            f"Vπ finite fraction: {finite_frac:.2%}\n\n"
            f"Edge-good fraction (±{EDGE_FRAC:.0%}): {edge_good_frac:.2%}\n"
            f"SNR-good fraction (>= {self.SNR_THRESHOLD}): {snr_good_frac:.2%}\n"
            f"Good fraction (edge-good & SNR-good): {good_frac:.2%}\n"
            f"Median SNR: {snr_med if np.isfinite(snr_med) else 'n/a'}\n"
            f"Jump rate (good points): {jump_rate if np.isfinite(jump_rate) else 'n/a'}\n"
        )]

        if (good_frac >= GOOD_FRAC_MIN) and smooth_pass:
            return OperationStatus.SUCCESS

        if good_frac < GOOD_FRAC_MIN:
            logger.warning("π extraction failed: not enough good flux points (edge/finite/SNR).")
        if not smooth_pass:
            logger.warning("π extraction warning: Vπ vs flux unstable on good points.")

        return OperationStatus.FAILURE
