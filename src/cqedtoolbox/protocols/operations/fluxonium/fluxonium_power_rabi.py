import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cqedtoolbox.fitfuncs.resonators import HangerResponseBruno
from scipy.optimize import curve_fit

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.record import record_as, independent, dependent

from labcore.protocols.base import ProtocolOperation, OperationStatus
from cqedtoolbox.protocols.operations.fluxonium.res_spec_vs_flux import _fluxonium_basis, _readout_frequencies
from cqedtoolbox.protocols.parameters import (
    Repetition, StartFlux, EndFlux, FluxSteps, NumGainSteps, GainPulseDuration, GainMultiplier,
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


def _fit_complex_rabi(voltage_1d, signal_1d):
    """
    Fit averaged complex signal directly with
        S(V) = C + D * cos(omega * V + phi)
    where C and D are complex, omega and phi are real.
    Returns
    v_pi : float
        Pi-pulse amplitude in same units as voltage_1d.Since p(V) ~ sin^2(kV+phi0), so V_pi = pi / (2k) = pi / omega.
    fit_dict : dict
        Fitted parameters and a simple residual-based quality metric.
    """

    v = np.asarray(voltage_1d, dtype=float)
    s = np.asarray(signal_1d, dtype=np.complex128)

    m = np.isfinite(v) & np.isfinite(s.real) & np.isfinite(s.imag)
    v = v[m]
    s = s[m]

    order = np.argsort(v)
    v = v[order]
    s = s[order]

    span = float(v[-1] - v[0])
    C0 = np.mean(s)
    D0 = 0.5 * (np.max(np.abs(s - C0)) + np.std(s))
    D0 = max(float(np.real(D0)), 1e-6)
    omega_min = (0.8 * np.pi) / span
    omega_max = (8.4 * np.pi) / span
    omega0 = (4.0 * np.pi) / span

    def model_concat(V, c_re, c_im, d_re, d_im, omega, phi):
        C = c_re + 1j * c_im
        D = d_re + 1j * d_im
        y = C + D * np.cos(omega * V + phi)
        return np.concatenate([y.real, y.imag])

    ydata = np.concatenate([s.real, s.imag])
    p0 = [float(C0.real), float(C0.imag), float(D0), 0.0, float(np.clip(omega0, omega_min, omega_max)), 0.0]
    bounds_lo = [-np.inf, -np.inf, -np.inf, -np.inf, omega_min, -np.pi]
    bounds_hi = [ np.inf,  np.inf,  np.inf,  np.inf, omega_max,  np.pi]

    best = None
    for phi0 in (0.0, np.pi/4, np.pi/2, -np.pi/4, -np.pi/2):
        p0_try = p0.copy()
        p0_try[-1] = phi0
        try:
            popt, _ = curve_fit(model_concat, v, ydata, p0=p0_try, bounds=(bounds_lo, bounds_hi), maxfev=50000)
            yfit = model_concat(v, *popt)
            resid = ydata - yfit
            sse = float(np.sum(resid**2))
            if (best is None) or (sse < best[0]):
                best = (sse, popt)
        except Exception:
            continue

    _, popt = best
    c_re, c_im, d_re, d_im, omega_fit, phi_fit = popt
    C_fit = c_re + 1j * c_im
    D_fit = d_re + 1j * d_im
    v_pi = float(np.pi / omega_fit)
    v_pi = float(np.clip(v_pi, span / 100.0, 2.0 * span))

    yfit = C_fit + D_fit * np.cos(omega_fit * v + phi_fit)
    resid_complex = s - yfit
    resid_rms = float(np.sqrt(np.mean(np.abs(resid_complex)**2)))
    amp = float(np.abs(D_fit))
    snr_like = float(amp / resid_rms) if resid_rms > 0 else np.nan

    fit_dict = {
        "C_re": float(C_fit.real),
        "C_im": float(C_fit.imag),
        "D_re": float(D_fit.real),
        "D_im": float(D_fit.imag),
        "omega": float(omega_fit),
        "phi": float(phi_fit),
        "resid_rms": resid_rms,
        "snr_like": snr_like,
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
            gain_multiplier=GainMultiplier(params),
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
        V_grid = self.gain_multiplier() * (Veff_vec[:, None]) * np.linspace(0.0, 1.0, n_Volts)[None, :]
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
        2) Fit the averaged complex signal directly with a complex cosine.
        3) Extract Vπ from the oscillation period.
        """

        signals = self.dependents["signal"]
        voltage = self.independents["voltages"][0, :, :]
        S = np.mean(signals, axis=0)   # shape: (n_flux, n_V)

        n_flux, n_V = S.shape
        V_grid = voltage

        pi_voltages = np.full(n_flux, np.nan)
        snr_flux = np.full(n_flux, np.nan)
        fit_results = [None] * n_flux

        for flux_idx in range(n_flux):
            sig_1d = S[flux_idx]
            pi_voltages[flux_idx], fit_results[flux_idx] = _fit_complex_rabi(V_grid[flux_idx], sig_1d)
            snr_flux[flux_idx] = fit_results[flux_idx]["snr_like"]

        self.pi_power_vs_flux = pi_voltages
        self.snr = snr_flux
        self.fit_results = np.asarray(fit_results, dtype=object)

        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ds.add(
                pi_power_vs_flux=np.asarray(self.pi_power_vs_flux),
                snr=np.asarray(self.snr),
                snr_threshold=float(self.SNR_THRESHOLD),
                fit_results=self.fit_results,
            )
            fig, ax = plt.subplots()
            ax.plot(self.independents["flux"][0, :, 0], self.pi_power_vs_flux, "o-")
            ax.set_xlabel("Flux (rad)")
            ax.set_ylabel("Pi pulse amplitude (V)")
            ax.set_title("Extracted π pulse amplitude vs flux")
            ds.add_figure(f"{self.name}_pi_vs_flux", fig=fig)
            image_path = ds._new_file_path(ds.savefolders[1], f"{self.name}_pi_vs_flux", suffix="png")
            self.figure_paths.append(image_path)


    def evaluate(self) -> OperationStatus:
        """
        Final evaluation using only:
        1) SNR quality of the direct complex Rabi fit
        2) Smoothness of extracted pi-power vs flux on SNR-good points

        Designed to catch the failure mode where the two IQ clusters are too close:
        the oscillation becomes weak (low SNR) and the extracted pi values become jagged.
        """

        if not len(self.pi_power_vs_flux):
            self.report_output = ["Empty π-pulse result."]
            return OperationStatus.FAILURE

        GOOD_FRAC_MIN = 0.80 # minimum fraction of flux points that must have good Vπ extraction (not edge/finite and good SNR) for overall success
        JUMP_FRAC = 0.40
        JUMP_RATE_MAX = 0.20

        snr_bad = (
            ~np.isfinite(self.snr)
            | ~np.isfinite(self.pi_power_vs_flux)
            | (self.snr < self.SNR_THRESHOLD)
        )
        good = ~snr_bad
        good_frac = float(np.mean(good))

        pi_g = self.pi_power_vs_flux[good]
        if pi_g.size >= 3:
            d = np.abs(np.diff(pi_g))
            scale = np.maximum(np.abs(pi_g[:-1]), np.abs(pi_g[1:]))
            scale = np.maximum(scale, 1e-12)
            jump_rate = float(np.mean(d > JUMP_FRAC * scale))
            smooth_pass = jump_rate <= JUMP_RATE_MAX
        else:
            jump_rate = np.nan
            smooth_pass = False

        finite_frac = float(np.mean(np.isfinite(self.pi_power_vs_flux)))
        snr_good_frac = float(np.mean(~snr_bad))
        snr_med = float(np.nanmedian(self.snr)) if np.any(np.isfinite(self.snr)) else np.nan

        self.report_output = [(
            "## Rabi π-pulse extraction\n"
            f"Flux points: {len(self.pi_power_vs_flux)}\n"
            f"Vπ finite fraction: {finite_frac:.2%}\n"
            f"SNR-good fraction (>= {self.SNR_THRESHOLD}): {snr_good_frac:.2%}\n"
            f"Good fraction: {good_frac:.2%}\n"
            f"Median SNR: {snr_med if np.isfinite(snr_med) else 'n/a'}\n"
            f"Jump rate (SNR-good points): {jump_rate if np.isfinite(jump_rate) else 'n/a'}\n"
        )]

        if (good_frac >= GOOD_FRAC_MIN) and smooth_pass:
            return OperationStatus.SUCCESS

        if good_frac < GOOD_FRAC_MIN:
            logger.warning(
                f"Power Rabi failed: only {good_frac:.1%} of flux points passed SNR threshold "
                f"{self.SNR_THRESHOLD}."
            )
        elif not smooth_pass:
            logger.warning(
                f"Power Rabi failed: extracted π-pulse amplitudes are too jagged "
                f"(jump rate {jump_rate:.2f} > {JUMP_RATE_MAX:.2f})."
            )

        return OperationStatus.FAILURE
