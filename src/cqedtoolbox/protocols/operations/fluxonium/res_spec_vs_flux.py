import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.sparse import diags, kron, identity, csc_matrix
from scipy.sparse.linalg import eigsh

plt.switch_backend("agg")

from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.analysis import DatasetAnalysis
from labcore.analysis.fit import Fit
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.record import record_as, independent, dependent

from labcore.protocols.base import ProtocolOperation, OperationStatus
from cqedtoolbox.protocols.parameters import (
    Repetition, ResonatorSpecSteps, StartReadoutFrequency, EndReadoutFrequency,
    StartFlux, EndFlux, FluxSteps, ZeroFluxCurrent, ECParam, ELParam, EJParam, CouplingG, ResonatorFr,
)
from cqedtoolbox.protocols.operations.single_qubit.res_spec import UnwindAndFitRet
from cqedtoolbox.fitfuncs.resonators import moving_average, HangerResponseBruno

logger = logging.getLogger(__name__)


class DoubleHangerResponseBruno(Fit):
    """For single Hanger fit of each dip, we use model from: https://arxiv.org/abs/1502.04082 - pages 6/7.
    S_g/e = A (1 + alpha * (x - f_0)/f_0) (1 - Q_l/|Q_e| exp(i \theta) / (1 + 2i Q_l (x-f_0)/f_0)) exp(i(\phi_v f_0+ phi_0))
    We assume all parameters (except fr_g and fr_e) are same for both dips since fr_g and fr_e are sent from same cable and resonator.
    And S_tot = p_g * S + (1 - p_g) * S_e """

    @staticmethod
    def model(coordinates: np.ndarray, A: float, fr_g: float, fr_e: float, p_g: float, Q_i: float, Q_e_mag: float, theta: float, phase_offset: float,
              phase_slope: float, transmission_slope: float):
        
        x = coordinates
        amp_correction_g = A * (1 + transmission_slope * (x - fr_g)/fr_g)
        amp_correction_e = A * (1 + transmission_slope * (x - fr_e)/fr_e)
        phase_correction = np.exp(1j*(phase_slope * x + phase_offset))

        if Q_e_mag == 0:
            Q_e_mag = 1e-12
        Q_e_complex = Q_e_mag * np.exp(-1j*theta)
        Q_c = 1./((1./Q_e_complex).real)
        Q_l = 1./(1./Q_c + 1./Q_i)
        response_g = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (1. + 2j*Q_l*(x-fr_g)/fr_g)
        response_e = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (1. + 2j*Q_l*(x-fr_e)/fr_e)
        
        return (p_g*response_g * amp_correction_g + (1-p_g) * response_e * amp_correction_e) * phase_correction

    @staticmethod
    def guess(coordinates, data) -> Dict[str, Any]:
        """Heuristic initial guess for the double-hanger Bruno model.
        This is Bruno's single-notch guess, extended to find a second dip and p_g.
        NOTE: to avoid errors, ensure that input arrays have length >~ 250"""

        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()
        data_smooth = moving_average(data)
        depth = amp - np.abs(data)
        loc = []
        margin = max(5, len(data) // 150)
        for i in range(margin, len(data) - margin):
            window = depth[i - margin : i + margin + 1]
            if depth[i] == window.max():  # in order to find the second dip we need to find two biggest local max and avoid noise on sholder of largest dip
                loc.append(i)
        loc = np.array(loc)
        vals = depth[loc]
        order = np.argsort(vals)
        dip_g = loc[order[0]]
        dip_e = loc[order[1]]
        guess_fr_g = coordinates[dip_g]
        guess_fr_e = coordinates[dip_e]
        [guess_transmission_slope, _] = np.polyfit(coordinates[:data.size // 10], np.abs(data[:data.size // 10]), 1)
        amp_correction_g = amp * (1+guess_transmission_slope*(coordinates-guess_fr_g)/guess_fr_g)

        depth_g = amp - np.abs(data_smooth[dip_g])
        depth_e = amp - np.abs(data_smooth[dip_e])
        guess_p_g = depth_g / (depth_g + depth_e)
        width_g = np.argmin(np.abs(amp - np.abs(data_smooth) - depth_g / 2))
        kappa = 2 * np.abs(coordinates[dip_g] - coordinates[width_g])
        guess_Q_l = guess_fr_g / kappa

        [slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data_smooth[:data.size // 10], deg=False), 1)
        phase_offset = np.angle(data_smooth[0], deg=False) - slope * (coordinates[0]-0)
        phase_correction = np.exp(1j*(slope * coordinates + phase_offset))
        correction_g = amp_correction_g * phase_correction

        guess_theta = 0.5
        guess_Q_e_mag = np.abs(-guess_Q_l * np.exp(1j*guess_theta) / (data_smooth[dip_g]/correction_g[dip_g]-1))
        eps = 1e-6
        if not np.isfinite(guess_Q_e_mag) or guess_Q_e_mag <= 0:
            guess_Q_e_mag = 5e4  # some reasonable default

        denom = np.real(1 / (guess_Q_e_mag * np.exp(-1j * guess_theta)))
        if not np.isfinite(denom) or abs(denom) < eps:
            denom = np.sign(denom) * eps
        guess_Q_c = 1.0 / denom

        denom2 = 1.0 / guess_Q_l - 1.0 / guess_Q_c
        if not np.isfinite(denom2) or abs(denom2) < eps:
            denom2 = np.sign(denom2) * eps
        guess_Q_i = 1.0 / denom2
        if not np.isfinite(guess_Q_i) or guess_Q_i <= 0:
            guess_Q_i = 1e6
        if guess_p_g < 0.5:
            guess_p_g = 1 - guess_p_g
            a = guess_fr_g
            b = guess_fr_e
            guess_fr_e = a
            guess_fr_g = b

        return dict(
            A = amp,
            fr_g = guess_fr_g,
            fr_e = guess_fr_e,
            p_g = guess_p_g,
            Q_i = guess_Q_i,
            Q_e_mag = guess_Q_e_mag,
            theta = guess_theta,
            phase_offset = phase_offset,
            phase_slope = slope,
            transmission_slope = guess_transmission_slope,
        )


def add_mag_and_unwind_and_choose_fit(frequencies, signal_raw, model_choice) -> UnwindAndFitRet:
    phase_unwrap = np.unwrap(np.angle(signal_raw))
    phase_slope = np.polyfit(frequencies, phase_unwrap, 1)[0]
    signal_unwind = signal_raw * np.exp(-1j * frequencies * phase_slope)
    magnitude = np.abs(signal_raw)
    phase = np.arctan2(signal_unwind.imag, signal_unwind.real)
    if model_choice == "single":
        fit = HangerResponseBruno(frequencies, signal_unwind)
    else:
        fit = DoubleHangerResponseBruno(frequencies, signal_unwind)
    fit_result = fit.run(fit)
    fit_curve = fit_result.eval()
    residuals = signal_unwind - fit_curve
    amp = fit_result.params["A"].value
    noise = np.std(residuals)
    snr = np.abs(amp / (4 * noise))

    ret = UnwindAndFitRet(
        signal_unwind=signal_unwind,
        magnitude=magnitude,
        phase=phase,
        fit_curve=fit_curve,
        fit_result=fit_result,
        residuals=residuals,
        snr=snr,
        fig=None,
        ax=None,
    )
    return ret


def _count_dips_fast(sig_raw) -> int:
    """Fast estimate of the count of dips for signal at a specific flux"""
    
    magnitude = np.abs(sig_raw)
    win = max(5, len(magnitude) // 200)
    mag_s = np.convolve(magnitude, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(-mag_s, prominence = 0.15 * np.ptp(mag_s), distance = max(5, len(magnitude) // 150))
    return int(len(peaks))


def _decide_model(sig_raw, sample_every: int = 10, frac_single_threshold: float = 0.70) -> str:
    """Decide to use single hanger fit if count of dips <= 1 for >= 75% of flux points (we assume fr_g and fr_e are indistinguishable,
    or fr_e not in range of frequencies scanned in this case), otherwise will use double notch fit"""
    
    idx = np.arange(0, len(sig_raw), sample_every)
    counts = np.array([_count_dips_fast(sig_raw[i]) for i in idx])
    return "single" if np.mean(counts <= 1) >= frac_single_threshold else "double"


def _fluxonium_basis(EC, EL, EJ, flux_ext, levels=8, grid=2000, flux_max=12*np.pi):
    """Creat Fluxonium Hamiltonian, return (E, n_Q projected into eigenbasis) using FD in phase."""
    
    flux = np.linspace(-flux_max, flux_max, grid)
    dflux = flux[1] - flux[0]
    main, off = 2.0*np.ones(grid), -1.0*np.ones(grid-1)
    lap = diags([off, main, off], offsets=[-1, 0, 1]) / (dflux**2)
    T = -4.0 * EC * lap
    V = 0.5*EL*(flux**2) - EJ*np.cos(flux-flux_ext)
    H = T + diags(V, 0)

    evals, evecs = eigsh(H, k=levels, which="SA")
    idx = np.argsort(evals)
    E = np.array(evals[idx])
    Psi = np.array(evecs[:, idx])

    offp =  np.ones(grid-1) / (2.0 * dflux)
    offm = -np.ones(grid-1) / (2.0 * dflux)
    d1 = diags([offm, np.zeros(grid), offp], [-1, 0, 1], dtype=np.complex128)
    n_grid = (-1j) * d1
    nQ = Psi.conj().T @ (n_grid @ Psi)
    return E, nQ


def _resonator_ops_from_fr(fr_bare, N_phot=6):
    """Creat Resonator Hamiltonian, return H_res (GHz), n_R = i(a†-a), N = a†a in N_phot Fock basis."""
    
    a = np.zeros((N_phot, N_phot), dtype=complex)
    for n in range(1, N_phot):
        a[n-1, n] = np.sqrt(n)
    adag = a.conj().T
    Hres = fr_bare * (adag @ a + 0.5 * np.eye(N_phot))
    nR = 1j * (adag - a)
    Nph = adag @ a
    return csc_matrix(Hres), csc_matrix(nR), csc_matrix(Nph)


def _build_total_H(EC, EL, EJ, flux_ext, fr_bare, g, q_levels=8, N_phot=6, grid=2000, flux_max=12*np.pi):
    EQ, nQ = _fluxonium_basis(EC, EL, EJ, flux_ext, levels=q_levels, grid=grid, flux_max=flux_max)
    Hq = diags(EQ, 0, format='csc')
    Iq = identity(q_levels, format='csc')
    Hres, nR, Nph = _resonator_ops_from_fr(fr_bare, N_phot=N_phot)
    Ir = identity(N_phot, format='csc')
    Hint = g * kron(csc_matrix(nQ), nR)
    Htot = kron(Hq, Ir) + kron(Iq, Hres) + Hint
    return Htot, Nph, q_levels, N_phot


def _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold, n_target):
    """Select the eigenstate closest to |manifold, n_target>, since they might not be lowest 4 eigenstates.
    For g/e state of qubit, expection value of P_g/P_e projector should close to 1. For 0/1 state of resonator expectation of N operator should close to 0/1"""
    
    P = Pg_full if manifold == "g" else Pe_full
    best_E = None
    best_score = np.inf
    for j in range(E.size):
        v = V[:, j]
        p = float(np.real_if_close(v.conj().T @ (P @ v)))
        nbar = float(np.real_if_close(v.conj().T @ (Nph_full @ v)))
        score = (1.0 - p) + abs(nbar - n_target)
        if score < best_score:
            best_score = score
            best_E = E[j]
    return best_E


def _readout_frequencies(EC, EL, EJ, flux_ext, fr, g, q_levels=10, N_phot=6, grid=2000, flux_max=12*np.pi):
    """Simulate frequencies fr_g and fr_e of specific flux point"""
    
    Htot, Nph, Lq, Nr = _build_total_H(EC, EL, EJ, flux_ext, fr, g, q_levels=q_levels,
        N_phot=N_phot, grid=grid, flux_max=flux_max)
    k_eval = min(4 * Lq, Htot.shape[0] - 2) # Make sure to include g0, g1, e0, e1
    evals, evecs = eigsh(Htot, k=k_eval, which="SA")
    idx = np.argsort(evals)
    E = evals[idx]
    V = evecs[:, idx]

    Pg = np.zeros((Lq, Lq))
    Pg[0, 0] = 1.0
    Pe = np.zeros((Lq, Lq))
    Pe[1, 1] = 1.0
    Pg_full = kron(csc_matrix(Pg), identity(Nr, format="csc"))
    Pe_full = kron(csc_matrix(Pe), identity(Nr, format="csc"))
    Nph_full = kron(identity(Lq, format="csc"), Nph)

    Eg0 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="g", n_target=0)
    Eg1 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="g", n_target=1)
    Ee0 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="e", n_target=0)
    Ee1 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="e", n_target=1)
    fr_g = Eg1 - Eg0
    fr_e = Ee1 - Ee0
    return fr_g, fr_e


def _estimate_current_period(currents, fr_g, min_period_fraction=0.80, max_period_fraction=1.05):
    """
    Estimate the period in the sweep variable (current or flux) from f_r_g, using fft autocorrelation.
    Assumesthe sweep spans of order ~1 period of the main oscillation.
    """

    n = len(currents)
    y = fr_g - np.nanmean(fr_g)
    y = np.nan_to_num(y, nan=0.0)
    nfft = 1 << ((2 * n - 1).bit_length())
    fft = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(fft * np.conj(fft))[:n].real
    ac /= ac[0]
    dI = float(np.mean(np.diff(currents)))
    span = float(currents[-1] - currents[0])
    lags = np.arange(n, dtype=float) * dI
    min_T = min_period_fraction * span
    max_T = max_period_fraction * span
    mask = (lags >= min_T) & (lags <= max_T)
    if not np.any(mask):
        return span
    ac_win = ac[mask]
    lag_win = lags[mask]
    if ac_win.size < 3:
        return span
    interior = (ac_win[1:-1] > ac_win[:-2]) & (ac_win[1:-1] >= ac_win[2:])
    peak_indices = np.where(interior)[0] + 1
    if peak_indices.size == 0:
        return span
    best_idx = peak_indices[np.argmax(ac_win[peak_indices])]
    period_est = lag_win[best_idx]
    return period_est


def _zero_flux_point(EC, EL, EJ, fr_bare, g, flux_vals, fr_g_exp):
    """Assume in experiment, we already know accurate qubit params for fr_bare, g(coupling strength), EC, EL.
    Use an estimation of EJ(+-10%) and input it to give a structure of fr_g vs flux. Then, scan a constant flux shift and pick
    the one that best matches experiment. """

    fr_g_thr = []
    for phi in flux_vals:
        fr_g, _ = _readout_frequencies(EC, EL, EJ, phi, fr_bare, g)
        fr_g_thr.append(fr_g)
    fr_g_thr = np.asarray(fr_g_thr, dtype=float)
    fr_g_exp = np.asarray(fr_g_exp, dtype=float)
    dphi = float(flux_vals[1] - flux_vals[0])
    thr0 = fr_g_thr - fr_g_thr.mean()
    exp0 = fr_g_exp - fr_g_exp.mean()
    F_thr = np.fft.fft(thr0)
    F_exp = np.fft.fft(exp0)
    corr = np.fft.ifft(F_thr * np.conj(F_exp)).real
    best_k = int(np.argmax(corr))
    shift_flux = best_k * dphi
    zero_flux = -shift_flux
    zero_flux_wrapped = zero_flux % 2*np.pi
    return zero_flux_wrapped


class ResonatorSpectroscopyVsFlux(ProtocolOperation):

    SNR_THRESHOLD = 2.5

    # True fake data params (may change to any)
    _SIM_QI = 8000.0
    _SIM_QE_MAG = 5000.0
    _SIM_THETA = 0.5
    _SIM_PHASE_OFF = 0.1
    _SIM_PHASE_SLOPE = 2e-9  # rad/Hz
    _SIM_TX_SLOPE = 5e-9  # /Hz
    _SIM_A = 1.0
    _SIM_PG = 0.9
    _SIM_EC = 1.0  # followings are params for fluxonium
    _SIM_EL = 0.5
    _SIM_EJ = 5.3
    _SIM_FR = 4.07
    _SIM_G = 0.067
    _SIM_NOISE_SIGMA = 0.2
    _SIM_EARTH_FLUX = 0.5

    def __init__(self, params):
        super().__init__()
    
        self._register_inputs(
            repetitions=Repetition(params),
            steps=ResonatorSpecSteps(params),
            start_freq=StartReadoutFrequency(params),
            end_freq=EndReadoutFrequency(params),
            start_flux=StartFlux(params),
            end_flux=EndFlux(params),
            flux_steps=FluxSteps(params),
            EC=ECParam(params),
            EL=ELParam(params),
            EJ=EJParam(params),
            g=CouplingG(params),
            fr=ResonatorFr(params),
        )
        self._register_outputs(
            zero_flux_current=ZeroFluxCurrent(params)
        )

        self.condition = f"Success if every trace has SNR ≥ {self.SNR_THRESHOLD}"
        self.independents = {"frequencies": [], "flux": []}
        self.dependents = {"signal": []}
        self.data_loc = None
        self.model_choice = None
        self.fr_g = []
        self.fr_e = []
        self.fit_results = []
        self.pg = []
        self.snr = []
        self.figure_paths = []
        self.symmetry_score = None
        self.zero_slope = None


    def _measure_dummy(self) -> Path:
        """
        Generate fake double-hanger res_spec-vs-flux data.
        Uses theoretical fr_g(flux) and fr_e(flux) to build a synthetic
        3D S21[flux, repetition, freq] map, with additive complex Gaussian noise.
        The interface matches _measure_quick: same inputs, same ddh5 layout, and measure the faked qubit with given scanned ranges.
        """

        logger.info("Starting dummy resonator spectroscopy vs flux (simulated fake data)")
        n_rep = self.repetitions()
        n_flux = self.flux_steps()
        n_freq = self.steps()
        start_flux = self.start_flux()
        end_flux = self.end_flux()
        start_freq = self.start_freq()
        end_freq = self.end_freq()
        freq = np.linspace(start_freq, end_freq, n_freq)
        flux_vals = np.linspace(start_flux, end_flux, n_flux)

        fr_g_vs_flux=[]
        fr_e_vs_flux=[]
        for flux_ext in flux_vals:
            fr_g, fr_e = _readout_frequencies(self._SIM_EC, self._SIM_EL, self._SIM_EJ, flux_ext + self._SIM_EARTH_FLUX, self._SIM_FR, self._SIM_G, q_levels=10, N_phot=5)
            fr_g_vs_flux.append(fr_g)
            fr_e_vs_flux.append(fr_e)
        fr_g_vs_flux = np.asarray(fr_g_vs_flux)
        fr_e_vs_flux = np.asarray(fr_e_vs_flux)

        def _dummy_double_hanger_signal():
            signals = np.empty((n_flux, n_freq), dtype=np.complex128)
            for i_flux in range(n_flux):
                noise = self._SIM_NOISE_SIGMA * (np.random.randn(n_freq) + 1j * np.random.randn(n_freq)) / np.sqrt(2.0)
                fr_g = fr_g_vs_flux[i_flux]
                fr_e = fr_e_vs_flux[i_flux]
                s21_clean = DoubleHangerResponseBruno.model(coordinates=freq, A=self._SIM_A, fr_g=fr_g, fr_e=fr_e, p_g=self._SIM_PG, Q_i=self._SIM_QI,
                                                            Q_e_mag=self._SIM_QE_MAG, theta=self._SIM_THETA, phase_offset=self._SIM_PHASE_OFF, phase_slope=self._SIM_PHASE_SLOPE, transmission_slope=self._SIM_TX_SLOPE)
                signals[i_flux, :] = s21_clean + noise
            return signals
        freq_grid = np.broadcast_to(freq[None, :], (n_flux, n_freq))
        flux_gird = np.broadcast_to(flux_vals[:, None], (n_flux, n_freq))
        sweep = (
            sweep_parameter("rep", range(n_rep))
            @ record_as(lambda:flux_gird, independent("current"))
            @ record_as(lambda: freq_grid, independent("frequency"))
            @ record_as(_dummy_double_hanger_signal, dependent("signal"))
        )
        loc, data = run_and_save_sweep(sweep, './data', 'my_data')
        logger.info(f"Dummy measurement complete, data saved to {loc}")
        return loc


    def _load_data_dummy(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)
        self.independents["frequencies"] = np.asarray(data["frequency"]["values"])
        self.independents["flux"] = np.asarray(data["current"]["values"])
        self.dependents["signal"] = np.asarray(data["signal"]["values"])


    def analyze(self):
        """Fit (single/double), compute zero-flux candidates with data having good snr."""
        with DatasetAnalysis(self.data_loc.parent, self.name) as ds:
            freqs = np.asarray(self.independents["frequencies"][0,0,:])
            flux_vals = np.asarray(self.independents["flux"][0,:,0])
            sig2d = np.asarray(np.mean(self.dependents["signal"], axis=0))
            self.model_choice = _decide_model(sig2d)
            logger.info(f"model_choice = {self.model_choice}")
            for sig_row in sig2d:
                ret = add_mag_and_unwind_and_choose_fit(freqs, sig_row, self.model_choice)
                p = ret.fit_result.params
                self.fit_results.append(p)
                self.snr.append(ret.snr)
                self.fr_g.append(p["f_0"] if self.model_choice == "single" else p["fr_g"])
                self.fr_e.append(np.nan if self.model_choice == "single" else p["fr_e"])
                self.pg.append(np.nan if self.model_choice == "single" else p["p_g"])
            phi_arr = np.asarray(flux_vals).astype(float)
            frg_arr = np.asarray(self.fr_g).astype(float)
            snr_arr = np.asarray(self.snr).astype(float)
            mask = (
                np.isfinite(phi_arr)
                & np.isfinite(frg_arr)
                & np.isfinite(snr_arr)
                & (snr_arr >= self.SNR_THRESHOLD)
            )
            I_period = _estimate_current_period(phi_arr[mask], frg_arr[mask])
            self.zero_flux_current = _zero_flux_point(self.EC(), self.EL(), self.EJ(), self.fr(), self.g(), phi_arr[mask]*2*np.pi/I_period, frg_arr[mask])
            ds.add(
                flux=phi_arr,
                fr_g=frg_arr,
                snr=snr_arr,
                model_choice=self.model_choice,
                zero_flux_current=np.asarray(self.zero_flux_current),
            )

            extent = [
            float(flux_vals.min()) if flux_vals.size else 0.0,
            float(flux_vals.max()) if flux_vals.size else 1.0,
            float(freqs.min()) if freqs.size else 0.0,
            float(freqs.max()) if freqs.size else 1.0,
        ]
            fig, ax = plt.subplots()
            plt.imshow(np.abs(sig2d.T), origin="lower", aspect="auto", extent=extent, cmap="inferno")
            plt.colorbar(label="|S| (a.u.)")
            ax.plot(phi_arr[mask], frg_arr[mask], ".-", label="Fitted resonator read out frequency (SNR > 2)")
            ax.axvline(self.zero_flux_current, linestyle="--", label="Zero flux point")
            ax.set_ylim(freqs.min(), freqs.max())
            ax.set_xlabel("Flux(rad)")
            ax.set_ylabel("Frequency(GHz)")
            ax.set_title("Resonator signal response vs frequencies vs flux")
            ax.legend()
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            fig.savefig(image_path)
            self.figure_paths.append(image_path)

            frg_good = frg_arr[mask]
            c_centered = phi_arr[mask] - float(self.zero_flux_current)
            idx0 = int(np.argmin(np.abs(c_centered)))
            n_sym_pairs =  10
            left  = frg_good[idx0-1 : idx0-n_sym_pairs-1 : -1]
            right = frg_good[idx0+1 : idx0+n_sym_pairs+1]
            diffs = right - left
            symmetry_score = float(np.sqrt(np.mean(diffs**2)))
            n_fit = 5
            i_start = idx0 - n_fit
            i_stop  = idx0 + n_fit + 1
            x = c_centered[i_start:i_stop]
            y = frg_good[i_start:i_stop]
            a, b = np.polyfit(x, y, 1)
            zero_slope = float(a)
            self.symmetry_score = symmetry_score
            self.zero_slope = zero_slope


    def evaluate(self) -> OperationStatus:
        """
        Final evaluation: determine quality of flux sweep fit.
        Criteria:
        1. Enough flux points have SNR above SNR_THRESHOLD.
        2. The fitted f_r_g(phi) is sufficiently symmetric around the
            extracted zero-flux point (symmetry_score small).
        3. The slope near the zero-flux point is small (flat minimum).
        """
        
        if not self.snr or len(self.snr) == 0:
            self.report_output = ["No SNR computed. Did analyze() run?"]
            logger.warning("No SNR computed. Did analyze() run?")
            return OperationStatus.FAILURE
        snr_arr = np.asarray(self.snr, dtype=float)
        success_mask = snr_arr >= self.SNR_THRESHOLD
        good_fraction = float(np.mean(success_mask))
        all_snr_good = good_fraction >= 0.8
        symmetry_ok = abs(self.symmetry_score) <= 1e-3
        slope_ok    = abs(self.zero_slope)     <= 1e-3

        msg = (
            "## Resonator Spectroscopy vs Flux\n"
            f"Flux points (traces): {len(self.snr)}\n"
            f"SNR min/median/max: {snr_arr.min():.2f} / "
            f"{np.median(snr_arr):.2f} / {snr_arr.max():.2f}\n"
            f"Pass threshold (per trace): SNR ≥ {self.SNR_THRESHOLD}\n"
            f"Good-SNR fraction: {good_fraction:.2%}\n\n"
            f"Zero-flux current estimate: {self.zero_flux_current}\n"
            f"Symmetry score (f(+φ) vs f(−φ)): {self.symmetry_score:.3g} "
            f"Slope near zero-flux: {self.zero_slope:.3g} "
        )
        self.report_output = [msg]
        if all_snr_good and symmetry_ok and slope_ok:
            return OperationStatus.SUCCESS
        elif symmetry_ok and slope_ok:
            logger.info(self.report_output)
            logger.warning(f"Some traces have SNR below threshold {self.SNR_THRESHOLD}")
            return OperationStatus.FAILURE
        elif all_snr_good:
            logger.info(self.report_output)
            logger.warning("Bad zero flux point estimation")
            return OperationStatus.FAILURE
        else:
            logger.info(self.report_output)
            logger.warning("Some traces have SNR below threshold and Bad zero flux point estimation")
            return OperationStatus.FAILURE

