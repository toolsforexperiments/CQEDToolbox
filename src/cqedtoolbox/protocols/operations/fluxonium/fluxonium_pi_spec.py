import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cqedtoolbox.fitfuncs.resonators import HangerResponseBruno

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.analysis.fitfuncs.generic import Gaussian
from labcore.measurement.sweep import sweep_parameter
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.record import record_as, independent, dependent

from labcore.protocols.base import ProtocolOperation, OperationStatus, serialize_fit_params
from cqedtoolbox.protocols.operations.fluxonium.res_spec_vs_flux import _readout_frequencies, _fluxonium_basis
from cqedtoolbox.protocols.parameters import (
    Repetition, StartFlux, EndFlux, FluxSteps, ResonatorSpecSteps, StartPiSpecFrequency, EndPiSpecFrequency,
    QubitFrequency, GainPulseDuration, ECParam, ELParam, EJParam, ZeroFluxCurrent, ReadoutFrequency, GainMultiplier
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
    k = (2* Q_E * eta / HBAR) * np.abs(n01)
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

def _fit_gaussian_center_snr(x, y):
    """
    Fit y(x) with Gaussian and return (center, snr, fit_result).
    Notes:
      - This assumes Gaussian has params like A and some center param (x0/mu/center/f0).
      - SNR defined as |A| / (4*std(residual)).
    """
    fit = Gaussian(x, y)
    res = fit.run(fit)
    yfit = res.eval()
    resid = y - yfit
    noise = float(np.std(resid)) + 1e-12
    A = res.params["A"].value
    center = res.params["x0"].value
    snr = np.abs(A) / (4.0 * noise)
    return center, snr, res


class FluxoniumPiSpectroscopy(ProtocolOperation):
    """
    Fluxonium 'pi spectroscopy' / qubit spectroscopy vs flux.
    Dummy mode: simulate signal[rep, flux, freq] and fit a Gaussian peak to find f01(flux).

    Outputs: stores f01_vs_flux and snr arrays; you can add a protocol output later if desired.
    """

    SNR_THRESHOLD = 1.5
    MIN_GOOD_FRACTION = 0.8  # succeed if >=50% of flux points have good SNR
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
            f0=ReadoutFrequency(params),
            flux_start=StartFlux(params),
            flux_end=EndFlux(params),
            flux_steps=FluxSteps(params),
            steps=ResonatorSpecSteps(params), # frequency steps for qubit drive, not readout spec
            start_freq=StartPiSpecFrequency(params),
            end_freq=EndPiSpecFrequency(params),
            rabi_duration=GainPulseDuration(params),
            gain_multiplier=GainMultiplier(params),
            EC=ECParam(params),
            EL=ELParam(params),
            EJ=EJParam(params),
            Earth_flux=ZeroFluxCurrent(params)
        )
        self._register_outputs(
            qubit_freq_vs_flux=QubitFrequency(params)
        )

        self.condition = (
            f"Success if at least {self.MIN_GOOD_FRACTION:.0%} of flux points have "
            f"SNR >= {self.SNR_THRESHOLD} for any component (re/imag/mag/phase)."
        )

        self.independents = {"flux": [], "frequencies": []}
        self.dependents = {"signal": []}

        self.snr_best_vs_flux = None
        self.best_component_vs_flux = None
        self.fit_params_vs_flux = None
        self.figure_paths = []


    def _measure_dummy(self) -> Path:
        """
        Generate fake double-hanger res_spec-vs-flux data for fluxonium after gain pulse to activate qubit.
        Uses theoretical fr_g(flux) and fr_e(flux) to build a synthetic
        3D S21[reps, flux, freq] map, with additive complex Gaussian noise.
        The interface matches _measure_quick: same inputs, same ddh5 layout, and measure the faked qubit with given scanned ranges.
        """

        logger.info("Starting dummy fluxonium pi spectroscopy vs flux (simulated data)")

        n_rep = self.repetitions()
        n_flux = self.flux_steps()
        start_freq = self.start_freq()
        end_freq = self.end_freq()
        n_freq = self.steps()
        f0_vs_flux = self.f0()
        start_flux = self.flux_start()
        end_flux = self.flux_end()
        flux_vals = np.linspace(start_flux, end_flux, n_flux)
        freq_vals = np.linspace(start_freq, end_freq, n_freq)
        V_drive = self.gain_multiplier()*_solve_for_amplitude(5*np.pi, self.rabi_duration(), self.EC(), self.EL(), self.EJ(), flux_vals[n_flux//2]+self.Earth_flux())
        fr_g_vs_flux=[]
        fr_e_vs_flux=[]
        for flux in flux_vals:
            fr_g, fr_e = _readout_frequencies(self.EC(), self.EL(), self.EJ(), flux, self._SIM_FR, self._SIM_G)
            fr_g_vs_flux.append(fr_g)
            fr_e_vs_flux.append(fr_e)
        fr_g_vs_flux = np.asarray(fr_g_vs_flux)
        fr_e_vs_flux = np.asarray(fr_e_vs_flux)
        p_e_grid = np.empty((n_flux, n_freq), dtype=float)
        for i_flux in range(n_flux):
            flux_eff = float(flux_vals[i_flux] + self._SIM_EARTH_FLUX)
            for i_f in range(n_freq):
                f_drive = float(freq_vals[i_f])
                p_e_grid[i_flux, i_f] = _pe_rabi_after_pulse(self._SIM_EC, self._SIM_EL, self._SIM_EJ, flux_eff, f_drive, V_drive, self.rabi_duration(), self._SIM_KELVIN)
        s21_g_vec = np.array([HangerResponseBruno.model(coordinates=f0_vs_flux[i], A=self._SIM_A, f_0=fr_g_vs_flux[i], Q_i=self._SIM_Q_I, Q_e_mag=self._SIM_Q_E_MAG,
            theta=self._SIM_THETA, phase_offset=self._SIM_PHASE_OFF, phase_slope=self._SIM_PHASE_SLOPE, transmission_slope=self._SIM_TX_SLOPE) for i in range(n_flux)], dtype=np.complex128)
        s21_e_vec = np.array([HangerResponseBruno.model(coordinates=f0_vs_flux[i], A=self._SIM_A, f_0=fr_e_vs_flux[i], Q_i=self._SIM_Q_I, Q_e_mag=self._SIM_Q_E_MAG,
            theta=self._SIM_THETA, phase_offset=self._SIM_PHASE_OFF, phase_slope=self._SIM_PHASE_SLOPE, transmission_slope=self._SIM_TX_SLOPE) for i in range(n_flux)], dtype=np.complex128)
        s21_g_grid = np.broadcast_to(s21_g_vec[:, None], (n_flux, n_freq))
        s21_e_grid = np.broadcast_to(s21_e_vec[:, None], (n_flux, n_freq))
        def _dummy_hanger_signal():
            x_grid = np.random.rand(n_flux, n_freq)
            choose_e = x_grid < p_e_grid
            noise = self._SIM_NOISE_SIGMA * (np.random.randn(n_flux, n_freq) + 1j*np.random.randn(n_flux, n_freq)) / np.sqrt(2.0)
            signals = np.where(choose_e, s21_e_grid, s21_g_grid) + noise
            return signals
        
        flux_grid = np.broadcast_to(flux_vals[:, None], (n_flux, n_freq))
        freq_grid = np.broadcast_to(freq_vals[None, :], (n_flux, n_freq))
        sweep = (
            sweep_parameter("rep", range(n_rep))
            @ record_as(lambda:flux_grid, independent("current"))
            @ record_as(lambda: freq_grid, independent("qubit_drive_frequency"))
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
        self.independents["frequencies"] = np.asarray(data["qubit_drive_frequency"]["values"])
        self.independents["flux"] = np.asarray(data["current"]["values"])
        self.dependents["signal"] = np.asarray(data["signal"]["values"])


    def analyze(self):
        """
        Pi spectroscopy analysis (fluxonium):
        1) average over repetitions -> mean complex signal per (flux, freq)
        2) for each flux point, Gaussian-fit (real/imag/mag/phase), pick best SNR -> f01 center
        3) save arrays + figures into DatasetAnalysis
        """

        signals = self.dependents["signal"]
        freq_vals = self.independents["frequencies"][0, 0, :]
        flux_vals = self.independents["flux"][0, :, 0]
        S = np.mean(signals, axis=0)
        n_flux, n_freq = S.shape
        f01 = np.full(n_flux, np.nan, float)
        snr_best = np.full(n_flux, np.nan, float)
        best_comp = np.empty(n_flux, dtype=object)
        fit_params_best = [None] * n_flux

        for i in range(n_flux):
            row = S[i, :]
            candidates = {
                "real": row.real,
                "imag": row.imag,
                "mag":  np.abs(row),
                "phase": np.unwrap(np.angle(row)),
            }
            best = None
            for name, y in candidates.items():
                center, snr, res = _fit_gaussian_center_snr(freq_vals, y)
                if (best is None) or (snr > best[0]):
                    best = (snr, center, name, res)
            snr_best[i], f01[i], best_comp[i] = best[0], best[1], best[2]
            fit_params_best[i] = serialize_fit_params(best[3].params)
        self.qubit_freq_vs_flux = f01
        self.snr_best_vs_flux = snr_best
        self.best_component_vs_flux = best_comp
        self.fit_params_vs_flux = fit_params_best
        re_map = S.real.T
        im_map = S.imag.T
        mag_map = np.abs(S).T
        ph_map = np.unwrap(np.angle(S), axis=1).T

        maps = {
            "real":  (re_map,  "Real(S)",  "Real(S21)"),
            "imag":  (im_map,  "Imag(S)",  "Imag(S21)"),
            "mag":   (mag_map, "|S|",      "|S21|"),
            "phase": (ph_map,  "Phase(S)", "Phase (rad)"),
        }
        figs = {}
        for tag, (Z, title, cbar_label) in maps.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                Z,
                aspect="auto",
                origin="lower",
                extent=[flux_vals[0], flux_vals[-1], freq_vals[0], freq_vals[-1]],
                cmap="magma",
            )
            ax.set(
                title=f"{self.name}: {title}",
                xlabel="Flux (rad)",
                ylabel="Drive frequency (GHz)",
            )
            fig.colorbar(im, ax=ax, label=cbar_label)
            fig.tight_layout()
            figs[tag] = fig
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(flux_vals, f01, "o-", lw=1)
        ax.set(
            title=f"{self.name}: f01 vs flux (best-SNR Gaussian)",
            xlabel="Flux (rad)",
            ylabel="Extracted f01 (GHz)",
        )
        ax.grid(True)
        fig.tight_layout()
        figs["f01"] = fig

        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ds.add(
                flux_vals=np.asarray(flux_vals, float),
                freq_vals=np.asarray(freq_vals, float),
                signal_mean_re=np.asarray(S.real.T, float),
                signal_mean_im=np.asarray(S.imag.T, float),
                signal_mean_mag=np.asarray(np.abs(S).T, float),
                signal_mean_phase=np.asarray(np.angle(S).T, float),
                f01_vs_flux=np.asarray(f01, float),
                snr_best_vs_flux=np.asarray(snr_best, float),
                best_component_vs_flux=np.asarray(best_comp, dtype=str),
                fit_params_vs_flux=np.asarray(fit_params_best, dtype=object),
            )

            for tag, fig in figs.items():
                ds.add_figure(f"{self.name}_{tag}", fig=fig)
                img_path = ds._new_file_path(ds.savefolders[1], f"{self.name}_{tag}", suffix="png")
                self.figure_paths.append(img_path)
                plt.close(fig)


    def evaluate(self) -> OperationStatus:
        """
        Success criteria:
        - compute good mask over flux: finite f01 and snr_best >= threshold
        - succeed if fraction(good) >= MIN_GOOD_FRACTION
        """

        if len(self.qubit_freq_vs_flux) == 0:
            self.report_output = ["Empty spectroscopy result."]
            return OperationStatus.FAILURE
        
        good = np.isfinite(self.qubit_freq_vs_flux) & np.isfinite(self.snr_best_vs_flux) & (self.snr_best_vs_flux >= self.SNR_THRESHOLD)
        frac = float(np.mean(good)) if good.size else 0.0

        header = (
            f"## Fluxonium Spectroscopy vs Flux\n"
            f"Flux range: {float(self.flux_start()):.6g} → {float(self.flux_end()):.6g}\n"
            f"Frequency range: {float(self.start_freq()):.6g} → {float(self.end_freq()):.6g}\n"
            f"SNR threshold: {self.SNR_THRESHOLD}\n"
            f"Good fraction: {frac:.1%} (need ≥ {self.MIN_GOOD_FRACTION:.1%})\n"
            f"Data Path: `{self.data_loc}`\n"
        )

        if frac >= self.MIN_GOOD_FRACTION:
            self.report_output = [header, "SUCCESS: enough flux points have good SNR.\n"]
            return OperationStatus.SUCCESS

        self.report_output = [header, "FAILURE: not enough flux points have good SNR.\n"]
        return OperationStatus.FAILURE
