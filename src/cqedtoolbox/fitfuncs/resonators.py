from typing import Dict, Any, Tuple

import lmfit
import numpy as np
import scipy
from matplotlib import pyplot as plt
from pathlib import Path

from matplotlib.ticker import MaxNLocator
from labcore.analysis.fit import Fit, FitResult


class ReflectionResponse(Fit):

    @staticmethod
    def model(coordinates: np.ndarray, A: float, f_0: float, Q_i: float, Q_e: float, phase_offset: float,
              phase_slope: float):
        """
        Reflection response model derived from input-output theory. For detail, see section 12.2.6 in "Quantum
        and Atom Optics" by Daniel Adam Steck

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of measurement
        A
            amplitude correction of the response
        f_0
            resonant frequency
        Q_i
            internal Q (coupling to losses of the cavity)
        Q_e
            external Q (coupling to output pins)
        phase_offset
            the offset of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response
        phase_slope
            the slope of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response

        Returns
        -------
        numpy array
            the ideal response calculated with the equation
        """
        x = coordinates
        s11_ideal = (1j * (1 - x / f_0) + (Q_i - Q_e) / (2 * Q_e * Q_i)) / (
                    1j * (1 - x / f_0) - (Q_i + Q_e) / (2 * Q_e * Q_i))
        correction = A * np.exp(1j * (phase_offset + phase_slope * (x - f_0)))
        return s11_ideal * correction

    @staticmethod
    def guess(coordinates: np.ndarray, data: np.ndarray):
        """ make an initial guess on parameters based on the measured reflection response data and the input-output
        theory. NOTE: to avoid errors, ensure that input arrays have length greater than 250.

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of measurement
        data
            1d numpy array containing the complex measured reflection response data

        Returns
        -------
        dict
            a dictionary whose values are the guess on A, f_0, Q_i, Q_e, phase_offset, and phase slope and keys
            contain their names
        """

        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()
        dip_loc = np.argmax(np.abs(np.abs(data) - amp))
        guess_f_0 = coordinates[dip_loc]

        data = moving_average(data)
        depth = amp - np.abs(data[dip_loc])
        width_loc = np.argmin(np.abs(amp - np.abs(data) - depth / 2))
        kappa = 2 * np.abs(coordinates[dip_loc] - coordinates[width_loc])
        guess_Q_tot = guess_f_0 / kappa
        # print(guess_Q_tot)

        [slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data[:data.size // 10], deg=False), 1)
        phase_offset = np.angle(data[0]) + slope * (coordinates[dip_loc] - coordinates[0])
        correction = amp * np.exp(1j * phase_offset)
        # print(data[dip_loc]/correction)
        guess_Q_e = 2 * guess_Q_tot / (1 - np.abs(data[dip_loc]/correction))
        guess_Q_i = 1 / (1 / guess_Q_tot - 1 / guess_Q_e)

        return dict(
            f_0=guess_f_0,
            A=amp,
            phase_offset=phase_offset,
            phase_slope=slope,
            Q_i=guess_Q_i,
            Q_e=guess_Q_e
        )


class HangerResponseBruno(Fit):
    """model from: https://arxiv.org/abs/1502.04082 - pages 6/7."""

    @staticmethod
    def model(coordinates: np.ndarray, A: float, f_0: float, Q_i: float, Q_e_mag: float, theta: float, phase_offset: float,
              phase_slope: float, transmission_slope: float):
        r"""A (1 + alpha * (x - f_0)/f_0) (1 - Q_l/|Q_e| exp(i \theta) / (1 + 2i Q_l (x-f_0)/f_0)) exp(i(\phi_v f_0
        + phi_0))"""

        x = coordinates
        amp_correction = A * (1 + transmission_slope * (x - f_0)/f_0)
        phase_correction = np.exp(1j*(phase_slope * x + phase_offset))

        if Q_e_mag == 0:
            Q_e_mag = 1e-12
        Q_e_complex = Q_e_mag * np.exp(-1j*theta)
        Q_c = 1./((1./Q_e_complex).real)
        Q_l = 1./(1./Q_c + 1./Q_i)
        response = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (1. + 2j*Q_l*(x-f_0)/f_0)

        return response * amp_correction * phase_correction

    @staticmethod
    def guess(coordinates, data) -> Dict[str, Any]:
        """NOTE: to avoid errors, ensure that input arrays have length greater than 250."""

        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()
        dip_loc = np.argmax(np.abs(np.abs(data) - amp))
        guess_f_0 = coordinates[dip_loc]
        [guess_transmission_slope, _] = np.polyfit(coordinates[:data.size // 10], np.abs(data[:data.size // 10]), 1)
        amp_correction = amp * (1+guess_transmission_slope*(coordinates-guess_f_0)/guess_f_0)

        data = moving_average(data)
        depth = amp - np.abs(data[dip_loc])
        width_loc = np.argmin(np.abs(amp - np.abs(data) - depth / 2))
        kappa = 2 * np.abs(coordinates[dip_loc] - coordinates[width_loc])
        guess_Q_l = guess_f_0 / kappa
        # print(guess_Q_tot)

        [slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data[:data.size // 10], deg=False), 1)
        phase_offset = np.angle(data[0], deg=False) - slope * (coordinates[0]-0)
        phase_correction = np.exp(1j*(slope * coordinates + phase_offset))
        correction = amp_correction * phase_correction
        # print(data[dip_loc]/correction)

        guess_theta = 0.5  # there are deterministic ways of finding it but looking at two symmetric points close to f_r in S21 , but it's kinda unnecessary so I just choose a small value and it works so far
        guess_Q_e_mag = np.abs(-guess_Q_l * np.exp(1j*guess_theta) / (data[dip_loc]/correction[dip_loc]-1))
        guess_Q_c = 1 / np.real(1 / (guess_Q_e_mag*np.exp(-1j*guess_theta)))
        guess_Q_i = 1 / (1 / guess_Q_l - 1 / guess_Q_c)

        return dict(
            A = amp,
            f_0 = guess_f_0,
            Q_i = guess_Q_i,
            Q_e_mag = guess_Q_e_mag,
            theta = guess_theta,
            phase_offset = phase_offset,
            phase_slope = slope,
            transmission_slope = guess_transmission_slope,
        )

        # return dict(
        #     A = 1,
        #     f_0 = 1,
        #     Q_i = 1e6,
        #     Q_e = 1e6,
        #     theta = 0,
        #     phase_offset=0,
        #     phase_slope=0,
        #     transmission_slope=0,
        # )


class TransmissionResponse(Fit):

    @staticmethod
    def model(coordinates: np.ndarray, f_0: float, Q_t: float, Q_e: float, phase_offset: float,
              phase_slope: float):
        """
        Reflection response model derived from input-output theory. For detail, see section 12.2.6 in "Quantum
        and Atom Optics" by Daniel Adam Steck

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of measurement
        f_0
            resonant frequency
        Q_t
            total Q
        Q_e
            geometric mean of the two coupling Qs (coupling to output pins) multiplied with the total attenuation
            of the signal path.
        phase_offset
            the offset of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response
        phase_slope
            the slope of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response

        Returns
        -------
        numpy array
            the ideal response calculated with the equation
        """
        x = coordinates
        # s21_ideal = (k_e1*k_e2)**0.5 / ( 1j*2*np.pi*(f_0-x) - (k_e1+k_e2+k_i)/2 )
        # s21_ideal = Q_e / (1j*(1-x/f_0)*Q_e**2 - (Q_e2+Q_e1+Q_e1*Q_e2/Q_i)/2)
        correction = np.exp(1j * (phase_offset + phase_slope * (x - f_0)))
        s21 = correction * (1j * Q_e * (1. - x/f_0) - .5 * Q_e / Q_t)**(-1)
        return s21

    @staticmethod
    def guess(coordinates: np.ndarray, data: np.ndarray) -> Dict[str, Any]:
        """ make an initial guess on parameters based on the measured reflection response data and the input-output
        theory. NOTE: to avoid errors, ensure that input arrays have length greater than 250.

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of measurement
        data
            1d numpy array containing the complex measured reflection response data

        Returns
        -------
        dict
            a dictionary whose values are the guess on A, f_0, Q_i, Q_e, phase_offset, and phase slope and keys
            contain their names
        """
        data = moving_average(data)

        # Average the first and last 10% of the data to get the base amplitude
        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()

        # Find the resonance frequency from the max point
        dip_loc = np.argmax(np.abs(np.abs(data) - amp))
        guess_f_0 = coordinates[dip_loc]

        # Find the depth to get kappa and from kappa and f_0 estimate Q_t
        depth = amp - np.abs(data[dip_loc])
        width_loc = np.argmin(np.abs(amp - np.abs(data) - depth / 2))
        kappa = np.abs(coordinates[dip_loc] - coordinates[width_loc])
        guess_Q_t = guess_f_0 / kappa

        # Use Q_t estimate and the max value to get an estimate for Q_e'
        guess_Q_e = 2 * guess_Q_t/np.abs(depth)

        # Guess the phase offset and slope
        [guess_slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data[:data.size // 10], deg=False), 1)
        guess_phase = np.angle(data[0]) + guess_slope * (coordinates[dip_loc] - coordinates[0])

        return dict(
            f_0=guess_f_0,
            Q_t=guess_Q_t,
            Q_e=guess_Q_e,
            phase_offset=guess_phase,
            phase_slope=guess_slope,
        )

    # @staticmethod
    # def nphoton(P_cold_dBm: float, Q_e1: float, Q_e2: float, Q_i: float, f_0: float):
    #     """ calculate the number of photons in the resonator
    #
    #     Parameters
    #     ----------
    #     P_cold_dBm
    #         the power (in unit of dBm) injected into the cold resonator, with cable loss taken into account
    #     Q_i
    #         internal Q (coupling to losses of the cavity)
    #     Q_e1
    #         input external Q (coupling to input pins)
    #     Q_e2
    #         output external Q (coupling to output pins)
    #     f_0
    #         resonant frequency
    #
    #     Returns
    #     -------
    #     float
    #         the number of photons
    #     """
    #     P_cold_W = 1e-3 * 10 ** (P_cold_dBm / 10.)
    #     Q_tot = 1 / (1 / Q_e1 + 1 / Q_e2 + 1 / Q_i)
    #     photon_number = 2. * P_cold_W * Q_tot ** 2 / (np.pi * scipy.constants.h * f_0 ** 2 * Q_e1)
    #     return photon_number


class AvoidedCrossing(Fit):

    @staticmethod
    def model(coordinates: Tuple[np.ndarray, ...], f_c: float, kappa: float, phase_offset: float, phase_slope: float, A: float,
              g: float, gamma: float, m: float, cc: float ):
        """
        This function represents the fitting model for an avoided crossing fit of two interacting energy modes
        intersecting with each other. One mode, f_c, is considered to be flat as a function of the varying external
        parameter, in our case magnetic field B. The mode that varies as a function of the external parameter, in this
        case what we define as f_ss in the model, is assumed to have a linear dependence on the externally varying
        parameter.

        Parameters
        ----------
        coordinates
            The frequencies and magnetic fields used in collecting the data. The convention is B-fields (T) first and
            frequencies (Hz) second.
        f_c
            The frequency of the stationary mode (GHz), what we call our cavity mode
        kappa
            The loss rate of the cavity mode (GHz)
        phase_offset
            A phase offset matching term (radians) to line up the phase offset between our data and model
        phase_slope
            A phase slope matching term (radians) to line up the phase evolution between our data and model
        A
            An amplitude term (unitless) to match the maximum peak height of our stationary mode response
        g
            The coupling (GHz) between the cavity mode and the varying mode ( f_ss or spin mode)
        gamma
            The loss rate or inhomogeneous broadening (GHz) of the spin mode
        m
            The slope of the linear response for the spin mode (GHz/T)
        cc
            The magnetic field (T) or varying parameter point where the two modes cross

        Returns
        -------
        numpy array
            A numpy array of the ideal response of the avoided crossing as a function of frequency and your varying
            parameter (B-field)

        """

        # This is how we are going to pass coordinates
        b_fields, freqs = coordinates

        # Our frequencies tend to be recorded in Hz, just bringing them into the same units as the rest of the parameters
        freqs = freqs / 1e9

        phase = np.exp(1j * (phase_offset + phase_slope * (freqs - f_c) + np.pi / 2))

        f_ss = m * (b_fields - cc) + f_c

        cal_results = A * phase * kappa * (freqs - f_ss - 1j * gamma) / ((freqs - f_c - 1j * kappa) * (freqs - f_ss - 1j * gamma) - g ** 2)
        return cal_results


    @staticmethod
    def guess(coordinates: Tuple[np.ndarray, ...], data: np.ndarray) -> Dict[str, Any]:
        """
        A function to generate a set of initial guesses for the parameters of our avoided crossing fit. It runs a
        transmission response fit on a set of data far from the avoided crossing point to extract some parameters and
        then provides educated guesses and arbitrary initial guesses for the remaining parameters.

        Parameters
        ----------
        coordinates
            The frequencies and magnetic fields used in collecting the data. The convention is B-fields (T) first and
            frequencies (Hz) second.
        data
            The complex response of the resonator as a function of frequency and externally applied magnetic field

        Returns
        -------
        dict
            A dictionary whose keys are the names of all the model parameters and whose values represent initial
            guesses for all the model parameters.

        """
        b_fields, freqs = coordinates
        fit = TransmissionResponse(freqs[0], data[0])
        fit_result = fit.run()

        res_params = fit_result.params

        f_c = res_params['f_0'].value / 1e9
        phase_offset = res_params['phase_offset'].value
        phase_slope = res_params['phase_slope'].value
        kappa = res_params['f_0'].value / res_params['Q_t'].value * 1e-9 / 2
        A = np.max(np.abs(data[0]))
        gamma = 2e-3
        g = 0.5e-3
        m = 80
        cc = (b_fields[0] + b_fields[-1])/2

        return dict(
            f_c = f_c,
            phase_offset = phase_offset,
            phase_slope = phase_slope,
            kappa = kappa,
            A = A,
            gamma = gamma,
            g = g,
            m = m,
            cc = cc
        )

    def analyze(self, coordinates: Tuple[np.ndarray, ...], data: np.ndarray,
                dry: bool = False, params: Dict[str, Any] = {},
                gamma_range: Tuple[float, float] = (0.001, 0.5),
                g_range: Tuple[float, float] = (1e-4, 5e-3),
                *args: Any, **fit_kwargs: Any) -> FitResult:
        """
        This is the function that runs when we do the command fit.run() for our fit object and produces the fitresult
        that tells us the results of the lmfit residual minimization based on the model for our class.
        We had to rewrite the base analysis function due to the restrictions we wanted to place on certain parameters.
        We wanted to have only certain parameters varying and to specify the range on the parameters that were varying.

        Parameters
        ----------
        coordinates
            The frequencies and magnetic fields used in collecting the data. The convention is B-fields (T) first and
            frequencies (Hz) second.
        data
            The complex response of the resonator as a function of frequency and externally applied magnetic field
        dry
            A boolean that tells you whether this is a dry run, a run with no residual minimization.
            A dry run would just spit out the fit_result based on the initial guesses.
        params
            A dictionary of all the parameters required for your model
        gamma_range
            The range of all potential values for the gamma parameter in the fit. We want to restrict the range to keep
            values within an expected range. Correlations between gamma and g can lead to runaway fits for weak coupling
        g_range
            The range of all potential values for the g parameter in the fit. We want to restrict the range to keep
            values within an expected range. Correlations between gamma and g can lead to runaway fits for weak coupling
        args
            any arguments you want to pass forward to any functions down the line
        fit_kwargs
            any keyword arguments for the fitting function that you would like to pass

        Returns
        -------
        FitResult
            A FitResult object that contains all the information required to run a fit and analyze its results

        """

        b_fields, freqs = coordinates

        cc_range = (b_fields[0], b_fields[-1])


        b_fields = np.expand_dims(b_fields, axis=1)
        b_fields = np.repeat(b_fields, freqs.shape[1], axis=1)

        model = lmfit.model.Model(self.model)

        _params = lmfit.Parameters()
        for pn, pv in self.guess(coordinates, data).items():
            _params.add(pn, value=pv)
        for pn, pv in params.items():
            if isinstance(pv, lmfit.Parameter):
                _params[pn] = pv
            else:
                _params[pn].set(value=pv)

        if dry:
            for pn, pv in _params.items():
                pv.set(vary=False)


        param_ranges = dict(
            g = g_range,
            gamma = gamma_range,
            cc = cc_range
        )

        for pn, pv in _params.items():
            if pn in param_ranges:
                pv.vary = True
                pv.min = param_ranges[pn][0]
                pv.max = param_ranges[pn][1]
            else:
                pv.vary = False

        lmfit_result = model.fit(data, params=_params,
                                 coordinates=(b_fields, freqs), **fit_kwargs)

        return FitResult(lmfit_result)


def moving_average(a):
    n = a.size//20*2+1
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return np.append(np.append(a[:int((n-1)/2-1)], ret[n - 1:] / n), a[int(-(n-1)/2-1):])


def _plot_mag_avoided_crossing(B, f, data, fit_result, title):
    """
    A helper function that produces a set of 2D plots corresponding to the magnitude of data and the fit of an avoided
    crossing alongside the residuals between the two

    Parameters
    ----------
    B
        1D array of floats containing magnetic field values
    f
        1D array of floats containing frequency values
    data
        2D numpy array of complex floats representing the cavity response as a function of frequency and magnetic field
    fit_result
        2D numpy array of complex floats representing the values from the model that was fit to the data
    title
        string representing the title you want on your figure

    Returns
    -------
    matplotlib.figure
        The three subplots of the magnitude of the data and fit with the residuals between them as a figure object

    """

    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    subfigs = fig.subfigures(1, 2, wspace=0.03, width_ratios=[1.7, 1])

    # Put my two color plots into the left set of figures
    axsLeft = subfigs[0].subplots(2, 1, sharex=True, sharey=True)
    # Data
    pcmd = axsLeft[0].pcolormesh(B, f, np.abs(data.transpose()),
                                 vmin=np.abs(data.transpose()).min(),
                                 vmax=np.abs(data.transpose()).max())
    axsLeft[0].xaxis.set_major_locator(MaxNLocator(2))
    axsLeft[0].set_title('Magnitude of Data')
    axsLeft[0].set_ylabel('Cavity Detuning (MHz)')

    # Fit Results
    pcmf = axsLeft[1].pcolormesh(B, f, np.abs(fit_result.transpose()),
                                 vmin=np.abs(data.transpose()).min(),
                                 vmax=np.abs(data.transpose()).max())
    axsLeft[1].set_xlabel('External Field (T)')
    axsLeft[1].set_ylabel('Cavity Detuning (MHz)')
    axsLeft[1].xaxis.set_major_locator(MaxNLocator(5))
    axsLeft[1].set_title('Magnitude of Fit Result')

    # Adding color bar
    subfigs[0].colorbar(pcmd, shrink=0.6, ax=axsLeft, location='bottom')

    # Plotting Residuals on the right side of the figure
    axsRight = subfigs[1].subplots(1,1)
    axsRight.set_xlabel('External Field (T)')
    axsRight.set_ylabel('Cavity Detuning (MHz)')
    axsRight.xaxis.set_major_locator(MaxNLocator(2))
    pcmr = axsRight.pcolormesh(B, f, (np.abs(fit_result) - np.abs(data)).transpose(), cmap='bwr')
    axsRight.set_title('Magnitude Residuals')
    subfigs[1].colorbar(pcmr, ax=axsRight, location='bottom')

    fig.suptitle(title, fontsize='x-large')

    plt.show()

    return fig


def fit_and_plot_avoided_crossing(coordinates: Tuple[np.ndarray, ...], data: np.ndarray,
                                  title="Avoided Crossing Fitting Magnitude Plots", *args, **kwargs):
    """
    This is a convenience function that runs an avoided crossing fit on a 1D set of magnetic fields and a 2D set of
    frequencies as coordinates and a 2D set of data. It then prints the fit report and plots the data, fit, and
    residuals for you.

    Parameters
    ----------
    coordinates
        b_fields - 1D numpy array of B fields
        freqs - 2D numpy array of frequency values, the shape should match the data array
    data
        2D numpy array of trace values that represent a resonator response as a function of applied field and frequency
    title
        String that represents the title of the figure that will be plotted
    args
        (optional) arguments to pass to other classes in the inheritance line
    kwargs
        (optional) keyword arguments, useful for passing values and ranges for the parameters that vary in the fit

    Returns
    -------
    FitResult, matplotlib.figure
        the result of the fit and the figure of the data, fit, and residuals
    """

    fit = AvoidedCrossing(coordinates, data)
    fit_result = fit.run(*args, **kwargs)

    print(fit_result.lmfit_result.fit_report())
    print('===================================')

    model_data = fit_result.eval()

    fig = _plot_mag_avoided_crossing(coordinates[0], (coordinates[1][0]/1e9 - fit_result.params['f_c'].value)*1000,
                                     data, model_data, title=title)

    return fit_result, fig


def plot_resonator_response(frequency: np.ndarray, figsize: tuple =(6, 3), f_unit: str = 'Hz', **sparams):
    """ plot the magnitude, phase, and polar diagrams of the data, the model with initially guessed parameters,
    and the fitted curve

    Parameters
    ----------
    coordinates
        1d numpy array containing the frequencies in range of measurement
    figsize
        size of the figure. Default is (6,3)
    f_unit
        the unit of frequency, often in Hz or GHz. Default is Hz
    sparams
        a dictionary whose values are either a few 1d arrays containing the measured data, values of model with
        initially guessed parameters, and values of the fitted curve, or a few dictionaries, each containing one of
        them and the corresponding plotting, and keys are their names.

    Returns
    -------
    matplotlib.figure
        a figure showing the magnitude, phase, and polar diagrams of the data, the model with initially guessed
        parameters, and the fitted curve
    """
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2,1])
    mag_ax = fig.add_subplot(gs[0,0])
    phase_ax = fig.add_subplot(gs[1,0], sharex=mag_ax)
    circle_ax = fig.add_subplot(gs[:,1], aspect='equal')

    for name, sparam in sparams.items():
        if isinstance(sparam, np.ndarray):
            data = sparam
            sparam = {}
        elif isinstance(sparam, dict):
            data = sparam.pop('data')
        else:
            raise ValueError(f"cannot accept data of type {type(sparam)}")

        mag_ax.plot(frequency, np.abs(data), label=name, **sparam)
        phase_ax.plot(frequency, np.angle(data, deg=False), **sparam)
        circle_ax.plot(data.real, data.imag, **sparam)

    mag_ax.legend(loc='best', fontsize='x-small')
    mag_ax.set_ylabel('Magnitude')

    phase_ax.set_ylabel('Phase (rad)')
    phase_ax.set_xlabel(f'Frequency ({f_unit})')

    circle_ax.set_xlabel('Re')
    circle_ax.set_ylabel('Im')

    return fig


def fit_and_plot_reflection(f_data: np.ndarray, s11_data: np.ndarray, fn=None, **guesses):
    """ convenience function which does the fitting and plotting (and saving to local directory if given the address)
     in a single call

    Parameters
    ----------
    f_data
        1d numpy array containing the frequencies in range of measurement
    s11_data
        1d numpy array containing the complex measured reflection response data
    guesses
        (optional) manual guesses on fit parameters

    Returns
    -------
    matplotlib.figure
        a figure showing the magnitude, phase, and polar diagrams of the data, the model with initially guessed
        parameters, and the fitted curve
    """
    fit = ReflectionResponse(f_data, s11_data)
    guess_result = fit.run(dry=True, **guesses)
    guess_y = guess_result.eval()

    fit_result = fit.run(**guesses)
    print(fit_result.lmfit_result.fit_report())

    fit_y = fit_result.eval()

    fig = plot_resonator_response(f_data * 1e-9, f_unit='GHz',
                                  data=dict(data=s11_data, lw=0, marker='.'),
                                  guess=dict(data=guess_y, lw=1, dashes=[1, 1]),
                                  fit=dict(data=fit_y))

    if fn is not None:
        with open(Path(fn.parent, 'fit.txt'), 'w') as f:
            f.write(fit_result.lmfit_result.fit_report())
        fig.savefig(Path(fn.parent, 'fit.png'))
        print('fit result and plot saved')

    return fig

def fit_and_plot_resonator_response(f_data: np.ndarray, s11_data: np.ndarray, response_type: str = 'transmission', fn=None, **guesses):
    """ convenience function which does the fitting and plotting (and saving to local directory if given the address)
     in a single call

    Parameters
    ----------
    f_data
        1d numpy array containing the frequencies in range of measurement
    s11_data
        1d numpy array containing the complex measured reflection response data
    response_type
        name of the response that we want to fit. The default is transmission response
    guesses
        (optional) manual guesses on fit parameters

    Returns
    -------
    matplotlib.figure
        a figure showing the magnitude, phase, and polar diagrams of the data, the model with initially guessed
        parameters, and the fitted curve
    """
    if response_type == 'transmission':
        fit = TransmissionResponse(f_data, s11_data)
    elif response_type == 'hanger':
        fit = HangerResponseBruno(f_data, s11_data)
    else:
        fit = ReflectionResponse(f_data, s11_data)
    guess_result = fit.run(dry=True, **guesses)
    guess_y = guess_result.eval()

    fit_result = fit.run(**guesses)
    print(fit_result.lmfit_result.fit_report())

    fit_y = fit_result.eval()

    fig = plot_resonator_response(f_data * 1e-9, f_unit='GHz',
                                  data=dict(data=s11_data, lw=0, marker='.'),
                                  guess=dict(data=guess_y, lw=1, dashes=[1, 1]),
                                  fit=dict(data=fit_y))
    print()
    print("===========")
    print()
    if response_type == 'transmission':
        print("Kappa total is", round(fit_result.params['f_0'].value / fit_result.params['Q_t'].value * 1e-6, 3), "MHz")
    if response_type == 'hanger':
        Q_e_mag = fit_result.params['Q_e_mag'].value
        theta = fit_result.params['theta'].value
        print("for your convenience: Q_c = 1/Re{1/Q_e} = ", 1 / np.real( 1 / (Q_e_mag*np.exp(-1j*theta)) ))


    if fn is not None:
        with open(Path(fn.parent, 'fit.txt'), 'w') as f:
            f.write(fit_result.lmfit_result.fit_report())
        fig.savefig(Path(fn.parent, 'fit.png'))
        print('fit result and plot saved')

    return fit, fit_result.lmfit_result, fig

def nphoton(P_cold_dBm: float, Q_e: float, theta: float, Q_i: float, f_0: float, f_p: None):
    """
    Calculates number of photons into a resonator.
    :param P_cold_dBm: Input power to the device in dBm.
    :param Q_e: External quality factor.
    :param theta: Rotation angle of the external quality factor.
    :param Q_i: Internal quality factor.
    :param f_0: Resonator frequency.
    :param f_p: Pump frequency. If None, resonant pumping/probing is assumed.
    """
    P_cold_W = 1e-3 * 10 ** (P_cold_dBm / 10.)
    Q_e_complex = Q_e * np.exp(-1j * theta)
    Q_c = 1. / ((1 / Q_e_complex).real)
    Q_tot = 1. / (1. / Q_c + 1. / Q_i)

    kappa_c = f_0 / Q_c
    kappa_tot = f_0 / Q_tot

    if f_p is None:
        f_p = f_0

    return 1 / (2 * np.pi)**2 * kappa_c / 2 * P_cold_W / (scipy.constants.hbar * f_p) / ((f_0 - f_p) ** 2 + kappa_tot ** 2 / 4)
