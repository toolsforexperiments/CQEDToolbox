# Fluxonium Protocol Notes

This folder contains fluxonium-specific protocol operations for auto-tuning and characterization in `CQEDToolbox`.

Before using these protocols, first read the main `CQEDToolbox` README for installation, dependencies, and the general package structure.

CQEDToolbox builds lab-specific measurement protocols on top of `labcore` and QCodes. `labcore` provides sweep framework, HDF5 data storage, and fitting tools, while CQEDToolbox connects these tools to real circuit-QED experiments.

---

# Folder Structure

This folder contains fluxonium-specific operations:

```text
fluxonium/
    res_spec_vs_flux.py
    fluxonium_pi_spec.py
    fluxonium_power_rabi.py
```

Coherence measurements are not duplicated here. After the fluxonium-specific tuning steps, use the parallel folder:

```text
single_qubit/
```

for:

```text
T1
T2 Ramsey
T2 Echo
```

---

# Fluxonium Auto-Tuning Workflow

The intended workflow is:

```text
Resonator spectroscopy vs flux
        ↓
Flux offset / zero-flux calibration (Use ML model "https://github.com/Jianyaogu/Fluxonium-offset-inverse-model/tree/main" if method in resonator spec vs flux works bad)
        ↓
Fluxonium qubit / pi spectroscopy vs flux
        ↓
Fluxonium power Rabi vs flux
        ↓
T1 / T2 measurements
        ↓
Use protocols in single_qubit/
```

The main difference between fluxonium and ordinary single-qubit protocols is that fluxonium calibration usually includes an additional flux/current sweep dimension.

---

# General Protocol Structure

Each protocol follows the `ProtocolOperation` structure:

```text
measure
load
analyze
evaluate
```

For testing, dummy measurement methods may exist:

```python
_measure_dummy(...)
```

For real hardware execution, the production path should use QICK methods such as:

```python
_measure_qick(...)
```

Dummy code should be replaced by written QICK implementation following the style of the corresponding `single_qubit/` protocol, but with the extra flux/current sweep added where needed.

---

# Parameter Manager / Instrument Server

Each protocol registers its inputs and outputs through parameter classes:

```python
self._register_inputs(...)
self._register_outputs(...)
```

These are connected to the parameter manager in `instrumentserver`.

Before running a protocol, open the parameter manager in instrumentserver and make sure the required parameters exist and are set correctly.

Typical required parameters include:

```text
repetitions
readout/drive frequency
start / end flux
flux steps
frequency sweep range
frequency steps
fluxonium parameters: EC, EL, EJ, g, fr
zero-flux current
gain pulse duration
```

---

# How to Run a Protocol in Dummy Mode

Example structure:

```python
%load_ext autoreload
%autoreload 2

from cqedtoolbox.protocols import base
from cqedtoolbox.protocols.operations.fluxonium.fluxonium_power_rabi import FluxoniumPowerRabi

base.PLATFORMTYPE = base.PlatformTypes.DUMMY
```

Then connect to the parameter manager:

```python
from instrumentserver.client import Client

cli = Client(host="127.0.0.1", port=5555, timeout=100)

pm = cli.find_or_create_instrument("parameter_manager")

p = FluxoniumPowerRabi(params=pm)
```

Run:

```python
p.execute()
```

The same structure can be used for other fluxonium protocols by changing the imported protocol class.

---

# 1. `res_spec_vs_flux.py`

## Purpose

Resonator spectroscopy vs flux/current.

This is the first step in the fluxonium auto-tuning flow.

It sweeps readout frequency and flux/current, then extracts the resonator frequency as a function of flux.

## Inputs

Typical inputs:

```text
repetitions
start / end readout frequency
readout frequency steps
start / end flux
flux steps
fluxonium guess parameters: EC, EL, EJ
coupling g
bare resonator frequency fr
```

## Outputs

Typical outputs:

```text
resonator frequency vs flux
zero-flux current estimate
fit parameters
SNR / fit quality
figures
```

---

# 2. `fluxonium_pi_spec.py`

## Purpose

Fluxonium qubit / pi spectroscopy vs flux/current.

This protocol sweeps qubit drive frequency and flux/current to extract the qubit transition frequency as a function of flux.

## Inputs

Typical inputs:

```text
repetitions
readout frequency
start / end qubit drive frequency
qubit drive frequency steps
start / end flux
flux steps
drive pulse duration
fluxonium parameters: EC, EL, EJ
zero-flux current
gain multiplier
```

## Outputs

Typical outputs:

```text
qubit frequency vs flux
best-SNR component
SNR vs flux
Gaussian fit parameters
figures
```

## Gain Multiplier

A gain multiplier can be used to scale the physical qubit drive amplitude during spectroscopy.

Example:

```text
gain_multiplier = 2.0
```

means twice the applied drive amplitude is used (due to 1/2 will be disspated before driving qubit). This is useful because the theoretical drive amplitude may not match the experimentally required amplitude due to attenuation, line loss, and device-dependent coupling.

---

# 3. `fluxonium_power_rabi.py`

## Purpose

Fluxonium power Rabi vs flux/current.

This protocol sweeps qubit drive gain and flux/current to extract the pi-pulse gain as a function of flux.

## Inputs

Typical inputs:

```text
repetitions
readout frequency
qubit drive frequency
start / end flux
flux steps
start / end gain
gain steps
drive pulse duration
fluxonium parameters: EC, EL, EJ
zero-flux current
```

## Outputs

Typical outputs:

```text
pi-pulse gain vs flux
Rabi fit parameters
SNR / fit quality
figures
```

---

# 4. T1 / T2 Measurements

After resonator spectroscopy, qubit spectroscopy, and power Rabi calibration are complete, use the corresponding protocols in:

```text
single_qubit/
```

for:

```text
T1
T2 Ramsey
T2 Echo
```

These protocols are shared with other single-qubit workflows and do not need to be duplicated inside the fluxonium folder unless fluxonium-specific behavior is added later.

---
