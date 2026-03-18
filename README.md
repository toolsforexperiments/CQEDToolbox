# CQEDToolbox

The Pfafflab's circuit QED measurement workflows, open-sourced.

CQEDToolbox implements lab-specific measurement protocols, instrument drivers, and analysis routines on top of [labcore](https://github.com/toolsforexperiments/labcore) and [QCodes](https://qcodes.github.io/Qcodes/). labcore provides the building blocks (sweep framework, data storage, fitting tools); CQEDToolbox is how we wire them together to run real experiments. You're welcome to use it as-is or as a starting point for your own lab's workflow.

---

## What's inside

### Measurement protocols

Ready-to-run experiments for single-qubit characterization, organized by qubit type:

**Transmon**
- Resonator spectroscopy (bare and dispersive, vs. readout gain)
- Qubit spectroscopy (saturation, π-pulse)
- T1, T2 Ramsey, T2 Echo
- Power Rabi
- Readout calibration
- Single-qubit tuneup sequences: AllXY, Pauli bars, pulse trains, interleaved randomized benchmarking (IRB)

**Fluxonium**
- Resonator spectroscopy vs. flux
- Qubit spectroscopy, power Rabi

### Control electronics

- **OPX** (Quantum Machines) — config generation, mixer calibration, sweep integration
- **QICK** (Xilinx RFSoC) — config and sweep support

### QCodes instrument drivers

| Instrument | Driver |
|---|---|
| SignalCore SC5511A / SC5521A | RF signal generators |
| Oxford Instruments Triton | Dilution refrigerator |
| Yokogawa GS200 | DC voltage/current source |
| Keysight N9030B / P937A | Spectrum analyzer / VNA |
| SignalHound Spike | Spectrum analyzer |
| ThorLabs TSP-01B | Temperature/humidity sensor |

### Fit functions

Resonator fit models built on top of labcore's lmfit-based fitting framework.

### Readout analysis

Qubit readout discrimination and calibration utilities.

---

## Installation

CQEDToolbox is not yet on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/toolsforexperiments/CQEDToolbox.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/toolsforexperiments/CQEDToolbox.git
pip install -e CQEDToolbox/
```

> **Note:** Some dependencies (`qick`, `qm-qua`) require hardware-specific installations and may not resolve cleanly on all platforms. See the dependency notes in `pyproject.toml`.

Requires Python ≥ 3.11.

---

## Relationship to labcore

CQEDToolbox depends on [labcore](https://github.com/toolsforexperiments/labcore), which provides:
- The sweep framework (`Sweep`, `sweep_parameter`, `@recording`)
- HDF5 data storage (`DDH5Writer`, `load_as_xr`)
- The base fitting and analysis infrastructure

If you are not from a CQED lab, labcore alone may be all you need. CQEDToolbox adds the physics-specific layer on top.

---

## Development

```bash
git clone https://github.com/toolsforexperiments/CQEDToolbox.git
cd CQEDToolbox
pip install -e ".[dev]"
pytest
```

Code quality:

```bash
ruff check --fix .
ruff format .
mypy .
```

---

## License

MIT. See [LICENSE](LICENSE) for details.