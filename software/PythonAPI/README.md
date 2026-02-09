# Open Micro Stage Python API

A Python API for controlling the Open Micro Stage micro-manipulator.

## Installation

### Development Installation

```bash
pip install -e .
```

### Production Installation

```bash
pip install open_micro_stage
```

## Usage

```python
from open_micro_stage import OpenMicroStageInterface

# Create interface and connect
oms = OpenMicroStageInterface(show_communication=True, show_log_messages=True)
oms.connect('/dev/ttyACM0')

# Home device
oms.home()

# Move to position
oms.move_to(0, 0, 0, f=10)
oms.wait_for_stop()

# Read device state
oms.read_device_state_info()
```

See `examples/` directory for more usage examples.

## Requirements

- Python >= 3.8
- numpy
- pyserial
- colorama

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Quality

The project uses [Ruff](https://github.com/astral-sh/ruff) for:
- Code formatting
- Import sorting
- Linting (pycodestyle, Pyflakes, flake8-bugbear, flake8-comprehensions)

#### Format and lint code

```bash
ruff format .
ruff check --fix .
```

## License

See LICENSE file for details.

