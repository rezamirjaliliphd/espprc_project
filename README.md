# ESPPRC Project

A high-performance solver for the **Elementary Shortest Path Problem with Resource Constraints (ESPPRC)** using a bidirectional label-setting algorithm. This solver is implemented in Python with Cython-accelerated routines for speed-critical operations like dominance checking and path concatenation.

## 🚀 Features

- Bidirectional label-setting framework
- Cython-accelerated dominance checks and label concatenation
- Efficient pruning of labels and upper bound estimation
- Easy integration into branch-and-price frameworks

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/rezamirjaliliphd/espprc_project.git
cd espprc_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the Cython extension

```bash
python setup.py build_ext --inplace
```

Or install in editable mode with modern tooling:

```bash
pip install -e .
```

## 📦 Requirements

- Python 3.8+
- NumPy
- Cython

## 🧪 Testing

We use `pytest`. You don't need any external dataset — test instances are generated on-the-fly:

```bash
pytest
```

## 💡 Usage Example

```python
from espprc.espprc import ESPPRC
import numpy as np

# Generate dummy instance
n, n_res = 5, 2
r = np.random.rand(n, n, n_res + 1) * 10
r_max = np.array([15.0] * n_res)

esp = ESPPRC(r_max=r_max, r=r)
esp.solve()

print("Best Path(s):", esp.best_path)
print("Best Cost(s):", esp.best_cost)
```

## 📚 Citation

If you use this project in your research, please cite:

> Mirjalili, Reza. *ESPPRC Solver with Cython Acceleration for Large-Scale Column Generation.* 2025.

## 📄 License

This project is licensed under the MIT License.
