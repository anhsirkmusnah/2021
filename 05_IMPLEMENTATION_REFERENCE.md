# Implementation Reference Guide

This document provides a detailed mapping between mathematical concepts and their code implementations, along with comprehensive parameter references and configuration options.

---

## Table of Contents

1. [Codebase Structure](#1-codebase-structure)
2. [Core Classes Reference](#2-core-classes-reference)
3. [Function Reference](#3-function-reference)
4. [Parameter Reference](#4-parameter-reference)
5. [Gate Encoding Specification](#5-gate-encoding-specification)
6. [Configuration Options](#6-configuration-options)
7. [File Format Specifications](#7-file-format-specifications)

---

## 1. Codebase Structure

### 1.1 Directory Layout

```
QML dataproc/
├── ITensor_C/                    # ITensor C++ backend
│   ├── helloitensor.cc          # C++ MPS simulation core
│   ├── qubit.h                  # Custom ITensor qubit site type
│   ├── main.py                  # Main execution script
│   ├── main_dlp.py              # Discrete log problem variant
│   ├── projected_kernel_ansatz.py  # Ansatz + kernel builder
│   ├── Makefile                 # Compilation rules
│   └── datasets/                # Data directory
│
├── QuantumLibs/                  # Python quantum simulation
│   ├── main.py                  # Execution entry point
│   ├── main_features.py         # Feature extraction mode
│   ├── main_memefficient.py     # Memory-optimized version
│   ├── projected_kernel_ansatz.py
│   ├── projected_quantum_features.py
│   ├── memefficient_pqk_ansatz.py
│   └── run_svm_largekernel.py   # Large-scale SVM
│
├── dataproc files/               # Production pipeline
│   ├── main.py                  # Orchestration script
│   ├── generate_pqf.py          # Feature generation
│   ├── projected_quantum_features.py
│   ├── train.py                 # Model training
│   ├── test.py                  # Model evaluation
│   └── helloitensor.cc          # Local ITensor build
│
├── Installation-Script/          # Deployment utilities
│   ├── Dockerfile               # Container definition
│   ├── install_bash.sh          # Unix installer
│   ├── install_windows.ps1      # Windows installer
│   ├── elliptic_preproc.py      # Data preprocessing
│   └── readme.md                # Installation guide
│
└── docs/                         # Documentation
    ├── 01_THEORETICAL_FOUNDATIONS.md
    ├── 02_PROJECTED_QUANTUM_KERNELS.md
    ├── 03_TENSOR_NETWORK_SIMULATION.md
    ├── 04_QMLOPS_PIPELINE.md
    ├── 05_IMPLEMENTATION_REFERENCE.md
    └── 06_FRAUD_DETECTION_APPLICATION.md
```

### 1.2 Module Dependency Graph

```
                    ┌─────────────────┐
                    │     main.py     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌─────────────┐ ┌──────────────┐
    │  mpi4py       │ │  pandas     │ │ sklearn      │
    │  (MPI_COMM)   │ │ (DataFrames)│ │ (SVM, Scale) │
    └───────────────┘ └─────────────┘ └──────────────┘
            │
            ▼
┌───────────────────────────────────────────────────┐
│         projected_kernel_ansatz.py                │
│  ┌─────────────────────────────────────────────┐ │
│  │    ProjectedKernelStateAnsatz               │ │
│  │    • __init__(): Build symbolic circuit     │ │
│  │    • circuit_for_data(): Substitute values  │ │
│  │    • circuit_to_list(): Serialize gates     │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │    build_kernel_matrix()                    │ │
│  │    • MPI parallelization                    │ │
│  │    • Round-robin distribution               │ │
│  │    • Checkpointing                          │ │
│  └─────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌─────────────┐ ┌──────────────┐
    │  pytket       │ │  sympy      │ │ helloitensor │
    │  (Circuits)   │ │ (Symbols)   │ │ (MPS sim)    │
    └───────────────┘ └─────────────┘ └──────────────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │   ITensor    │
                                      │   (C++ lib)  │
                                      └──────────────┘
```

---

## 2. Core Classes Reference

### 2.1 ProjectedKernelStateAnsatz

**Location:** `projected_kernel_ansatz.py`

**Purpose:** Creates and manages symbolic quantum circuits for data encoding.

```python
class ProjectedKernelStateAnsatz:
    """
    Creates parameterized quantum circuits for projected kernel computation.

    Attributes:
        ansatz_circ (Circuit): The pytket circuit with symbolic parameters
        feature_symbol_list (List[Symbol]): Symbols f_0, f_1, ..., f_{n-1}
        reps (int): Number of circuit layer repetitions
        gamma (float): Rotation scaling parameter
        num_features (int): Number of qubits/features
        hadamard_init (bool): Whether to apply initial Hadamard layer
        entanglement_map (List[Tuple[int, int]]): Qubit connectivity
    """
```

#### Constructor

```python
def __init__(
    self,
    num_features: int,       # Number of qubits = features
    reps: int,               # Layer repetitions
    gamma: float,            # Rotation scaling [0.1, 1.0]
    entanglement_map: List[Tuple[int, int]],  # Connectivity
    ansatz: str,             # "hamiltonian" or "magic"
    hadamard_init: bool = True  # Initial H layer
):
```

#### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `circuit_for_data` | `(feature_values: List[float]) -> Circuit` | Substitute symbols with data values |
| `circuit_to_list` | `(circuit: Circuit) -> List[List]` | Serialize circuit to gate list |
| `hamiltonian_ansatz` | `() -> int` | Build Rz + XXPhase circuit |
| `magic_ansatz` | `() -> int` | Build H + T + CZ circuit |

### 2.2 ProjectedQuantumFeatures

**Location:** `dataproc files/projected_quantum_features.py`

**Purpose:** Feature extraction mode (returns quantum features instead of kernel).

```python
class ProjectedQuantumFeatures:
    """
    Similar to ProjectedKernelStateAnsatz but optimized for
    direct feature extraction rather than kernel computation.
    """
```

### 2.3 C++ Circuit Simulator

**Location:** `helloitensor.cc`

**Purpose:** ITensor-based MPS simulation of quantum circuits.

```cpp
// Main simulation function
template<typename T1, typename T2>
std::vector<std::vector<T2>> circuit_xyz_exp(
    std::vector<std::tuple<T1,T1,T1,T2>> tensor_vec,  // Gate list
    int no_sites                                       // Number of qubits
);

// Returns: [[⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩], ..., [⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩]]
```

---

## 3. Function Reference

### 3.1 Data Handling Functions

#### `draw_sample`

```python
def draw_sample(
    df: pd.DataFrame,    # Full dataset
    ndmin: int,          # Minority class samples
    ndmaj: int,          # Majority class samples
    test_frac: float = 0.2,  # Test split fraction
    seed: int = 123      # Random seed
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample and split data with stratification.

    Returns:
        (train_features, train_labels, test_features, test_labels)
    """
```

#### `entanglement_graph`

```python
def entanglement_graph(
    nq: int,  # Number of qubits
    nn: int   # Nearest neighbor depth
) -> List[Tuple[int, int]]:
    """
    Generate entanglement connectivity map.

    Example:
        entanglement_graph(5, 1) -> [(0,1), (2,3), (1,2), (3,4)]
    """
```

### 3.2 Kernel Construction Functions

#### `build_kernel_matrix`

```python
def build_kernel_matrix(
    mpi_comm,                          # MPI communicator
    ansatz: ProjectedKernelStateAnsatz,
    X: np.ndarray,                     # Data matrix (N, d)
    Y: Optional[np.ndarray] = None,    # Optional second dataset
    alpha: float = 1,                  # Kernel bandwidth
    info_file: Optional[str] = None,   # Profiling output file
    cpu_max_mem: int = 6,              # Memory limit (GB)
    minutes_per_checkpoint: Optional[int] = None
) -> np.ndarray:
    """
    Construct kernel matrix using MPI parallelization.

    Returns:
        Kernel matrix of shape (len(Y) or len(X), len(X))

    Algorithm:
        1. Distribute X across processes
        2. Each process simulates its chunk's circuits
        3. Round-robin pass Y chunks to compute all K[i,j]
        4. MPI_Reduce to combine partial matrices
    """
```

#### `build_qf_matrix`

```python
def build_qf_matrix(
    mpi_comm,
    ansatz: ProjectedQuantumFeatures,
    X: np.ndarray,
    info_file: Optional[str] = None,
    cpu_max_mem: int = 6
) -> np.ndarray:
    """
    Extract quantum features (not kernel) from data.

    Returns:
        Feature matrix of shape (N, 3*n_qubits)
    """
```

### 3.3 Training Functions

#### `model_train`

```python
def model_train(
    features: np.ndarray,      # Feature matrix
    target_label: np.ndarray,  # Labels
    version: str = '0',        # Model version
    model_name: str = 'default'  # Model name prefix
) -> int:
    """
    Train LightGBM classifier and save model.

    Returns:
        0 on success
    """
```

#### `model_test`

```python
def model_test(
    model_filename: str,       # Path to saved model
    features: np.ndarray,      # Test features
    actual_label_with_index: np.ndarray  # Labels with indices
) -> Dict[str, Any]:
    """
    Evaluate model on test data.

    Returns:
        Dictionary with metrics: Accuracy, Precision, Recall, F1, AUC, etc.
    """
```

### 3.4 Preprocessing Functions

#### `apply_scaling`

```python
def apply_scaling(
    classical_features: np.ndarray,
    train_flag: bool
) -> np.ndarray:
    """
    Apply StandardScaler to features.

    If train_flag=True: Fit and save scaler
    If train_flag=False: Load saved scaler and transform

    Returns:
        Scaled feature array
    """
```

#### `generate_projectedQfeatures`

```python
def generate_projectedQfeatures(
    data_feature: pd.DataFrame,
    reps: int,              # Circuit repetitions
    gamma: float,           # Rotation scaling
    target_label: str,      # Label column name
    info: str,              # 'train' or 'test'
    slice_size: int = 50000,  # Batch size for large data
    train_flag: bool = False
) -> int:
    """
    Generate quantum features for entire dataset in batches.

    Returns:
        0 (features saved to disk)
    """
```

---

## 4. Parameter Reference

### 4.1 Quantum Circuit Parameters

| Parameter | Symbol | Type | Range | Default | Description |
|-----------|--------|------|-------|---------|-------------|
| `num_features` | $n$ | int | [2, 100+] | - | Number of qubits/features |
| `reps` | $r$ | int | [1, 20] | 2 | Circuit layer repetitions |
| `gamma` | $\gamma$ | float | (0, 1] | 1.0 | Rotation scaling factor |
| `alpha` | $\alpha$ | float | (0, 10] | 1.0 | Kernel bandwidth |
| `ansatz` | - | str | {"hamiltonian", "magic"} | "hamiltonian" | Circuit type |
| `hadamard_init` | - | bool | {True, False} | True | Initial H gates |

### 4.2 Data Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `n_illicit` | int | [1, N/2] | Fraud class sample size |
| `n_licit` | int | [1, N/2] | Legitimate class sample size |
| `data_seed` | int | Any | Random seed for reproducibility |
| `test_frac` | float | (0, 1) | Test set fraction (default 0.2) |

### 4.3 MPI Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_procs` | int | Number of MPI processes |
| `rank` | int | Current process rank [0, n_procs-1] |
| `root` | int | Root process (always 0) |
| `mpi_comm` | MPI.Comm | MPI communicator (COMM_WORLD) |

### 4.4 SVM Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `C` | float | [0.01, 2.0] | Regularization parameter |
| `kernel` | str | "precomputed" | Must be precomputed for PQK |
| `tol` | float | 1e-5 | Optimization tolerance |

### 4.5 LightGBM Parameters

```python
default_params = {
    'subsample': 1.0,           # Row sampling ratio
    'reg_lambda': 0.5,          # L2 regularization
    'reg_alpha': 0.1,           # L1 regularization
    'num_leaves': 40,           # Max leaves per tree
    'n_estimators': 300,        # Number of trees
    'min_child_samples': 30,    # Min samples per leaf
    'max_depth': 15,            # Max tree depth
    'learning_rate': 0.1,       # Step size
    'colsample_bytree': 0.8,    # Feature sampling ratio
}
```

### 4.6 Parameter Interaction Guidelines

| Scenario | Recommended Settings |
|----------|---------------------|
| Few features (n < 10) | γ=1.0, reps=2-3, α=0.5-1.0 |
| Many features (n > 20) | γ=0.3-0.5, reps=5-10, α=0.1-0.5 |
| Small data (N < 500) | Higher C (1.0-2.0), α=0.5-1.0 |
| Large data (N > 5000) | Lower C (0.1-0.5), α=0.1-0.3 |
| High entanglement needed | reps=5-10, nn=2-3 |
| Fast simulation | reps=1-2, nn=1 |

---

## 5. Gate Encoding Specification

### 5.1 Gate Code Table

| Code | Gate | Qubits | Parameter | ITensor Operator |
|------|------|--------|-----------|------------------|
| 0 | H | 1 | None | `"H"` |
| 1 | Rx | 1 | angle | `"Rx"` with `alpha=angle` |
| 2 | Rz | 1 | angle | `"Rz"` with `alpha=angle` |
| 3 | XXPhase | 2 | angle | `expHermitian(X⊗X, -i*θ)` |
| 4 | ZZPhase | 2 | angle | `expHermitian(Z⊗Z, -i*θ)` |
| 5 | SWAP | 2 | None | Manual matrix |
| 6 | T | 1 | None | `"T"` |
| 7 | CZ | 2 | None | Manual matrix |

### 5.2 Gate List Format

```python
# Each gate is a list: [code, qubit1, qubit2, parameter]
# qubit2 = -1 for single-qubit gates

example_circuit = [
    [0, 0, -1, 0],      # H on qubit 0
    [0, 1, -1, 0],      # H on qubit 1
    [2, 0, -1, 0.5],    # Rz(0.5) on qubit 0
    [2, 1, -1, 0.3],    # Rz(0.3) on qubit 1
    [3, 0, 1, 0.25],    # XXPhase(0.25) on qubits 0,1
]
```

### 5.3 Circuit Serialization

```python
def circuit_to_list(self, circuit):
    """Convert pytket Circuit to gate list for ITensor."""
    gates = []
    for gate in circuit.get_commands():
        if gate.op.type == OpType.H:
            gates.append([0, gate.qubits[0].index[0], -1, 0])
        elif gate.op.type == OpType.Rz:
            gates.append([2, gate.qubits[0].index[0], -1, gate.op.params[0]])
        elif gate.op.type == OpType.XXPhase:
            gates.append([3, gate.qubits[0].index[0],
                         gate.qubits[1].index[0], gate.op.params[0]])
        # ... other gates
    return gates
```

---

## 6. Configuration Options

### 6.1 Environment Variables (Docker)

```bash
# MPI Configuration
NUM_NODES=4                  # Number of MPI processes
OMPI_ALLOW_RUN_AS_ROOT=1     # Allow root execution

# OpenMP Configuration
OMP_NUM_THREADS=24           # Threads per process
OMP_PROC_BIND=close          # Thread affinity
OMP_PLACES=cores             # Bind to physical cores

# ITensor Configuration
ITENSOR_USE_OMP=1            # Enable OpenMP in ITensor
MKL_NUM_THREADS=4            # MKL thread count
OPENBLAS_NUM_THREADS=4       # OpenBLAS thread count

# Memory Optimization
OMP_STACKSIZE=2M             # Per-thread stack size
MALLOC_MMAP_THRESHOLD_=131072
```

### 6.2 Compilation Flags

**Linux (GCC):**
```bash
CXXFLAGS="-m64 -std=c++17 -fconcepts -fPIC -O2 -DNDEBUG -Wall"
LDFLAGS="-litensor -lpthread -lblas -llapack"
```

**macOS (Clang):**
```bash
CXXFLAGS="-std=c++17 -fPIC -Wno-gcc-compat -O2 -DNDEBUG"
LDFLAGS="-litensor -framework Accelerate"
```

### 6.3 ITensor Configuration

**options.mk (Linux):**
```makefile
PLATFORM = lapack
BLAS_LAPACK_LIBFLAGS = -lopenblas -llapack
```

**options.mk (macOS):**
```makefile
PLATFORM = macos
BLAS_LAPACK_LIBFLAGS = -framework Accelerate
```

### 6.4 MPI Runtime Options

```bash
# Basic execution
mpirun -n 4 python main.py [args]

# With binding
mpirun -n 4 --bind-to core python main.py [args]

# With hostfile
mpirun -n 16 --hostfile hosts.txt python main.py [args]

# OpenMPI options for large jobs
mpirun -n 96 \
    --mca btl_openib_if_include ib0 \
    --mca btl_tcp_if_include eth0 \
    python main.py [args]
```

---

## 7. File Format Specifications

### 7.1 Input Data Format

**CSV Structure:**
```csv
Class,Feature 1,Feature 2,...,Feature N
0,1.234,5.678,...,9.012
1,3.456,7.890,...,1.234
...
```

- First column: Class label (0 = fraud, 1 = legitimate)
- Remaining columns: Numerical features
- No header row for elliptic dataset (added programmatically)

### 7.2 Kernel Matrix Format (NumPy)

```python
# Training kernel
# Shape: (N_train, N_train)
# dtype: float64
kernel_train = np.load("kernels/TrainKernel_Nf-12_r-10_g-1_Ntr-100.npy")

# Test kernel
# Shape: (N_test, N_train)
kernel_test = np.load("kernels/TestKernel_Nf-12_r-10_g-1_Ntr-100.npy")
```

**Filename Convention:**
```
{Type}Kernel_Nf-{num_features}_r-{reps}_g-{gamma}_Ntr-{n_train}.npy
```

### 7.3 Profiling JSON Format

```json
{
    "lenX": [800, "entries"],
    "lenY": [200, "entries"],
    "r0_circ_gen": [1.23, "seconds"],
    "r0_circ_sim": [45.67, "seconds"],
    "avg_circ_sim": [0.0571, "seconds"],
    "median_circ_sim": [0.0543, "seconds"],
    "kernel_mat_time": [120.5, "seconds"],
    "total_time": [180.3, "seconds"],
    "r0_product": [30.2, "seconds"],
    "avg_product": [0.0012, "seconds"],
    "median_product": [0.0011, "seconds"],
    "r_nonRR_recv": [5.2, "seconds"],
    "r0_RR_recv": [15.8, "seconds"]
}
```

### 7.4 Model Pickle Format

```python
# Model file: ./model/qiml_fraud_model_v0.pkl
# Contains: sklearn SVC or LightGBM classifier

# Scaler file: ./model/scaler.pkl
# Contains: sklearn StandardScaler fitted on training data

# Load example:
with open('./model/qiml_fraud_model_v0.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### 7.5 Results CSV Format

```csv
Actual_Label,QML_prob,Light_GBM_prob,Value
0,0.92,0.85,1500.00
1,0.15,0.22,250.00
...
```

| Column | Description |
|--------|-------------|
| Actual_Label | Ground truth (0=fraud, 1=legitimate) |
| QML_prob | Quantum model fraud probability |
| Light_GBM_prob | Classical model fraud probability |
| Value | Transaction amount |

---

## Quick Reference Card

### Command Line Usage

```bash
# Kernel mode (SVM)
mpirun -n <P> python main.py <n> <r> <γ> <α> <n_fraud> <n_legit> <seed> <file>

# Feature mode (LightGBM)
mpirun -n <P> python main.py train <train.csv> <test.csv> <label_col> True
mpirun -n <P> python main.py test <train.csv> <test.csv> <label_col> False
mpirun -n <P> python main.py generate <train.csv> <test.csv> <label_col> <flag>

# Docker
docker build -t qiml .
docker run --env NUM_NODES=4 --env NUM_FEATURES=12 ... qiml
```

### Key Files by Function

| Function | Primary File | Secondary Files |
|----------|-------------|-----------------|
| Circuit construction | `projected_kernel_ansatz.py` | `pytket` |
| MPS simulation | `helloitensor.cc` | ITensor library |
| Kernel building | `projected_kernel_ansatz.py` | `mpi4py` |
| Feature extraction | `generate_pqf.py` | `projected_quantum_features.py` |
| Model training | `train.py` | `lightgbm` |
| Model evaluation | `test.py` | `sklearn.metrics` |
| Data preprocessing | `elliptic_preproc.py` | `pandas` |

---

*Next: [06_FRAUD_DETECTION_APPLICATION.md](06_FRAUD_DETECTION_APPLICATION.md) - Domain-specific fraud detection documentation*
