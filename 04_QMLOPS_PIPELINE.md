# QMLOps Pipeline: End-to-End Workflow

This document provides a comprehensive overview of the Quantum-Inspired Machine Learning Operations (QMLOps) pipeline, from raw data ingestion to production model deployment.

---

## Table of Contents

1. [Pipeline Architecture Overview](#1-pipeline-architecture-overview)
2. [Stage 1: Data Ingestion & Preprocessing](#2-stage-1-data-ingestion--preprocessing)
3. [Stage 2: Quantum Feature Engineering](#3-stage-2-quantum-feature-engineering)
4. [Stage 3: Kernel Matrix Construction](#4-stage-3-kernel-matrix-construction)
5. [Stage 4: Model Training](#5-stage-4-model-training)
6. [Stage 5: Model Evaluation & Inference](#6-stage-5-model-evaluation--inference)
7. [Parallelization Strategy](#7-parallelization-strategy)
8. [Pipeline Orchestration](#8-pipeline-orchestration)
9. [Production Deployment](#9-production-deployment)

---

## 1. Pipeline Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QiML PIPELINE ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │   RAW DATA   │    │ PREPROCESSED │    │   QUANTUM    │    │   KERNEL     │
 │              │───▶│    DATA      │───▶│   FEATURES   │───▶│   MATRIX     │
 │  (CSV/DB)    │    │  (Scaled)    │    │   (3n-dim)   │    │   (NxN)      │
 └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                     │
                                                                     ▼
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │  PRODUCTION  │    │   TRAINED    │    │    MODEL     │    │     SVM      │
 │  INFERENCE   │◀───│    MODEL     │◀───│   TRAINING   │◀───│   FITTING    │
 │              │    │   (.pkl)     │    │              │    │              │
 └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 1.2 Pipeline Variants

The codebase supports two operational modes:

| Mode | Purpose | Primary Output | Key Files |
|------|---------|----------------|-----------|
| **Kernel Mode** | SVM with precomputed PQK | Kernel matrices + SVM | `main.py`, `main_dlp.py` |
| **Feature Mode** | LightGBM with quantum features | Feature vectors + GBDT | `dataproc files/main.py` |

### 1.3 Component Interaction

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PYTHON LAYER                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │     pytket     │  │   mpi4py       │  │  scikit-learn  │        │
│  │   (Circuits)   │  │ (Parallelism)  │  │    (SVM)       │        │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘        │
│          │                   │                   │                  │
│  ┌───────▼───────────────────▼───────────────────▼────────┐        │
│  │            projected_kernel_ansatz.py                   │        │
│  │   ProjectedKernelStateAnsatz + build_kernel_matrix()   │        │
│  └───────────────────────────┬────────────────────────────┘        │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ pybind11
┌──────────────────────────────▼──────────────────────────────────────┐
│                          C++ LAYER                                  │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │                   helloitensor.cc                        │       │
│  │   ITensor MPS simulation + circuit_xyz_exp()            │       │
│  └─────────────────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │                   ITensor Library                        │       │
│  │         MPS, SVD, Tensor Operations, Qubit Sites        │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage 1: Data Ingestion & Preprocessing

### 2.1 Data Flow

```
Raw Data (CSV)
     │
     ▼
┌─────────────────────────────────────────┐
│         DATA INGESTION                  │
│  • Load CSV with pandas                 │
│  • Handle missing values                │
│  • Filter unknown labels                │
│  • Class label encoding                 │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         CLASS BALANCING                 │
│  • Stratified sampling                  │
│  • n_illicit samples from class 0       │
│  • n_licit samples from class 1         │
│  • Reproducible via data_seed           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         TRAIN/TEST SPLIT                │
│  • Stratified split (default 80/20)     │
│  • Maintains class distribution         │
│  • Deterministic via seed               │
└─────────────────────────────────────────┘
     │
     ▼
Preprocessed DataFrames
```

### 2.2 Implementation

```python
def draw_sample(df, ndmin, ndmaj, test_frac=0.2, seed=123):
    """
    Sample and split data maintaining class balance.

    Args:
        df: Full dataset with 'Class' column
        ndmin: Number of minority class samples (fraud)
        ndmaj: Number of majority class samples (legitimate)
        test_frac: Fraction for test set
        seed: Random seed for reproducibility
    """
    # Stratified sampling from each class
    data_reduced = pd.concat([
        df[df['Class']==0].sample(ndmin, random_state=seed*20+2),
        df[df['Class']==1].sample(ndmaj, random_state=seed*46+9)
    ], axis=0)

    # Stratified train/test split
    train_df, test_df = train_test_split(
        data_reduced,
        stratify=data_reduced['Class'],
        test_size=test_frac,
        random_state=seed*26+19
    )

    return (np.array(train_df.drop('Class', axis=1)),
            np.array(train_df['Class'], dtype='int'),
            np.array(test_df.drop('Class', axis=1)),
            np.array(test_df['Class'], dtype='int'))
```

### 2.3 Feature Scaling Pipeline

```
Raw Features
     │
     ▼
┌─────────────────────────────────────────┐
│     QUANTILE TRANSFORM                  │
│  • Maps to Gaussian distribution        │
│  • Handles outliers and skewed data     │
│  output_distribution='normal'           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│     STANDARD SCALER                     │
│  • Zero mean, unit variance             │
│  • x' = (x - μ) / σ                     │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│     MINMAX SCALER                       │
│  • Scale to [-π, π] or [-π/4, π/4]      │
│  • Matches quantum rotation range       │
└─────────────────────────────────────────┘
     │
     ▼
Scaled Features (ready for quantum encoding)
```

### 2.4 Mathematical Transformations

**Step 1: Quantile Transform**
$$x' = \Phi^{-1}(F(x))$$
where $F$ is the empirical CDF and $\Phi^{-1}$ is the inverse normal CDF.

**Step 2: Standardization**
$$x'' = \frac{x' - \mu}{\sigma}$$

**Step 3: MinMax Scaling**
$$x''' = a + \frac{(x'' - \min(x''))(\cdot)(b - a)}{\max(x'') - \min(x'')}$$
where $[a, b] = [-\pi, \pi]$ or $[-\pi/4, \pi/4]$.

### 2.5 Feature Selection

```python
# Select top num_features columns
reduced_train_features = train_features[:, 0:num_features]
reduced_test_features = test_features[:, 0:num_features]
```

**Rationale:** Number of qubits = Number of features. Selecting top-k features:
- Reduces circuit complexity
- Features typically ordered by importance in preprocessed datasets
- Allows scaling experiments by varying `num_features`

---

## 3. Stage 2: Quantum Feature Engineering

### 3.1 Circuit Construction

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CIRCUIT CONSTRUCTION FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

Classical Feature Vector x = [x₁, x₂, ..., xₙ]
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CREATE SYMBOLIC CIRCUIT (pytket)                       │
│                                                                     │
│    Symbols: f_0, f_1, ..., f_{n-1}                                 │
│                                                                     │
│    Layer Structure (Hamiltonian Ansatz):                           │
│    ┌───┐                                                           │
│    │ H │ on all qubits (initialization)                            │
│    └───┘                                                           │
│         │                                                           │
│         ▼  × r repetitions                                         │
│    ┌──────────────┐                                                │
│    │ Rz(γ·fᵢ/π)  │ on each qubit i                                │
│    └──────────────┘                                                │
│    ┌──────────────────────────────┐                                │
│    │ XXPhase(γ²(1-fᵢ)(1-fⱼ))    │ on entangled pairs (i,j)       │
│    └──────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SYMBOL SUBSTITUTION                                    │
│                                                                     │
│    symbol_map = {f_0: x₁, f_1: x₂, ..., f_{n-1}: xₙ}              │
│    circuit.symbol_substitution(symbol_map)                         │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CIRCUIT COMPILATION (pytket)                           │
│                                                                     │
│    • Map to linear architecture                                    │
│    • Decompose BRIDGE gates (for non-adjacent qubits)              │
│    • Insert SWAPs as needed                                        │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SERIALIZE TO GATE LIST                                 │
│                                                                     │
│    Output: [(0, 0, -1, 0),     # H on qubit 0                      │
│             (0, 1, -1, 0),     # H on qubit 1                      │
│             (2, 0, -1, θ₀),   # Rz on qubit 0                      │
│             (3, 0, 1, θ₀₁),   # XXPhase on qubits 0,1              │
│             ...]                                                    │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    Gate List for ITensor
```

### 3.2 Entanglement Graph Construction

```python
def entanglement_graph(nq, nn):
    """
    Generate nearest-neighbor entanglement map.

    Args:
        nq: Number of qubits
        nn: Neighborhood depth (1 = immediate neighbors only)

    Returns:
        List of qubit pairs [(i, j), ...]
    """
    map = []
    for d in range(1, nn+1):
        busy = set()
        # First layer: non-overlapping pairs
        for i in range(nq):
            if i not in busy and i+d < nq:
                map.append((i, i+d))
                busy.add(i+d)
        # Second layer: remaining pairs
        for i in busy:
            if i+d < nq:
                map.append((i, i+d))
    return map
```

**Example for n=5, nn=1:**
```
Qubits:  0 --- 1 --- 2 --- 3 --- 4
Pairs:   [(0,1), (2,3), (1,2), (3,4)]
         └─Layer 1─┘  └─Layer 2─┘
```

### 3.3 MPS Simulation

```
Gate List
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ITensor MPS SIMULATION (C++)                           │
│                                                                     │
│    1. Initialize |0⟩^⊗n as MPS with χ=1                            │
│    2. For each gate in list:                                       │
│       • Position orthogonality center                              │
│       • Apply gate (contract + SVD for 2-qubit)                    │
│       • Truncate to cutoff (default 1E-10)                         │
│    3. Final state: |ψ(x)⟩ as MPS                                   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
MPS State |ψ(x)⟩
```

### 3.4 Expectation Value Extraction

```
MPS State |ψ(x)⟩
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              EXTRACT PAULI EXPECTATION VALUES                       │
│                                                                     │
│    For each qubit i = 1, ..., n:                                   │
│        ⟨X_i⟩ = ⟨ψ|X_i|ψ⟩                                           │
│        ⟨Y_i⟩ = ⟨ψ|Y_i|ψ⟩                                           │
│        ⟨Z_i⟩ = ⟨ψ|Z_i|ψ⟩                                           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
Quantum Feature Vector Φ(x) = [⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ..., ⟨Xₙ⟩, ⟨Yₙ⟩, ⟨Zₙ⟩]
                              └────────────────────────────────────────┘
                                           3n dimensions
```

---

## 4. Stage 3: Kernel Matrix Construction

### 4.1 Kernel Matrix Types

| Matrix | Dimensions | Formula | Use |
|--------|------------|---------|-----|
| Training Kernel | $N_{train} \times N_{train}$ | $K_{ij} = k(x_i, x_j)$ | SVM fitting |
| Test Kernel | $N_{test} \times N_{train}$ | $K_{ij} = k(x'_i, x_j)$ | SVM prediction |

### 4.2 Kernel Entry Computation

```
For data points x, x':
    │
    ├─► Simulate circuit U(x)  → Get Φ(x)
    │
    ├─► Simulate circuit U(x') → Get Φ(x')
    │
    └─► Compute kernel:

        D = Σᵢ 2·[(⟨Xᵢ⟩ˣ - ⟨Xᵢ⟩ˣ')² + (⟨Yᵢ⟩ˣ - ⟨Yᵢ⟩ˣ')² + (⟨Zᵢ⟩ˣ - ⟨Zᵢ⟩ˣ')²]

        k(x, x') = exp(-α · D)
```

### 4.3 MPI Parallelization Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLEL KERNEL CONSTRUCTION                     │
└─────────────────────────────────────────────────────────────────────┘

Data X = [x₁, x₂, ..., xₙ] distributed across P processes:

Process 0: [x₁, ..., x_{N/P}]         → Chunk X₀
Process 1: [x_{N/P+1}, ..., x_{2N/P}] → Chunk X₁
...
Process P-1: [x_{(P-1)N/P+1}, ..., xₙ] → Chunk X_{P-1}

Each process:
1. Simulates circuits for its X chunk
2. Stores quantum features locally

Kernel Matrix Tiling:
┌─────┬─────┬─────┬─────┐
│ 0,0 │ 0,1 │ 0,2 │ 0,3 │  ← Process 0 computes column 0
├─────┼─────┼─────┼─────┤
│ 1,0 │ 1,1 │ 1,2 │ 1,3 │  ← Process 1 computes column 1
├─────┼─────┼─────┼─────┤
│ 2,0 │ 2,1 │ 2,2 │ 2,3 │  ← Process 2 computes column 2
├─────┼─────┼─────┼─────┤
│ 3,0 │ 3,1 │ 3,2 │ 3,3 │  ← Process 3 computes column 3
└─────┴─────┴─────┴─────┘

Round Robin Y-chunk passing:
Iteration 1: Process i has Y chunk i
Iteration 2: Process i has Y chunk (i+1) mod P
...
```

### 4.4 Symmetry Exploitation

For training kernel (X = Y):
$$K_{ij} = K_{ji}$$

Only compute upper triangle and copy:
```python
if Y is None:  # X == Y case
    if not (this_iteration == 0 or  # Skip block diagonal
            x_chunks % 2 == 0 and this_iteration == iterations - 1):
        kernel_mat[x_index, y_index] = kernel_entry  # Symmetry copy
```

### 4.5 Checkpointing

Long-running computations save periodic checkpoints:
```python
if minutes_per_checkpoint is not None:
    if last_checkpoint_time + 60*minutes_per_checkpoint < MPI.Wtime():
        np.save(checkpoint_file, kernel_mat)
        last_checkpoint_time = MPI.Wtime()
```

---

## 5. Stage 4: Model Training

### 5.1 SVM with Precomputed Kernel

```python
from sklearn.svm import SVC

# Kernel mode: Use precomputed kernel matrix
svc = SVC(
    kernel="precomputed",  # Use our PQK kernel matrix
    C=regularization,      # Misclassification penalty
    tol=1e-5,              # Optimization tolerance
    verbose=False
)

svc.fit(kernel_train, train_labels)
```

### 5.2 Feature Mode: LightGBM

```python
import lightgbm as lgb

# Feature mode: Use quantum features directly
params = {
    'subsample': 1.0,
    'reg_lambda': 0.5,
    'reg_alpha': 0.1,
    'num_leaves': 40,
    'n_estimators': 300,
    'min_child_samples': 30,
    'max_depth': 15,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
}

model = lgb.LGBMClassifier(**params)
model.fit(combined_features, train_labels)
```

### 5.3 Hyperparameter Grid Search

```
For SVM regularization C:
    C ∈ [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]

For each C:
    1. Fit SVC on training kernel
    2. Predict on test kernel
    3. Compute metrics (accuracy, precision, recall, AUC)
    4. Store results

Select C with best validation performance
```

### 5.4 Model Persistence

```python
import pickle

def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

# Save trained model
save_model(model, f"./model/{model_name}{version}.pkl")

# Save scaler for inference
with open('./model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

---

## 6. Stage 5: Model Evaluation & Inference

### 6.1 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | $(TP + TN) / N$ | Overall correctness |
| Precision | $TP / (TP + FP)$ | Fraud detection reliability |
| Recall | $TP / (TP + FN)$ | Fraud capture rate |
| F1 Score | $2 \cdot \frac{P \cdot R}{P + R}$ | Balanced metric |
| AUC-ROC | Area under ROC curve | Discrimination ability |

### 6.2 Confusion Matrix Analysis

```
                    Predicted
                  Fraud | Legit
              ┌─────────┬─────────┐
Actual Fraud  │   TP    │   FN    │  → Recall = TP/(TP+FN)
              ├─────────┼─────────┤
Actual Legit  │   FP    │   TN    │  → Specificity = TN/(TN+FP)
              └─────────┴─────────┘
                  ↓
              Precision = TP/(TP+FP)
```

### 6.3 Inference Pipeline

```
New Transaction x_new
        │
        ▼
┌───────────────────────────────────────┐
│         PREPROCESSING                 │
│  • Load saved scaler                  │
│  • Apply same transformations         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    QUANTUM FEATURE EXTRACTION         │
│  • Build circuit U(x_new)             │
│  • Simulate with MPS                  │
│  • Extract Φ(x_new)                   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    KERNEL/FEATURE COMPUTATION         │
│                                       │
│  Kernel Mode:                         │
│    Compute k(x_new, x_train) for all  │
│    training points (support vectors)  │
│                                       │
│  Feature Mode:                        │
│    Use Φ(x_new) directly              │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│         MODEL PREDICTION              │
│  • Load saved model                   │
│  • Predict label and probability      │
└───────────────────────────────────────┘
        │
        ▼
Prediction: Fraud/Legit + Confidence Score
```

---

## 7. Parallelization Strategy

### 7.1 MPI Communication Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ROUND ROBIN COMMUNICATION                        │
└─────────────────────────────────────────────────────────────────────┘

Iteration 0:                    Iteration 1:
┌─────┐  ┌─────┐  ┌─────┐      ┌─────┐  ┌─────┐  ┌─────┐
│ P0  │  │ P1  │  │ P2  │      │ P0  │  │ P1  │  │ P2  │
│ Y0  │  │ Y1  │  │ Y2  │  →   │ Y2  │  │ Y0  │  │ Y1  │
└─────┘  └─────┘  └─────┘      └─────┘  └─────┘  └─────┘
   ↓        ↓        ↓            ↓        ↓        ↓
Compute  Compute  Compute      Compute  Compute  Compute
K[0,0]   K[1,1]   K[2,2]       K[0,2]   K[1,0]   K[2,1]

Final: MPI_Reduce to sum all partial kernel matrices
```

### 7.2 Load Balancing

```python
entries_per_chunk = int(np.ceil(len(X) / n_procs))

# Handle uneven distribution
if rank == n_procs - 1:
    # Last process may have fewer entries
    actual_entries = len(X) - (rank * entries_per_chunk)
else:
    actual_entries = entries_per_chunk
```

### 7.3 Scalability Characteristics

| Data Size | Processes | Kernel Entries | Time Complexity |
|-----------|-----------|----------------|-----------------|
| N | P | N²/2 (symmetric) | O(N²/P) |
| 100 | 4 | 5,050 | ~1 min |
| 500 | 8 | 125,250 | ~15 min |
| 1000 | 16 | 500,500 | ~1 hour |

---

## 8. Pipeline Orchestration

### 8.1 Command-Line Interface

**Kernel Mode:**
```bash
mpirun -n <nodes> python main.py \
    <num_features> \
    <reps> \
    <gamma> \
    <alpha> \
    <n_illicit> \
    <n_licit> \
    <data_seed> \
    <data_file>
```

**Feature Mode:**
```bash
mpirun -n <nodes> python main.py \
    <method: train|test|generate> \
    <train_data_info> \
    <test_data_info> \
    <target_label> \
    <train_flag: True|False>
```

### 8.2 Docker Orchestration

```yaml
# Environment variables for containerized execution
ENV NUM_NODES=4
ENV NUM_FEATURES=12
ENV REPS=10
ENV GAMMA=1
ENV ALPHA=0.5
ENV N_ILLICIT=100
ENV N_LICIT=100
ENV DATA_SEED=456
ENV DATA_FILE=/opt/datasets/bitstrings_20_preproc.csv

CMD mpirun -n ${NUM_NODES} python3 main_dlp.py \
    ${NUM_FEATURES} ${REPS} ${GAMMA} ${ALPHA} \
    ${N_ILLICIT} ${N_LICIT} ${DATA_SEED} ${DATA_FILE}
```

### 8.3 Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Training Kernel | `kernels/TrainKernel_*.npy` | N×N kernel matrix |
| Test Kernel | `kernels/TestKernel_*.npy` | M×N kernel matrix |
| Profiling Data | `*.json` | Performance metrics |
| Checkpoints | `tmp/checkpoint_*.npy` | Recovery files |
| Trained Model | `model/*.pkl` | Serialized classifier |
| Feature Arrays | `pqf_arr/*.npy` | Quantum features |
| Results | `Result_*.csv` | Predictions and metrics |

---

## 9. Production Deployment

### 9.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────────────────────────────────────┐
│   Client    │────▶│              API Gateway                    │
│  (REST/gRPC)│     │  • Rate limiting                            │
└─────────────┘     │  • Authentication                           │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │           Inference Service                  │
                    │  ┌─────────────────────────────────────────┐│
                    │  │         Feature Extraction               ││
                    │  │  • Load scaler (model/scaler.pkl)       ││
                    │  │  • Preprocess input                      ││
                    │  │  • Run MPS simulation                    ││
                    │  └─────────────────────────────────────────┘│
                    │  ┌─────────────────────────────────────────┐│
                    │  │         Model Inference                  ││
                    │  │  • Load model (model/*.pkl)             ││
                    │  │  • Compute kernel/features               ││
                    │  │  • Generate prediction                   ││
                    │  └─────────────────────────────────────────┘│
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │           Response                           │
                    │  {                                           │
                    │    "prediction": "fraud" | "legitimate",    │
                    │    "confidence": 0.95,                      │
                    │    "quantum_features": [...],               │
                    │    "processing_time_ms": 150                │
                    │  }                                           │
                    └─────────────────────────────────────────────┘
```

### 9.2 Inference Latency Considerations

| Component | Typical Latency | Optimization |
|-----------|-----------------|--------------|
| Preprocessing | 1-5 ms | Vectorized ops |
| Circuit Simulation | 50-200 ms | Pre-compiled circuits |
| Kernel Computation | 10-50 ms | Cached support vectors |
| Model Prediction | 1-10 ms | Native code |
| **Total** | **~100-300 ms** | Batch processing |

### 9.3 Batch Processing for Throughput

For high-throughput scenarios:
```python
def batch_predict(transactions: List[np.ndarray], batch_size: int = 100):
    """Process transactions in batches for efficiency."""
    results = []
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i+batch_size]

        # Parallel feature extraction
        features = parallel_extract_features(batch)

        # Batch kernel computation
        kernel_batch = compute_kernel_batch(features, support_vectors)

        # Batch prediction
        predictions = model.predict(kernel_batch)
        results.extend(predictions)

    return results
```

---

## Pipeline Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE QiML PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

TRAINING PHASE:
═══════════════
Raw Data ──► Preprocess ──► Quantum Encoding ──► MPS Simulation
                │                │                     │
                │                ▼                     ▼
                │         Circuit U(x)         Expectation Values
                │                                      │
                ▼                                      ▼
        Scaled Features ────────────────────► Quantum Features Φ(x)
                                                       │
                                                       ▼
                                              Kernel Matrix K
                                                       │
                                                       ▼
                                              SVM Training
                                                       │
                                                       ▼
                                              Trained Model (.pkl)


INFERENCE PHASE:
════════════════
New Transaction ──► Load Scaler ──► Preprocess ──► Quantum Encoding
                         │              │                │
                         ▼              ▼                ▼
                    scaler.pkl    Scaled x_new     Circuit U(x_new)
                                                         │
                                                         ▼
                                                  MPS Simulation
                                                         │
                                                         ▼
                                                  Φ(x_new)
                                                         │
                         ┌───────────────────────────────┤
                         │                               │
                         ▼                               ▼
                  Kernel Mode                     Feature Mode
                  k(x_new, x_sv)                  Φ(x_new) directly
                         │                               │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                                  Load Model (.pkl)
                                         │
                                         ▼
                                    Prediction
                                         │
                                         ▼
                              Fraud / Legitimate + Score
```

---

*Next: [05_IMPLEMENTATION_REFERENCE.md](05_IMPLEMENTATION_REFERENCE.md) - Code mapping and parameter reference*
