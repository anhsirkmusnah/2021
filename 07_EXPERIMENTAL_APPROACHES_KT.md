# Projected Quantum Features for Fraud Detection: Experimental Approaches KT

## Knowledge Transfer Document for Quantum Engineers / Scientists

---

## 1. Executive Summary

This document details three experimental approaches explored for integrating projected quantum features into a production fraud detection pipeline. Each approach uses tensor-network-simulated quantum circuits to derive features from classical transaction data, differing in how those features are consumed downstream.

| # | Approach | Classifier | Scale Tested | Outcome |
|---|----------|-----------|--------------|---------|
| 1 | PQK-SVM | SVM (precomputed kernel) | Up to ~30K samples | Did not scale; compute cost O(N^2), degraded performance, memory limits |
| 2 | Quantum Feature Projection (QFP) | LightGBM | ~2.4M train / ~140M test | Quantum+classical features did not significantly outperform classical-only |
| 3 | QFP with X/Y/Z Basis Decomposition | LightGBM + Neural Networks | ~2.4M train / ~140M test | X-basis most aligned with ansatz; Y and Z bases introduced noise; no consistent gain |

**Bottom line:** Across all three approaches, the addition of quantum-derived features did not yield statistically meaningful improvement over the classical baseline for this dataset and scale.

---

## 2. Common Technical Foundation

### 2.1 Quantum Circuit Design

All approaches share the same core quantum encoding pipeline: classical feature values are embedded into a parameterized quantum circuit, the circuit is simulated via Matrix Product States (MPS), and single-qubit Pauli expectation values are extracted.

#### 2.1.1 Angle Encoding

Each classical feature x_i is mapped to a single qubit via a rotation gate. One qubit per feature means d features require d qubits.

**Initialization:**

All qubits begin in the computational basis state |0>. A Hadamard gate is applied to each qubit to create an equal superposition:

```
|0> --[H]--> |+> = (|0> + |1>) / sqrt(2)
```

This is necessary because rotation gates around the Z-axis have no observable effect on |0> (an eigenstate of Z). The Hadamard moves the state to the equator of the Bloch sphere where rotations produce distinguishable states.

**Feature encoding (single-qubit):**

Two encoding schemes were tested:

| Scheme | Gate | Rotation Angle | Bloch Sphere Effect |
|--------|------|---------------|-------------------|
| Hamiltonian (primary) | Rz(x_i) | theta = gamma * x_i / pi | Rotation around Z-axis |
| Alternative | Rx(x_i) | theta = x_i | Rotation around X-axis |

For the **Rz encoding** used in the hamiltonian ansatz:

```
Rz(theta) = | e^(-i*theta/2)    0           |
            |      0          e^(i*theta/2)  |
```

The gate applies a relative phase between |0> and |1> components proportional to the feature value. In the Bloch sphere picture, this rotates the state vector around the Z-axis by angle theta.

The parameter gamma (rotation scaling factor) controls the sensitivity of the encoding:
- gamma = 1: full rotation range (used in production)
- gamma < 1: compressed rotation range (prevents over-rotation for many qubits)
- The division by pi normalizes to the pytket half-turns convention

For the **Rx encoding**:

```
Rx(theta) = | cos(theta/2)    -i*sin(theta/2) |
            | -i*sin(theta/2)  cos(theta/2)   |
```

This rotates around the X-axis, which directly modulates the |0> vs |1> occupation probability rather than the relative phase. The choice between Rx and Rz affects which Pauli measurement bases are most sensitive to the encoded data.

#### 2.1.2 Entanglement Structure

After single-qubit encoding, two-qubit entangling gates create correlations between feature-encoding qubits. Two entangling gate types were tested:

**XXPhase gate** (primary, hamiltonian ansatz):

```
R_XX(theta) = exp(-i * theta * X_1 (x) X_2 / 2)
```

where X_1 (x) X_2 is the tensor product of Pauli-X operators on qubits 1 and 2.

The rotation angle encodes a **feature interaction term**:

```
theta_ij = gamma^2 * (1 - x_i) * (1 - x_j)
```

This means:
- When both features are near 1 (after normalization): minimal entanglement (theta ~ 0)
- When both features are near 0: maximum entanglement (theta ~ gamma^2)
- The interaction is **non-linear** in the feature values

**ZZPhase gate** (alternative):

```
R_ZZ(theta) = exp(-i * theta * Z_1 (x) Z_2 / 2)
```

The ZZ interaction commutes with Z-basis measurements, meaning it primarily affects X and Y expectation values while leaving Z expectations largely unchanged. This has implications for which measurement bases carry information (discussed in Approach 3).

**Entanglement topology — linear nearest-neighbor:**

```
Qubits:  q_0 --- q_1 --- q_2 --- ... --- q_{n-1}

Entanglement pairs: [(0,1), (2,3), (1,2), (3,4), ...]
```

The `entanglement_graph(nq, nn=1)` function generates non-overlapping gate layers for parallel execution:
- Layer 1: non-overlapping pairs (0,1), (2,3), (4,5), ...
- Layer 2: remaining pairs (1,2), (3,4), (5,6), ...

This linear connectivity is chosen because:
1. It matches MPS tensor network structure (enabling efficient simulation)
2. It is physically realizable on near-term quantum hardware
3. Deeper entanglement is achievable through repeated layers (reps parameter)

**Number of entanglement layers:** In production, `reps = 5` was used for QFP approaches and `reps = 1` was also tested. Increasing reps increases expressivity but exponentially increases tensor network complexity due to repeated SVD operations.

#### 2.1.3 Complete Ansatz Structure

For a single data point x = [x_1, x_2, ..., x_n], the circuit is:

```
U(x) = PROD_{l=1}^{reps} [ PROD_{(i,j) in E} R_XX(gamma^2 * (1-x_i)(1-x_j))
                            * PROD_{i=1}^{n} Rz(gamma * x_i / pi) ]
        * H^{(x)n}

|psi(x)> = U(x) |0>^{(x)n}
```

Where E is the entanglement map (nearest-neighbor pairs).

The circuit is constructed symbolically using **pytket** (Quantinuum's quantum SDK):
- Symbolic parameters (sympy Symbols) represent feature values
- At runtime, symbols are substituted with actual data values
- The circuit is compiled to a linear architecture using `DefaultMappingPass`
- Non-adjacent two-qubit gates are decomposed via `DecomposeBRIDGE` (SWAP insertion)

### 2.2 Measurement: Bloch Vector Extraction

After circuit simulation, we extract the **Bloch vector** for each qubit. For qubit i in the n-qubit state |psi(x)>:

```
<X_i> = Tr(rho_i * sigma_X)    in [-1, +1]
<Y_i> = Tr(rho_i * sigma_Y)    in [-1, +1]
<Z_i> = Tr(rho_i * sigma_Z)    in [-1, +1]
```

Where rho_i = Tr_{all other qubits}(|psi><psi|) is the **reduced density matrix** of qubit i, obtained by tracing out all other qubits.

The Pauli matrices:

```
sigma_X = | 0  1 |    sigma_Y = | 0  -i |    sigma_Z = | 1   0 |
          | 1  0 |              | i   0 |              | 0  -1 |
```

These three expectation values are the (x, y, z) coordinates of qubit i's state on the Bloch sphere. Together they completely characterize the single-qubit reduced state.

**For n qubits, this yields 3n real-valued features:**

```
Phi(x) = [<X_1>, <Y_1>, <Z_1>, <X_2>, <Y_2>, <Z_2>, ..., <X_n>, <Y_n>, <Z_n>]  in R^{3n}
```

**Physical interpretation:**
- **<Z_i>**: Measures the "computational basis" population. For |0>, <Z> = +1; for |1>, <Z> = -1. This is most directly related to the probability of each basis state.
- **<X_i>**: Measures the "superposition balance" between |+> and |->. Sensitive to the real part of the off-diagonal coherence.
- **<Y_i>**: Measures the "phase coherence" between |0> and |1>. Sensitive to the imaginary part of the off-diagonal coherence.

**Key insight for the Rz-encoded hamiltonian ansatz:** Since Rz rotations act on the phase (Z-axis rotation), and the initial Hadamard places qubits on the equator, the Rz encoding primarily modulates X and Y expectations. The Z expectation is less directly coupled to the input features in this ansatz.

### 2.3 Tensor Network Simulation with ITensor

#### 2.3.1 Why Tensor Networks

Simulating a d-qubit quantum circuit naively requires storing a complex vector of dimension 2^d. For d = 216 qubits, this is 2^216 ~ 10^65 complex numbers — physically impossible.

**Matrix Product States (MPS)** exploit the fact that states with bounded entanglement can be represented efficiently:

```
|psi> = SUM_{i1,i2,...,in} A^1[i1] * A^2[i2] * ... * A^n[in]  |i1 i2 ... in>
```

Where each A^k[i_k] is a matrix of dimension chi_{k-1} x chi_k, and chi_k is the **bond dimension** at bond k.

**Storage:** O(n * chi^2) instead of O(2^n)

For our circuits with linear nearest-neighbor entanglement and shallow depth, the bond dimension chi remains manageable (empirically chi < 50 for typical configurations), making simulation tractable even for 216 qubits.

#### 2.3.2 ITensor C++ Implementation

The simulation engine is a custom C++ library (`helloitensor.cc`) using the **ITensor** library (Fishman et al., SciPost Physics Codebases 4, 2022) with a custom `qubit.h` site type, exposed to Python via **pybind11**.

**Core function:** `circuit_xyz_exp(gate_list, n_qubits) -> [[<X>, <Y>, <Z>], ...]`

**Gate application algorithm:**

1. **Single-qubit gates** (H, Rz, Rx, T):
   ```
   psi.position(i)          // Move orthogonality center to site i
   G = op(sites, "Rz", i)   // Get gate operator
   new_tensor = G * psi(i)   // Contract gate with MPS tensor
   psi.set(i, new_tensor)    // Update MPS
   ```
   Complexity: O(chi^2). Does not increase bond dimension.

2. **Two-qubit gates** (XXPhase, ZZPhase, SWAP, CZ):
   ```
   psi.position(i)
   wf = psi(i) * psi(j)     // Merge two adjacent MPS tensors
   wf = G * wf              // Apply two-qubit gate
   [U, S, V] = SVD(wf)      // Decompose back into two tensors
   psi.set(i, U)             // Left tensor
   psi.set(j, S * V)         // Right tensor (singular values absorbed)
   ```
   Complexity: O(chi^3) due to SVD. Bond dimension can increase up to 2*chi before truncation.

**SVD truncation:** Singular values below `Cutoff = 1E-10` are discarded. This controls the trade-off between accuracy and bond dimension growth. For our shallow circuits, truncation errors are negligible.

**Gate encoding scheme** (Python to C++ bridge):

| Code | Gate | Type | Parameters |
|------|------|------|-----------|
| 0 | H (Hadamard) | 1-qubit | None |
| 1 | Rx | 1-qubit | angle (half-turns) |
| 2 | Rz | 1-qubit | angle (half-turns) |
| 3 | XXPhase | 2-qubit | angle (half-turns) |
| 4 | ZZPhase | 2-qubit | angle (half-turns) |
| 5 | SWAP | 2-qubit | None |
| 6 | T | 1-qubit | None |
| 7 | CZ | 2-qubit | None |

**Expectation value extraction** uses the mixed-canonical form of MPS:

```cpp
tensor_mps.position(i+1);  // Move orthogonality center to site i
<X_i> = Re( <psi| X_half |psi> )  // Local contraction at site i
<Y_i> = Re( <psi| Y_half |psi> )
<Z_i> = Re( <psi| Z_half |psi> )
```

The `X_half`, `Y_half`, `Z_half` operators in `qubit.h` are the Pauli matrices scaled by 1/2 (consistent with spin-1/2 convention). In canonical form, the expectation value reduces to a local contraction at the single site, with complexity O(chi^2) per qubit.

#### 2.3.3 Computational Bottleneck: SVD in Entanglement

Every two-qubit gate application requires an SVD decomposition:

```
Total SVD operations per data point = reps * |E| (entanglement pairs)
```

For 216 qubits, 1 layer: |E| = 215 pairs, so 215 SVDs per data point.
For 216 qubits, 5 layers: 1,075 SVDs per data point.

Each SVD has complexity O(chi^3), and chi grows with entanglement. This is the **primary computational bottleneck** of the entire pipeline, and the reason why:
- Circuit depth (reps) is kept shallow
- Only nearest-neighbor entanglement is used
- Bond dimension truncation is aggressive (cutoff = 1E-10)

### 2.4 Infrastructure and Hardware

**Compute platform:** Google Cloud Dataproc (CPU-based)

| Resource | Specification |
|----------|--------------|
| vCPUs | 48-96 per cluster |
| Parallelization | MPI (mpi4py + OpenMPI) |
| Simulation | ITensor C++ (CPU, no GPU) |
| Compilation | g++/clang++ with pybind11 |

**MPI parallelization strategy:**

Data points are distributed across MPI ranks using round-robin chunking:
```
entries_per_chunk = ceil(N / n_procs)
rank k processes: data[k*chunk : (k+1)*chunk]
```

For the SVM approach, kernel matrix tiles are computed in round-robin with message passing of Y chunks between ranks. For the QFP approach, each rank independently simulates circuits for its data chunk, then results are gathered via `MPI.reduce(SUM)`.

### 2.5 Dataset

**Production dataset** (proprietary):

| Property | Value |
|----------|-------|
| Training set | ~2.4 million transactions |
| Test set | ~140 million transactions |
| Features | 216 per transaction |
| Task | Binary classification (fraud vs. legitimate) |
| Class distribution | Highly imbalanced (production-grade) |
| Baseline model | Production vendor LightGBM |

The 216 features are normalized via StandardScaler before quantum encoding. For the SVM approach, an additional QuantileTransformer and MinMaxScaler([-pi, pi]) pipeline was applied.

**Initial validation dataset:** Elliptic Bitcoin (Weber et al., 2019)
- 203,769 transactions, 46,564 labeled
- 166 features, binary labels (illicit/licit)
- Used for initial proof-of-concept and parameter tuning

---

## 3. Approach 1: Projected Quantum Kernel with SVM (PQK-SVM)

### 3.1 Method Overview

This approach constructs a **kernel matrix** from the projected quantum features and uses it directly in an SVM classifier with a precomputed kernel.

**Pipeline:**

```
Classical Data  -->  Preprocessing  -->  Quantum Circuit U(x)
                                              |
                                              v
                                         MPS Simulation
                                              |
                                              v
                                    Bloch Vectors: [<X>, <Y>, <Z>]_i
                                              |
                                              v
                         Pairwise Kernel: K(x, x') = exp(-alpha * D(x, x'))
                                              |
                                              v
                                    SVM(kernel="precomputed")
                                              |
                                              v
                                      Classification
```

### 3.2 Mathematical Formulation

#### 3.2.1 Projected Quantum Kernel Formula

Given two data points x and x', each encoded into quantum states |psi(x)> and |psi(x')>, the projected quantum kernel is:

```
K_PQK(x, x') = exp( -alpha * D(x, x') )
```

Where D is the **projected distance**:

```
D(x, x') = SUM_{i=1}^{n} 2 * || rho_i(x) - rho_i(x') ||^2_F
```

Expanding using Bloch vectors:

```
D(x, x') = SUM_{i=1}^{n} [ (<X_i>^x - <X_i>^{x'})^2
                           + (<Y_i>^x - <Y_i>^{x'})^2
                           + (<Z_i>^x - <Z_i>^{x'})^2 ]
```

This is the **squared Euclidean distance** between the projected feature vectors Phi(x) and Phi(x') in R^{3n}, wrapped in an RBF kernel.

**The factor of 2** comes from the relationship between Frobenius norm of density matrix differences and Bloch vector distances: ||rho - sigma||^2_F = (1/2) * ||r - s||^2 for single-qubit states.

#### 3.2.2 Kernel Properties

The PQK satisfies Mercer's condition (positive semi-definite) because:
- It is an RBF kernel applied to a valid feature map
- K(x, x) = 1 for all x (self-similarity)
- K(x, x') = K(x', x) (symmetry)
- 0 < K(x, x') <= 1 for all pairs

**alpha (kernel bandwidth):** Controls the sharpness of the kernel.
- Large alpha: only very similar quantum states yield high kernel values (sharp boundary)
- Small alpha: broader similarity, smoother decision boundary
- Analogous to gamma in the classical RBF kernel

#### 3.2.3 Relationship to Classical RBF

The PQK is mathematically equivalent to an RBF kernel in a 3n-dimensional feature space:

```
K_PQK(x, x') = K_RBF(Phi(x), Phi(x'))

where Phi(x) = [<X_1>, <Y_1>, <Z_1>, ..., <X_n>, <Y_n>, <Z_n>]
```

The "quantum advantage" claim is that the non-linear feature map x -> Phi(x) (mediated by the quantum circuit) captures correlations that are hard to construct classically.

### 3.3 Implementation Details

**Codebase:** `ITensor_C/main.py` + `ITensor_C/projected_kernel_ansatz.py`

**Preprocessing pipeline:**

```python
# 1. Quantile transform to Gaussian distribution (handles outliers)
transformer = QuantileTransformer(output_distribution='normal')
X = transformer.fit_transform(X)

# 2. Standardize to zero mean, unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Scale to quantum rotation range [-pi, pi]
minmax = MinMaxScaler((-np.pi, np.pi))
X = minmax.fit_transform(X)

# 4. Select top-k features (k = num_qubits)
X_reduced = X[:, 0:num_features]
```

**Kernel matrix construction:**

The N x N training kernel matrix requires N^2 / 2 unique entries (exploiting symmetry). Each entry requires:
1. Simulating circuit for data point x (if not cached): O(n * reps * chi^3)
2. Simulating circuit for data point x': O(n * reps * chi^3)
3. Computing the distance D(x, x'): O(3n)
4. Computing exp(-alpha * D): O(1)

With MPI parallelization, the kernel matrix is tiled across P processes. Y chunks are passed in round-robin to compute all tiles without storing all data on one rank.

**SVM training:**

```python
# Grid search over regularization parameter C
C_values = [2, 1.5, 1, 0.5, 0.1, 0.05, 0.01]
for C in C_values:
    svm = SVC(kernel="precomputed", C=C, tol=1e-5)
    svm.fit(K_train, train_labels)
    predictions = svm.predict(K_test)
```

A classical RBF-SVM baseline was also trained for direct comparison:

```python
svm_rbf = SVC(kernel="rbf", C=C, tol=1e-5)
svm_rbf.fit(X_train, train_labels)
```

### 3.4 Parameters Used

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_features | 12-20 | Limited by SVM scaling |
| reps | 2-10 | Tested range; more reps = deeper circuit |
| gamma | 0.5-1.0 | Rotation scaling |
| alpha | 0.5 | Kernel bandwidth |
| Entanglement | XXPhase (Hamiltonian), ZZPhase (alt.) | Both tested |
| Encoding | Rz (primary), Rx (alternative) | Both tested |
| SVM C | Grid search [0.01, 2.0] | Regularization |
| Preprocessing | QuantileTransformer + StandardScaler + MinMaxScaler | Full pipeline |

### 3.5 Scale and Performance

**Tested scale:** Up to approximately 30,000 samples from the production dataset.

**Computational cost analysis for N = 30,000:**

```
Kernel matrix entries:      N^2 / 2 = 450,000,000
Circuit simulations:        2N = 60,000  (unique data points)
Per-simulation cost:        O(n * reps * chi^3) ~ 50-100 ms (for n=12-20)
Total simulation time:      ~50-100 minutes (with 48 MPI ranks)
Kernel distance computation: ~minutes (dominated by simulation)
Memory for kernel matrix:   N^2 * 8 bytes = ~7.2 GB (float64)
```

**Why this approach did not scale:**

1. **O(N^2) kernel matrix computation:** For 2.4M training samples, the kernel matrix would have ~2.88 * 10^12 entries, requiring ~23 TB of memory and years of compute time.

2. **SVM scaling:** Even with a precomputed kernel, SVM training is O(N^2) to O(N^3) in the number of samples, making it infeasible for millions of data points.

3. **Performance degradation:** At 30K samples, classification metrics did not justify the computational cost compared to classical baselines. The projected quantum kernel did not demonstrate a clear advantage over the RBF kernel at this scale.

4. **Memory constraints:** The 48-96 vCPU Dataproc cluster could not hold kernel matrices larger than ~30K x 30K in distributed memory.

### 3.6 Outcome

The PQK-SVM approach was abandoned for the production pipeline due to the fundamental O(N^2) scaling barrier. It remains viable for small-sample regimes (N < 5,000) or when kernel-based interpretability is valued, but cannot serve production-scale fraud detection with millions of transactions.

---

## 4. Approach 2: Quantum Feature Projection (QFP)

### 4.1 Method Overview

To overcome the O(N^2) scaling limitation of the kernel approach, Approach 2 **explicitly computes the projected feature vectors** and passes them directly to a tree-based classifier. This reduces the quantum computation from O(N^2) kernel entries to O(N) feature extractions.

**Pipeline:**

```
Classical Data (N x d)  -->  StandardScaler  -->  Quantum Circuit U(x_i)
                                                       |
                                                       v
                                                  MPS Simulation
                                                       |
                                                       v
                                            Bloch Vectors: Phi(x_i) in R^{3d}
                                                       |
                                                       v
                                        Concatenate: [x_i, Phi(x_i)] in R^{4d}
                                                       |
                                                       v
                                                 LightGBM Classifier
                                                       |
                                                       v
                                                 Classification
```

### 4.2 Key Difference from Approach 1

Instead of computing pairwise kernel values between all N^2 pairs, we compute the quantum feature vector Phi(x) for each of the N data points independently. The kernel is now **implicit** — LightGBM learns its own decision boundaries in the augmented feature space.

| Aspect | Approach 1 (PQK-SVM) | Approach 2 (QFP) |
|--------|---------------------|-------------------|
| Output of quantum step | N x N kernel matrix | N x 3d feature matrix |
| Scaling | O(N^2) | O(N) |
| Classifier | SVM (precomputed) | LightGBM |
| Feature representation | Implicit (kernel) | Explicit (feature vectors) |
| Production scalable | No (>30K infeasible) | Yes (tested on 2.4M/140M) |

### 4.3 Implementation Details

**Codebase:** `dataproc files/main.py` + `dataproc files/generate_pqf.py` + `dataproc files/projected_quantum_features.py`

#### 4.3.1 Preprocessing

Simpler than Approach 1 — only StandardScaler is applied:

```python
# StandardScaler only (no QuantileTransformer, no MinMaxScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Scaler is saved to ./model/scaler.pkl for consistent test-time normalization
```

The rationale for simpler preprocessing: LightGBM is invariant to monotonic transformations of features, so the additional QuantileTransformer and MinMaxScaler steps that were necessary for SVM (which is sensitive to feature scaling) are not required.

However, the rotation angles in the quantum circuit are still directly proportional to the feature values, so the StandardScaler ensures features are on a comparable scale for the quantum encoding.

#### 4.3.2 Quantum Feature Generation

The `generate_projectedQfeatures()` function orchestrates the full pipeline:

```python
def generate_projectedQfeatures(data_feature, reps, gamma, target_label, info,
                                 slice_size=50000, train_flag=False):
    # 1. Scale features
    classical_features = apply_scaling(data_feature, train_flag)

    # 2. Create quantum ansatz (hamiltonian, reps=5, gamma=1)
    ansatz = create_ansatz(num_features, reps, gamma)

    # 3. Process in slices of 50,000 to manage memory
    for each slice:
        quantum_features = build_qf_matrix(mpi_comm, ansatz, X=slice)
        combined = np.concatenate((classical_slice, quantum_features), axis=1)

    # 4. Save combined feature array
    np.save(f'./pqf_arr/{info}_mini_arr.npy', final_features)
```

**Sliced processing:** Data is processed in chunks of 50,000 samples to avoid memory exhaustion. Each chunk is independently MPI-parallelized.

**`build_qf_matrix()` algorithm:**

```
For each MPI rank:
    For each data point in this rank's chunk:
        1. circuit = ansatz.circuit_for_data(x_i)     # Substitute feature values
        2. gates = ansatz.circuit_to_list(circuit)      # Convert to gate encoding
        3. exp_xyz = circuit_xyz_exp(gates, n_qubits)   # C++ MPS simulation
        4. Store [<X_1>, <Y_1>, <Z_1>, ..., <X_n>, <Y_n>, <Z_n>]

    MPI.reduce(SUM) to collect all results at root
```

**Output feature dimensions:**

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Classical features | d = 216 | Original normalized transaction features |
| Quantum features | 3d = 648 | X, Y, Z expectations per qubit |
| **Combined** | **4d = 864** | Concatenation of classical + quantum |

#### 4.3.3 Model Training

Two LightGBM models are trained in parallel for direct comparison:

**1. Classical-only model:**

```python
model_train(classical_features, labels, model_name='classical_fraud_model_v')
```

**2. Quantum-augmented model:**

```python
model_train(combined_features, labels, model_name='qiml_fraud_model_v')
```

**LightGBM hyperparameters:**

```python
params = {
    'subsample': 1.0,
    'reg_lambda': 0.5,        # L2 regularization
    'reg_alpha': 0.1,         # L1 regularization
    'num_leaves': 40,
    'n_estimators': 300,
    'min_child_samples': 30,
    'max_depth': 15,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
}
```

**Note on hyperparameter optimization:** GridSearchCV was impractical due to the large feature space (864 dimensions) and dataset scale (~2.4M samples). The parameters above were hand-tuned. This is a known limitation — it is possible that more exhaustive tuning could reveal small performance differences between classical and quantum-augmented models.

### 4.4 Parameters Used

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_features (qubits) | 216 | One qubit per feature (all features used) |
| reps | 5 | Moderate depth; balance expressivity vs. compute |
| gamma | 1 | Full rotation scaling (alpha=1 in user notation) |
| ansatz | hamiltonian | Rz encoding + XXPhase entanglement |
| hadamard_init | True | Required for Rz encoding sensitivity |
| slice_size | 50,000 | Memory management for MPI processing |
| Preprocessing | StandardScaler only | Sufficient for tree-based models |

### 4.5 Results

**Classical + Quantum features did not significantly outperform classical-only features.**

The LightGBM models trained on the 864-dimensional combined feature space and the 216-dimensional classical feature space achieved nearly identical performance across all metrics (accuracy, precision, recall, F1, AUC-ROC).

**Possible explanations:**

1. **Redundant information:** The quantum features, being non-linear transformations of the same classical features, may not add information beyond what LightGBM can already extract through its own non-linear splits.

2. **Shallow entanglement:** With only nearest-neighbor connectivity and limited depth, the quantum circuit may not generate sufficiently complex feature interactions to surpass what a 300-tree ensemble can learn.

3. **Feature dilution:** Adding 648 potentially noisy quantum features to 216 informative classical features may cause LightGBM to waste splits on uninformative dimensions, even with colsample_bytree=0.8.

4. **Ansatz mismatch:** The hamiltonian ansatz may not be well-suited to the specific data distribution of this fraud detection dataset.

---

## 5. Approach 3: QFP with X/Y/Z Basis Decomposition

### 5.1 Motivation

Given that the full 648 quantum features (X + Y + Z for 216 qubits) did not improve performance, Approach 3 investigates **which measurement bases carry useful information** by decomposing the quantum features into their X, Y, and Z components and testing them independently.

### 5.2 Feature Decomposition

The 648 quantum features per data point are structured as:

```
Phi(x) = [<X_1>, <Y_1>, <Z_1>, <X_2>, <Y_2>, <Z_2>, ..., <X_216>, <Y_216>, <Z_216>]
```

This is decomposed into three basis-specific vectors:

| Basis Vector | Dimension | Content |
|-------------|-----------|---------|
| Phi_X(x) | 216 | [<X_1>, <X_2>, ..., <X_216>] |
| Phi_Y(x) | 216 | [<Y_1>, <Y_2>, ..., <Y_216>] |
| Phi_Z(x) | 216 | [<Z_1>, <Z_2>, ..., <Z_216>] |

### 5.3 Experimental Configurations

Multiple feature combinations were tested:

| Configuration | Total Dims | Components |
|--------------|-----------|------------|
| Classical only | 216 | Original features |
| Classical + X | 432 | 216 classical + 216 X-basis |
| Classical + Y | 432 | 216 classical + 216 Y-basis |
| Classical + Z | 432 | 216 classical + 216 Z-basis |
| Classical + X + Y + Z | 864 | Full combined (Approach 2) |
| X-basis only | 216 | Quantum X features alone |
| Y-basis only | 216 | Quantum Y features alone |
| Z-basis only | 216 | Quantum Z features alone |
| X + Y + Z only | 648 | All quantum features, no classical |

### 5.4 Results and Analysis

#### 5.4.1 Basis-wise Performance

**X-basis + classical:** Performed closest to the classical-only baseline. Of the three Pauli bases, X-basis features were the most informative.

**Y-basis and Z-basis:** Often added noise, and in some configurations actively degraded performance relative to classical-only.

**Reduced feature space:** Even with the most informative basis (X-only), the reduction from 864 to 432 features did not yield a consistent performance gain.

#### 5.4.2 Why X-basis Dominates

For the **hamiltonian ansatz with Rz encoding**:

1. The initial Hadamard gate places each qubit at the equator of the Bloch sphere: |+> = (|0> + |1>)/sqrt(2), which has Bloch coordinates (1, 0, 0) — pointing along the **X-axis**.

2. **Rz(theta)** rotates the state around the Z-axis on the equator, moving it in the X-Y plane. The X and Y expectations oscillate as cos(theta) and sin(theta) respectively.

3. The **XXPhase entanglement** creates correlations primarily in the X-X direction, which directly modulates the X expectations of both qubits.

4. **Z expectations** are largely unaffected by Rz rotations (Z is an eigenaxis of Rz). The Z-component of the Bloch vector changes primarily through entanglement-induced mixed-state effects, which are weaker in a shallow circuit.

Therefore:
- **<X_i>** directly encodes the feature value x_i via cos(gamma * x_i / pi) plus entanglement corrections → most informative
- **<Y_i>** encodes the feature via sin(gamma * x_i / pi) plus entanglement corrections → partially informative but phase-sensitive
- **<Z_i>** primarily reflects entanglement-induced decoherence, not direct feature encoding → least informative, mostly noise

#### 5.4.3 Neural Network Experiments

To test whether the apparent redundancy of quantum features was an artifact of tree-based models (which cannot easily exploit correlations between X, Y, Z components of the same qubit), fully connected neural networks were also evaluated.

**Hypothesis:** X, Y, Z components represent correlated quantum information (they are constrained to the Bloch sphere: <X>^2 + <Y>^2 + <Z>^2 <= 1). Tree-based models split on individual features independently and cannot easily exploit this joint structure. Neural networks, with their dense layers, could potentially learn useful cross-basis interactions.

**Outcome:** Neural network performance closely matched LightGBM results. No clear advantage was observed from including the full Bloch vector (X + Y + Z) over X-basis alone. This suggests the lack of improvement is not a classifier limitation but a fundamental property of the quantum features themselves.

#### 5.4.4 Measurement Constraints on Real Hardware

An important consideration for future hardware deployment:

On **real quantum hardware**, you can only measure one basis per circuit execution. To obtain X, Y, and Z expectations, you need three separate circuit preparations:
- X measurement: append H before measurement
- Y measurement: append S^dag then H before measurement
- Z measurement: measure directly in computational basis

Due to quantum uncertainty (measurement collapse), these three measurements correspond to **different instances** of the quantum state, not the same one. The (X, Y, Z) vector obtained from hardware is therefore a statistical estimate from independent samples, not a single-shot Bloch vector.

In tensor network simulation, we have simultaneous access to all three components from the same MPS. This is a simulation advantage that would not transfer to hardware. The finding that only X-basis features are useful is therefore **favorable for hardware deployment** — it reduces the required circuit executions by 3x.

### 5.5 Implications

1. **Dimensionality reduction is justified:** Using only Classical + X-basis (432 features) is as good as the full 864-dimensional space, with half the dimensionality and 3x fewer quantum measurements needed.

2. **Ansatz design matters:** The choice of encoding gate (Rz) directly determines which measurement basis is most informative. An Rx-encoded ansatz would likely shift the informativeness to a different basis.

3. **Shallow circuits limit expressivity:** With only nearest-neighbor entanglement and a few layers, the quantum circuit acts as a relatively simple non-linear transformation. LightGBM with 300 trees over 216 features may already be capturing equivalent or superior non-linear structure.

---

## 6. Comparative Analysis

### 6.1 Approach Comparison

| Dimension | Approach 1 (PQK-SVM) | Approach 2 (QFP) | Approach 3 (QFP + Basis Split) |
|-----------|---------------------|-------------------|-------------------------------|
| **Quantum output** | Pairwise kernel K(x,x') | Feature vector Phi(x) in R^{648} | Decomposed: Phi_X, Phi_Y, Phi_Z each in R^{216} |
| **Classical input** | 12-20 features | 216 features | 216 features |
| **Classifier** | SVM | LightGBM | LightGBM + Neural Network |
| **Scaling** | O(N^2) | O(N) | O(N) |
| **Max samples tested** | ~30K | ~2.4M train / 140M test | ~2.4M train / 140M test |
| **Preprocessing** | QT + SS + MMS | SS only | SS only |
| **Quantum features used** | All (implicit in kernel) | All X,Y,Z | Decomposed by basis |
| **Performance vs baseline** | Did not justify compute cost | Nearly identical to classical-only | X-basis closest; Y,Z add noise |
| **Compute cost** | Very high (kernel matrix) | High (per-sample simulation) | Same as Approach 2 |
| **Production viable** | No | Marginal (no gain) | Marginal (no gain) |

### 6.2 Scaling Analysis

**Per data-point quantum simulation cost:**

```
Circuit: 216 qubits, 5 reps, ~1075 two-qubit gates
Time per simulation: ~100-500 ms (depending on bond dimension)
```

**Total quantum computation time estimates (48 MPI ranks):**

| Dataset | N | Simulations | Est. Time (48 ranks) |
|---------|---|-------------|---------------------|
| SVM kernel (30K) | 30,000 | 60,000 + N^2/2 kernel ops | Hours to days |
| QFP train (2.4M) | 2,400,000 | 2,400,000 | ~14-70 hours |
| QFP test (140M) | 140,000,000 | 140,000,000 | ~34-170 days |

The test-set inference time for 140M samples is a major practical concern, even with aggressive parallelization.

### 6.3 Bond Dimension Behavior

For 216 qubits with linear nearest-neighbor entanglement:

| reps | Typical max chi | Memory per MPS | SVD cost factor |
|------|----------------|----------------|-----------------|
| 1 | ~4-8 | ~50 KB | Low |
| 5 | ~16-50 | ~5 MB | Moderate |
| 10 | ~50-200 | ~50 MB | High |

The shallow circuits keep bond dimension manageable, which is what makes 216-qubit simulation feasible on CPUs. However, this also limits the entanglement (and thus the expressivity) of the quantum features.

---

## 7. Key Findings

### 7.1 What We Learned

1. **PQK-SVM does not scale to production volumes.** The O(N^2) kernel matrix is fundamentally incompatible with millions of training samples, regardless of parallelization.

2. **Projected quantum features from a shallow hamiltonian ansatz do not improve fraud detection over classical features alone** at the scale tested (2.4M train / 140M test).

3. **X-basis Pauli expectations are the most informative** quantum features for the Rz-encoded hamiltonian ansatz. Y and Z bases primarily contribute noise.

4. **The lack of improvement is classifier-agnostic.** Both LightGBM (tree-based) and neural networks (dense layers) converge to the same conclusion — quantum features are not adding predictive value beyond what is already captured by the 216 classical features.

5. **Tensor network simulation is computationally feasible** for 216-qubit shallow circuits on CPU clusters, but the per-sample cost (~100-500 ms) creates practical bottlenecks at production scale.

### 7.2 Possible Explanations

1. **Insufficient circuit depth:** Shallow circuits with 1-5 layers of nearest-neighbor entanglement may not generate features complex enough to surpass 300-tree LightGBM ensembles. Deeper circuits would increase expressivity but are computationally prohibitive with MPS simulation.

2. **Ansatz-data mismatch:** The hamiltonian ansatz may not be well-suited to the statistical structure of this fraud dataset. Different ansatz designs (e.g., data-re-uploading, hardware-efficient ansatz, problem-specific circuits) may yield different results.

3. **Feature saturation:** LightGBM with 216 features, 300 trees, and max depth 15 may already be near the Bayes-optimal classifier for this data distribution, leaving no room for quantum feature improvement.

4. **Limited hyperparameter tuning:** The LightGBM hyperparameters were not exhaustively optimized due to scale constraints. GridSearchCV over the 864-dimensional quantum-augmented space was impractical.

### 7.3 Recommendations for Future Work

1. **Data-re-uploading circuits:** Instead of encoding features once, re-upload classical features at each layer. This creates more complex non-linear feature maps without increasing qubit count.

2. **Trainable quantum circuits:** Use variational circuits where gate parameters are optimized (e.g., via gradient descent) to maximize a classification objective, rather than using fixed data-dependent rotations.

3. **Feature selection before encoding:** Instead of encoding all 216 features, use classical feature importance to select a smaller set of highly informative features and encode them into a deeper circuit.

4. **Alternative entanglement topologies:** Star graphs, all-to-all connectivity (via SWAP networks), or problem-specific connectivity informed by feature correlation structure.

5. **GPU-accelerated simulation:** Migrating from CPU-based ITensor to GPU-accelerated tensor network libraries (cuTensorNet, TensorCircuit) could reduce per-sample simulation time by 10-100x.

---

## 8. References

1. Huang, H.-Y., Broughton, M., Mohseni, M., et al. (2021). "Power of data in quantum machine learning." *Nature Communications*, 12, 2631. — Introduced projected quantum kernels.

2. Havlicek, V., Corcoles, A.D., Temme, K., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209-212. — Original quantum feature map framework.

3. Fishman, M., White, S.R., & Stoudenmire, E.M. (2022). "The ITensor Software Library for Tensor Network Calculations." *SciPost Physics Codebases*, 4. — ITensor library.

4. Weber, M., et al. (2019). "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics." KDD Workshop. — Elliptic Bitcoin dataset.

5. Heredge, J., et al. (2023). "Quantum Multiple Kernel Learning in Financial Classification Tasks." arXiv:2312.00260.

6. Vasquez, A.C., et al. (2023). "Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models." arXiv:2308.05237.

---

*Document Version: 1.0*
*Last Updated: February 2026*
*Classification: Internal Technical KT*
