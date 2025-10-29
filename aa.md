<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# UNIVERSAL QUANTUM SIMULATION LIBRARY: COMPREHENSIVE TECHNICAL REFERENCE

## LIBRARY 1: QUANTUM SIMULATION METHODS - PARAMETERS - CAPABLE FRAMEWORKS

### **State Vector Simulation Methods**

#### **Dense State Vector (Exact)**

**Description**: Stores complete quantum state as $2^n$ complex amplitudes in memory.[^1][^2][^3]

**Controllable Parameters**:

- **Precision**: `complex64` (single, 8 bytes/amplitude), `complex128` (double, 16 bytes/amplitude), `complex256` (quad precision)[^4][^5]
- **Memory alignment**: 256-byte alignment recommended for optimal GPU performance[^5]
- **Batch size** (`nSVs`): Number of state vectors for batched simulation[^6][^5]
- **State vector stride** (`svStride`): Offset between batched state vectors in memory[^5]
- **Index bits** (`nIndexBits`): Number of qubits per state vector[^5]

**Memory Requirements**: $2^n \times 16$ bytes (double precision) - 32 qubits = 64 GB, 44 qubits = 256 TB[^3]

**Capable Frameworks**:

- **Qiskit Aer**: `method='statevector'`, GPU support via CUDA[^2][^1]
- **Cirq/TensorFlow Quantum**: State vector backend with AVX2/SSE instructions[^7][^8]
- **QuEST**: Hybrid OpenMP/MPI/CUDA/HIP statevector simulator[^9][^4]
- **NVIDIA cuStateVec**: GPU-accelerated batched state vector operations[^6][^5]
- **PennyLane**: `default.qubit` device (CPU), `lightning.qubit` (optimized C++)[^10]
- **Yao.jl**: Pure Julia statevector with SIMD optimization[^11][^12]
- **ProjectQ**: C++ backend with OpenMP/MPI parallelization[^11]

**Performance**: Scales to 44 qubits on 1024 nodes (4096 GPUs) using distributed memory. Single GPU limited to ~30 qubits.[^3]

***

#### **Density Matrix Simulation**

**Description**: Simulates mixed quantum states as $2^n \times 2^n$ density matrices, enabling noise modeling.[^1][^2][^4]

**Controllable Parameters**:

- **Precision**: `complex64`, `complex128`
- **Noise model parameters**: Gate error rates (1Q: 0.5%, 2Q: 1-5% typical)[^13]
- **Decoherence times**: T1 (amplitude damping), T2 (dephasing) in microseconds
- **Readout error**: Measurement error matrices (1-3% typical)[^13]
- **Error mitigation**: Measurement error mitigation (M.E.M.) flag[^13]

**Memory Requirements**: $2^{2n} \times 16$ bytes - exponentially harder than state vector (16 qubits = 64 GB for density matrix vs 1 MB for state vector)

**Capable Frameworks**:

- **Qiskit Aer**: `method='density_matrix'`, full noise model support[^2][^1]
- **QuEST**: Density matrix mode with arbitrary noise channels[^4][^9]
- **Cirq**: DensityMatrixSimulator with custom noise[^14][^15]

**Use Case**: Noisy intermediate-scale quantum (NISQ) device simulation with realistic errors.[^13]

***

### **Tensor Network Simulation Methods**

#### **Matrix Product State (MPS) / Tensor Train**

**Description**: Represents quantum state as chain of 3D tensors connected by bond indices, exploiting area-law entanglement.[^16][^17][^18]

**Controllable Parameters**:

- **Max bond dimension** (`max_bond_dim`, `χ`): Controls entanglement capacity
    - Qiskit: `matrix_product_state_max_bond_dimension` (default: None = unlimited)[^19][^20][^21]
    - PennyLane: `max_bond_dim=128` (default)[^22][^10]
    - TeNPy: `chi_max` parameter in DMRG/TEBD[^18]
- **Truncation threshold** (`cutoff`, SVD threshold):
    - Qiskit: `matrix_product_state_truncation_threshold` (singular value cutoff)[^20][^21][^19]
    - PennyLane: `cutoff=0` (default, no truncation), range $10^{-16}$ to $10^{-4}$[^10][^22]
    - ITensor: `cutoff=1e-8` to `1e-12` typical for precision control[^23][^24][^25]
- **Cutoff mode** (PennyLane):
    - `"abs"`: Absolute singular value threshold
    - `"rel"`: Relative to largest singular value[^10]
- **Truncation behavior**:
    - `enable_truncation=True/False` (Qiskit)[^20]
    - Automatic reduced extent based on lowest cutoff among multiple criteria[^10]
- **Zero threshold** (`zero_threshold`): Elements below this treated as zero[^20]
- **Validation threshold**: Check for MPS validity[^20]
- **Chop threshold**: Remove small tensor elements[^20]
- **MPS logging**: `mps_log_data=True` for bond dimension tracking[^21]

**Memory Scaling**: Linear in circuit depth and qubits: $O(n \cdot d \cdot \chi^2)$ where $n$=qubits, $d$=circuit depth, $\chi$=bond dimension[^26]

**Capable Frameworks**:

- **Qiskit Aer**: `method='matrix_product_state'`, note: bond dimension currently has global static bug (Issue \#2286)[^19][^20]
- **PennyLane Lightning Tensor**: `method='mps'`, GPU-accelerated via cuTensorNet[^22][^10]
- **ITensor**: MPS/MPO via Julia or C++, DMRG optimization[^24][^25][^27][^28][^23]
- **TeNPy**: Canonical MPS forms, TEBD/DMRG algorithms[^18]
- **NVIDIA cuQuantum**: MPS backend with GPU acceleration[^29]
- **quimb**: Python MPS library with exact simulation capability[^20]

**Optimization Algorithms on MPS**:

- **TEBD** (Time Evolving Block Decimation): Time evolution with Trotter decomposition[^18]
- **DMRG** (Density Matrix Renormalization Group): Ground state optimization with sweeps parameter[^18]
- **iTEBD/iDMRG**: Infinite system variants[^18]

***

#### **Projected Entangled Pair States (PEPS)**

**Description**: Generalizes MPS to 2D/higher-dimensional lattices with tensor at each site.[^16]

**Controllable Parameters**:

- **Bond dimension** (`χ`): Controls 2D entanglement, much more expensive than MPS
- **Lattice topology**: Square, triangular, hexagonal geometries
- **Boundary conditions**: Open, periodic, cylindrical

**Memory Scaling**: Polynomial but rapidly growing with bond dimension

**Capable Frameworks**:

- **ITensor**: 2D PEPS support with custom contraction orders[^28]
- **cuTensorNet**: General PEPS contraction via optimized tensor network[^29]
- **TeNPy**: Limited PEPS support, primarily 1D focus[^18]

**Challenge**: Contraction is NP-hard; requires approximation methods for >~10×10 lattices.[^16]

***

#### **Tree Tensor Network States (TTNS)**

**Description**: Hierarchical tree structure for non-local correlations, logarithmic correlation length in tree depth.[^16]

**Controllable Parameters**:

- **Tree topology**: Balanced vs unbalanced (MPS is maximally unbalanced tree)
- **Bond dimension**: At each tree edge
- **Branching factor**: Number of children per node

**Capable Frameworks**:

- **ITensor**: General tensor network support includes TTNS
- **TeNPy**: Can construct custom tree networks
- **cuTensorNet**: General contraction handles arbitrary tree topologies

***

#### **Exact Tensor Network (Full Contraction)**

**Description**: Expresses quantum circuit as tensor network, finds optimal contraction path, contracts to exact result.[^30][^17][^10]

**Controllable Parameters**:

- **Contraction path optimization**:
    - `opt_einsum` methods: 'greedy', 'optimal', 'branch-bound', 'dp'[^29]
    - cuTensorNet hyper-optimizer: Automatic path finding dramatically faster[^31][^29]
- **Slicing configuration**:
    - **Slicing indices**: Select mode subsets for memory/parallelism tradeoff[^31][^29]
    - **Number of slices**: Total slices to divide computation[^31]
    - **Workspace size constraint**: Maximum GPU memory available[^31]
- **Compute type**:
    - `CUTENSORNET_COMPUTE_32F`: FP32 standard
    - `CUTENSORNET_COMPUTE_TF32`: TensorFloat-32 (Ampere+)
    - `CUTENSORNET_COMPUTE_3XTF32`: 3× TF32 for higher accuracy
    - `CUTENSORNET_COMPUTE_16F`: FP16 (Volta+)
    - `CUTENSORNET_COMPUTE_16BF`: BFloat16 (Ampere+)
    - `CUTENSORNET_COMPUTE_64F`: FP64 double precision[^32][^31]
- **Data type**: `CUDA_R_16F`, `CUDA_R_16BF`, `CUDA_R_32F`, `CUDA_R_64F`, `CUDA_C_32F`, `CUDA_C_64F`[^31]
- **Tensor Core usage**: Automatic on supported data type/compute type combinations[^31]

**Capable Frameworks**:

- **PennyLane Lightning Tensor**: `method='tn'`, exact TN simulation[^22][^10]
- **NVIDIA cuTensorNet**: Full contraction optimizer API with hyper-optimization[^32][^29][^31]
- **TensorNetwork library** (Google): General TN contraction[^30]

**Performance**: Dramatically faster path finding than classical methods; exact results but high memory cost.[^29][^10]

***

### **Stabilizer Simulation Methods**

#### **Clifford Stabilizer Formalism**

**Description**: Efficient simulation of Clifford circuits (H, S, CNOT, Pauli) using stabilizer tableaux, polynomial scaling.[^1][^2]

**Controllable Parameters**:

- **Stabilizer representation**: Tableau size $2n \times (2n+1)$ for $n$ qubits
- **Noise restriction**: Only Clifford errors allowed (Pauli channels, depolarizing)[^1]

**Memory Scaling**: $O(n^2)$ - can simulate thousands of qubits efficiently

**Capable Frameworks**:

- **Qiskit Aer**: `method='stabilizer'`, handles noisy Clifford circuits[^2][^1]
- **Stim**: Ultra-fast stabilizer simulator for quantum error correction[^33]
- **Qrack**: Hybrid stabilizer/state vector approach[^34]

***

#### **Extended Stabilizer (CH-form)**

**Description**: Approximate simulation for Clifford+T circuits via stabilizer state rank decomposition.[^34][^2][^1]

**Controllable Parameters**:

- **Number of stabilizer terms**: Grows as $2^{t}$ where $t$ = number of T gates
- **Approximation threshold**: Schmidt decomposition rounding parameter (SDRP)[^34]
- **Sampling method**: Monte Carlo samples from stabilizer mixture

**Memory Scaling**: Exponential in T-gate count but polynomial in qubits

**Capable Frameworks**:

- **Qiskit Aer**: `method='extended_stabilizer'`[^2][^1]
- **Qrack**: Schmidt decomposition optimization with tunable SDRP[^34]

**Use Case**: 54 qubits, 10 layers random circuits with ~4% fidelity on single GPU.[^34]

***

### **Unitary/Superoperator Simulation**

#### **Unitary Matrix Simulation**

**Description**: Simulates unitary evolution of circuit itself (not state evolution).[^1][^2]

**Controllable Parameters**:

- **Precision**: `complex64`, `complex128`
- **Matrix size**: $2^n \times 2^n$ complex elements

**Limitations**: Cannot simulate measurement, reset, or noise. Ideal circuits only.[^1]

**Capable Frameworks**:

- **Qiskit Aer**: `method='unitary'`[^2][^1]

***

#### **Superoperator Simulation**

**Description**: Simulates quantum channel (superoperator) including noise, represented as $4^n \times 4^n$ matrix.[^2][^1]

**Controllable Parameters**:

- **Kraus operators**: Noise channel decomposition
- **Channel composition**: Sequential application of superoperators

**Memory Requirements**: $4^n$ scaling - even worse than density matrix

**Capable Frameworks**:

- **Qiskit Aer**: `method='superop'`, supports gates and reset but not measurement[^1][^2]

***

### **Stochastic/Approximate Methods**

#### **Stochastic Quantum Circuit Simulation**

**Description**: Monte Carlo sampling of error channels, avoids tracking full density matrix.[^33]

**Controllable Parameters**:

- **Number of trajectories** (`n_shots`): Monte Carlo samples for statistical averaging
- **Error probability** (`p`): Apply error with probability $p$, leave untouched with $1-p$[^33]
- **Sampling strategy**: Importance sampling, stratified sampling

**Memory Scaling**: Same as state vector simulation ($O(2^n)$) but avoids $O(4^n)$ of density matrices

**Capable Frameworks**:

- **Qiskit Aer**: Stochastic noise sampling in statevector method[^1]
- **Decision Diagram simulators**: Stochastic path sampling through quantum circuits[^33]

**Accuracy**: Forms empirical averages from sampled trajectories to approximate true distribution.[^33]

***

#### **Quantum Monte Carlo (QMC)**

**Description**: Extends classical Monte Carlo with quantum effects for many-body systems.[^35]

**Controllable Parameters**:

- **Walkers**: Number of Monte Carlo walkers
- **Time step** (`τ`): Imaginary time evolution step size
- **Projector/Diffusion parameters**: For ground state projection

**Capable Frameworks**:

- **Classiq**: Quantum Monte Carlo implementations[^35]
- **Custom implementations**: Often domain-specific for condensed matter

***

### **Hybrid/Optimization Algorithm Methods**

#### **Variational Quantum Eigensolver (VQE)**

**Description**: Variational algorithm for ground state finding with parameterized quantum circuits.[^36]

**Controllable Parameters**:

- **Ansatz type**: Hardware-efficient, UCCSD, problem-specific
- **Circuit depth** (`p` or `d`): Number of ansatz layers
- **Optimizer**: COBYLA, SPSA, Adam, L-BFGS-B
- **Learning rate** (`α`): For gradient-based optimizers (0.01-0.1 typical)
- **Gradient method**: Parameter-shift rule, finite differences, adjoint[^36]
- **Measurements per step** (`M`): Shot budget per optimization iteration[^36]
- **Total function calls** (`n_calls`): Optimization budget[^36]

**Scaling Challenges**: Requires $M \ll 2^L$ for practical regime; basic optimizers fail without smart initialization.[^36]

**Capable Frameworks**:

- **Qiskit Algorithms**: VQE module with various optimizers
- **PennyLane**: `qml.VQECost` with automatic differentiation[^37]
- **Cirq/TFQ**: VQE via TensorFlow integration[^7]

***

#### **Quantum Approximate Optimization Algorithm (QAOA)**

**Description**: Variational algorithm for combinatorial optimization with alternating mixer/problem Hamiltonians.[^36]

**Controllable Parameters**:

- **QAOA depth** (`p`): Number of (mixer, problem) layer pairs
- **Beta parameters** (`β`): Mixer Hamiltonian angles (p values)
- **Gamma parameters** (`γ`): Problem Hamiltonian angles (p values)
- **Initialization strategy**:
    - Random: Poor performance[^36]
    - Annealing-inspired: Good performance, enables deeper circuits[^36]
    - INTERP: Interpolate from shallow to deep
- **Optimizer**: Same as VQE options
- **Measurement shots**: 8192 typical for statistical accuracy[^13]

**Performance**: Deep QAOA (p≥5) with smart initialization achieves exponential exponent $k \approx 0.5$ vs $k=1$ for random.[^36]

**Capable Frameworks**:

- **Qiskit Algorithms**: QAOA implementation
- **PennyLane**: `qaoa` module
- **Cirq**: QAOA examples and utilities

***

## LIBRARY 2: FRAMEWORKS/PLATFORMS - SIMULATION TECHNIQUES - CONTROLLABLE PARAMETERS

### **Qiskit Aer (IBM)**

**Programming Language**: Python (C++ backend)

**Simulation Methods Supported**:

1. **Statevector** (`method='statevector'`)
2. **Density Matrix** (`method='density_matrix'`)
3. **MPS** (`method='matrix_product_state'`)
4. **Stabilizer** (`method='stabilizer'`)
5. **Extended Stabilizer** (`method='extended_stabilizer'`)
6. **Unitary** (`method='unitary'`)
7. **Superoperator** (`method='superop'`)
8. **Automatic** (`method='automatic'` - selects based on circuit)[^2][^1]

**Global Configuration Parameters**:

- **Device**: `device='CPU'` or `device='GPU'` (NVIDIA CUDA support for statevector, density_matrix, unitary)[^38][^2]
- **Precision**: Single vs double (backend dependent)
- **Shots**: Number of measurement samples
- **Memory**: Max memory allocation
- **Seed**: RNG seed for reproducibility
- **Optimization level**: Circuit transpilation optimization (0-3)

**MPS-Specific Parameters**:

- `matrix_product_state_max_bond_dimension`: Maximum χ (default: None = unlimited)[^21][^19][^20]
- `matrix_product_state_truncation_threshold`: SVD cutoff threshold[^19][^21][^20]
- `enable_truncation`: Enable/disable truncation (boolean)[^20]
- `zero_threshold`: Treat elements below this as zero[^20]
- `validation_threshold`: MPS validity check threshold[^20]
- `chop_threshold`: Remove small tensor elements[^20]
- `mps_log_data`: Enable bond dimension logging (boolean)[^21]

**Noise Model Parameters**:

- **Gate errors**: `thermal_relaxation_error(T1, T2, gate_time)`, `depolarizing_error(p, num_qubits)`
- **Readout errors**: Measurement confusion matrix $2 \times 2$ per qubit[^13]
- **Error mitigation**: `measurement_error_mitigation=True`[^13]

**Hardware Backend Parameters** (Real QPU):

- **Backend name**: IBM quantum hardware (e.g., 'ibm_cairo', 'ibm_hanoi')[^13]
- **Coupling map**: Device connectivity graph
- **Basis gates**: Native gate set of device
- **Qubit properties**: T1, T2, gate fidelities per qubit[^13]

**Multi-GPU Distribution**: Via MPI for statevector method, supports up to 1024 nodes[^3]

**Can Build Custom Simulations**: Yes, via custom C++ simulators inheriting from Aer base classes

***

### **PennyLane (Xanadu)**

**Programming Language**: Python (with C++/CUDA backends)

**Simulation Methods via Devices**:

#### **default.qubit** (CPU software simulator)

- Pure Python/NumPy statevector simulation
- All quantum operations supported


#### **default.mixed**

- Density matrix simulation with noise


#### **lightning.qubit** (Optimized C++ statevector)

- AVX2/AVX512 vectorization
- OpenMP multithreading
- Parameter-shift and adjoint differentiation


#### **lightning.gpu** (CUDA statevector)

- NVIDIA GPU acceleration
- cuStateVec backend integration


#### **lightning.tensor** (Tensor network simulator)

- **MPS mode**: `method='mps'`
- **Exact TN mode**: `method='tn'`
- GPU-accelerated via cuTensorNet[^22][^10]


#### **lightning.kokkos** (Multi-backend portability)

- CPU, NVIDIA GPU, AMD GPU support via Kokkos abstraction

**Lightning Tensor MPS Parameters**:

- `max_bond_dim`: Maximum bond dimension (default: 128)[^10][^22]
- `cutoff`: Singular value truncation threshold (default: 0)[^22][^10]
- `cutoff_mode`: 'abs' (absolute) or 'rel' (relative)[^10][^22]
- `wires`: Number of qubits (100+ supported with MPS)[^10]
- `c_dtype`: `numpy.complex64` or `numpy.complex128` (default)[^22][^10]

**Differentiation Methods**:

- `diff_method='parameter-shift'`: Quantum parameter-shift rule[^22][^10]
- `diff_method='finite-diff'`: Finite difference
- `diff_method='adjoint'`: Adjoint state method (fast, exact for statevector)
- `diff_method='backprop'`: Classical backpropagation (for classical preprocessing)

**Quantum Neural Network Support**:

- Built-in QNN layers: `AngleEmbedding`, `AmplitudeEmbedding`, `IQPEmbedding`
- Trainable quantum templates
- Automatic differentiation through quantum circuits[^37]

**Quantum Feature Encoding Classes**:

- `AngleEmbedding`: Rotation encoding
- `AmplitudeEmbedding`: Amplitude encoding
- `IQPEmbedding`: Instantaneous Quantum Polynomial encoding
- `BasisEmbedding`: Computational basis encoding
- Custom feature maps via templates

**Hybrid Framework Integration**: Native integration with PyTorch, TensorFlow, JAX for hybrid quantum-classical optimization[^37]

**Hardware Backend Support**: IBM Quantum, Rigetti, IonQ, Amazon Braket (75% of public QPUs)[^39][^37]

**Can Build Custom Simulations**: Yes, via custom device plugins and quantum templates

***

### **NVIDIA CUDA-Q + cuQuantum Ecosystem**

**Programming Languages**: Python, C++[^40][^41][^42]

**Architecture**: Unified programming model for QPU-CPU-GPU heterogeneous computing[^43][^40]

#### **CUDA-Q Platform Components**:

**Compilation Pipeline**:

- MLIR (Multi-Level Intermediate Representation)
- LLVM-based optimization
- QIR (Quantum Intermediate Representation) output[^41][^40]

**Execution Backends**:

- **State vector simulators**: CPU and GPU
- **Tensor network simulators**: Via cuTensorNet
- **75% of public QPUs**: Direct hardware integration[^40]
- **Quantum Photonics support**: Integrated with Quantum Circuits Inc.[^44]

**CUDA-Q Specific Features**:

- **Qubit-agnostic programming**: Works across all qubit modalities (superconducting, ion trap, photonic)[^40]
- **Kernel-based execution**: Mark quantum functions as `__qpu__` kernels[^41]
- **Asynchronous execution**: Non-blocking quantum kernel launches[^42]
- **Dynamic parallelism**: Distribute shots across GPUs automatically


#### **cuQuantum SDK - cuStateVec (State Vector Library)**

**Purpose**: GPU-accelerated state vector operations[^45][^6][^5]

**Key Parameters**:

- **Data types**: `complex64` (CUDA_C_32F), `complex128` (CUDA_C_64F)[^5]
- **State vector size**: $2^n$ complex elements, n = number of qubits
- **Device memory**: Must fit in single GPU for single-GPU mode (~40 qubits max on A100 80GB)

**Batched State Vector Parameters**:

- `nSVs`: Number of state vectors in batch[^6][^5]
- `nIndexBits`: Qubits per state vector (all must be same)[^6][^5]
- `svStride`: Offset between consecutive state vectors (≥ $2^{nIndexBits}$)[^5][^6]
- Contiguous device memory allocation required[^5]

**Gate Application APIs**:

- Single-qubit gates: Parametric rotations (RX, RY, RZ), Pauli, Hadamard
- Multi-qubit gates: CNOT, CZ, SWAP, Toffoli, custom unitary matrices
- Batched gate application for multiple state vectors simultaneously

**Measurement APIs**:

- Computational basis measurement with collapse
- Batched measurement across multiple state vectors
- Sampling without collapse

**Multi-GPU Support**:

- Qubit reordering for optimized memory distribution
- MPI + CUDA for distributed state vector[^6]

**Integration**: Accessible via QuEST v4, PennyLane, CUDA-Q[^9][^4]

#### **cuQuantum SDK - cuTensorNet (Tensor Network Library)**

**Purpose**: Optimized tensor network contraction for quantum circuits[^46][^45][^32][^29][^31]

**Network Creation Parameters**:

- **Number of input tensors** (`numInputs`)[^31]
- **Modes per tensor** (`numModesIn[i]`): Tensor rank
- **Extents per mode** (`extentsIn[i][j]`): Dimension of each mode
- **Strides** (`stridesIn`): Memory layout (NULL = Fortran/column-major)[^31]
- **Mode labels** (`modesIn`): Einstein summation notation indices[^31]
- **Output modes** (`numModesOut`): -1 for auto-infer, 0 for full reduction[^31]

**Data Type Parameters**:

- **Data types**: `CUDA_R_16F`, `CUDA_R_16BF`, `CUDA_R_32F`, `CUDA_R_64F`, `CUDA_C_32F`, `CUDA_C_64F`[^31]
- **Compute types**:
    - `CUTENSORNET_COMPUTE_32F` (FP32, default for R_32F)
    - `CUTENSORNET_COMPUTE_TF32` (TensorFloat-32, Ampere+)
    - `CUTENSORNET_COMPUTE_3XTF32` (3× TF32 precision)
    - `CUTENSORNET_COMPUTE_16F` (FP16, Volta+)
    - `CUTENSORNET_COMPUTE_16BF` (BFloat16, Ampere+)
    - `CUTENSORNET_COMPUTE_64F` (FP64 double precision)[^32][^31]
- **Tensor Core usage**: Automatic based on data/compute type combination[^31]

**Contraction Optimizer Parameters**:

- **Workspace size constraint** (`workspaceSizeConstraint`): Max GPU memory available[^32][^31]
- **Optimizer attributes** (via `cutensornetContractionOptimizerConfig`):
    - Graph partitioning method
    - Seed for randomized optimization
    - Maximum iterations for path search
    - Time budget for optimization[^32][^31]

**Slicing Parameters**:

- **Slicing indices**: Select which modes to slice for parallelization[^29][^31]
- **Number of slices**: Auto-computed or user-specified
- **Memory/parallelism tradeoff**: More slices = less memory, more parallel work[^29]

**Contraction Path Optimization**:

- **Simplification phase**: Rank reduction, redundancy elimination[^29]
- **Division phase**: Recursive graph partitioning into contraction tree
- **Slicing phase**: Edge subset selection
- **Reconfiguration phase**: Subtree cost optimization with dynamic slicing[^29]
- **Hyper-optimization**: Dramatically faster than opt-einsum classical methods[^29]

**Autotune Parameters** (`cutensornetContractionAutotunePreference`):

- Time budget for autotuning kernel selection
- Coverage mode: Quick vs thorough[^32][^31]

**Output Information Available**:

- Total FLOP count
- Largest intermediate tensor size
- Number of sliced modes and their extents
- Estimated runtime
- Mode labels for all intermediate tensors[^31]

**Performance**: Achieves state-of-the-art contraction path finding; production-ready for QSVM with 10-100× speedup over CPU[^45][^46]

#### **cuQuantum SDK - cuTENSOR (Low-level Tensor Operations)**

- Pairwise tensor contractions (Einstein summation)
- Building block for cuTensorNet
- Direct GPU memory operations with optimal kernel selection

***

### **ITensor (Tensor Network Framework)**

**Programming Languages**: C++ (original), Julia (ITensor.jl)[^25][^27][^23][^24][^28]

**Core Capabilities**:

- Matrix Product States (MPS) and Matrix Product Operators (MPO)
- DMRG (Density Matrix Renormalization Group) optimization
- TEBD (Time Evolving Block Decimation)
- 2D tensor networks (PEPS support)
- Custom tensor network architectures[^27][^23][^25]

**MPS/MPO Parameters**:

- **Bond dimension** (`maxdim`): Maximum χ allowed during truncation[^23][^24]
- **Cutoff** (`cutoff`): SVD truncation threshold ($10^{-8}$ to $10^{-12}$ typical)[^24][^23]
- **Minimum dimension** (`mindim`): Minimum bond dimension to keep
- **Absolute cutoff** (`absoluteCutoff`): Boolean for absolute vs relative cutoff
- **Respect degenerate**: Keep degenerate singular values together

**DMRG Parameters**:

- **Number of sweeps**: Optimization iterations through the system
- **Energy convergence**: Stop when energy change < threshold
- **Noise term**: Add noise to escape local minima (0-1e-10 range)
- **Observer objects**: Custom measurements during DMRG sweeps[^23][^24][^18]

**TEBD Parameters**:

- **Time step** (`τ` or `dt`): Trotter decomposition step size
- **Trotter order**: First-order, second-order, fourth-order
- **Total time**: Evolution duration
- **Gate cutoff**: Truncation after each gate application[^18]

**Quantum Circuit Simulation**:

- Express circuits as sequences of MPO gates
- Contract MPO into MPS for state evolution
- Not pre-built for ML; requires custom implementation[^25]

**Index/Tensor Manipulation**:

- Prime indices for tensor product differentiation
- Automatic index contraction via matching labels
- QN (quantum number) conserving tensors for symmetry exploitation[^24][^18]

**Charge Conservation (Abelian Symmetries)**:

- U(1) symmetry: Particle number conservation
- Z2 symmetry: Parity conservation
- Tensor blocking by quantum numbers for massive speedup[^18]

**Can Build Custom Simulations**: Yes, highly flexible tensor network library for custom algorithm development. Production-ready for classical tensor network simulation, not turnkey for quantum ML[^28][^25]

***

### **Cirq (Google Quantum AI)**

**Programming Language**: Python[^15][^14]

**Simulation Methods**:

1. **Cirq Simulator**: Pure state vector simulation with NumPy backend[^14][^15]
2. **DensityMatrixSimulator**: Mixed state simulation
3. **CliffordSimulator**: Stabilizer tableau simulation
4. **Clifford+T Simulator**: For fault-tolerant circuits

**Simulator Parameters** (`cirq.Simulator`):

- **Seed**: RNG seed for stochastic operations
- **Split unitary**: Automatically split large multi-qubit gates
- **Noise model**: Custom noise via `cirq.NoiseModel` class
- **Dtype**: `numpy.complex64` or `numpy.complex128`

**Simulation Methods**:

- `simulate()`: Return final state vector
- `simulate_expectation_values()`: Compute observable expectations
- `simulate_sweep()`: Parameter sweep over circuit parameters[^14]

**Batched Execution**:

- `cirq.ParamResolver`: Map symbolic parameters to values
- Loop over parameter sets (no native batching, requires manual iteration)[^7]

**Noise Modeling**:

- `cirq.DepolarizingChannel(p)`
- `cirq.AmplitudeDampingChannel(gamma)`
- `cirq.PhaseDampingChannel(gamma)`
- Custom Kraus operators

**Hardware Integration**:

- Google Quantum Engine (Sycamore, Weber processors)
- Access via `cirq_google` package
- Device-specific gate sets and topology constraints

**Can Build Custom Simulations**: Yes, implement custom `cirq.SimulatesSamples` or `cirq.SimulatesIntermediateState` classes[^15][^14]

***

### **TensorFlow Quantum (Google)**

**Programming Language**: Python (C++ optimized backend)[^8][^47][^7]

**Architecture**: Cirq circuits as TensorFlow tensors for hybrid quantum-classical ML[^47][^8][^7]

**Primitives**:

- **Quantum circuit batches**: `cirq.Circuit` objects as `tf.Tensor` with varying sizes[^8][^47]
- **Pauli sum operators**: `cirq.PauliSum` as `tf.Tensor` batches[^47][^8]

**Core Operations**:

1. **tfq.layers.Expectation**: Batched expectation value computation[^7]
2. **tfq.layers.Sample**: Sample measurement outcomes from circuits
3. **tfq.layers.State**: Output state vectors from circuits

**Simulator Backend**:

- Optimized C++ simulators from Google's quantum supremacy experiment[^8]
- AVX2 and SSE instruction sets for vectorization[^8]
- Significantly faster than pure Python/NumPy Cirq simulation

**Batching Semantics**:

- Native batching following TensorFlow conventions
- Circuits of different sizes in same batch (dynamic graph support)[^8]
- Parallel execution across batch dimension

**Differentiation**:

- Parameter-shift rule for quantum gradients
- Integration with `tf.GradientTape`
- Hybrid quantum-classical backpropagation[^7]

**Hardware Backend**:

- Swap between simulator and physical QPU by changing one line[^8]
- Seamless Cirq device integration

**Limitations**:

- Requires TensorFlow ecosystem
- Less flexible than standalone Cirq for custom simulators
- Primarily focused on hybrid QML workflows[^47][^8]

***

### **QuEST (Quantum Exact Simulation Toolkit)**

**Programming Languages**: C and C++[^48][^49][^50][^4][^9]

**Architecture**: Hybrid multithreading + GPU + distributed simulation[^49][^4][^9]

**Parallelization Backends** (can combine any combination):

- **OpenMP**: Multithreading across CPU cores
- **MPI**: Distribution across compute nodes
- **CUDA**: NVIDIA GPU acceleration
- **HIP**: AMD GPU acceleration
- **Thrust**: GPU algorithm library
- **cuQuantum**: NVIDIA cuStateVec/cuTensorNet integration (v4)[^4][^9]
- **GPUDirect**: Direct GPU-GPU communication without CPU[^9][^4]

**Simulation Types**:

1. **Statevector** (`Qureg` in state vector mode)
2. **Density matrix** (`Qureg` in density matrix mode)

**Precision Parameters**:

- **qreal**: Single, double, or quad precision for real numbers[^4]
- **qcomp**: Single, double, or quad precision for complex numbers[^4]
- Compile-time selection via CMake flags

**Qureg Parameters** (`createQureg`):

- **Number of qubits**: System size
- **Environment**: `QuESTEnv` containing hardware config
- **Precision**: Selected at compile time

**Automatic Hardware Selection**:

- Runtime detection of optimal hardware (CPU cores, GPUs, network)[^4]
- Abstracted behind single unified API[^4]
- User doesn't specify parallelization in code

**Gate Operations**:

- All standard single and multi-qubit gates
- Parametric rotations with arbitrary angles
- Custom unitary matrices
- Controlled gates with arbitrary number of controls

**Density Matrix Specific**:

- `mixDamping`: Damping/amplitude damping noise[^4]
- `mixDepolarising`: Depolarizing noise
- `mixPauli`: Pauli channel noise
- Custom Kraus operators via `mixKrausMap`

**Performance**:

- Achieved 45 qubit simulation (2017 record)[^49][^11]
- Featured in Scientific Reports Top 100 Physics papers[^9]
- Comparable performance to ProjectQ (state-of-art)[^11]

**Hamiltonian Support**:

- `initDiagonalOpFromPauliHamil`: Initialize diagonal operators from Pauli Hamiltonians[^4]
- Efficient time evolution under Hamiltonians

**Can Build Custom Simulations**: Yes, flexible C/C++ API for custom quantum algorithms. Direct access to state vector amplitudes. Suitable for HPC deployment[^49][^9][^4]

***

### **Yao.jl (Julia Quantum Framework)**

**Programming Language**: Pure Julia[^12][^11]

**Architecture**: Generic and differentiable quantum programming[^12]

**Core Abstractions**:

- **Quantum block intermediate representation**: Circuits as composable blocks[^12]
- **Batched quantum registers**: GPU acceleration for multiple quantum states
- **Builtin automatic differentiation**: Optimized for reversible computing[^12]

**Performance**:

- State-of-art performance comparable to ProjectQ
- Significantly less overhead than ProjectQ (benefits from Julia JIT)[^11][^12]
- Multiple dispatch enables algorithmic specialization without touching backend[^11]

**Repeated Blocks Optimization**:

- Dramatically better performance for repeated gate patterns
- Automatic detection and optimization of common subcircuits[^11]

**Simulation Methods**:

- State vector (full and batched)
- Density matrix
- MPS (via tensor network extensions)

**Differentiation**:

- Native automatic differentiation for variational algorithms
- Parameter-shift rule
- Adjoint method

**GPU Support**:

- CUDA backend for GPU acceleration
- Batched register operations for parallel state simulation[^12]

**Extensibility**:

- Highly extensible framework design
- Easy to add custom gates, noise models, measurements
- Generic programming paradigm[^12]

**Can Build Custom Simulations**: Yes, Julia metaprogramming enables deep customization and performance optimization[^11][^12]

***

### **ProjectQ**

**Programming Language**: Python (C++ backend)[^11]

**Architecture**: Quantum computing software framework with emphasis on compilation and optimization[^11]

**Parallelization**:

- **SIMD**: Single instruction multiple data vectorization
- **OpenMP**: Multithreading
- **MPI**: Distributed memory parallelism[^11]

**Performance**: Held 45-qubit simulation record for several months; state-of-art performance[^11]

**Compiler Optimization**:

- Automatic circuit optimization passes
- Gate fusion and cancellation
- Platform-specific compilation

**Backend Support**:

- Local simulator (C++ optimized)
- IBM Quantum hardware
- Rigetti hardware
- Ion trap systems

**Parameters**: Standard quantum circuit parameters (gates, qubits, measurements)

**Can Build Custom Simulations**: Yes, via custom compiler engines and backend simulators[^11]

***

### **Qrack (High-performance Simulator)**

**Programming Language**: C++ with Python bindings[^34]

**Unique Features**:

- **Hybrid stabilizer/state vector**: Automatic switching between representations[^34]
- **Schmidt decomposition optimization**: Decomposes state into tensor product when possible[^34]
- **Tunable approximation** (SDRP): Schmidt Decomposition Rounding Parameter for controlled accuracy/performance tradeoff[^34]

**Performance Achievements**:

- 27 qubit exact QFT on single laptop
- 54 qubit approximate random circuits (10 layers, ~4% fidelity) on single A100 GPU[^34]

**Optimization Techniques**:

- Circuit simplification before execution
- Dynamic detection of separable subsystems
- Aggressive gate cancellation[^34]

**GPU Support**: NVIDIA CUDA acceleration

**Approximation Control**: SDRP parameter controls accuracy vs speed (lower SDRP = more approximation, faster execution)[^34]

**Can Build Custom Simulations**: Yes, C++ library with extensive API[^34]

***

### **TeNPy (Tensor Network Python)**

**Programming Language**: Pure Python with NumPy/SciPy[^18]

**Focus**: One-dimensional tensor product state algorithms for condensed matter physics[^18]

**Core Algorithms**:

1. **TEBD** (Time Evolving Block Decimation)[^18]
2. **DMRG** (Density Matrix Renormalization Group)[^18]
3. **iTEBD** (infinite TEBD)[^18]
4. **iDMRG** (infinite DMRG)[^18]

**MPS Parameters**:

- **chi_max**: Maximum bond dimension
- **svd_min**: Minimum singular value to keep (cutoff)
- **trunc_cut**: Truncation error threshold
- **Canonical form**: Left, right, mixed canonical MPS forms[^18]

**Charge Conservation**:

- Abelian symmetries (U(1), Z2, etc.)
- Tensor blocking by quantum numbers
- Accelerates operations via symmetry sectors[^18]

**Lattice Models**:

- Predefined lattices: Chain, ladder, square, triangular, honeycomb
- Custom lattice definitions
- Periodic, open, infinite boundary conditions

**Hamiltonian Representation**:

- MPO (Matrix Product Operator) format
- NearestNeighborModel for local interactions
- Long-range interaction support

**Can Build Custom Simulations**: Yes, modular Python library. Educational and research-oriented. Not optimized for quantum ML specifically[^18]

***

### **Quantum Circuit Simulation Memory/Qubit Limits Summary**

| **Hardware Configuration** | **State Vector Qubits** | **MPS Qubits (χ=128)** | **Memory** |
| :-- | :-- | :-- | :-- |
| Laptop (64 GB RAM) | 31 | 100+ | 64 GB |
| Desktop (256 GB RAM) | 33 | 100+ | 256 GB |
| Single GPU Node (LUMI, 480 GB usable) | 34 | 200+ | 480 GB |
| Single A100 GPU (80 GB VRAM) | 35-36 | 150+ | 80 GB |
| 64 GPU Nodes (32 TB total) | 40 | 500+ | 32 TB |
| 1024 GPU Nodes (256 TB total) | 44 | 1000+ | 256 TB |

[^3]

***

## **Custom-Buildable Simulation Techniques**

### **cuTENSOR + Custom MPS/MPO Gates**

**Approach**: Use NVIDIA cuTENSOR for low-level tensor contractions, manually implement MPS evolution with quantum gates as MPOs.[^29]

**Parameters You Control**:

- Tensor contraction order (einsum notation)
- Memory layout (row-major vs column-major)
- GPU streams for asynchronous execution
- Bond dimension cutoff policy
- SVD algorithm selection
- Truncation strategy (absolute, relative, entropy-based)

**Advantages**: Maximum performance control, GPU acceleration, custom truncation schemes

**Complexity**: High - requires deep understanding of tensor networks and CUDA programming

***

### **ITensor + Custom Quantum Feature Encoding**

**Approach**: Use ITensor's MPS/MPO infrastructure, implement quantum gates as MPOs, build custom feature encoding circuits.[^27][^25][^23]

**Parameters You Control**:

- All MPS/DMRG parameters (cutoff, maxdim, sweeps, noise)
- Custom gate decompositions
- Feature map design (rotation angles, entanglement patterns)
- SVD truncation precisely to optimize recall (as you've done for fraud detection)

**Advantages**: Flexible, production-ready tensor network backend, precise control for your use case

**Complexity**: Medium - good documentation, active community, Julia/C++ expertise required

***

### **Multi-Framework Hybrid Pipelines**

**Example Architecture**:

1. **cuTensorNet path optimization** → Find optimal contraction order
2. **Export to Qiskit/PennyLane** → Execute with established quantum ML features
3. **cuStateVec acceleration** → GPU-accelerate statevector components

**Parameters**: Combination of all framework parameters across pipeline stages

***

## **Recommendations for Your Fraud Detection System**

Given your background with **ITensor, SVD cutoff optimization, and production fraud detection**, here's the optimal technical stack:

### **Primary Framework: Qiskit with cuQuantum Backend**

- Use **Qiskit** for quantum feature encoding (ZZFeatureMap with depth=2, circular entanglement)
- Enable **NVIDIA cuQuantum** backend for GPU acceleration during training
- Leverage proven 94.3% accuracy configuration from CERN[^13]


### **Feature Engineering: Custom ITensor Pipeline**

- Continue using **ITensor** for SVD-based feature selection (complementary to quantum features)
- Your cutoff tuning expertise (10⁻⁸ to 10⁻¹²) directly applies to MPS bond dimension optimization
- Classical tensor network preprocessing before quantum encoding


### **Production Deployment: PennyLane Lightning Tensor**

- **MPS mode** with `max_bond_dim=64-128`, `cutoff=1e-10` for controllable approximation
- GPU acceleration for real-time inference
- Hybrid integration with your existing classical pipeline[^10][^22]


### **Parameter Configuration for Optimal Fraud Detection**:

```python
# Qiskit Quantum Feature Encoding
ZZFeatureMap(
    feature_dimension=7,
    reps=2,
    entanglement='circular'
)

# PennyLane Production MPS
device_kwargs = {
    'max_bond_dim': 64,
    'cutoff': 1e-10,
    'cutoff_mode': 'abs'
}

# ITensor Classical TN Preprocessing
mps_params = {
    'maxdim': 50,
    'cutoff': 1e-10,
    'sweeps': 5
}
```

This configuration provides **maximum controllability** for the realistic fraud detection parameters that have **proven evidence** of superior performance (81.7% fraud detection rate, 94.3% accuracy).[^51][^13]
<span style="display:none">[^52][^53][^54][^55][^56][^57]</span>

<div align="center">⁂</div>

[^1]: https://www.scribd.com/document/838453751/Simulators-Qiskit-Aer-0-15-0

[^2]: https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html

[^3]: https://fiqci.fi/publications/2025-04-01-LUMI-quantum-simulations-qiskit-aer

[^4]: https://quest-kit.github.io/QuEST/index.html

[^5]: https://docs.nvidia.com/cuda/cuquantum/24.03.1/custatevec/overview.html

[^6]: https://docs.nvidia.com/cuda/cuquantum/23.06.1/custatevec/overview.html

[^7]: https://www.tensorflow.org/quantum/tutorials/hello_many_worlds

[^8]: https://www.tensorflow.org/quantum/design

[^9]: https://github.com/quest-kit/QuEST

[^10]: https://docs.pennylane.ai/projects/lightning/en/latest/lightning_tensor/device.html

[^11]: https://docs.yaoquantum.org/v0.1/dev/benchmark/

[^12]: https://quantum-journal.org/papers/q-2020-10-11-341/

[^13]: https://quantum.cern/sites/default/files/2022-11/Mixed_QuantumClassical_Method_for_Fraud_Detection_With_Quantum_Feature_Selection.pdf

[^14]: https://quantumai.google/cirq/simulate/simulation

[^15]: https://quantumai.google/reference/python/cirq/Simulator

[^16]: https://arxiv.org/html/2503.08626v3

[^17]: https://pennylane.ai/qml/demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks

[^18]: https://scipost.org/SciPostPhysLectNotes.5/pdf

[^19]: https://github.com/Qiskit/qiskit-aer/issues/2286

[^20]: https://stackoverflow.com/questions/79280458/qiskit-exact-simulation-with-matrix-product-states-method

[^21]: https://qiskit.qotlabs.org/docs/api/qiskit-addon-aqc-tensor/simulation-aer-qiskit-aer-simulation-settings

[^22]: https://docs.pennylane.ai/projects/lightning/en/stable/code/api/pennylane_lightning.lightning_tensor.LightningTensor.html

[^23]: https://itensor.github.io/ITensors.jl/v0.1/MPSandMPO.html

[^24]: https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html

[^25]: https://scipost.org/SciPostPhysCodeb.4/pdf

[^26]: https://arxiv.org/html/2401.06188v2

[^27]: https://github.com/ITensor/ITensorMPS.jl

[^28]: https://itensor.org

[^29]: https://www.nersc.gov/assets/Uploads/11-cuQuantum-Kim.pdf

[^30]: https://dl.acm.org/doi/10.1145/3547334

[^31]: https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/functions.html

[^32]: https://docs.nvidia.com/cuda/cuquantum/22.03.0/cutensornet/api/functions.html

[^33]: https://www.cda.cit.tum.de/files/eda/2021_stochastic_quantum_circuit_simulation_using_decision_diagrams.pdf

[^34]: https://arxiv.org/pdf/2304.14969.pdf

[^35]: https://www.classiq.io/algorithms/quantum-monte-carlo

[^36]: https://arxiv.org/html/2308.00044v2

[^37]: https://wqs.events/sdk-showdown-pennylane-0-42-vs-qiskit-1-0-a-comprehensive-comparison/

[^38]: https://docs.quantum.ibm.com/api/qiskit/0.40/qiskit_aer.AerSimulator

[^39]: https://abbdm.com/index.php/Journal/article/download/232/182/1226

[^40]: https://developer.nvidia.com/cuda-q

[^41]: https://nvidia.github.io/cuda-quantum/latest/using/cpp.html

[^42]: https://developer.nvidia.com/blog/nvidia-cuda-q-0-12-expands-toolset-for-developing-hardware-performant-quantum-applications/

[^43]: https://developer.nvidia.com/blog/new-nvidia-cuda-q-features-boost-quantum-application-performance/

[^44]: https://quantumcircuits.com/resources/quantum-circuits-integrates-with-nvidia/

[^45]: https://developer.nvidia.com/cuquantum-sdk

[^46]: https://arxiv.org/html/2405.02630v1

[^47]: https://blog.mlq.ai/tensorflow-quantum-introduction/

[^48]: https://github.com/QuEST-Kit/QuEST

[^49]: https://www.nature.com/articles/s41598-019-47174-9

[^50]: https://github.com/QuEST-Kit/QuEST/blob/main/README.md

[^51]: https://arxiv.org/html/2509.25245v1

[^52]: https://arxiv.org/pdf/2503.04423.pdf

[^53]: https://library.oapen.org/bitstream/handle/20.500.12657/96026/9788898587049.pdf?sequence=4

[^54]: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html

[^55]: https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aer_simulator.py

[^56]: https://github.com/tensorflow/quantum/issues/235

[^57]: https://www.cs.ucdavis.edu/~bai/QUEST/manual.pdf

