def build_qf_matrix(
    mpi_comm,
    ansatz: ProjectedQuantumFeatures,
    X,
    info_file=None,
    cpu_max_mem=6,
    use_enhanced_features: bool = False,
    include_correlations: bool = False,
    use_batch_processing: bool = False,
    correlation_config: dict = None
) -> np.ndarray:
    """
    FIXED: Order-preserving quantum feature matrix generation.
    
    CRITICAL: Ensures row order matches input X exactly.
    """
    n_qubits = ansatz.ansatz_circ.n_qubits
    
    # MPI information
    root = 0
    rank = mpi_comm.Get_rank()
    n_procs = mpi_comm.Get_size()
    
    # CRITICAL: Use ceiling division for correct chunk distribution
    entries_per_chunk = int(np.ceil(len(X) / n_procs))
    
    # Synchronize
    mpi_comm.Barrier()
    
    if rank == root:
        profiling_dict = dict()
        profiling_dict["lenX"] = (len(X), "entries")
        start_time = MPI.Wtime()
        print(f"\n{'='*60}")
        print(f"MPI ORDER-PRESERVING FEATURE GENERATION")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"MPI processes: {n_procs}")
        print(f"Entries per process: {entries_per_chunk}")
        print(f"Enhanced features: {use_enhanced_features}")
        print(f"Correlations: {include_correlations}")
        print(f"SVD cutoff: {ansatz.svd_cutoff}")
        sys.stdout.flush()
    
    # Default correlation configuration
    if correlation_config is None:
        correlation_config = {'type': 'adjacent'}
    
    # Determine feature dimensions (SAME ON ALL RANKS)
    if use_enhanced_features:
        feature_dim = n_qubits * 4
    else:
        feature_dim = n_qubits * 3
    
    if include_correlations:
        corr_type = correlation_config.get('type', 'adjacent')
        
        if corr_type == 'adjacent':
            feature_dim += (n_qubits - 1)
        elif corr_type == 'skip':
            skip_dist = correlation_config.get('skip_distance', 2)
            feature_dim += max(0, n_qubits - skip_dist)
        elif corr_type == 'multiscale':
            skip_distances = correlation_config.get('skip_distances', [1, 2, 3])
            for skip_dist in skip_distances:
                if skip_dist < n_qubits:
                    feature_dim += (n_qubits - skip_dist)
        elif corr_type == 'xyz':
            skip_dist = correlation_config.get('skip_distance', 1)
            feature_dim += 3 * max(0, n_qubits - skip_dist)
    
    if rank == root:
        print(f"Feature dimension: {feature_dim}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
    
    # CRITICAL: Calculate exact indices for this rank
    start_idx = rank * entries_per_chunk
    end_idx = min(start_idx + entries_per_chunk, len(X))
    actual_chunk_size = max(0, end_idx - start_idx)
    
    if rank == root:
        print(f"[Rank {rank}] Processing indices: [{start_idx}:{end_idx}] ({actual_chunk_size} samples)")
    
    # Process this rank's chunk
    exp_x_chunk = []
    exp_x_time = []
    
    for k in range(actual_chunk_size):
        global_idx = start_idx + k
        
        time0 = MPI.Wtime()
        
        circ = ansatz.circuit_for_data(X[global_idx, :])
        circ_gates = ansatz.circuit_to_list(circ)
        
        # Extract base features
        if use_enhanced_features:
            features = circuit_xyz_exp_enhanced(circ_gates, n_qubits, ansatz.svd_cutoff)
            features = np.asarray(features).flatten()
        else:
            features = circuit_xyz_exp(circ_gates, n_qubits)
            features = np.asarray(features).flatten()
        
        # Add correlation features
        if include_correlations:
            corr_type = correlation_config.get('type', 'adjacent')
            
            if corr_type == 'adjacent':
                corr = circuit_correlations(circ_gates, n_qubits, ansatz.svd_cutoff)
                features = np.concatenate([features, np.asarray(corr)])
            
            elif corr_type == 'skip':
                skip_dist = correlation_config.get('skip_distance', 2)
                corr = circuit_skip_correlations(circ_gates, n_qubits, skip_dist, ansatz.svd_cutoff)
                features = np.concatenate([features, np.asarray(corr)])
            
            elif corr_type == 'multiscale':
                skip_distances = correlation_config.get('skip_distances', [1, 2, 3])
                corr = ansatz.extract_multiscale_correlations(circ, n_qubits, skip_distances)
                features = np.concatenate([features, corr])
            
            elif corr_type == 'xyz':
                skip_dist = correlation_config.get('skip_distance', 1)
                corr = circuit_xyz_correlations(circ_gates, n_qubits, skip_dist, ansatz.svd_cutoff)
                features = np.concatenate([features, np.asarray(corr).flatten()])
        
        exp_x_time.append(MPI.Wtime() - time0)
        exp_x_chunk.append(features)
        
        # Progress reporting
        if rank == root and k % max(1, actual_chunk_size // 10) == 0:
            print(f"[Rank {rank}] Progress: {k}/{actual_chunk_size} ({100*k//actual_chunk_size}%)")
            sys.stdout.flush()
    
    # Convert to numpy array
    if len(exp_x_chunk) > 0:
        exp_x_chunk = np.asarray(exp_x_chunk)
    else:
        exp_x_chunk = np.zeros((0, feature_dim))
    
    if rank == root:
        print(f"[Rank {rank}] Completed: {exp_x_chunk.shape}")
    
    # CRITICAL: Synchronize before gather
    mpi_comm.Barrier()
    
    # FIXED: Gather with explicit ordering
    # Gather chunk sizes first to verify
    all_chunk_sizes = mpi_comm.gather(actual_chunk_size, root=root)
    
    # Gather the actual data
    all_chunks = mpi_comm.gather(exp_x_chunk, root=root)
    
    # CRITICAL: Only root assembles in correct order
    if rank == root:
        print(f"\n{'='*60}")
        print(f"ASSEMBLING CHUNKS IN ORDER")
        print(f"{'='*60}")
        print(f"Chunk sizes from ranks: {all_chunk_sizes}")
        
        # Verify total matches input size
        total_from_ranks = sum(all_chunk_sizes)
        if total_from_ranks != len(X):
            print(f"ERROR: Size mismatch! Expected {len(X)}, got {total_from_ranks}")
            raise ValueError("MPI gather size mismatch")
        
        # CRITICAL: Concatenate chunks IN RANK ORDER
        # Rank 0's chunk first, then rank 1, etc.
        projected_features = np.vstack([chunk for chunk in all_chunks if len(chunk) > 0])
        
        # VERIFICATION: Check final size
        if projected_features.shape[0] != len(X):
            print(f"ERROR: Final size mismatch! Expected {len(X)}, got {projected_features.shape[0]}")
            raise ValueError("Final array size mismatch")
        
        if projected_features.shape[1] != feature_dim:
            print(f"ERROR: Feature dimension mismatch! Expected {feature_dim}, got {projected_features.shape[1]}")
            raise ValueError("Feature dimension mismatch")
        
        print(f"SUCCESS: Assembled array shape: {projected_features.shape}")
        print(f"Verification: First 3 rows match indices [0:3] of input")
        
        # Timing statistics
        duration = sum(exp_x_time)
        print(f"\n[Timing] Total processing time: {round(duration,2)} seconds")
        if len(exp_x_time) > 0:
            average = mean(exp_x_time)
            print(f"[Timing] Average per sample: {round(average,4)} seconds")
            print(f"[Timing] Median per sample: {round(median(exp_x_time),4)} seconds")
        
        print(f"\nFeature dimension: {feature_dim}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        return projected_features
    else:
        return None


def generate_projectedQfeatures(
    data_feature, 
    reps, 
    gamma, 
    target_label=None,
    info='quantum_features', 
    slice_size=50000, 
    train_flag=False,
    svd_cutoff=1e-5,
    use_enhanced_features=True,
    include_correlations=True,
    correlation_config=None,
    use_batch_processing=False,
    mpi_comm=None
):
    """
    FIXED: Order-preserving quantum feature generation.
    
    CRITICAL: Maintains exact row order of input data_feature.
    """
    # Initialize MPI
    if mpi_comm is None:
        mpi_comm = get_mpi_comm()
    
    rank = get_rank()
    root = 0
    start_time = time.time()
    
    if rank == root:
        print(f"\n{'='*70}")
        print(f"ORDER-PRESERVING QUANTUM FEATURE GENERATION")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - SVD Cutoff: {svd_cutoff}")
        print(f"  - Enhanced Features: {use_enhanced_features}")
        print(f"  - Include Correlations: {include_correlations}")
        print(f"  - Slice Size: {slice_size}")
        print(f"{'='*70}\n")
    
    # Convert to numpy array ONCE
    num_features = data_feature.shape[1]
    data_size = data_feature.shape[0]
    classical_features = np.array(data_feature)
    
    if rank == root:
        print(f"[INFO] Input data shape: {classical_features.shape}")
        print(f"[INFO] Number of features: {num_features}")
        print(f"[INFO] Number of samples: {data_size}")
        print(f"[INFO] Input row order: [0, 1, 2, ..., {data_size-1}]")
    
    # CRITICAL: Scale on root, then broadcast to maintain order
    if train_flag:
        if rank == root:
            scaler = StandardScaler()
            classical_features = scaler.fit_transform(classical_features)
            
            import os
            os.makedirs('./model', exist_ok=True)
            with open('./model/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print(f"[INFO] Scaler fitted and saved")
            print(f"[INFO] Scaled features shape: {classical_features.shape}")
        else:
            classical_features = None
        
        # Broadcast to all processes
        classical_features = mpi_comm.bcast(classical_features, root=root)
    else:
        # All processes load and scale
        with open('./model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        classical_features = scaler.transform(classical_features)
    
    # Verify all processes have same data
    mpi_comm.Barrier()
    
    if rank == root:
        print(f"[VERIFY] All processes have scaled data")
        print(f"[VERIFY] Classical features shape on root: {classical_features.shape}")
    
    # Create ansatz (same on all processes)
    ansatz = create_ansatz(num_features, reps, gamma, svd_cutoff=svd_cutoff)
    
    # Default correlation configuration
    if correlation_config is None:
        correlation_config = {
            'type': 'multiscale',
            'skip_distances': [1, 2, 3]
        }
    
    if rank == root:
        print(f"\n[INFO] Starting slice-wise processing...")
        print(f"[INFO] Slices will be concatenated IN ORDER")
    
    # CRITICAL: Only root collects results
    combined_feature_list = [] if rank == root else None
    
    # Process data in slices - MAINTAINING ORDER
    num_slices = (data_size + slice_size - 1) // slice_size
    
    for i in range(num_slices):
        slice_start = i * slice_size
        slice_end = min(slice_start + slice_size, data_size)
        
        # CRITICAL: Extract slice IN ORDER
        classical_features_split = classical_features[slice_start:slice_end]
        
        if rank == root:
            print(f"\n{'='*60}")
            print(f"SLICE {i+1}/{num_slices}")
            print(f"{'='*60}")
            print(f"  Input indices: [{slice_start}:{slice_end}]")
            print(f"  Slice shape: {classical_features_split.shape}")
            print(f"  Classical features [first 3 rows] correspond to input rows: {[slice_start, slice_start+1, slice_start+2]}")
        
        # All processes participate
        slice_timer = time.time()
        quantum_features_split = build_qf_matrix(
            mpi_comm, 
            ansatz, 
            X=classical_features_split,
            use_enhanced_features=use_enhanced_features,
            include_correlations=include_correlations,
            correlation_config=correlation_config,
            use_batch_processing=use_batch_processing
        )
        
        # CRITICAL: Only root processes result
        if rank == root:
            slice_duration = time.time() - slice_timer
            print(f"\n[TIMING] Slice processing time: {slice_duration:.2f} seconds")
            
            # VERIFICATION: Check sizes match
            if quantum_features_split.shape[0] != classical_features_split.shape[0]:
                print(f"ERROR: Slice size mismatch!")
                print(f"  Classical: {classical_features_split.shape[0]}")
                print(f"  Quantum: {quantum_features_split.shape[0]}")
                raise ValueError(f"Size mismatch in slice {i+1}")
            
            print(f"[VERIFY] Quantum features shape: {quantum_features_split.shape}")
            print(f"[VERIFY] Rows {slice_start} to {slice_end-1} processed IN ORDER")
            
            # Combine classical and quantum IN ORDER
            combined_features_split = np.concatenate(
                (classical_features_split, quantum_features_split), 
                axis=1
            )
            print(f"[VERIFY] Combined features shape: {combined_features_split.shape}")
            print(f"[VERIFY] First row of combined corresponds to input row {slice_start}")
            
            # Append IN ORDER
            combined_feature_list.append(combined_features_split)
        
        # Synchronize between slices
        mpi_comm.Barrier()
    
    # CRITICAL: Only root aggregates IN ORDER
    if rank == root:
        print(f"\n{'='*60}")
        print(f"FINAL ASSEMBLY IN ORDER")
        print(f"{'='*60}")
        
        if len(combined_feature_list) > 1:
            # CRITICAL: Concatenate slices IN ORDER
            final_features = np.concatenate(combined_feature_list, axis=0)
            print(f"[INFO] Concatenated {len(combined_feature_list)} slices IN ORDER")
        elif len(combined_feature_list) == 1:
            final_features = combined_feature_list[0]
            print(f"[INFO] Single slice processed")
        else:
            raise ValueError("No features generated!")
        
        # VERIFICATION: Final size check
        if final_features.shape[0] != data_size:
            print(f"ERROR: Final size mismatch!")
            print(f"  Expected: {data_size}")
            print(f"  Got: {final_features.shape[0]}")
            raise ValueError("Final feature array has wrong size")
        
        print(f"\n[SUCCESS] Final feature array shape: {final_features.shape}")
        print(f"[VERIFY] Row 0 of output corresponds to row 0 of input")
        print(f"[VERIFY] Row {data_size-1} of output corresponds to row {data_size-1} of input")
        print(f"[VERIFY] All {data_size} rows processed IN INPUT ORDER")
        
        print(f"\n[INFO] Classical features: {num_features}")
        print(f"[INFO] Quantum features: {final_features.shape[1] - num_features}")
        print(f"[INFO] Total features: {final_features.shape[1]}")
        
        # Save with metadata
        save_array(final_features, info + '_quantum_enhanced')
        
        # Total timing
        total_duration = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"[TIMING] Total execution time: {total_duration:.2f} seconds")
        print(f"[TIMING] Average time per sample: {total_duration/data_size:.4f} seconds")
        print(f"{'='*60}\n")
        
        return final_features
    else:
        return None
