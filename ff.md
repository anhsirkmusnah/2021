import numpy as np
import pandas as pd
import pickle

def validate_order_preservation(
    original_csv_path='100k.csv',
    pqf_array_path='./pqf_arr/train_1.1.25_quantum_enhanced.npy',
    scaler_path='./model/scaler.pkl',
    target_label='fraud_label',
    top_n_features=216,
    n_samples_to_check=100
):
    """
    Comprehensive validation that quantum feature generation preserves row order.
    
    Returns:
        dict: Validation results with detailed diagnostics
    """
    print("\n" + "="*70)
    print("ORDER PRESERVATION VALIDATION")
    print("="*70 + "\n")
    
    # 1. Load original data
    print("[STEP 1/5] Loading original CSV data...")
    original_data = pd.read_csv(original_csv_path)
    print(f"  Original data shape: {original_data.shape}")
    
    # 2. Extract and scale classical features (EXACTLY as pipeline does)
    print("\n[STEP 2/5] Extracting and scaling classical features...")
    if target_label in original_data.columns:
        classical_features = original_data.drop([target_label], axis=1)
    else:
        classical_features = original_data
    
    print(f"  Classical features shape: {classical_features.shape}")
    
    # Apply same scaling as pipeline
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    classical_features_scaled = scaler.transform(classical_features)
    print(f"  Scaled features shape: {classical_features_scaled.shape}")
    
    # 3. Load PQF array
    print("\n[STEP 3/5] Loading PQF array...")
    pqf_array = np.load(pqf_array_path)
    print(f"  PQF array shape: {pqf_array.shape}")
    
    # 4. Extract classical features from PQF array
    print("\n[STEP 4/5] Extracting classical features from PQF array...")
    n_classical_features = classical_features.shape[1]
    pqf_classical_part = pqf_array[:, :n_classical_features]
    pqf_quantum_part = pqf_array[:, n_classical_features:]
    
    print(f"  Classical part shape: {pqf_classical_part.shape}")
    print(f"  Quantum part shape: {pqf_quantum_part.shape}")
    
    # 5. Validate row-by-row
    print("\n[STEP 5/5] Validating row order preservation...")
    print("="*70)
    
    validation_results = {
        'total_samples': len(original_data),
        'samples_checked': min(n_samples_to_check, len(original_data)),
        'mismatches': [],
        'match_count': 0,
        'all_match': False
    }
    
    # Check random sample of rows
    import random
    random.seed(42)
    indices_to_check = random.sample(range(len(original_data)), 
                                     min(n_samples_to_check, len(original_data)))
    indices_to_check = sorted(indices_to_check)
    
    print(f"\nChecking {len(indices_to_check)} random rows...")
    print(f"Sample indices: {indices_to_check[:10]}...\n")
    
    for idx in indices_to_check:
        # Compare scaled classical features
        expected = classical_features_scaled[idx, :top_n_features]
        actual = pqf_classical_part[idx, :top_n_features]
        
        # Check if they match (with floating point tolerance)
        matches = np.allclose(expected, actual, rtol=1e-5, atol=1e-8)
        
        if matches:
            validation_results['match_count'] += 1
            if idx < 5 or idx in [len(original_data)-1]:  # Print first few and last
                print(f"✓ Row {idx}: MATCH")
                print(f"  Expected (first 3): {expected[:3]}")
                print(f"  Actual   (first 3): {actual[:3]}")
        else:
            validation_results['mismatches'].append(idx)
            print(f"✗ Row {idx}: MISMATCH")
            print(f"  Expected (first 5): {expected[:5]}")
            print(f"  Actual   (first 5): {actual[:5]}")
            print(f"  Max diff: {np.max(np.abs(expected - actual))}")
            
            # Check if it's a shift/scrambling issue
            # Try to find where this row actually is
            for search_idx in range(max(0, idx-10), min(len(pqf_array), idx+10)):
                if search_idx != idx:
                    test = pqf_classical_part[search_idx, :top_n_features]
                    if np.allclose(expected, test, rtol=1e-5, atol=1e-8):
                        print(f"  ⚠️  Found matching row at index {search_idx} (offset: {search_idx - idx})")
                        break
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Total samples: {validation_results['total_samples']}")
    print(f"Samples checked: {validation_results['samples_checked']}")
    print(f"Matches: {validation_results['match_count']}")
    print(f"Mismatches: {len(validation_results['mismatches'])}")
    print(f"Match rate: {100*validation_results['match_count']/validation_results['samples_checked']:.2f}%")
    
    validation_results['all_match'] = (len(validation_results['mismatches']) == 0)
    
    if validation_results['all_match']:
        print("\n✓✓✓ ORDER PRESERVED CORRECTLY ✓✓✓")
    else:
        print("\n✗✗✗ ORDER NOT PRESERVED ✗✗✗")
        print(f"\nMismatched rows: {validation_results['mismatches'][:20]}...")
        
        # Diagnose the issue
        print("\nDiagnosing issue...")
        if len(validation_results['mismatches']) == validation_results['samples_checked']:
            print("  → All rows mismatch: Likely systematic issue")
            print("  → Check: Scaling applied correctly?")
            print("  → Check: Same scaler used?")
            print("  → Check: Feature columns in same order?")
        else:
            print("  → Partial mismatch: Likely row ordering issue")
            print("  → Check: MPI gather order")
            print("  → Check: Slice concatenation order")
    
    print("="*70 + "\n")
    
    return validation_results


# Additional diagnostic function
def diagnose_feature_mismatch(
    original_csv_path='100k.csv',
    pqf_array_path='./pqf_arr/train_1.1.25_quantum_enhanced.npy',
    scaler_path='./model/scaler.pkl',
    target_label='fraud_label'
):
    """
    Deep diagnostic to find WHERE the mismatch occurs.
    """
    print("\n" + "="*70)
    print("DEEP DIAGNOSTIC - FEATURE MISMATCH ANALYSIS")
    print("="*70 + "\n")
    
    # Load data
    original_data = pd.read_csv(original_csv_path)
    classical_features = original_data.drop([target_label], axis=1)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    classical_scaled = scaler.transform(classical_features)
    
    pqf_array = np.load(pqf_array_path)
    n_classical = classical_features.shape[1]
    pqf_classical = pqf_array[:, :n_classical]
    
    # Test 1: Check if ANY row matches
    print("[TEST 1] Searching for ANY matching rows...")
    found_matches = []
    for i in range(min(100, len(original_data))):
        expected = classical_scaled[i]
        for j in range(min(len(pqf_array), i+50)):
            actual = pqf_classical[j]
            if np.allclose(expected, actual, rtol=1e-5, atol=1e-8):
                if i != j:
                    print(f"  Row {i} found at index {j} (offset: {j-i})")
                    found_matches.append((i, j))
                break
    
    if not found_matches:
        print("  No matches found - data completely different!")
        print("  → Scaling issue or wrong scaler file")
    else:
        print(f"  Found {len(found_matches)} matches with offsets")
        print("  → Row ordering issue, not data corruption")
    
    # Test 2: Check hash of first column
    print("\n[TEST 2] Comparing first column statistics...")
    print(f"  Original first column mean: {classical_scaled[:, 0].mean():.6f}")
    print(f"  PQF first column mean: {pqf_classical[:, 0].mean():.6f}")
    print(f"  Original first column std: {classical_scaled[:, 0].std():.6f}")
    print(f"  PQF first column std: {pqf_classical[:, 0].std():.6f}")
    
    if np.allclose(classical_scaled[:, 0].mean(), pqf_classical[:, 0].mean(), rtol=1e-3):
        print("  → Statistics match: Same data, possibly reordered")
    else:
        print("  → Statistics differ: Different data or different scaling")
    
    # Test 3: Check if features are shifted
    print("\n[TEST 3] Checking for feature column shifts...")
    for shift in range(-10, 11):
        if shift == 0:
            continue
        if 0 <= shift < n_classical:
            test_similarity = np.allclose(
                classical_scaled[0, :100], 
                pqf_classical[0, shift:shift+100], 
                rtol=1e-5, atol=1e-8
            )
            if test_similarity:
                print(f"  Features appear shifted by {shift} columns!")
                break
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run validation
    results = validate_order_preservation(
        original_csv_path='100k.csv',
        pqf_array_path='./pqf_arr/train_1.1.25_quantum_enhanced.npy',
        scaler_path='./model/scaler.pkl',
        target_label='fraud_label',
        top_n_features=216,
        n_samples_to_check=100
    )
    
    # If validation fails, run deep diagnostic
    if not results['all_match']:
        print("\nRunning deep diagnostic...\n")
        diagnose_feature_mismatch(
            original_csv_path='100k.csv',
            pqf_array_path='./pqf_arr/train_1.1.25_quantum_enhanced.npy',
            scaler_path='./model/scaler.pkl',
            target_label='fraud_label'
        )
