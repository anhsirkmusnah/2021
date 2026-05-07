# LLM Hardware — Gradual Upgrade Scenarios

**Use case:** CPU-only LLM inference (Qwen 2.5 7B / Phi-3.5 Mini 3.8B Q4_K_M) for 2-5 demo users.
**Critical metric:** Memory bandwidth (LLM token generation is bandwidth-bound).

---

## Single Decision Table — All Scenarios

| # | Scenario | Platform | DIMMs | Total RAM | Channels | Mem BW | Qwen 7B<br/>single | Qwen 7B<br/>3-concurrent | Phi-3.5<br/>single | Verdict |
|---|----------|----------|-------|-----------|----------|--------|------------|---------------------|------------|---------|
| **A0** | SR645 as-is | EPYC 7413 | 4×16GB DDR4-3200 | 64 GB | 4 of 8 | 102 GB/s | 14-20 tok/s | 9-13 each | 30-40 tok/s | Working baseline |
| **A1** | SR645 +2 DIMMs | EPYC 7413 | 6×16GB DDR4-3200 | 96 GB | 6 of 8 ⚠️ | ~140 GB/s | 18-25 tok/s | 11-15 each | 38-50 tok/s | Asymmetric — avoid |
| **A2** | **SR645 +4 DIMMs** | EPYC 7413 | 8×16GB DDR4-3200 | 128 GB | **8 of 8** ✅ | 204 GB/s | **27-35 tok/s** | **14-20 each** | 50-65 tok/s | **Best balanced upgrade** |
| **B0** | HP Z6 as-is ⚠️ | TR PRO 7945WX | 2×32GB DDR5-5200 | 64 GB | 2 of 8 | 83 GB/s | 10-14 tok/s | 4-6 each | 20-28 tok/s | Bandwidth-starved |
| **B1** | HP Z6 +2 DIMMs | TR PRO 7945WX | 4×32GB DDR5-5200 | 128 GB | 4 of 8 | 166 GB/s | 22-30 tok/s | 8-12 each | 40-55 tok/s | Decent middle tier |
| **B2** | HP Z6 +4 DIMMs | TR PRO 7945WX | 6×32GB DDR5-5200 | 192 GB | 6 of 8 ⚠️ | ~250 GB/s | 32-42 tok/s | 11-15 each | 55-70 tok/s | Asymmetric — avoid |
| **B3** | **HP Z6 +6 DIMMs** | TR PRO 7945WX | 8×32GB DDR5-5200 | 256 GB | **8 of 8** ✅ | **333 GB/s** | **40-55 tok/s** | **13-18 each** | **70-90 tok/s** | **Maximum performance** |
| **C** | Ryzen 9950X (max) | Zen 5 desktop | 2×32GB DDR5-6000 | 64 GB | 2 of 2 (cap) | 96 GB/s | 22-30 tok/s | 9-13 each | 50-65 tok/s | Platform capped |
| **D** | Ryzen 7950X (max) | Zen 4 desktop | 2×32GB DDR5-5600 | 64 GB | 2 of 2 (cap) | 89 GB/s | 18-25 tok/s | 8-12 each | 40-55 tok/s | Platform capped |

---

## Reading the Table

### ⚠️ Asymmetric configs (6 channels) — why to skip them

EPYC and Threadripper PRO platforms expect populated channels in powers of 2 (2, 4, 8). At 6 channels, the memory controller runs in mixed/unbalanced mode:
- Some address ranges get 6-way interleave, others get 2-way
- Real bandwidth lands ~10-15% below the theoretical sum
- Tail latency increases under load
- **Skip 6-channel tiers — go straight from 4-ch to 8-ch**

### ✅ The upgrade paths that actually matter

| Path | From → To | Δ Speed (single-user) | Why this tier |
|------|-----------|----------------------|---------------|
| **A0 → A2 (SR645)** | 4-ch → 8-ch | 14-20 → 27-35 (+~75%) | Doubles bandwidth; balanced 8-channel |
| **B0 → B1 (HP Z6 first step)** | 2-ch → 4-ch | 10-14 → 22-30 (+~115%) | Cheapest way to make HP Z6 not-terrible |
| **B0 → B3 (HP Z6 full)** | 2-ch → 8-ch | 10-14 → 40-55 (+~285%) | Unleashes the platform |

### Crossover points — where one beats the other

| Question | Answer |
|----------|--------|
| When does HP Z6 beat SR645 (single-user)? | At 4-channel and above (B1+ beats A0/A2) |
| When does HP Z6 beat SR645 (concurrent)? | Never — SR645's 24 cores + 128 MB L3 always wins multi-user |
| When does HP Z6 4-ch (B1) beat SR645 8-ch (A2)? | Only on single-user. SR645 still wins concurrent |
| When does Ryzen 9950X beat anything? | Beats SR645 as-is on single-user (22-30 vs 14-20). Loses to all upgraded server platforms |

---

## Decision Logic — One Sentence Per Tier

| If you can spend... | Pick | Why |
|--------------------|------|-----|
| Nothing | **A0 (SR645 as-is)** | 14-20 tok/s is acceptable for demo; switch default to Phi-3.5 for snappier UX |
| Minimum upgrade budget | **A2 (SR645 8-ch)** | Doubles speed for 4 small DIMMs; still wins concurrent users vs everything |
| Mid-range upgrade | **B1 (HP Z6 4-ch)** | Faster single-user than A2, but loses concurrent and costs much more |
| Maximum performance | **B3 (HP Z6 8-ch)** | The only way to break 40+ tok/s on Qwen 7B; best single-user platform |

---

## Single Most Important Insight

**Memory channels matter more than CPU choice.** A Threadripper PRO at 2-channel (B0) loses to a 24-core EPYC at 4-channel (A0). Once both are at 8-channel, the Threadripper PRO wins single-user (B3 > A2) but the EPYC still wins concurrent. The Ryzen desktops hit a wall at 2 channels and can't be upgraded — they're the floor of what the platform allows, not the ceiling.

---

## Architectural Notes

### Why memory bandwidth dominates LLM inference

Every token generated requires reading the entire model's weight matrix from RAM:
- Qwen 2.5 7B Q4_K_M ≈ 4.5 GB read per token
- At 102 GB/s: theoretical max ~22 tok/s (real-world ~14-20 with overhead)
- At 333 GB/s: theoretical max ~74 tok/s (real-world ~40-55 with overhead)

### Why core count matters less than expected

llama.cpp parallelizes across cores, but token generation is fundamentally a sequential, memory-bound operation. Beyond ~12 cores, additional cores mainly help with:
- Concurrent user handling (each request gets its own thread group)
- Prompt prefill phase (compute-bound, scales with cores)
- L3 cache pressure relief

### Why AVX-512 matters

llama.cpp uses AVX-512 for matrix multiplication kernels — ~30-40% faster than AVX2 fallback:
- **EPYC 7413 (Zen 3)**: AVX2 only — falls back to slower kernels
- **TR PRO 7945WX (Zen 4)**: AVX-512 (double-pumped 256-bit)
- **Ryzen 9950X (Zen 5)**: Native 512-bit AVX-512 (best implementation)
- **Ryzen 7950X (Zen 4)**: AVX-512 (double-pumped 256-bit)

### Platform memory channel limits

| Platform | Max channels | DIMMs needed for max |
|----------|-------------|----------------------|
| Lenovo SR645 (EPYC SP3) | 8 | 8 DIMMs |
| HP Z6 G5 (WRX90) | 8 | 8 DIMMs |
| Ryzen AM5 (consumer) | 2 | 2 DIMMs (already maxed) |
