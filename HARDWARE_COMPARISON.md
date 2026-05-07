# LLM Hardware Comparison — UK Server Node

**Use case:** CPU-only LLM inference (Qwen 2.5 7B / Phi-3.5 Mini 3.8B, GGUF Q4_K_M) for 2-5 demo users.
**Critical factor:** Memory bandwidth (LLM token generation is bandwidth-bound, not compute-bound).

---

## TL;DR

| Rank | Option | Single-user tok/s | Cost delta | Verdict |
|------|--------|-------------------|------------|---------|
| 🥇 | **SR645 as-is** | 14-20 | $0 (owned) | Use this — it works |
| 🥈 | Ryzen 9950X build | 22-30 | +$1,500 | Best new-build value |
| 🥉 | HP Z6 G5 (TR PRO 7945WX) as-is | 10-14 ⚠️ | +$5-7K | **Worse than SR645** — bandwidth-starved |
| — | Ryzen 7950X build | 18-25 | +$1,200 | Originally proposed, fine |

---

## 1. AS-IS COMPARISON (No Upgrades)

### 1.1 Headline Specs

| Spec | **Lenovo SR645** (own) | **HP Z6 G5** (option) | Ryzen 9950X (new) | Ryzen 7950X (new) |
|------|------------------------|----------------------|-------------------|-------------------|
| CPU | EPYC 7413 (Zen 3) | TR PRO 7945WX (Zen 4) | Ryzen 9950X (Zen 5) | Ryzen 7950X (Zen 4) |
| Cores / Threads | **24C / 48T** | 12C / 24T | 16C / 32T | 16C / 32T |
| Boost clock | 3.6 GHz | **5.3 GHz** | **5.7 GHz** | **5.7 GHz** |
| L3 cache | **128 MB** | 64 MB | 64 MB | 64 MB |
| AVX-512 | ❌ AVX2 only | ✅ Zen 4 (double-pumped) | ✅ Zen 5 (native 512-bit) | ✅ Zen 4 (double-pumped) |
| RAM (as-is) | 4× 16 GB DDR4-3200 | 2× 32 GB DDR5-5200 | 2× 32 GB DDR5-6000 | 2× 32 GB DDR5-5600 |
| Channels populated | 4 of 8 | **2 of 8** ⚠️ | 2 of 2 (max) | 2 of 2 (max) |
| Memory bandwidth | 102 GB/s | **83 GB/s** ⚠️ | 96 GB/s | 89 GB/s |
| Storage | 2× 960 GB SSD | (varies) | 1 TB NVMe Gen4 | 1 TB NVMe Gen4 |
| TDP | 180 W | 350 W | 170 W | 170 W |
| Form factor | 1U rack | Tower workstation | Desktop | Desktop |
| Mgmt features | IPMI / XClarity, ECC | iLO, ECC | None (consumer) | None (consumer) |
| **Hardware cost** | **$0** (owned) | $5,000-7,000 | $1,400-1,700 | $1,100-1,300 |

### 1.2 LLM Performance — As-Is

| Workload | SR645 (4-ch) | HP Z6 (2-ch) | 9950X | 7950X |
|----------|-------------|--------------|-------|-------|
| **Qwen 2.5 7B — single user** | 14-20 tok/s | **10-14 tok/s** ⚠️ | 22-30 tok/s | 18-25 tok/s |
| **Qwen 2.5 7B — 3 concurrent** | 9-13 each | 4-6 each | 9-13 each | 8-12 each |
| **Phi-3.5 Mini 3.8B — single** | 30-40 tok/s | 20-28 tok/s | 50-65 tok/s | 40-55 tok/s |
| **Prompt processing (prefill)** | Decent (24 cores, no AVX-512) | OK (12 cores + AVX-512) | **Best** (16 cores + native AVX-512) | Good (16 cores + AVX-512) |

### 1.3 Key Insights

| Finding | Detail |
|---------|--------|
| 🚨 **HP Z6 as-is is *worse* than SR645** | Only 2 of 8 memory channels populated → bandwidth-starved despite faster CPU |
| ✅ **SR645 wins on concurrent users** | 24 cores + 128 MB L3 spreads load better than 12-16 cores |
| ⚠️ **EPYC 7413 has no AVX-512** | Milan/Zen 3 — llama.cpp falls back to AVX2 (~30-40% slower kernels) |
| 🏎️ **9950X best for single-user speed** | Zen 5 native AVX-512 + highest boost clock |
| 💰 **SR645 = $0 cost** | Already owned — beats everything on price/perf |

---

## 2. WITH UPGRADES (For Reference Only)

### 2.1 Memory Channel Upgrades — The Big Lever

| Option | Upgrade | Cost | Bandwidth | Single-user tok/s (Qwen 7B) |
|--------|---------|------|-----------|-----------------------------|
| SR645 8-ch | +4× 16GB DDR4-3200 RDIMM | ~$100-150 | 204 GB/s (+100%) | **27-35** |
| HP Z6 8-ch | +6× 32GB DDR5-5200 RDIMM | ~$1,200-2,000 | 333 GB/s (+300%) | **40-55** |
| 9950X | (already at max — 2-ch only) | — | 96 GB/s | 22-30 |
| 7950X | (already at max — 2-ch only) | — | 89 GB/s | 18-25 |

**Note:** Consumer Ryzen platforms (AM5) are 2-channel only — no upgrade path beyond DDR5-6000+ overclocking.

### 2.2 Performance Per Dollar (Upgrade Cost vs Speed Gain)

| Upgrade | $/(tok/s gained) | ROI |
|---------|------------------|-----|
| **SR645 → 8-ch** | **~$10 per tok/s** | 🏆 Best ROI — minimal spend, near-doubles speed |
| HP Z6 buy + 8-ch | ~$130-160 per tok/s | Best raw performance, expensive entry |
| 9950X buy | ~$50-65 per tok/s | Strong middle ground |
| 7950X buy | ~$50-65 per tok/s | Solid value |

---

## 3. DECISION FACTORS

### What dominates LLM speed
1. **Memory bandwidth** (linear scaling with token gen speed)
2. **AVX-512 support** (~30-40% on prompt processing)
3. **Boost clock** (sequential token generation)
4. **Core count** (mainly for concurrent users + prompt prefill)
5. **L3 cache** (reduces RAM traffic for hot weights)

### Form-factor / operational
| Factor | Best | Worst |
|--------|------|-------|
| Datacenter rack-mount | SR645 (1U) | Ryzen builds (desktop) |
| Enterprise mgmt (IPMI/iLO) | SR645, HP Z6 | Ryzen builds |
| ECC memory | SR645, HP Z6 | Ryzen builds (non-ECC) |
| Sustained 24/7 load | SR645, HP Z6 | Ryzen builds (consumer thermals) |
| Power efficiency | Ryzen 7950X (170W) | TR PRO 7945WX (350W) |

---

## 4. BOTTOM LINE

### Recommended path
**Use the SR645 as-is.** $0 cost, 14-20 tok/s on Qwen 2.5 7B is acceptable for a 2-user demo.

### If demo speed is too slow
Default to **Phi-3.5 Mini 3.8B** (30-40 tok/s on SR645) — quality is still excellent for the productivity tasks (email refine, tone change, translate, summarize). Keep Qwen 7B as a "high quality" toggle.

### Free tuning levers (any hardware)
- BIOS: NPS=1, C-states off, Performance mode
- OS: CPU governor=performance, hugepages enabled, NUMA balancing off
- Ollama: `OLLAMA_NUM_THREADS=24`, `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_FLASH_ATTENTION=1`
- Docker: pin Ollama to physical cores 0-23

Expected gain from free tuning alone: **+15-25%** on top of the as-is numbers.

### Avoid
- **Buying the HP Z6 G5 with only 2× 32GB.** It would underperform your existing SR645 while costing $5-7K.
