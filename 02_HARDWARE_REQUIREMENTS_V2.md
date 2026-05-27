# LLM Server Hardware — Comprehensive Comparison (V2)

**Updated:** May 2026
**Use case:** CPU/iGPU/dGPU LLM inference for Qwen 2.5 7B Q4_K_M (~4.5 GB weights) and Phi-3.5 Mini 3.8B Q4_K_M
**Ranking priority:** Single-user response speed (tokens/sec)
**Architecture target:** Linux/Docker/XFRM full-stack (with split-arch noted where Apple Silicon would be LLM-only)

---

## Tier Definitions

LLM inference is **memory-bandwidth-bound**. Tiers reflect realistic single-user Qwen 7B Q4_K_M token-generation rates.

| Tier | Symbol | tok/s on Qwen 7B Q4_K_M | User Experience | Verdict |
|------|--------|-------------------------|-----------------|---------|
| **Reference Class** | 👑 | 60+ tok/s | Near-instant for any prompt | Best-in-class; demo-flagship |
| **Best-in-Class** | 🌟 | 40-60 tok/s | Snappy; ~5-6 sec for 200-word reply | Production-grade single-user |
| **Production** | 🟢 | 25-40 tok/s | Responsive; ~7-10 sec for 200-word reply | Solid for 2-5 user demos |
| **Demo-Workable** | 🟡 | 12-25 tok/s | Slow but acceptable; ~12-18 sec | Demo-only; below acceptable production |
| **Inadequate** | ❌ | < 12 tok/s | Painful (>20 sec); audience disengages | Not suitable for stakeholder demo |

---

## MASTER MATRIX — Single-User Qwen 7B Q4_K_M (ranked by tier, then speed)

| Tier | # | Option | Form | CPU | Cores | Mem BW | RAM (Max) | AVX-512 | Linux | Cost (USD) | Qwen 7B tok/s | Phi-3.5 tok/s | Critical Note |
|------|---|--------|------|-----|-------|--------|-----------|---------|-------|-----------|--------------|--------------|---------------|
| 👑 | 1 | **Mac Studio M3 Ultra 256GB** | Compact 3.7L | M3 Ultra 32c | 32C (24P+8E) | **819 GB/s** | 256 GB UMA | ❌ ARM | ❌ macOS | ~$11,000 | **~100-110** | 200+ | Split-arch needed; no Docker-native; 70B at 12-18 t/s |
| 👑 | 2 | **Mac Studio M3 Ultra 96GB** | Compact 3.7L | M3 Ultra 28c | 28C (20P+8E) | 819 GB/s | 96 GB UMA | ❌ ARM | ❌ macOS | $3,999 | **~80-100** | 180+ | Split-arch; best $/perf on Ultra tier |
| 👑 | 3 | **Mac Studio M4 Max 16c/128GB** | Compact 3.7L | M4 Max 16c | 16C (12P+4E) | **546 GB/s** | 128 GB UMA | ❌ ARM | ❌ macOS | $5,799 | **~70-85** | 150+ | Split-arch; can run 70B Q4 at 8-15 t/s |
| 👑 | 4 | **Lenovo ThinkStation P3 Ultra G2 + RTX 4000 SFF Ada 20GB** | 3.9L | Core Ultra 9 285 + dGPU | 24C + GPU | 280 GB/s VRAM | 20 GB VRAM (+ 128 GB sys ECC) | ❌ Arrow Lake | ✅ Cert | ~$5,000 | **~70-100** | 140+ | dGPU path; VRAM-capped at 20GB; full Linux stack |
| 👑 | 5 | **Mac Studio M4 Max 14c/36-64GB** | Compact 3.7L | M4 Max 14c | 14C (10P+4E) | **410 GB/s** | 36-64 GB UMA | ❌ ARM | ❌ macOS | $1,999-2,799 | **~58-75** | 120+ | Split-arch; best value Studio config |
| 🌟 | 6 | **HP Z6 G5 (TR PRO 7945WX) — 8-ch DDR5-5200 (8×32GB)** | Tower | TR PRO 7945WX | 12C/24T Zen 4 | **333 GB/s** | 256 GB ECC | ✅ Zen 4 | ✅ | $7,000-9,000 | **~40-55** | 70-90 | Best non-Apple, non-Strix tier; needs +6 DIMMs |
| 🌟 | 7 | **HP Z2 Mini G1a — 395 PRO 128GB ECC** | 2.7L mini-WS | Ryzen AI Max+ 395 PRO | 16C/32T Zen 5 | **256 GB/s** (LPDDR5X-8533) | 128 GB ECC soldered | ✅ Native | ✅ Cert 24.04 | **~$6,718** | **~45-55** (iGPU) | 80-120 | Best enterprise LLM mini; HP Care; DASH mgmt |
| 🌟 | 8 | **Framework Desktop 395/128GB** | 4.5L Mini-ITX | Ryzen AI Max+ 395 | 16C/32T Zen 5 | **215 GB/s** (measured) | 128 GB soldered | ✅ Native | ✅ Best in class | $1,999 | **~45-55** (iGPU) | 80-120 | Strix Halo champion; can run 70B Q4 @ 4.8 t/s |
| 🌟 | 9 | **Beelink GTR9 Pro 395/128GB** | Mini | Ryzen AI Max+ 395 | 16C/32T Zen 5 | 215 GB/s | 128 GB soldered | ✅ Native | ✅ | $1,985 | **~45-55** (iGPU) | 80-120 | 140W TDP cap; dual 10GbE; sustains boost |
| 🌟 | 10 | **Minisforum MS-S1 Max 395/128GB** | Mini (2U opt) | Ryzen AI Max+ 395 | 16C/32T Zen 5 | 215 GB/s | 128 GB soldered | ✅ Native | ✅ | $1,899-2,399 | **~45-55** | 80-120 | 2U rack option for clusters; dual 10GbE on some SKUs |
| 🌟 | 11 | **GMKtec EVO-X2 395/128GB** | ~1L mini | Ryzen AI Max+ 395 | 16C/32T Zen 5 | 215 GB/s | 128 GB soldered | ✅ Native | ✅ | $2,199-2,299 | **~45-55** (iGPU) | 80-120 | Most established Strix Halo; 140W peak |
| 🌟 | 12 | **Framework Desktop 395/64GB** | 4.5L Mini-ITX | Ryzen AI Max+ 395 | 16C/32T Zen 5 | 215 GB/s | 64 GB soldered | ✅ Native | ✅ Best | **$1,599** | **~45-55** (iGPU) | 80-120 | 💎 **Sweet spot $/perf**; can't run 70B at this RAM |
| 🌟 | 13 | **GMKtec EVO-X2 395/64GB** | ~1L mini | Ryzen AI Max+ 395 | 16C/32T Zen 5 | 215 GB/s | 64 GB soldered | ✅ Native | ✅ | $1,499 | **~45-55** | 80-120 | Cheapest 395 entry |
| 🌟 | 14 | **Minisforum AtomMan G7 PT** | 1.7L mini | Ryzen 9 7945HX + RX 7600M XT dGPU | 16C/32T + GPU | 288 GB/s VRAM | 8 GB VRAM (+96 GB DDR5) | ✅ Zen 4 | ✅ | $999-1,499 | **~35-50** (dGPU) | 80-100 | Only mini PC with dGPU OOB; VRAM-capped at 8GB |
| 🟢 | 15 | **Mac mini M4 Pro 14c/48-64GB** | 0.8L | M4 Pro 14c | 14C (10P+4E) | **273 GB/s** | 48-64 GB UMA | ❌ ARM | ❌ macOS | $1,999-2,799 | **~35-48** | 90-120 | Split-arch; remarkable perf/$ ratio |
| 🟢 | 16 | **HP Z6 G5 — 4-ch DDR5-5200 (4×32GB)** | Tower | TR PRO 7945WX | 12C/24T Zen 4 | **166 GB/s** | 128 GB ECC | ✅ Zen 4 | ✅ | ~$6,000 | **~22-30** | 40-55 | Mid-tier HP Z6 upgrade; +2 DIMMs from as-is |
| 🟢 | 17 | **Mac mini M4 Pro 12c/24GB** | 0.8L | M4 Pro 12c | 12C (8P+4E) | 273 GB/s | 24 GB UMA | ❌ ARM | ❌ macOS | **$1,399** | **~30-45** | 80-110 | Split-arch; cheapest Pro tier |
| 🟢 | 18 | **Lenovo SR645 (EPYC 7413) — 8-ch DDR4-3200 (8×16GB)** | 1U rack | EPYC 7413 | 24C/48T Zen 3 | **204 GB/s** | 128 GB ECC | ❌ AVX2 only | ✅ | (own) + $100 | **~27-35** | 50-65 | 🎯 Owned; +4 DIMM upgrade; best multi-user perf |
| 🟡 | 19 | **Custom Ryzen 9 9950X build** | Desktop | 9950X | 16C/32T Zen 5 | **96 GB/s** (DDR5-6000) | 64 GB (192 max) | ✅ Native (best) | ✅ | $1,400-1,700 | **~22-30** | 50-65 | Best AVX-512; bandwidth-capped at 2-ch |
| 🟡 | 20 | **Beelink SER9 (Strix Point HX 370)** | 0.27L | Ryzen AI 9 HX 370 | 12C (4Z5 + 8Z5c) | **120 GB/s** LPDDR5X | 32 GB soldered | ✅ 256-bit | ✅ | $899-1,249 | **~18-22** (iGPU) | 40-55 | LPDDR5X-7500 soldered; non-upgradable RAM |
| 🟡 | 21 | **Mac mini M4 base 24-32GB** | 0.8L | M4 10c | 10C (4P+6E) | **120 GB/s** | 32 GB UMA | ❌ ARM | ❌ macOS | $799-1,599 | **~21-33** | 60-80 | Split-arch; entry Apple Silicon LLM |
| 🟡 | 22 | **Minisforum MS-A2 (9955HX 96GB)** | 1.78L | Ryzen 9 9955HX | 16C/32T Zen 5 | **89 GB/s** DDR5-5600 SODIMM | 96 GB | ✅ Native | ✅* | $1,100-1,400 | **~15-22** | 35-50 | * Linux boost-clock bug on some kernels; PCIe x16 + dual 10GbE |
| 🟡 | 23 | **Custom Ryzen 9 7950X build** | Desktop | 7950X | 16C/32T Zen 4 | **89 GB/s** (DDR5-5600) | 64 GB (192 max) | ✅ Zen 4 | ✅ | $1,100-1,300 | **~18-25** | 40-55 | Original PoC pick; bandwidth-capped |
| 🟡 | 24 | **Lenovo SR645 (EPYC 7413) — 4-ch DDR4-3200 (4×16GB)** | 1U rack | EPYC 7413 | 24C/48T Zen 3 | **102 GB/s** | 64 GB ECC | ❌ AVX2 only | ✅ | (own) $0 | **~14-20** | 30-40 | 🎯 Current owned config; switch to Phi-3.5 default |
| 🟡 | 25 | **Lenovo ThinkCentre M90q Gen 6 (Ultra 9 285)** | 1.35L | Core Ultra 9 285 | 24C | **102 GB/s** (DDR5-6400 SODIMM) | 64 GB | ❌ Arrow Lake | ✅ Cert | $1,600-2,100 | **~18-22** CPU / ~30-34 Arc iGPU | 40-55 | Arc iGPU via IPEX-LLM/OpenVINO path |
| 🟡 | 26 | **ASUS NUC 15 Pro+ (Ultra 9 285H)** | **0.7L** | Core Ultra 9 285H | 16C/16T (6P+8E+2LPE) | 89 GB/s | 96 GB SODIMM | ❌ | ✅ Excellent | $550-1,200 | **~12-18** CPU / ~25-35 Arc iGPU | 30-50 | Smallest Intel option; refined firmware |
| 🟡 | 27 | **Geekom IT15 (Ultra 9 285H)** | **0.46L** | Core Ultra 9 285H | 16C/16T | 89 GB/s | 32 GB DDR5-5600 | ❌ | ✅ | $1,100-1,399 | **~12-18** / ~25-35 iGPU | 30-50 | Tiny; audible under load |
| 🟡 | 28 | **HP EliteDesk 805 G9 Mini (Ryzen 7 PRO 8700G)** | 2.0L | Ryzen 7 PRO 8700G | 8C/16T Zen 4 | 89 GB/s | 64 GB SODIMM | ✅ Zen 4 | ✅ | $900-1,300 | **~12-17** CPU / ~15-20 Radeon 780M | 30-40 | DASH mgmt; Radeon 780M iGPU usable |
| 🟡 | 29 | **Lenovo ThinkCentre M75q Gen 5 (Ryzen 7 PRO 8700GE)** | 1L | Ryzen 7 PRO 8700GE | 8C/16T Zen 4 | 83 GB/s | 64 GB SODIMM | ✅ Zen 4 | ✅ | $900-1,400 | **~12-17** / ~15-20 iGPU | 30-40 | Compact AMD enterprise; DASH |
| 🟡 | 30 | **Minisforum UM890 Pro (8945HS 96GB)** | Small | Ryzen 9 8945HS | 8C/16T Zen 4 | 89 GB/s | 96 GB | ✅ Zen 4 | ✅ | $649-879 | **~12-17** | 30-40 | OCuLink for eGPU upgrade path |
| 🟡 | 31 | **Lenovo ThinkStation P3 Ultra G2 (CPU-only)** | 3.9L | Core Ultra 9 285 | 24C | 102 GB/s | 128 GB DDR5-6400 ECC | ❌ | ✅ Cert | $2,500-3,500 | **~12-18** | 30-50 | Without dGPU; ECC; add RTX A1000/4000 SFF |
| 🟡 | 32 | **Dell Precision 3280 Compact (CPU-only)** | 2.7L | i9-14900 | 24C/32T | 89 GB/s | 128 GB ECC | ❌ | ✅ RHEL | $1,029-2,500 | **~10-15** | 30-40 | ProSupport + RHEL cert; add RTX A1000 |
| 🟡 | 33 | **HP EliteDesk 800 G9 Mini (i9-14900T)** | 2.0L | i9-14900T | 24C/32T | 89 GB/s | 64 GB SODIMM | ❌ | ✅ | $1,400-1,800 | **~10-15** | 25-35 | vPro Ent; no AVX-512; no useful iGPU |
| 🟡 | 34 | **Lenovo ThinkCentre M90q Gen 5 (i9-14900)** | 1.35L | i9-14900 | 24C/32T | 89 GB/s | 64 GB SODIMM | ❌ | ✅ | $1,500-2,000 | **~10-15** | 25-35 | vPro Ent; UHD 770 iGPU weak |
| 🟡 | 35 | **Dell OptiPlex 7020 MFF (i9-14900)** | 1.2L | i9-14900 | 24C/32T | 89 GB/s | 64 GB SODIMM | ❌ | ✅ | $1,300-1,700 | **~10-15** | 25-35 | vPro Ent; bandwidth-limited |
| 🟡 | 36 | **Lenovo ThinkStation P3 Tiny (Ultra 9)** | 1L | Core Ultra 9 285 | 24C | 89 GB/s | 96 GB SODIMM | ❌ | ✅ | $1,500-3,500 | **~10-15** | 25-35 | Tiny WS; no ECC despite WS branding |
| 🟡 | 37 | **Minisforum MS-01 (i9-13900H)** | Mini | i9-13900H | 14C/20T Raptor | 83 GB/s | 64 GB DDR5-5200 | ❌ (consumer) | ✅ | $649-900 | **~10-13** | 25-35 | Dual SFP+ 10GbE; PCIe x16 |
| 🟡 | 38 | **Beelink GTR7 Pro (7940HS)** | Small | Ryzen 9 7940HS | 8C/16T Zen 4 | 89 GB/s | 32-64 GB | ✅ Zen 4 | ✅ | $720-869 | **~12-15** | 30-40 | Older Zen 4; bandwidth-limited |
| 🟡 | 39 | **Khadas Mind 2S (Ultra 7 255H)** | 0.2L pocket | Core Ultra 7 255H | 16C | 128 GB/s (LPDDR5X) | 64 GB soldered | ❌ | ✅ | $1,599 | **~15-20** | 40-55 | Premium portable; modular dock |
| ❌ | 40 | **HP Z6 G5 — AS-SHIPPED 2-ch (2×32GB)** | Tower | TR PRO 7945WX | 12C/24T Zen 4 | **83 GB/s** | 64 GB ECC | ✅ Zen 4 | ✅ | $5,000-7,000 | **~10-14** | 20-28 | ⚠️ Bandwidth-starved; underperforms own SR645 |
| ❌ | 41 | **HP ProDesk 405 G9 Mini** | 2.0L | Ryzen 7 PRO entry | 8C/16T Zen 4 | 76 GB/s | 64 GB | ✅ Zen 4 | ✅ | $700-1,000 | **~10-13** | 25-35 | Entry-tier business; underpowered for production demo |
| ❌ | 42 | **Lenovo ThinkCentre Neo 50q** | 2.0L | i7 (12C) | 12C | 76 GB/s | 64 GB | ❌ | ✅ | $500-800 | **~7-10** | 18-25 | Entry tier; not suitable for stakeholder demo |
| ❌ | 43 | **Lenovo ThinkEdge SE30** | Industrial | i5-1145GRE | 4C/8T | ~50 GB/s | 32 GB DDR4 | ❌ | ✅ | $900-1,500 | **~3-5** | 8-12 | MIL-SPEC rugged; not LLM-capable |

---

## Category-by-Category Detail

### A. Apple Silicon (Mac Studio, Mac mini)

**Architecture caveat:** All Apple Silicon options require a **split architecture** — Ollama runs on the Mac, the rest of the stack (FastAPI, ChromaDB, QKD daemon, IPsec/XFRM, Docker) on a separate small Linux box, connected via REST over LAN. Apple does not run Linux containers natively (Docker uses an internal hypervisor VM) and **no QKD vendor SDK is known to support macOS**.

| Field | Mac mini M4 base | Mac mini M4 Pro | Mac Studio M4 Max | Mac Studio M3 Ultra |
|-------|------------------|-----------------|-------------------|---------------------|
| CPU cores | 10 (4P+6E) | 12-14 (8/10P + 4E) | 14-16 (10/12P + 4E) | 28-32 (20/24P + 8E) |
| GPU cores | 10 | 16 / 20 | 32 / 40 | 60 / 80 |
| NPU | 38 TOPS | 38 TOPS | 38 TOPS | ~36 TOPS (M3 gen) |
| **Memory BW** | **120 GB/s** | **273 GB/s** | **410-546 GB/s** | **819 GB/s** |
| Memory type | LPDDR5-7500 | LPDDR5X-8533 | LPDDR5X-8533 | LPDDR5-6400 |
| RAM tiers | 16/24/32 GB | 24/48/64 GB | 36/48/64/128 GB | 96/256 GB (512 pulled) |
| Ethernet | 1 GbE (10 BTO) | 1 GbE (10 BTO) | 10 GbE std | 10 GbE std |
| Idle power | ~3-4 W | ~3.5-5 W | ~7-10 W | ~10-15 W |
| Load power | 65 W (155W PSU) | 155 W | 480 W | 480 W |
| Form factor | 127×127×50mm / 0.8L | 127×127×50mm / 0.8L | 197×197×95mm / 3.7L | 197×197×95mm / 3.7L |
| Price (min viable LLM) | $799 (24 GB) | $1,399 (24 GB) | $1,999 (36 GB) | $3,999 (96 GB) |
| Price (max) | $1,599 (32 GB) | $4,699 (64 GB) | $5,799 (128 GB) | $11,000+ (256 GB) |

**Major gotchas:**
- 🚨 **Mac Pro discontinued March 26, 2026** — no M4 Ultra exists; M3 Ultra is the current top
- 🚨 **512 GB M3 Ultra option pulled** due to DRAM shortage (256 GB is current max)
- ❌ **No PCIe expansion** on mini or Studio — no HSM, no QKD interface cards (Sonnet xMac TB5 enclosures workaround at $1,500-2,500)
- ❌ **No ECC memory** on any Apple Silicon
- ⚠️ **Prefill (TTFT) is Apple's weak point** for RAG — for long retrieved contexts, M3 Ultra can be 10× slower than equivalent NVIDIA GPU
- ⚠️ **macOS hard to air-gap** — Apple Push, Spotlight, Software Update, Gatekeeper OCSP, XProtect all phone home
- ✅ **MLX framework is 10-50% faster than llama.cpp** on identical Apple HW (for Qwen 3 Coder, llama.cpp was ~⅓ MLX speed)

---

### B. AMD Strix Halo (Ryzen AI Max+ 395)

**Why these matter:** First non-Apple platform with Apple-class memory bandwidth (256-bit LPDDR5X-8000 = 256 GB/s theoretical, ~215 GB/s measured) AND Linux-native. Best CPU/iGPU LLM platform of 2026.

**Common chip specs (Ryzen AI Max+ 395):**
- 16C/32T Zen 5 (no 5c), 3.0 GHz base, 5.1 GHz boost
- Radeon 8060S iGPU (40 CUs RDNA 3.5, 2.9 GHz)
- XDNA 2 NPU (50 TOPS)
- LPDDR5X-8000 256-bit (soldered, no upgrade)
- 256 GB/s theoretical, ~215 GB/s measured
- Configurable 45-120 W (140 W in Beelink GTR9 Pro)

| Platform | RAM | Network | Price | Best For |
|----------|-----|---------|-------|----------|
| **Framework Desktop 395/64** | 64 GB | 5 GbE, 2× USB4 | **$1,599** | 💎 Sweet spot $/perf |
| **Framework Desktop 395/128** | 128 GB | 5 GbE, 2× USB4 | $1,999 | 70B model use; future-proof |
| **GMKtec EVO-X2 395/64** | 64 GB | 2.5 GbE, USB4 | $1,499 | Cheapest entry |
| **GMKtec EVO-X2 395/128** | 128 GB | 2.5 GbE, USB4 | $2,199-2,299 | 70B + most established |
| **Beelink GTR9 Pro 395/128** | 128 GB | **2× 10 GbE**, 2× USB4 | $1,985 preorder | Best networking; 140W TDP |
| **Minisforum MS-S1 Max 395/128** | 128 GB | 2× 10 GbE (some SKUs) | $1,899-2,399 | **2U rack option** for clusters |
| **HP Z2 Mini G1a 395 PRO/128 ECC** | 128 GB ECC LPDDR5X-8533 | 2.5 GbE std (10 opt) | $4,781-$6,718 | 🏆 Enterprise — DASH, ECC, HP warranty |
| **HP ZBook Ultra G1a 395 PRO/128** | 128 GB | 14" laptop | $4,049 | Only mobile 128GB Strix Halo |

**Real-world LLM benchmarks (128 GB Strix Halo, Linux kernel ≥ 6.15, Vulkan/RADV + FA):**

| Model | Quant | Vulkan tg | Notes |
|-------|-------|-----------|-------|
| Llama 2 7B (≈ Qwen 7B) | Q4_0 | **52.7 t/s** | With Flash Attention |
| Llama 2 7B | Q4_K_M | 47.9 t/s | |
| Llama 3.1 8B | Q4_K_M | ~45-50 t/s | |
| Qwen 3 30B-A3B MoE | UD-Q4_K_XL | 72 t/s | MoE benefits from BW headroom |
| Qwen3-Coder 30B MoE | Q4_K_S | 98.5 t/s | |
| GPT-OSS 120B MoE | Q4 | 55.6 t/s | **Runs fully on 128 GB unified** |
| Llama 3.3 70B dense | Q4_K_M | 4.8 t/s | 94% of BW ceiling; usable but slow |
| Llama 3.3 70B | Q6_K | 3.7-3.8 t/s | Sluggish |

**Linux readiness assessment (May 2026):**

| Status | Item |
|--------|------|
| ✅ **Works well** | Vulkan/RADV (default rec), Ubuntu 24.04.3 LTS, Kernel 6.15+ (15% perf jump from 6.14) |
| ⚠️ **Watch out** | gfx1151 is "Preview" in ROCm; kernel <6.18.4 has stability bugs; `linux-firmware-20251125` breaks ROCm |
| ⚠️ **Tune required** | UMA mapping not automatic — need `amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=31457280` |
| ⚠️ **Software lag** | Ollama bundled llama.cpp is ~56% slower than upstream on Vulkan (missing Wave32 FA patches) |
| ✅ **BIOS setup** | UMA frame buffer 512 MB, disable IOMMU |

**Critical gotcha:** RAM is soldered LPDDR5X. **Choose 128 GB at purchase** if 70B / MoE models matter — no field upgrade possible.

---

### C. Custom Desktop Builds (Ryzen)

| Build | CPU | RAM | Mem BW | Qwen 7B | Cost |
|-------|-----|-----|--------|---------|------|
| Ryzen 9 9950X | Zen 5 16C/32T, 5.7 GHz, native AVX-512 | 64 GB DDR5-6000 (2-ch max) | 96 GB/s | 22-30 t/s | $1,400-1,700 |
| Ryzen 9 7950X | Zen 4 16C/32T, 5.7 GHz, AVX-512 (double-pumped) | 64 GB DDR5-5600 (2-ch max) | 89 GB/s | 18-25 t/s | $1,100-1,300 |

**Platform ceiling:** Consumer AM5 is 2-channel only. These cap at ~96 GB/s no matter what RAM you buy. Zen 5's full-width AVX-512 helps prompt processing ~20-40% vs Zen 4 double-pumped.

---

### D. AMD Consumer Mini PCs

| Model | CPU | Cores | Mem BW | RAM | AVX-512 | Cost | Qwen 7B | Note |
|-------|-----|-------|--------|-----|---------|------|---------|------|
| **Minisforum MS-A2** | Ryzen 9 9955HX (Zen 5) | 16C/32T | 89 GB/s | 96 GB SODIMM | ✅ Native | $839-1,400 | 15-22 t/s | PCIe x16 + 2× SFP+ 10GbE; best expandability |
| **Minisforum AtomMan G7 PT** | Ryzen 9 7945HX + RX 7600M XT dGPU | 16C + GPU | 288 GB/s VRAM | 96 GB DDR5 + 8 GB VRAM | ✅ Zen 4 | $999-1,499 | 35-50 t/s (dGPU) | Only mini PC with dGPU OOB; VRAM-capped at 8GB |
| **Beelink SER9** | Ryzen AI 9 HX 370 (Strix Point) | 12C (4 Z5 + 8 Z5c) | 120 GB/s LPDDR5X | 32 GB soldered | ✅ 256-bit | $899-1,249 | 18-22 t/s | LPDDR5X-7500 soldered; no upgrade |
| **Minisforum UM890 Pro** | Ryzen 9 8945HS (Zen 4) | 8C/16T | 89 GB/s | 96 GB SODIMM | ✅ Zen 4 | $479-879 | 12-17 t/s | OCuLink for eGPU |
| **GMKtec K11** | Ryzen 9 8945HS | 8C/16T | 89 GB/s | 96 GB | ✅ Zen 4 | ~$650 | 12-17 t/s | OCuLink + USB4 |
| **Beelink GTR7 Pro** | Ryzen 9 7940HS | 8C/16T | 89 GB/s | 64 GB | ✅ Zen 4 | $720-869 | 12-15 t/s | Older Zen 4 |
| **Geekom A7 Max / AX8 Pro** | Ryzen 9 7940HS / 8945HS | 8C/16T | 76-89 GB/s | 32-64 GB | ✅ Zen 4 | $700-1,000 | 12-15 t/s | Sub-0.5L ultra-compact |

---

### E. Intel Consumer Mini PCs

🚨 **Critical limitation:** **Intel Core Ultra 200-series (Arrow Lake) does NOT support AVX-512.** llama.cpp falls back to AVX2 (~30-40% slower kernels) on pure-CPU paths. Arc iGPU via IPEX-LLM/OpenVINO is the workaround.

| Model | CPU | Cores | Mem BW | RAM | iGPU Path | Cost | Qwen 7B |
|-------|-----|-------|--------|-----|-----------|------|---------|
| **ASUS NUC 15 Pro+** | Core Ultra 9 285H | 16C (6P+8E+2LPE) | 89 GB/s | 96 GB SODIMM | Arc 140T, ~25-35 t/s via IPEX | $550-1,200 | 12-18 CPU / 25-35 iGPU |
| **Geekom IT15** | Core Ultra 9 285H | 16C | 89 GB/s | 32 GB SODIMM | Arc 140T | $1,100-1,399 | 12-18 / 25-35 iGPU |
| **Khadas Mind 2S** | Core Ultra 7 255H | 16C | **128 GB/s LPDDR5X** | 64 GB soldered | Arc iGPU | $1,599 | 15-20 / 25-35 iGPU |
| **Minisforum MS-01** | i9-13900H (Raptor) | 14C/20T (6P+8E) | 83 GB/s | 64 GB DDR5-5200 | UHD weak | $649-900 | 10-13 |

---

### F. Enterprise Business Mini PCs

| Model | Top CPU | Mem BW | RAM | AVX-512 | ECC | Mgmt | Linux Cert | Cost | Qwen 7B |
|-------|---------|--------|-----|---------|-----|------|------------|------|---------|
| **HP Z2 Mini G1a (Strix Halo)** | Ryzen AI Max+ PRO 395 | **256 GB/s** | 128 GB ECC | ✅ Native | ✅ | DASH + HP Sure Start | ✅ 24.04 | $4,781-6,718 | **40-48 (iGPU)** |
| **Lenovo ThinkStation P3 Ultra G2** | Core Ultra 9 285 | 102 GB/s | 128 GB ECC SODIMM | ❌ | ✅ | vPro Ent | ✅ | $2,500-5,000 | 12-18 CPU / **70-100 with RTX 4000 SFF Ada 20GB** |
| **Dell Precision 3280 Compact** | i9-14900 / Xeon W | 89 GB/s | 128 GB ECC | ❌ | ✅ | vPro Ent | ✅ RHEL cert | $1,029-4,500 | 10-15 CPU / **35-50 with RTX A1000 8GB** |
| **HP EliteDesk 805 G9 Mini** | Ryzen 7 PRO 8700G | 89 GB/s | 64 GB | ✅ Zen 4 | ❌ | DASH | ✅ | $900-1,300 | 12-17 |
| **HP EliteDesk 800 G9 Mini** | i9-14900T | 89 GB/s | 64 GB | ❌ | ❌ | vPro Ent | ✅ | $1,400-1,800 | 10-15 |
| **Lenovo ThinkCentre M90q Gen 6** | Core Ultra 9 285 | 102 GB/s | 64 GB SODIMM | ❌ | ❌ | vPro Ent | ✅ | $1,600-2,100 | 18-22 / 30-34 Arc iGPU |
| **Lenovo ThinkCentre M75q Gen 5** | Ryzen 7 PRO 8700GE | 83 GB/s | 64 GB SODIMM | ✅ Zen 4 | ❌ | DASH | ✅ | $900-1,400 | 12-17 |
| **Lenovo ThinkStation P3 Tiny G2** | Core Ultra 9 285 | 89 GB/s | 96 GB SODIMM | ❌ | ❌ | vPro Ent | ✅ | $1,500-3,500 | 10-15 |
| **Dell OptiPlex 7020 MFF** | i9-14900 | 89 GB/s | 64 GB SODIMM | ❌ | ❌ | vPro Ent | ✅ | $1,300-1,700 | 10-15 |

**Workstation + dGPU path (notable):**
- **Lenovo ThinkStation P3 Ultra G2 + RTX 4000 SFF Ada (20 GB VRAM)** at ~$5,000 — delivers **70-100 tok/s on 8B Q4_K_M** via CUDA. Fastest "small box" option, VRAM-capped at 20 GB so can't run 70B.
- **Dell Precision 3280 Compact + RTX A1000 (8 GB)** at ~$3-4K — ProSupport + RHEL cert, dGPU-bound to 8GB so Qwen 7B Q4 fits with some context.

---

### G. Existing Hardware (Owned)

| Config | RAM Config | Mem BW | Cost Delta | Qwen 7B tok/s | Concurrent (3 users) |
|--------|-----------|--------|-----------|--------------|---------------------|
| **SR645 (EPYC 7413) AS-IS** | 4×16GB DDR4-3200 (4 of 8 ch) | **102 GB/s** | $0 | **14-20** | 9-13 each |
| **SR645 +4 DIMMs** | 8×16GB DDR4-3200 (8 of 8 ch) | **204 GB/s** | ~$100 | **27-35** | **14-20 each** |
| **HP Z6 G5 (TR PRO 7945WX) AS-IS** | 2×32GB DDR5-5200 (2 of 8 ch) ⚠️ | **83 GB/s** | $0 | **10-14** | 4-6 each |
| **HP Z6 G5 +2 DIMMs (4-ch)** | 4×32GB DDR5-5200 (4 of 8 ch) | 166 GB/s | ~$600-1,000 | **22-30** | 8-12 each |
| **HP Z6 G5 +6 DIMMs (8-ch)** | 8×32GB DDR5-5200 (8 of 8 ch) | **333 GB/s** | ~$1,800-2,500 | **40-55** | 13-18 each |

---

## Decision Logic — Recommended Picks by Scenario

### Best single-user speed (max performance, any cost)
1. **Mac Studio M3 Ultra 256GB ($11K)** — 100+ tok/s on 7B (but split-arch required for QKD stack)
2. **Lenovo ThinkStation P3 Ultra G2 + RTX 4000 SFF Ada 20GB (~$5K)** — 70-100 tok/s, full Linux stack
3. **HP Z6 G5 fully upgraded 8-ch ($7-9K)** — 40-55 tok/s, enterprise WS form factor

### Best $/tok/s value
1. **Framework Desktop 395/64 ($1,599)** — 45-55 tok/s; 💎 best raw value
2. **GMKtec EVO-X2 395/64 ($1,499)** — same Strix Halo perf, less premium
3. **SR645 + $100 RAM** — 27-35 tok/s for ~$100 spend (already-owned advantage)

### Best for enterprise (HSBC environment, ECC, mgmt, support)
1. **HP Z2 Mini G1a 395 PRO/128 ECC ($4,781-6,718)** — 🏆 only enterprise mini with 256 GB/s bandwidth + ECC + DASH + HP warranty + Ubuntu cert
2. **Lenovo ThinkStation P3 Ultra G2 + dGPU ($5K)** — vPro Ent, ECC, RTX option
3. **Dell Precision 3280 Compact + RTX A1000 ($3-4K)** — ProSupport, RHEL cert, classic enterprise

### Best for owned hardware path (zero new spend)
**SR645 as-is + Phi-3.5 Mini as default model.** 30-40 tok/s on Phi-3.5 is acceptable for demo, free.

### Best for full-stack Linux (no split architecture)
1. **Framework Desktop 395/64 ($1,599)** — best Linux maturity in Strix Halo class
2. **HP Z2 Mini G1a (~$5K)** — best enterprise Linux Strix Halo
3. **SR645 8-ch ($100 upgrade)** — server-grade, owned

### Avoid for LLM workload
- ❌ **HP Z6 G5 as-shipped (2-ch)** — underperforms own SR645 at $5-7K cost
- ❌ **Any Core Ultra 200 CPU-only path** — no AVX-512 disqualifies CPU LLM (Arc iGPU OK)
- ❌ **Entry business mini PCs** (Neo 50q, ProDesk 405) — too underpowered
- ❌ **ThinkEdge SE30** — industrial-rugged but not LLM-capable

---

## Architectural Truths (Why the Rankings Land Here)

### 1. Memory bandwidth dominates LLM token generation
Every token regenerates the full forward pass, reading the entire model from RAM. Speed scales nearly linearly with bandwidth. A 24-core EPYC at 4-ch beats a 12-core Threadripper at 2-ch despite half the cores.

### 2. Apple Silicon and Strix Halo broke the bandwidth barrier
Both use wide unified memory buses (256-bit+) instead of consumer 2-channel DIMMs. This is the architectural shift: 215-819 GB/s vs 89-100 GB/s in 2-channel desktop / server-class 4-channel.

### 3. AVX-512 matters but is a secondary lever
~30-40% speedup on llama.cpp matrix kernels:
- **Zen 5 desktop (9950X)**: native 512-bit — best implementation
- **Zen 4 (7950X, 7945WX)**: AVX-512 double-pumped 256-bit
- **Zen 4 mobile (8945HS, 7940HS)**: same
- **EPYC 7413 (Zen 3)**: AVX2 only — falls back
- **Intel Arrow Lake / Core Ultra 200**: AVX-512 disabled by Intel

### 4. iGPU/dGPU paths bypass CPU bandwidth limits
- Strix Halo Radeon 8060S via Vulkan: uses full 215 GB/s unified
- RTX 4000 SFF Ada via CUDA: 280 GB/s VRAM, but VRAM-capped
- Intel Arc iGPU via IPEX-LLM/OpenVINO: viable workaround for Core Ultra 200

### 5. MoE models are the future of large local LLMs
MoE (Mixture of Experts) only activates a subset of weights per token. GPT-OSS 120B hits 55 t/s on 128 GB Strix Halo because only ~20B params are active per token, but the full 120B is loaded. **128 GB unified memory** opens this entire model class.

### 6. Concurrent users favor high core count over bandwidth
For 3-5 concurrent users, the SR645's 24 cores + 128 MB L3 outperforms 12-16 core Strix Halo even when single-user is reversed. If concurrent-user perf matters more than peak single-user, server platforms remain competitive.

---

## Quick-Pick Cheat Sheet

| Need | Pick |
|------|------|
| 💰 Zero new spend | SR645 as-is + Phi-3.5 default |
| 🪙 Minimum spend, big jump | SR645 + 4 DIMMs ($100) → 27-35 tok/s |
| 🎯 Best $/perf for new buy | Framework Desktop 395/64 ($1,599) → 45-55 tok/s |
| 🏆 Best enterprise (HSBC-friendly) | HP Z2 Mini G1a 128GB ECC (~$6K) → 45-55 tok/s + ECC + mgmt |
| ⚡ Fastest single-user demo | Mac Studio M3 Ultra OR P3 Ultra + RTX 4000 SFF Ada |
| 👥 Best for 3-5 concurrent users | SR645 + 4 DIMMs OR HP Z6 G5 8-ch |
| 🧠 70B model use | Strix Halo 128 GB (any vendor) — only sub-$2K path |
| 🌐 Smallest footprint | ASUS NUC 15 Pro+ (0.7L) — limited LLM perf though |
| 🍎 No-compromise Apple | Mac Studio M3 Ultra 96 GB ($4K) — split-arch caveat |

---

## Notes & Caveats

- **All tok/s estimates are for single-user Qwen 2.5 7B Q4_K_M.** Phi-3.5 Mini 3.8B Q4_K_M is roughly 1.8-2.5× faster.
- **Prices are May 2026 USD.** DRAM shortage has pushed many prices up vs Q1 2026; Apple's 256 GB upgrade rose $400 in March 2026.
- **For full QKD/Docker stack:** all non-Apple options run Ubuntu 24.04 + Docker + XFRM natively. Apple requires split-arch.
- **For 70B model use:** only 128 GB+ unified memory configs work (Strix Halo 128 GB or Mac Studio Max 128 GB / Ultra 96 GB+).
- **iGPU paths** (Strix Halo Vulkan, Intel Arc IPEX-LLM) all require careful Linux setup; consult vendor-specific guides.
- **dGPU options** (RTX 4000 SFF Ada, RTX A1000) are VRAM-bound — fast for small models but can't load 70B.

---

## Source Documents
- Apple Silicon: Apple specs pages, Mac mini M4 reviews (Jeff Geerling), llama.cpp GitHub discussions, MacRumors forums, LocalAIMaster benchmarks
- Strix Halo: Phoronix, ServeTheHome, Level1Techs forum, Hardware Corner, AMD developer articles, kyuz0/amd-strix-halo-toolboxes GitHub
- Enterprise minis: HP QuickSpecs, Lenovo PSREF, Dell spec sheets, StorageReview, ServeTheHome, Ubuntu/RHEL certification catalogs
- Consumer minis: Minisforum/Beelink/GMKtec/Geekom product pages, ServeTheHome reviews, Tom's Hardware, Phoronix Linux benchmarks
