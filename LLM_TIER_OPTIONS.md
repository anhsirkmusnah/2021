# LLM Server Options by Performance Tier — Single-Machine Full-Stack (Priced)

**Updated:** May 2026
**Hard requirement:** The **entire stack runs on ONE machine** — Ollama + FastAPI + ChromaDB + QKD KMS + QKD daemon + IPsec/XFRM + Nginx, in Docker on Ubuntu 24.04.
**Basis:** Pure technical merit. Each option at its sensible deployment spec.
**Ranking metric:** Single-user Qwen 2.5 7B Q4_K_M tokens/sec.

> 🚫 **Apple Silicon excluded** — macOS cannot host the Linux/Docker/XFRM QKD stack (would need a 2nd machine). All options below run everything on one x86/Linux box.

> 💸 **2026 DRAM shortage in effect.** Memory prices are 2-3× historical norms (DDR5 32GB SODIMM ~$380, DDR4-3200 16GB ECC RDIMM ~$225, DDR5 ECC RDIMM 32GB ~$300). On 128GB+ server configs, RAM now costs as much as the base machine. GPU prices are near-MSRP. Prices are volatile street estimates — verify at purchase.

---

## Field Key

| Field | Meaning |
|-------|---------|
| **Make / Model** | The platform |
| **CPU (Arch)** | Processor and microarchitecture |
| **Cores** | Total CPU cores — concurrent-user headroom |
| **Path** | CPU / iGPU-Vulkan / iGPU-IPEX / dGPU-CUDA / dGPU-ROCm |
| **RAM** | Configured memory for this row |
| **ECC** | Error-correcting memory: Yes / No / Optional |
| **BW (GB/s)** | Memory (or VRAM) bandwidth — dominant LLM speed factor |
| **7B t/s** | Qwen 2.5 7B Q4_K_M single-user tokens/sec |
| **3B t/s** | Phi-3.5 Mini 3.8B Q4_K_M single-user tokens/sec |
| **As-Is $** | Base machine price (barebone, or sold config; for dGPU rows includes the GPU — broken out in Notes) |
| **+RAM $** | Cost of memory to reach the RAM in this row (`incl` = soldered or already in base) |
| **Total $** | Deploy-ready price = As-Is + RAM |

**Tiers (Qwen 7B single-user):** Reference 60+ · Best-in-Class 40–60 · Production 25–40 · Demo-Workable 12–25

> 🔑 Without Apple, the single-box iGPU/CPU ceiling is **~55 t/s (Strix Halo)**. To exceed 60 t/s you need a **discrete GPU** in the same chassis — so the entire Reference tier is dGPU-equipped.

---

## 👑 TIER 1 — REFERENCE (60+ tok/s) — all require a discrete GPU

| Make / Model | CPU (Arch) | Cores | Path | RAM | ECC | BW (GB/s) | 7B t/s | 3B t/s | As-Is $ | +RAM $ | Total $ | Notes |
|--------------|-----------|-------|------|-----|-----|-----------|--------|--------|---------|--------|---------|-------|
| Lenovo P3 Ultra G2 + RTX 4000 SFF Ada 20GB | Core Ultra 9 285 + dGPU | 24 | dGPU-CUDA | 128 GB | Yes | 280 (VRAM) | 90 | 200 | 2,750 | 1,160 | **3,910** | As-Is incl RTX 4000 Ada ($1,250); fastest compact box; VRAM-capped 20GB |
| HP Z6 G5 + RTX 4000 SFF Ada (8-ch) | TR PRO 7945WX (Zen 4) + dGPU | 12 | dGPU-CUDA | 128 GB (8×16 RDIMM) | Yes | 280 (VRAM) | 90 | 200 | 3,650 | 1,520 | **5,170** | As-Is incl GPU ($1,250); full WS + 8-ch CPU fallback |
| SR645 EPYC 7413 (8-ch) + RTX 4000 SFF Ada | EPYC 7413 (Zen 3) + dGPU | 24 | dGPU-CUDA | 128 GB (8×16 RDIMM) | Yes | 280 (VRAM) | 80 | 160 | 3,750 | 1,800 | **5,550** | ⚠️ SR645 refurb-only (~$2,500, withdrawn new) + GPU $1,250; best concurrent box |
| Custom Ryzen 9 9950X + RTX 4070 Ti Super | Ryzen 9 9950X (Zen 5) + dGPU | 16 | dGPU-CUDA | 64 GB | Optional | 504 (VRAM) | 75 | 150 | 2,110 | 420 | **2,530** | As-Is = build $1,060 + GPU $1,050; cheapest dGPU path |

---

## 🌟 TIER 2 — BEST-IN-CLASS (40–60 tok/s)

| Make / Model | CPU (Arch) | Cores | Path | RAM | ECC | BW (GB/s) | 7B t/s | 3B t/s | As-Is $ | +RAM $ | Total $ | Notes |
|--------------|-----------|-------|------|-----|-----|-----------|--------|--------|---------|--------|---------|-------|
| HP Z2 Mini G1a PRO | Ryzen AI Max+ PRO 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | **Yes** | 256 | 55 | 115 | 3,900 | incl | **3,900** | 🏆 Only enterprise Strix; ECC LPDDR5X-8533, DASH, runs 70B/MoE. Range $3,343–4,995 |
| Beelink GTR9 Pro | Ryzen AI Max+ 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | No | 215 | 53 | 108 | 1,985 | incl | **1,985** | 💎 Best $/perf at 128GB; dual 10GbE; 2TB incl |
| GMKtec EVO-X2 395 | Ryzen AI Max+ 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | No | 215 | 52 | 105 | 2,249 | incl | **2,249** | Most established Strix Halo; 2TB incl |
| Framework Desktop 395 | Ryzen AI Max+ 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | No | 215 | 52 | 105 | 2,851 | incl | **2,851** | Best Linux maturity; incl 1TB. RAM-shortage pushed 128GB SKU +$460 |
| Minisforum MS-S1 Max | Ryzen AI Max+ 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | No | 215 | 52 | 105 | 2,959 | incl | **2,959** | 2U rack option; 2TB incl; premium WS pricing |
| HP ZBook Ultra G1a PRO | Ryzen AI Max+ PRO 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | No | 215 | 50 | 100 | 4,049 | incl | **4,049** | Self-contained laptop; unusual for server |
| HP Z6 G5 (8-ch, CPU only) | TR PRO 7945WX (Zen 4) | 12 | CPU | 128 GB (8×16 RDIMM) | **Yes** | 333 | 48 | 90 | 2,400 | 1,520 | **3,920** | Highest CPU-only bandwidth; PCIe headroom for dGPU |
| Dell Precision 3280 + RTX A1000 8GB | i9-14900 (Raptor) + dGPU | 24 | dGPU-CUDA | 128 GB | **Yes** | 144 (VRAM) | 45 | 110 | 1,858 | 1,160 | **3,018** | As-Is incl A1000 ($450); ProSupport, RHEL cert, ECC |
| Minisforum AtomMan G7 PT | Ryzen 9 7945HX (Zen 4) + RX 7600M XT | 16 | dGPU-ROCm | 32 GB | No | 288 (VRAM) | 42 | 85 | 1,499 | incl | **1,499** | Only mini PC with dGPU OOB; 8GB VRAM cap; 1TB incl |

---

## 🟢 TIER 3 — PRODUCTION (25–40 tok/s)

| Make / Model | CPU (Arch) | Cores | Path | RAM | ECC | BW (GB/s) | 7B t/s | 3B t/s | As-Is $ | +RAM $ | Total $ | Notes |
|--------------|-----------|-------|------|-----|-----|-----------|--------|--------|---------|--------|---------|-------|
| Lenovo M90q Gen 6 | Core Ultra 9 285 (Arrow Lake) | 24 | iGPU-IPEX | 64 GB | No | 102 | 32 | 68 | 1,800 | 380 | **2,180** | Arc iGPU via IPEX-LLM/OpenVINO; vPro |
| Lenovo P3 Tiny G2 | Core Ultra 9 285 (Arrow Lake) | 24 | iGPU-IPEX | 64 GB | No | 102 | 32 | 68 | 1,500 | 380 | **1,880** | vPro Ent; 1L |
| Lenovo P3 Ultra G2 (CPU/iGPU) | Core Ultra 9 285 (Arrow Lake) | 24 | iGPU-IPEX | 128 GB | **Yes** | 102 | 32 | 68 | 1,500 | 1,160 | **2,660** | ECC; dGPU slot → Reference (+$2,000 GPU+upg) |
| ASUS NUC 15 Pro+ | Core Ultra 9 285H (Arrow Lake) | 16 | iGPU-IPEX | 64 GB | No | 89 | 32 | 68 | 839 | 760 | **1,599** | Smallest (0.7L); barebone + RAM |
| SR645 EPYC 7413 (8-ch, CPU) | EPYC 7413 (Zen 3) | 24 | CPU | 128 GB (8×16 RDIMM) | **Yes** | 204 | 31 | 58 | 2,500 | 1,800 | **4,300** | ⚠️ Refurb-only; best concurrent-user CPU box; 128MB L3 |
| Geekom IT15 | Core Ultra 9 285H (Arrow Lake) | 16 | iGPU-IPEX | 32 GB | No | 89 | 30 | 65 | 1,199 | incl | **1,199** | Tiniest (0.46L); 32GB+2TB incl |
| Khadas Mind 2S | Core Ultra 7 255H (Arrow Lake) | 16 | iGPU-IPEX | 64 GB | No | 128 | 28 | 60 | 1,599 | incl | **1,599** | Pocketable; LPDDR5X soldered; 2TB incl |
| HP Z6 G5 (4-ch, CPU only) | TR PRO 7945WX (Zen 4) | 12 | CPU | 128 GB (4×32 RDIMM) | **Yes** | 166 | 26 | 50 | 2,400 | 1,200 | **3,600** | Mid-channel; upgrade to 8-ch (8×16) for 333 GB/s |
| Custom Ryzen 9 9950X (CPU) | Ryzen 9 9950X (Zen 5) | 16 | CPU | 64 GB | Optional | 96 | 25 | 55 | 1,060 | 420 | **1,480** | Best Zen 5 IPC + native AVX-512; PCIe for dGPU |

---

## 🟡 TIER 4 — DEMO-WORKABLE (12–25 tok/s)

| Make / Model | CPU (Arch) | Cores | Path | RAM | ECC | BW (GB/s) | 7B t/s | 3B t/s | As-Is $ | +RAM $ | Total $ | Notes |
|--------------|-----------|-------|------|-----|-----|-----------|--------|--------|---------|--------|---------|-------|
| Framework Desktop 385 | Ryzen AI Max 385 (Zen 5) | 8 | iGPU-Vulkan | 32 GB | No | 215 | 25 | 65 | 1,139 | incl | **1,139** | High BW but only 8 cores; soldered RAM |
| Custom Ryzen 9 7950X (CPU) | Ryzen 9 7950X (Zen 4) | 16 | CPU | 64 GB | Optional | 89 | 22 | 50 | 1,000 | 420 | **1,420** | Original PoC pick; PCIe for dGPU |
| Minisforum MS-A2 | Ryzen 9 9955HX (Zen 5) | 16 | CPU | 64 GB | No | 89 | 22 | 45 | 799 | 760 | **1,559** | PCIe x16 + dual SFP+ 10GbE; barebone + RAM |
| Beelink SER9 Pro | Ryzen AI 9 HX 370 (Strix Point) | 12 | iGPU-Vulkan | 32 GB | No | 120 | 20 | 50 | 999 | incl | **999** | LPDDR5X-7500 soldered; 1TB incl |
| Minisforum UM890 Pro | Ryzen 9 8945HS (Zen 4) | 8 | iGPU-Vulkan | 64 GB | No | 89 | 17 | 40 | 479 | 760 | **1,239** | OCuLink for eGPU → Best-in-Class; barebone + RAM |
| HP EliteDesk 805 G9 Mini | Ryzen 7 PRO 8700G (Zen 4) | 8 | iGPU-Vulkan | 64 GB | No | 89 | 16 | 38 | 1,017 | 380 | **1,397** | DASH mgmt; Radeon 780M iGPU |
| GMKtec K11 | Ryzen 9 8945HS (Zen 4) | 8 | iGPU-Vulkan | 32 GB | No | 89 | 15 | 38 | 599 | incl | **599** | 💰 Cheapest viable; 32GB+1TB incl; OCuLink |
| Lenovo M75q Gen 5 | Ryzen 7 PRO 8700GE (Zen 4) | 8 | iGPU-Vulkan | 64 GB | No | 83 | 15 | 36 | 800 | 760 | **1,560** | Compact AMD enterprise; DASH |
| Beelink GTR7 Pro | Ryzen 9 7940HS (Zen 4) | 8 | iGPU-Vulkan | 32 GB | No | 89 | 13 | 35 | 699 | incl | **699** | Older Zen 4; 32GB+1TB incl |
| HP EliteDesk 800 G9 Mini | i9-14900T (Raptor) | 24 | CPU | 64 GB | No | 89 | 13 | 32 | 1,400 | incl | **1,400** | vPro Ent; no AVX-512, weak iGPU; 64GB incl |
| Lenovo M90q Gen 5 | i9-14900 (Raptor) | 24 | CPU | 64 GB | No | 89 | 13 | 32 | 2,395 | incl | **2,395** | vPro Ent; brand premium; bandwidth-capped |
| Dell OptiPlex 7020 MFF | i9-14900 (Raptor) | 24 | CPU | 64 GB | No | 89 | 13 | 32 | 1,200 | 380 | **1,580** | vPro Ent |
| Minisforum MS-01 | i9-13900H (Raptor) | 14 | CPU | 64 GB | No | 83 | 12 | 32 | 649 | 760 | **1,409** | Dual SFP+ 10GbE + PCIe x16 → Reference w/dGPU |

> **Excluded (Inadequate, <12 t/s):** HP ProDesk 405 G9 Mini, Lenovo ThinkCentre Neo 50q, Lenovo ThinkEdge SE30, HP Z6 G5 at 2-channel.

---

## ⭐ TOP 5 OPTIONS (Single-Machine Full-Stack, Pure Technical Merit)

| Rank | Make / Model | CPU (Arch) | Cores | Path | RAM | ECC | BW (GB/s) | 7B t/s | 3B t/s | As-Is $ | +RAM $ | Total $ | Why It Wins |
|------|--------------|-----------|-------|------|-----|-----|-----------|--------|--------|---------|--------|---------|-------------|
| 🥇 | **Lenovo P3 Ultra G2 + RTX 4000 SFF Ada** | Core Ultra 9 285 + dGPU | 24 | dGPU-CUDA | 128 GB | **Yes** | 280 | 90 | 200 | 2,750 | 1,160 | **3,910** | Fastest single box running the whole stack; ECC + vPro; CUDA = most mature path |
| 🥈 | **HP Z2 Mini G1a PRO 128GB** | Ryzen AI Max+ PRO 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | **Yes** | 256 | 55 | 115 | 3,900 | incl | **3,900** | Best no-dGPU enterprise box: ECC, DASH, HP warranty, Ubuntu-cert, runs 70B/MoE in 2.7L |
| 🥉 | **Framework Desktop 395 128GB** | Ryzen AI Max+ 395 (Zen 5) | 16 | iGPU-Vulkan | 128 GB | No | 215 | 52 | 105 | 2,851 | incl | **2,851** | Best Linux maturity; upstream kernel/Mesa support; repairable; runs 70B Q4 |
| 4 | **HP Z6 G5 (TR PRO 7945WX, 8-ch)** | TR PRO 7945WX (Zen 4) | 12 | CPU (+dGPU opt) | 128 GB | **Yes** | 333 | 48 (90 w/dGPU) | 90 | 2,400 | 1,520 | **3,920** | Highest CPU-only BW; true WS with ECC + full PCIe → +$1,250 GPU reaches Reference |
| 5 | **SR645 EPYC 7413 (8-ch)** | EPYC 7413 (Zen 3) | 24 | CPU (+dGPU opt) | 128 GB | **Yes** | 204 | 31 (80 w/dGPU) | 58 | 2,500 | 1,800 | **4,300** | Concurrent-user champion (24c, 128MB L3); rack-native. ⚠️ Refurb-only |

### Value flag
- **Cheapest Best-in-Class:** Beelink GTR9 Pro 128GB — **$1,985 all-in** (53 t/s, dual 10GbE). Best raw $/perf if ECC/enterprise-support aren't required.
- **Cheapest viable at all:** GMKtec K11 — **$599 all-in** (15 t/s). Demo-tier only.

---

## Pricing Methodology

| Convention | Detail |
|------------|--------|
| **As-Is** | Base machine as commonly sold. Barebone (no RAM) for DIY-RAM platforms; configured price (RAM included) for soldered/configured machines. For dGPU rows, includes the GPU (cost broken out in Notes). |
| **+RAM** | Cost of memory modules to reach the row's RAM. `incl` = soldered LPDDR5X or already in the sold config. |
| **Total** | As-Is + RAM = deploy-ready hardware cost. Excludes OS (free, Ubuntu), shipping, warranty add-ons. |
| **Memory unit costs (May 2026, shortage-inflated)** | DDR4-3200 16GB ECC RDIMM ~$225 · DDR5-5600 16GB ECC RDIMM ~$190 · DDR5-5600 32GB ECC RDIMM ~$300 · DDR5-5600 32GB SODIMM ~$380 · DDR5-5600 32GB ECC SODIMM ~$290 · DDR5-6000 2×32GB desktop ~$420 |
| **GPU costs (near-MSRP)** | RTX 4000 SFF Ada 20GB ~$1,250 · RTX A1000 8GB ~$450 · RTX 4070 Ti Super 16GB ~$1,050 |
| **SR645 caveat** | EPYC 7413 generation is withdrawn from new sale; priced as refurb (~$2,500 base). New-configured would be ~$7-9.5K. |
| **HP Z6 G5 caveat** | True no-GPU is CTO-only; off-shelf bundles a T400. Base ~$2,400 used. |

---

## Architectural Reality (Single-Machine Constraint)

### Three viable inference paths on one Linux box

| Path | Ceiling (7B t/s) | Maturity | Examples |
|------|------------------|----------|----------|
| **dGPU-CUDA** | 90+ (VRAM-limited model size) | ⭐⭐⭐⭐⭐ | P3 Ultra / HP Z6 / SR645 / custom build + RTX 4000 Ada |
| **iGPU-Vulkan (Strix Halo)** | ~55 | ⭐⭐⭐⭐ (kernel ≥6.15 + tuning) | HP Z2 Mini G1a, Framework Desktop 395 |
| **iGPU-IPEX (Intel Arc)** | ~32 | ⭐⭐⭐ (IPEX-LLM/OpenVINO) | NUC 15 Pro+, M90q Gen 6, P3 Tiny |
| **CPU only** | ~48 (8-ch server) / ~25 (2-ch desktop) | ⭐⭐⭐⭐⭐ trivial | HP Z6 8-ch, SR645 8-ch, Ryzen 9950X |

### Decision splits three ways
1. **Max single-user speed** → dGPU-CUDA box (Tier 1). Accept VRAM cap + GPU driver maintenance. Cheapest: custom 9950X + RTX 4070 Ti Super (**$2,530**).
2. **Simplest reliable deploy + 70B capability** → Strix Halo. One box, no dGPU, 128GB unified. Cheapest: Beelink GTR9 Pro (**$1,985**); enterprise: HP Z2 Mini G1a (**$3,900**).
3. **Best multi-user (2-5 concurrent)** → high-core CPU box. SR645 24c (**$4,300**, refurb) or HP Z6 (**$3,920**). Add dGPU later for Reference single-user.

### Cost observations under the DRAM shortage
- **128GB server RAM (8×16 ECC) now costs $1,500-1,800** — as much as a refurb SR645 chassis. This erodes the server platforms' value vs soldered-RAM Strix Halo (where 128GB is baked into a $1,985-2,851 price).
- **Soldered-RAM Strix Halo boxes are now the value play** at 128GB — you avoid buying inflated DIMMs separately. Beelink GTR9 Pro at $1,985 all-in (128GB) undercuts a 128GB HP Z6 ($3,920) by nearly half while delivering more single-user tok/s.
- **ECC carries a steep premium** now: HP Z2 Mini G1a ($3,900) vs non-ECC Beelink GTR9 Pro ($1,985) — ~$1,900 for ECC + enterprise wrapper on the same silicon.

### Notes
- All tok/s are single-user. For 2-5 concurrent, divide by ~1.8–2.5×; 24-core options degrade least.
- **dGPU is a tier-elevator** for any PCIe platform (P3 Ultra, HP Z6, SR645, custom builds, MS-A2, MS-01) — adds ~$450-1,250 + reaches Reference, staying single-machine.
- **70B / large-MoE** requires 128GB+ unified (Strix Halo) or a high-VRAM dGPU. Most boxes here run 7B-32B comfortably.
