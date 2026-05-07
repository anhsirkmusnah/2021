# Satellite QKD-Secured GenAI Enterprise Productivity Application
# Hardware Requirements & Procurement Guide

**Version:** 1.0
**Date:** March 2026
**Status:** PoC/Pilot
**Budget Envelope:** USD $10,000–20,000 (compute hardware only; QKD ground terminals under separate budget)

---

## 1. Summary

This document specifies the hardware required for two compute nodes:

| Node | Location | Role | Form Factor |
|------|----------|------|-------------|
| **Server Node** | United Kingdom | Runs LLM inference, application backend, vector DB, QKD KMS, QKD-TLS gateway | Desktop workstation or high-performance mini-PC |
| **Client Node** | Singapore | Runs QKD KMS, QKD-TLS gateway, reverse proxy, optional frontend | Mini-PC or NUC-class device |

**Key constraint:** CPU-only inference (no GPU). LLM model size: 3.8B–7B parameters. Quantized to 4-bit (GGUF Q4_K_M).

---

## 2. UK Server Node — Detailed Specifications

This is the heavy-lifting machine. It runs the LLM inference engine, which is the most resource-intensive workload.

### 2.1 Workload Analysis

| Workload | CPU | RAM | Storage | Network |
|----------|-----|-----|---------|---------|
| Ollama + Qwen 2.5 7B Q4_K_M | High (all cores during inference) | ~6-8 GB | ~5 GB (model weights) | Localhost only |
| Ollama + Phi-3.5 Mini 3.8B Q4_K_M | Medium-High | ~4-5 GB | ~2.5 GB | Localhost only |
| Ollama + nomic-embed-text v1.5 | Low-Medium | ~1 GB | ~300 MB | Localhost only |
| FastAPI Backend + RAG pipeline | Low | ~1-2 GB | ~500 MB (app + deps) | Localhost only |
| ChromaDB | Low | ~1-2 GB (depends on doc count) | ~1-5 GB | Localhost only |
| Nginx + React Frontend | Minimal | ~200 MB | ~100 MB | Internal |
| QKD KMS | Minimal | ~200 MB | ~100 MB (key store) | QKD terminal NIC |
| QKD-TLS Gateway | Low | ~200 MB | ~50 MB | WAN NIC |
| Docker Engine + OS overhead | Low | ~2-3 GB | ~10 GB | — |
| **TOTAL** | **Sustained high** | **~16-22 GB active** | **~20 GB minimum** | **2 NICs** |
| **With headroom (recommended)** | — | **32-64 GB** | **500 GB - 1 TB** | **2 NICs** |

### 2.2 CPU Requirements (Critical)

CPU performance directly determines LLM inference speed. Key factors:

| Factor | Why It Matters | Requirement |
|--------|---------------|-------------|
| **Core count** | llama.cpp parallelizes across all cores during token generation | **Minimum 8 cores / 16 threads; Recommended 16 cores / 32 threads** |
| **AVX-512 / AVX2 support** | llama.cpp uses SIMD instructions heavily; AVX-512 provides ~30-50% speedup over AVX2 | **Required: AVX2; Preferred: AVX-512** |
| **Clock speed** | Higher single-core speed helps with sequential token generation | **Recommended: Base clock >= 3.5 GHz, Boost >= 5.0 GHz** |
| **Cache** | Large L3 cache improves memory access patterns for LLM inference | **Recommended: >= 32 MB L3 cache** |
| **Memory bandwidth** | LLM inference is memory-bandwidth-bound; DDR5 is significantly faster than DDR4 | **Required: DDR5 support** |

**Expected CPU inference performance (estimates for Qwen 2.5 7B Q4_K_M):**

| CPU | Cores/Threads | Est. Tokens/sec (single user) | Est. Tokens/sec (3 concurrent) |
|-----|---------------|-------------------------------|-------------------------------|
| AMD Ryzen 9 7950X | 16C/32T | 18-25 tok/s | 8-12 tok/s each |
| AMD Ryzen 9 9950X | 16C/32T | 20-28 tok/s | 10-14 tok/s each |
| Intel Core i7-14700K | 20C/28T | 15-22 tok/s | 7-10 tok/s each |
| Intel Core i9-14900K | 24C/32T | 18-25 tok/s | 9-12 tok/s each |
| AMD Ryzen 7 7800X | 8C/16T | 12-16 tok/s | 5-7 tok/s each |

> **Reading those numbers:** At 15 tokens/sec, a 200-word response (~270 tokens) takes ~18 seconds. At 10 tokens/sec, ~27 seconds. This is acceptable for a demo with 2-5 users but not instant.

> **For 3B model (Phi-3.5 Mini):** Roughly double the above speeds. A Ryzen 9 7950X would do ~35-50 tok/s single user.

### 2.3 Full Specification

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | AMD Ryzen 7 7800X (8C/16T) or Intel Core i7-14700 | **AMD Ryzen 9 7950X (16C/32T)** or AMD Ryzen 9 9950X | Ryzen 9 7950X offers best price/performance for llama.cpp workloads with AVX-512 support |
| **RAM** | 32 GB DDR5-4800 (2x16GB) | **64 GB DDR5-5600 (2x32GB)** | DDR5 is critical for memory-bandwidth-bound LLM inference. 64 GB allows comfortable headroom for concurrent users + vector DB |
| **Storage** | 512 GB NVMe PCIe Gen4 SSD | **1 TB NVMe PCIe Gen4 SSD** | Model weights ~5 GB, ChromaDB grows with documents, OS + Docker images ~20 GB. 1 TB provides ample space for model experimentation |
| **Motherboard** | B650 (AMD) or B760 (Intel) with 2+ M.2 slots | B650 or X670 with dual NIC or PCIe slot for add-in NIC | Must support DDR5; preferably has 2x Ethernet or PCIe slot for second NIC |
| **Network** | 1x Gigabit Ethernet (onboard) + 1x USB Ethernet adapter | **2x Gigabit Ethernet** (onboard or 1 onboard + 1 PCIe add-in) | NIC 1: QKD terminal. NIC 2: WAN/classical channel. PCIe Intel I210/I225 GbE card ~$30 if motherboard has only 1 NIC |
| **PSU** | 550W 80+ Bronze | 650W 80+ Gold | No GPU, so power draw is modest. 650W provides headroom |
| **Case** | Any ATX mid-tower or SFF | Compact/rack-friendly if datacenter deployment | Fractal Design Node 304, Silverstone SG13, or similar for small footprint |
| **Cooling** | Stock AMD cooler (Wraith Prism) | Tower cooler (Noctua NH-D15 or be quiet! Dark Rock 4) | Sustained LLM inference will keep CPU at high load continuously; good cooling prevents throttling |
| **OS** | Ubuntu Server 24.04 LTS (headless) | Ubuntu Server 24.04 LTS (headless) | Minimal install, no desktop environment |

### 2.4 Important CPU Notes

**AMD Ryzen 7000/9000 series and AVX-512:**
- Ryzen 7000 series (Zen 4) supports AVX-512, which llama.cpp uses for significant performance gains
- Ryzen 9000 series (Zen 5) also supports AVX-512 with further IPC improvements
- Intel 12th/13th/14th gen desktop chips have AVX-512 fused off on most SKUs (disabled by Intel)
- Intel Xeon and some specific SKUs support AVX-512 but are more expensive
- **Recommendation: AMD Ryzen 7000/9000 series for best CPU LLM inference performance**

**Memory bandwidth matters more than capacity:**
- LLM inference is memory-bandwidth-bound, not compute-bound
- DDR5-5600 provides ~89 GB/s dual-channel bandwidth vs DDR4-3200's ~51 GB/s
- This translates directly to ~40-50% faster token generation
- **Always use DDR5 for LLM workloads**

---

## 3. Singapore Client Node — Detailed Specifications

This is a lightweight machine. It runs no AI workloads — only network proxying, key management, and optionally the frontend web server.

### 3.1 Workload Analysis

| Workload | CPU | RAM | Storage | Network |
|----------|-----|-----|---------|---------|
| QKD KMS | Minimal | ~200 MB | ~100 MB | QKD terminal NIC |
| QKD-TLS Gateway | Low | ~200 MB | ~50 MB | WAN NIC |
| Nginx reverse proxy (user-proxy) | Minimal | ~100 MB | ~50 MB | User LAN NIC |
| React Frontend (optional, Nginx served) | Minimal | ~100 MB | ~100 MB | Internal |
| Docker Engine + OS overhead | Low | ~1-2 GB | ~8 GB | — |
| **TOTAL** | **Low** | **~2-3 GB active** | **~10 GB** | **2-3 NICs** |
| **With headroom** | — | **16 GB** | **256-512 GB** | **2 NICs** |

### 3.2 Full Specification

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | Intel Core i5 / AMD Ryzen 5 (4+ cores) | **Intel N100/N305 or AMD Ryzen 5 7530U class** | Any modern quad-core is sufficient; this machine is not compute-intensive |
| **RAM** | 8 GB DDR4/DDR5 | **16 GB DDR4/DDR5** | Headroom for Docker overhead and future expansion |
| **Storage** | 256 GB NVMe/SATA SSD | **512 GB NVMe SSD** | Minimal storage needs; 512 GB avoids any future pressure |
| **Network** | 1x Gigabit Ethernet + 1x USB Ethernet | **2x Gigabit Ethernet** (onboard or 1 + USB 3.0 Ethernet) | NIC 1: QKD terminal. NIC 2: WAN + user LAN (can share with VLAN). 3rd NIC if separate user LAN needed |
| **Form Factor** | Mini-PC / NUC | **Mini-PC / NUC** | Compact, low-power, easy to deploy in datacenter |
| **PSU** | Included with mini-PC | Included | Typically 65-120W external adapter |
| **Cooling** | Passive or integrated fan | Passive preferred (silent, no moving parts) | Low workload = low heat |
| **OS** | Ubuntu Server 24.04 LTS | Ubuntu Server 24.04 LTS | Same as server node for consistency |

---

## 4. Recommended Products & Estimated Pricing

### 4.1 UK Server Node — Option A: Custom Build (Best Performance/Dollar)

| Part | Specific Product | Est. Price (USD) |
|------|-----------------|-----------------|
| CPU | AMD Ryzen 9 7950X (16C/32T, 4.5-5.7 GHz, 64MB L3) | $450-550 |
| Motherboard | ASUS TUF GAMING B650-PLUS WiFi or Gigabyte B650 AORUS Elite AX | $150-180 |
| RAM | 64 GB (2x32GB) DDR5-5600 CL36 (G.Skill Ripjaws, Corsair Vengeance, or Kingston Fury) | $130-170 |
| Storage | 1 TB WD Black SN770 or Samsung 990 EVO NVMe Gen4 | $70-90 |
| PSU | Corsair RM650e or be quiet! Pure Power 12 M 650W | $70-90 |
| Case | Fractal Design Pop Mini Silent or Silverstone FARA R1 | $60-80 |
| CPU Cooler | Noctua NH-D15 or Thermalright Peerless Assassin 120 | $50-90 |
| NIC (2nd) | Intel I225-V 2.5GbE PCIe Card or TP-Link TX201 | $25-35 |
| **Subtotal** | | **$1,005-1,285** |

### 4.2 UK Server Node — Option B: Pre-Built Mini-PC/Barebone (Easier Procurement)

| Product | Specs | Est. Price (USD) |
|---------|-------|-----------------|
| **Minisforum MS-A1** (or similar Ryzen 9 mini-PC) | AMD Ryzen 9 7945HX (16C/32T), barebone (add RAM/SSD) | $550-700 |
| RAM upgrade | 64 GB DDR5-5600 SO-DIMM (2x32GB) | $150-200 |
| Storage | 1 TB NVMe SSD | $70-90 |
| USB-C to Ethernet adapter (if needed for 2nd NIC) | USB 3.0 Gigabit Ethernet | $15-20 |
| **Subtotal** | | **$785-1,010** |

> **Note on mini-PCs:** Many mini-PCs use laptop-class CPUs (HX/HS series). The Ryzen 9 7945HX has 16 cores and is very close to the desktop 7950X in multi-threaded performance, but may throttle under sustained load due to thermal constraints. A desktop build (Option A) provides better sustained performance for continuous LLM inference. For a demo that runs intermittently, a mini-PC is perfectly adequate.

### 4.3 UK Server Node — Option C: Used/Refurbished Workstation (Lowest Cost)

| Product | Specs | Est. Price (USD) |
|---------|-------|-----------------|
| Refurbished HP Z4 G4 or Dell Precision 5820 | Intel Xeon W-2200 series (10-18 cores), DDR4, tower workstation | $400-700 |
| RAM upgrade to 64 GB DDR4 ECC | 4x16GB DDR4-2666 ECC | $80-120 |
| NVMe SSD 1 TB | Add PCIe NVMe adapter if needed | $70-90 |
| **Subtotal** | | **$550-910** |

> **Caution:** These older Xeon chips support AVX-512 (good) but have lower clock speeds and DDR4 only (bad for LLM bandwidth). Expect ~30-40% slower inference than Ryzen 9 7950X. Acceptable for PoC if budget is very tight.

### 4.4 Singapore Client Node

| Product | Specs | Est. Price (USD) |
|---------|-------|-----------------|
| **Beelink SER5 / Minisforum UM560** (or similar) | AMD Ryzen 5 5600H, 16 GB DDR4, 500 GB SSD | $250-350 |
| **Intel NUC 12/13 (or ASUS NUC)** | Intel Core i5-1240P/1340P, 16 GB, 512 GB | $350-500 |
| **Beelink EQ12/EQ13** (ultra-budget) | Intel N100/N305, 16 GB DDR5, 500 GB SSD | $180-250 |
| USB Ethernet adapter (if 2nd NIC needed) | USB 3.0 Gigabit | $15-20 |

> **Recommendation:** Beelink/Minisforum AMD mini-PC at ~$300 — more than sufficient for proxy/KMS workloads.

### 4.5 Accessories & Peripherals (Both Sites)

| Item | Qty | Est. Price (USD) | Notes |
|------|-----|-----------------|-------|
| USB keyboard + mouse (for initial setup) | 2 | $30 | Can be shared/borrowed; not needed after setup if SSH is configured |
| HDMI display or portable monitor (for initial setup) | 2 | $0-100 | Borrow existing or use portable USB-C monitor |
| Ethernet cables (Cat6) | 4-6 | $20 | Various lengths for QKD terminal and WAN connections |
| USB flash drive (Ubuntu installer) | 2 | $10 | 8 GB minimum |
| UPS / Surge protector (optional) | 2 | $50-150 | Recommended for continuous pilot operation |
| **Subtotal** | | **$110-310** |

---

## 5. Budget Summary

### 5.1 Recommended Configuration

| Item | Est. Cost (USD) |
|------|----------------|
| UK Server Node (Option A — Custom Build, Ryzen 9 7950X) | $1,100-1,300 |
| Singapore Client Node (Mini-PC, Ryzen 5 class) | $280-350 |
| Accessories & peripherals | $110-310 |
| **Shipping (international, to UK + Singapore)** | $100-300 |
| **Contingency (10%)** | $160-230 |
| **TOTAL** | **$1,750-2,490** |

### 5.2 Budget Configuration

| Item | Est. Cost (USD) |
|------|----------------|
| UK Server Node (Option B — Mini-PC, Ryzen 9 7945HX) | $800-1,000 |
| Singapore Client Node (Budget Mini-PC, Intel N100) | $180-250 |
| Accessories & peripherals | $110-250 |
| Shipping | $100-200 |
| Contingency (10%) | $120-170 |
| **TOTAL** | **$1,310-1,870** |

### 5.3 Lowest Cost Configuration

| Item | Est. Cost (USD) |
|------|----------------|
| UK Server Node (Option C — Refurbished Workstation) | $550-900 |
| Singapore Client Node (Budget Mini-PC) | $180-250 |
| Accessories & peripherals | $60-150 |
| Shipping | $100-200 |
| Contingency (10%) | $90-150 |
| **TOTAL** | **$980-1,650** |

> **All configurations are well within the $10,000-20,000 budget.** The remaining budget can cover: software licenses (none needed for this stack), travel for on-site setup, additional testing equipment, or future expansion (e.g., adding a GPU later for faster inference).

---

## 6. Network Requirements

### 6.1 Inter-Site Connectivity (Classical Channel)

| Requirement | Specification |
|-------------|--------------|
| **Bandwidth** | Minimum 10 Mbps; Recommended 50+ Mbps |
| **Latency** | UK-Singapore typical: 160-200ms RTT. Acceptable for this application (LLM inference time dominates, not network latency) |
| **Reliability** | Stable connection required for continuous pilot. Redundant path recommended |
| **Type** | Corporate WAN, MPLS, site-to-site VPN over internet, or dedicated link |
| **Ports** | Configurable; default: TCP 443 (QKD-encrypted tunnel), TCP 8443 (KMS sync) |

> **Note:** The QKD-encrypted tunnel runs OVER this classical channel. The classical channel itself does not need to be encrypted (QKD provides the encryption), but it must be authenticated and reliable. If using internet, a basic VPN for routing is acceptable.

### 6.2 QKD Ground Terminal Connectivity

| Requirement | Specification |
|-------------|--------------|
| **Interface** | Gigabit Ethernet (confirm with SpeQtral/RAL Space) |
| **Protocol** | ETSI QKD 014 key delivery API over HTTPS (confirm with vendor) |
| **Network** | Private/isolated subnet (e.g., 10.0.1.0/24) between compute node and QKD terminal |
| **Firewall** | Only KMS container communicates with QKD terminal; all other containers blocked |

### 6.3 User LAN (Singapore Only)

| Requirement | Specification |
|-------------|--------------|
| **Interface** | Gigabit Ethernet or WiFi (via USB adapter if needed) |
| **Users** | 2-5 concurrent users with web browsers |
| **Bandwidth per user** | ~1-5 Mbps (text-based application, no video/large downloads) |
| **Access** | Users connect to `https://<client-node-ip>` in their browser |

---

## 7. Power & Environmental

| Site | Est. Power Draw | Heat Output | Notes |
|------|----------------|-------------|-------|
| UK Server (Desktop build) | 150-250W under LLM inference load; ~80W idle | ~850 BTU/hr peak | No special cooling needed for a single machine in a datacenter |
| UK Server (Mini-PC) | 80-150W under load; ~30W idle | ~500 BTU/hr peak | Very low power footprint |
| Singapore Client | 15-45W under load; ~10W idle | ~150 BTU/hr peak | Negligible |

- Standard 220-240V (UK) / 220-240V (Singapore) power outlets
- No special power requirements — standard datacenter PDU or wall outlet
- UPS recommended for continuous pilot operation (500VA unit for server, 300VA for client)

---

## 8. Software Requirements (Pre-Installation Checklist)

These are free/open-source and downloaded during initial setup:

| Software | Version | Source | Size |
|----------|---------|--------|------|
| Ubuntu Server 24.04 LTS | 24.04.x | ubuntu.com | ~2 GB (ISO) |
| Docker Engine (CE) | Latest stable | docker.com | ~300 MB |
| Docker Compose | v2.x (included with Docker) | docker.com | Included |
| Ollama | Latest | ollama.com | ~200 MB |
| Qwen 2.5 7B Instruct Q4_K_M | GGUF | Ollama registry / HuggingFace | ~4.5 GB |
| Phi-3.5 Mini 3.8B Instruct Q4_K_M | GGUF | Ollama registry / HuggingFace | ~2.3 GB |
| nomic-embed-text v1.5 | GGUF | Ollama registry | ~270 MB |
| Python 3.12 | Slim Docker image | Docker Hub | ~150 MB |
| Node.js 20 | Slim Docker image | Docker Hub | ~200 MB |
| ChromaDB | Latest | Docker Hub | ~500 MB |
| Nginx | Alpine | Docker Hub | ~40 MB |
| **Total download** | | | **~10 GB** |

> **If machines cannot access internet during setup:** Pre-download all images and model weights onto a USB drive or external SSD (~32 GB USB drive is sufficient). Docker images can be exported/imported via `docker save/load`. Ollama models can be copied as files.

---

## 9. Procurement Checklist

### 9.1 Hardware to Order

- [ ] UK Server Node: AMD Ryzen 9 7950X build or equivalent mini-PC (choose from Section 4)
- [ ] Singapore Client Node: Mini-PC (choose from Section 4.4)
- [ ] Ethernet cables: 4-6x Cat6 (1m and 3m lengths)
- [ ] USB Ethernet adapters: 2x (if machines have fewer than 2 built-in NICs)
- [ ] USB flash drives: 2x 16GB (for Ubuntu installer)
- [ ] USB keyboard + mouse: 1-2 sets (for initial setup, can reuse existing)
- [ ] UPS (optional): 2x basic units (500VA for UK, 300VA for Singapore)

### 9.2 Pre-Procurement Verification

Before ordering, confirm:

- [ ] Datacenter rack space or desk space available at both UK and Singapore sites
- [ ] Power outlets available (standard AC) at both sites
- [ ] Network ports available at both sites for WAN connection
- [ ] QKD ground terminal Ethernet interface specification from SpeQtral/RAL Space
- [ ] Shipping logistics and import requirements for both countries
- [ ] IT approval for connecting compute nodes to datacenter network

### 9.3 Recommended Vendor Channels

| Region | Vendor Options | Notes |
|--------|---------------|-------|
| UK | Amazon UK, Scan Computers, Overclockers UK, eBuyer | Custom build parts or pre-built systems |
| Singapore | Amazon SG, Lazada, Shopee, Sim Lim Square retailers | Mini-PCs widely available |
| Global | Minisforum.com, Beelink official store | Ship worldwide; good mini-PC selection |

---

## 10. Future Expansion Path (Beyond PoC)

If the PoC succeeds and the system moves to pilot or production:

| Upgrade | Impact | Est. Cost |
|---------|--------|-----------|
| Add GPU (NVIDIA RTX 4060/4070) to UK server | 10-50x faster LLM inference; enables larger models (13B-30B) | $300-600 |
| Upgrade to 70B model (with GPU) | Dramatically better output quality | GPU required |
| Add second UK server node | Redundancy, handle more concurrent users | $1,000-1,500 |
| Replace desktop with 1U rack server | Proper datacenter form factor, IPMI remote management | $2,000-4,000 |
| Add monitoring stack (Prometheus + Grafana) | Production-grade observability | Free (software) |
| Add authentication (SSO/SAML) | Multi-user access control | Free (Keycloak) |
