# Request for Quotation — On-Premises AI Inference Server

**Prepared:** May 2026
**Quantity:** 1 unit (of the selected configuration)
**Purpose:** Procurement RFQ — IT hardware only

---

**To:** [Reseller / Distributor / Account Manager]
**From:** [Name, Title, Department, Company]
**Contact:** [Phone / Email]
**Response requested by:** [Date]

---

**Subject:** RFQ — On-Premises AI Inference Server (3 configurations for evaluation; single-unit order)

Dear [Account Manager name],

We are evaluating compact, single-node compute platforms for an on-premises local AI (LLM) inference workload to be deployed in a **rack-based datacenter environment**. Please provide a formal quotation for the **three configurations below**, quoted individually.

We will order **one (1) unit** of the single selected configuration. Please quote each as a single unit.

All three configurations **must be rack-deployable** and meet the **common requirements** listed at the end.

---

## Configuration A — Compact Workstation + Pro GPU (performance pick)

| Item | Requirement |
|------|-------------|
| Platform | Lenovo ThinkStation P3 Ultra Gen 2 (or equivalent compact workstation) |
| CPU | Intel Core Ultra 9 285 (24-core) |
| Memory | 128 GB DDR5 **ECC** (4 × 32 GB SODIMM) |
| GPU | NVIDIA RTX 4000 SFF Ada Generation, 20 GB GDDR6 |
| Storage | 2 × 2 TB NVMe PCIe Gen4 SSD (RAID-1 capable) |
| Networking | Dual Ethernet preferred (min 2.5GbE) |
| Remote mgmt | Intel vPro Enterprise (KVM-over-IP) |
| OS | Ubuntu 24.04 LTS pre-installed, or no-OS |

## Configuration B — Integrated-GPU Workstation (no discrete GPU)

| Item | Requirement |
|------|-------------|
| Platform | HP Z2 Mini G1a (or equivalent) |
| CPU | AMD Ryzen AI Max+ PRO 395 (16-core) |
| Memory | 128 GB LPDDR5X-8533 **ECC** (unified/soldered — confirm exact ECC SKU) |
| GPU | Integrated AMD Radeon 8060S (no discrete GPU required) |
| Storage | 2 TB NVMe PCIe Gen4 SSD (2 × 1 TB if dual-slot available) |
| Networking | 2.5GbE minimum; 10GbE option if available |
| Remote mgmt | AMD DASH + HP Sure Start |
| OS | Ubuntu 24.04 LTS pre-installed, or no-OS |

## Configuration C — Compact Workstation + Entry Pro GPU (value pick)

| Item | Requirement |
|------|-------------|
| Platform | Dell Precision 3280 Compact (or equivalent) |
| CPU | Intel Core i9-14900 (24-core) |
| Memory | 128 GB DDR5 **ECC** (4 × 32 GB) |
| GPU | NVIDIA RTX A1000, 8 GB GDDR6 |
| Storage | 2 × 2 TB NVMe PCIe Gen4 SSD (RAID-1 capable) |
| Networking | Dual Ethernet preferred (min 2.5GbE) |
| Remote mgmt | Intel vPro Enterprise (KVM-over-IP) |
| OS | Ubuntu 24.04 LTS pre-installed, or no-OS |

---

## Common Requirements (mandatory — all three configurations)

- **Rack deployment (required):** Each unit must be deployable in a standard 19-inch rack. Please include the appropriate **rack-mount shelf, tray, or kit** in the quote and state the **rack units (U)** consumed per unit. Non-rack-deployable configurations will not be considered.
- **Continuous operation:** Rated for sustained 24/7 full-load operation. Confirm operating temperature range and thermal/throttling behaviour under sustained CPU+GPU load.
- **Power:** Auto-sensing 100–240 V, 50/60 Hz. State maximum power draw (W), PSU rating, and plug type. Note whether a redundant PSU is available.
- **Security baseline:** TPM 2.0 and UEFI Secure Boot required.
- **Linux support:** Confirm official **Ubuntu 24.04 LTS certification** for the exact SKU (and RHEL if available), including GPU driver support.
- **Warranty:** 3-year next-business-day on-site support. Please also quote 5-year and premium/4-hour-response options.

---

## Commercial / Procurement Information Requested

1. **Itemised per-unit pricing** (hardware + warranty + rack kit broken out separately).
2. **Lead time / availability** for each configuration — please flag any components affected by current memory (DRAM) supply constraints.
3. **Delivery** to [datacenter address — to be confirmed]; include shipping and any import/duty handling.
4. **Payment terms** and quote validity period.
5. Confirmation that all configurations are **currently orderable** (not end-of-life); propose the nearest equivalent if any line item is unavailable.

Please direct any technical clarification questions to me. We would appreciate the quotation by **[date]**.

Kind regards,
[Name]
[Title / Department]
[Company]
[Phone / Email]
