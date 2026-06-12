# Mid-Year OKR Update — Satellite QKD GenAI PoC

**Period:** H1 2026 (January – May 2026)
**Project:** Satellite-QKD-secured GenAI enterprise productivity application (UK ↔ Singapore)
**Reviewer:** [Name]
**Date:** May 2026

---

## Executive Summary

PoC technical foundation is in place. Architecture is fully documented and stakeholder-ready, hardware is procurement-ready, and external dependencies are mapped. **All four objectives are on track or substantially complete; one is at-risk pending vendor response.** The gating items for H2 are SpeQtral's reply to the integration questionnaire and the hardware procurement cycle.

---

## Objective 1 — Define a complete, defensible technical architecture for the QKD GenAI PoC

**Status:** 🟢 **On Track — 95% complete**

| KR | Target | Result | Status |
|----|--------|--------|--------|
| 1.1 | Multi-layer architecture document covering system, network, container, security, and QKD layers | 30+ page document, 13 sections + appendices | ✅ |
| 1.2 | Enterprise-grade diagram suite | 10 diagrams produced (system, UK/SG container topology, QKD key lifecycle, network topology, end-to-end request flow, rekey sequence, RAG pipeline, security layers, C4 context) | ✅ |
| 1.3 | Stakeholder-ready document package, HSBC-branded | DOCX (44 tables), PPTX (30 slides), PDF (23 pages), editable draw.io — all branded, classification-marked HSBC INTERNAL | ✅ |
| 1.4 | IPsec/XFRM + QKD key lifecycle deep-dive | Full coverage: make-before-break rekey, control channel HMAC, watchdog, ETSI 004/014 API spec, simulator mode | ✅ |

**Evidence:** `01_ARCHITECTURE.md`, `/diagrams/output/*.png`, `QKD_WorkAssist_Architecture.{docx,pptx,pdf,drawio}`

---

## Objective 2 — Make a defensible, cost-aware LLM-server hardware decision

**Status:** 🟢 **On Track — 90% complete** (awaits vendor quotation)

| KR | Target | Result | Status |
|----|--------|--------|--------|
| 2.1 | Evaluate ≥10 candidate compute platforms | **43 options** evaluated across Apple Silicon, AMD Strix Halo, custom Ryzen builds, consumer/enterprise mini PCs, workstations, and rack servers | ✅ Exceeded |
| 2.2 | Produce upgrade-aware comparison matrix | Flat 60-row Excel-friendly matrix (CSV + MD) showing as-is, +RAM, +dGPU configs with bandwidth and tier transitions | ✅ |
| 2.3 | Classify options into actionable performance tiers | 5-tier system (Reference / Best-in-Class / Production / Demo-Workable / Inadequate) ranked by single-user Qwen 7B tok/s | ✅ |
| 2.4 | Cost-effectiveness analysis under 2026 DRAM shortage | Memory + GPU + base-machine prices verified from current sources; total deploy cost calculated per option | ✅ |
| 2.5 | Constrain to full-stack-on-single-machine + rack-deployable | Apple Silicon (SplitArch) excluded; non-rack configs flagged | ✅ |
| 2.6 | Issue procurement-ready RFQ for shortlisted options | 3-config RFQ drafted (Lenovo P3 Ultra + RTX 4000 SFF Ada / HP Z2 Mini G1a 128GB ECC / Dell Precision 3280 + RTX A1000) | ✅ |
| 2.7 | Receive vendor quotation | Pending issuance | 🟡 Pending |

**Key findings:**
- 2026 DRAM shortage has 2–3× inflated server memory prices; this flips the value equation toward **soldered-RAM Strix Halo platforms** vs traditional DIMM servers.
- Full-stack-on-one-machine excludes all Apple Silicon (macOS cannot host the QKD/Docker/XFRM stack).
- Top performance-per-dollar enterprise pick: **Lenovo P3 Ultra G2 + RTX 4000 SFF Ada — $3,910 / 90 tok/s**.

**Evidence:** `LLM_TIER_OPTIONS.md`, `HARDWARE_MATRIX.{md,csv}`, `HARDWARE_UPGRADE_SCENARIOS.md`, `02_HARDWARE_REQUIREMENTS_V2.md`, `RFQ_LLM_INFERENCE_SERVERS.md`

---

## Objective 3 — Resolve external dependencies and integration unknowns

**Status:** 🟡 **At Risk — 65% complete** (gated on SpeQtral response)

| KR | Target | Result | Status |
|----|--------|--------|--------|
| 3.1 | Catalog all open SpeQtral integration questions | **80 questions** across 12 categories (API specifics, auth, key rates, network, hardware, ops, failure modes, testing-without-simulator, logistics, SLA, compliance) — prioritized P0–P3 | ✅ |
| 3.2 | Define mitigation plan for missing simulator | Plan documented: build ETSI-spec-compliant simulator + adapter shim layer; mock terminal as FastAPI service | ✅ |
| 3.3 | Decide inter-site WAN connectivity strategy | **EE Business 5G + Fixed Public IP** (UK) and Singtel/M1 Business equivalent (SG); avoids CGNAT and third-party VPN/SaaS dependencies | ✅ |
| 3.4 | Receive SpeQtral answers to P0 (build-blocking) questions | Awaiting response — request issued | 🔴 Pending |
| 3.5 | Confirm 5G + Fixed IP availability at both datacenter sites | Carrier shortlist defined; in-flight | 🟡 In progress |

**Risk:** Vendor delay on P0 questions (exact ETSI 014 JSON schema, authentication model) could push build timeline. **Mitigation in place:** build against strict ETSI standard text, insert adapter shim when actual responses arrive.

**Evidence:** `SPEQTRAL_QUESTIONS.md`

---

## Objective 4 — Validate end-user demo experience for QKD security indicator

**Status:** 🟢 **On Track — 85% complete**

| KR | Target | Result | Status |
|----|--------|--------|--------|
| 4.1 | QKD security indicator (floater) shows live rekey state | Deployed: simulated QKD daemon writes `status.json` every 5s; `REKEYING → UP` transitions render in real time; force-rekey trigger functional | ✅ |
| 4.2 | Full-page topology modal | Redesigned full-screen 4-zone layout (header / topology / stats bar / timeline+detail); previous scroll-clipping bug resolved | ✅ |
| 4.3 | Topology SVG render quality at scale | viewBox `0 0 1000 200` with `overflow:visible` and explicit filter regions; no edge clipping; 5-step animated progression | ✅ |
| 4.4 | Project demo-grade response speed on shortlisted hardware | ≥30 tok/s on Qwen 7B achievable on all three RFQ configs (Lenovo P3 Ultra: 90, HP Z2 Mini G1a: 55, Dell Precision 3280: 45) | ✅ |
| 4.5 | End-to-end stakeholder demo dry-run | Scheduled for H2 once hardware lands | 🟡 Pending HW |

**Evidence:** `services/app-frontend/src/components/TunnelVisualization.tsx`, `services/app-backend/src/routers/admin.py`

---

## Overall H1 Score

| Objective | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| O1 Architecture | 25% | 0.95 | 0.24 |
| O2 Hardware decision | 30% | 0.90 | 0.27 |
| O3 External dependencies | 25% | 0.65 | 0.16 |
| O4 Demo UX validation | 20% | 0.85 | 0.17 |
| **Total** | **100%** | — | **0.84** |

**H1 cumulative score: 0.84** — strong delivery on internally-owned work; the gap to 1.0 is concentrated in externally-gated items (vendor response, procurement cycle, carrier confirmation).

---

## H2 2026 Priorities

1. **Unblock SpeQtral P0 questions** — request a technical walkthrough call with SpeQtral engineering; target ETSI 014 JSON schema + auth model clarity by end of June.
2. **Issue RFQ and complete hardware procurement** — 4–6 week cycle including delivery to both UK and SG sites.
3. **Build ETSI 014 simulator + client adapter** — run in parallel with vendor responses so we are not blocked.
4. **Confirm EE Business 5G + Fixed IP** at the UK datacenter and Singtel/M1 Business equivalent at the SG site; order SIMs.
5. **Site survey readiness** for QKD ground terminal install (per SpeQtral logistics).
6. **End-to-end stakeholder demo** once hardware lands and tunnel is operational on real keys.

---

## Backlog / Stretch

- Re-baseline performance projections with canned API captures from a real SpeQtral terminal (if provided).
- Document multi-user concurrent performance under load.
- Decide on PCIe dGPU expansion path post-PoC (if scaling to pilot).
- Compliance traceability matrix (ETSI / NIST 800-57 / FIPS 197 / ISO 27001 / IMDA SG) mapped to delivered controls.

---

## Deliverable Inventory (H1)

| Category | Artifacts |
|----------|-----------|
| Architecture | `01_ARCHITECTURE.md`, `QKD_WorkAssist_Architecture.{docx,pptx,pdf,drawio}` |
| Diagrams | 10 PNG diagrams (`/diagrams/output/`) covering system, container, network, QKD lifecycle, data flow, security |
| Hardware analysis | `02_HARDWARE_REQUIREMENTS_V2.md`, `HARDWARE_MATRIX.{md,csv}`, `HARDWARE_UPGRADE_SCENARIOS.md`, `HARDWARE_COMPARISON.md`, `LLM_TIER_OPTIONS.md` |
| Procurement | `RFQ_LLM_INFERENCE_SERVERS.md` |
| Integration | `SPEQTRAL_QUESTIONS.md` (80 prioritized questions) |
| Application code | QKD floater + full-screen topology modal + simulated QKD daemon |







OKR 2 : MID YEAR REVIEW, PROVIDED TECHNICAL TALK AS A PART OF FDP - FACULTY DEVELOPMENT PROGRAM 

  Objective: Represent HSBC as a FACULTY TRAINER, delivering a generative-AI workshop to ~100 students of the as part of HSBC's volunteering engagement with supporting students from economically disadvantaged backgrounds.

  Scope & contribution:
  - Designed an original 90-minute (extensible to 2-hour) technical workshop, "Be the Algorithm: How Large Language Models Really Work — and How Banks Make
  Them Safe," demystifying GenAI from first principles — tokenization, embeddings, attention, next-token generation — through to production concerns
  relevant to financial services: hallucination, RAG, fine-tuning/SLMs, prompt-injection guardrails, and bias mitigation.
  - Built a custom, phone-based live interactive tool that lets a 100-person room collectively experience how an LLM generates a response in real time —
  chosen specifically to keep the session inclusive and accessible regardless of students' hardware access.
  - Authored a take-home Google Colab notebook for self-paced hands-on exploration, and prepared supporting slides and run-of-show.
  - Produced the formal workshop proposal (title, outline, prerequisites, logistics) for 

  Outcome / impact: Extends HSBC's brand and technical credibility into a leading academic institution, contributes to the technical upskilling of students
  from underrepresented backgrounds, and strengthens HSBC's industry–academia engagement with 

  Status: Workshop content and tooling complete; proposal submitted; delivery scheduled for 12–13 June 2026 at the 



