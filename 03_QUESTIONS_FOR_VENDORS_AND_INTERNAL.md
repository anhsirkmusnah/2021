# Satellite QKD-Secured GenAI PoC
# Questions for SpeQtral/RAL Space & Internal Teams

**Version:** 1.0
**Date:** March 2026
**Purpose:** Collect answers required to finalize architecture, integration, and deployment before hardware procurement and development begin.

---

## Section A: Questions for SpeQtral / RAL Space (QKD Vendor)

### A1. Ground Station Equipment

1. **Ground terminal provisioning:** Will SpeQtral/RAL Space provide QKD ground receiver terminals at both the UK and Singapore sites? Or does one site receive satellite keys and relay to the other?

2. **Ground terminal form factor:** What is the physical form factor, power, cooling, and space requirement for the ground terminal equipment? (Rack-mounted? Outdoor antenna + indoor unit? Size and weight?)

3. **Ground terminal connectivity:** What is the network interface for key delivery from the ground terminal to our compute node? (Ethernet assumed — please confirm interface type, speed, and protocol.)

4. **Installation requirements:** What are the installation prerequisites? (Roof access for antenna? Clear sky view? Specific orientation? Is professional installation provided by SpeQtral/RAL Space?)

5. **On-site support:** Will SpeQtral/RAL Space engineers be available on-site during initial setup and integration at both locations?

### A2. Key Delivery Interface

6. **API specification:** Does the ground terminal expose an ETSI GS QKD 014-compliant key delivery API? If not, what API/protocol is used to deliver keys to our Key Management Service?

7. **API documentation:** Can you provide API documentation, SDK, or sample code for integrating with the ground terminal's key delivery interface?

8. **Authentication:** How does our KMS authenticate with the ground terminal to receive keys? (Certificate-based? Pre-shared token? Physical network isolation only?)

9. **Key format:** What is the format of delivered keys? (Raw bytes? Base64? JSON? What metadata accompanies each key — key ID, timestamp, quality metrics?)

10. **Key ID synchronization:** How are key IDs synchronized between the UK and Singapore ground terminals so that both sides can identify matching keys? (Handled internally by SpeQtral system? Or must our KMS coordinate?)

### A3. Satellite Pass & Key Generation

11. **Pass schedule:** What is the expected satellite overpass schedule for the UK-Singapore link? (Passes per day? Duration per pass? Any seasonal/orbital variations?)

12. **Key generation rate:** What is the expected effective key generation rate? (Bits per pass? Bits per second during a pass? What is the usable key rate after sifting, error correction, and privacy amplification?)

13. **Key buffer:** What is the expected key pool size after a typical pass? Is this sufficient for continuous application use between passes? (Our estimated consumption: ~1 key per API request, 256 bits per key, ~100-500 requests per day.)

14. **Key persistence:** Are generated keys stored in the ground terminal until retrieved by our KMS? What happens if our KMS is temporarily offline during a pass — are keys buffered?

15. **Pass prediction:** Is there an API or schedule feed for upcoming pass times? (Useful for our status dashboard showing "next pass" to demo audiences.)

### A4. Security & Certification

16. **Security certification:** What security certifications does the QKD system hold? (Common Criteria? FIPS? ETSI QKD compliance certification?)

17. **Quantum bit error rate (QBER):** What is the typical QBER for the satellite link? What is the threshold at which the system aborts key generation?

18. **Tamper detection:** Does the ground terminal have tamper detection/response mechanisms?

### A5. Timeline & Logistics

19. **Equipment delivery timeline:** What is the lead time for ground terminal delivery and installation at UK and Singapore sites?

20. **Operational readiness:** After installation, what is the expected time to achieve first successful key exchange?

21. **Support model:** What ongoing support is provided? (Remote monitoring? On-call engineers? SLA for hardware issues?)

22. **Cost structure:** What is the cost model for the ground terminals and satellite access for a PoC/pilot duration of 3-6 months? (Lease? One-time? Per-pass?)

---

## Section B: Questions for Internal Enterprise IT / Infrastructure

### B1. Network & Connectivity

1. **WAN between UK and Singapore datacenters:** Is there existing WAN connectivity (MPLS, VPN, dedicated link, or internet) between the UK and Singapore datacenter sites? If yes, what bandwidth and latency? If no, what is the process to provision connectivity?

2. **Classical channel for QKD:** QKD requires a classical authenticated channel alongside the quantum channel. Can the existing WAN serve this purpose? Are there firewall rules or policies that would restrict custom TCP traffic (e.g., ports 443, 8443)?

3. **Network segmentation:** Can we create isolated network segments (VLANs) at each site for the QKD system? (We need: a QKD terminal network segment, a WAN/transport segment, and optionally a user-access segment in Singapore.)

4. **IP addressing:** Can static IP addresses or IP ranges be assigned for the QKD compute nodes at both sites?

5. **DNS:** Is internal DNS available, or should we use static IP addressing for the PoC?

### B2. Datacenter / Physical Space

6. **Rack space:** Is there available rack space or desk/shelf space at both the UK and Singapore datacenters for:
   - 1x desktop-sized compute node (UK: approximately 45cm x 20cm x 40cm, or mini-PC ~12cm x 12cm x 5cm)
   - 1x mini-PC (Singapore: ~12cm x 12cm x 5cm)
   - QKD ground terminal equipment (dimensions TBD from SpeQtral/RAL Space)

7. **Power:** Are standard AC power outlets available at both sites? (UK: 230V/13A BS 1363; Singapore: 230V/13A BS 1363 or Type G). Estimated power draw: UK server ~250W max, Singapore client ~50W max.

8. **Physical security:** Are the datacenter locations physically secured (access control, CCTV)? Any additional physical security requirements for the QKD equipment?

9. **Roof/antenna access:** Does the QKD satellite ground terminal require outdoor antenna placement? If so, is roof or outdoor access available at both datacenters? (Question for SpeQtral/RAL Space first, then confirm feasibility here.)

### B3. Security & Compliance

10. **Security approval:** Is there a security review or approval process for deploying new compute equipment and custom software in the datacenters? What documentation is required?

11. **Air-gap requirements:** The LLM server is designed to run with no internet access at runtime. Is the entire UK server node expected to be air-gapped (no internet at any time, including initial setup)? Or can it have internet access during provisioning and then be network-isolated for runtime?

12. **Data handling:** The PoC uses non-sensitive/public data. Are there still data handling policies that apply (e.g., data residency requirements for UK or Singapore)?

13. **Audit logging:** Are there requirements for centralized audit logging? The system generates local logs; should they be forwarded to a corporate SIEM?

14. **Penetration testing:** Will the PoC undergo any security testing or penetration testing? If so, when and by whom?

### B4. Procurement & Logistics

15. **Procurement process:** What is the fastest procurement channel for small IT equipment ($1,000-2,000 per site)? Can items be purchased directly (e.g., Amazon) or must they go through a corporate procurement system?

16. **Shipping to datacenters:** What is the process for shipping equipment to both datacenter locations? Any customs/import considerations for Singapore?

17. **Remote access:** After initial setup, will SSH access to the compute nodes be available remotely for ongoing maintenance? Or must all administration be done on-site?

18. **Existing equipment:** Is there any existing compute equipment at either datacenter that could be repurposed for this PoC? (e.g., unused servers, NUCs, desktops with sufficient specs — see Hardware Requirements document for minimum specifications.)

---

## Section C: Questions for Project Stakeholders

### C1. Demo & Success Criteria

1. **Demo format:** What is the intended demo format? (Live walkthrough? Recorded video? Self-service for stakeholders to try? All of the above?)

2. **Success criteria:** What specific outcomes define PoC success? Examples:
   - Successfully encrypt/decrypt LLM traffic using QKD keys
   - Demonstrate all productivity features (chat, email, translate, tone, summarize)
   - Achieve acceptable response latency (<X seconds)
   - Run continuously for X weeks without failure
   - Produce a technical report / whitepaper

3. **Stakeholder access:** Should all stakeholders (leadership, government customers, investors) see the same demo? Or are there different versions/emphasis for different audiences?

4. **Branding:** Should the application UI carry specific branding (company logo, project name, etc.)?

5. **Documentation deliverables:** Beyond the running system, what documents are expected? (Architecture document, security assessment, performance benchmarks, user guide, technical whitepaper?)

### C2. Timeline

6. **Target demo date:** Is there a specific event, meeting, or deadline driving the timeline?

7. **Phased delivery:** Is it acceptable to demonstrate with QKD simulator first (showing full application + encrypted tunnel with simulated keys) and then switch to real satellite QKD keys when ground terminals are ready?

8. **Development resources:** Who will be developing/deploying the software? (Internal team? External contractors? How many developers? Their experience with Docker, Python, React, LLM deployments?)

---

## Priority Summary

**Must answer before hardware procurement:**
- B1.1 (WAN connectivity)
- B2.6 (rack/desk space availability)
- B3.11 (air-gap requirements — affects setup process)
- B4.15 (procurement process)
- B4.18 (existing equipment)

**Must answer before software development begins:**
- A2.6-7 (API specification from SpeQtral)
- A2.9-10 (key format and ID synchronization)
- C1.2 (success criteria)
- C2.8 (development resources)

**Must answer before deployment:**
- A1.1-4 (ground terminal logistics)
- A3.11-12 (pass schedule and key rates)
- A5.19-20 (equipment delivery timeline)
- B1.2-4 (network configuration details)
- B2.9 (antenna access)

**Can be answered later (before pilot goes live):**
- A4.16-18 (certifications)
- B3.10-14 (security compliance details)
- C1.3-5 (demo format details)
