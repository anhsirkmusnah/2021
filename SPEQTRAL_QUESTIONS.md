# Questions for SpeQtral / RAL Space — Build Completion

**Context:** SpeQtral has confirmed they follow **ETSI GS QKD 014** but cannot provide simulators or documentation.
We need answers below to complete the integration without breaking on day-of-deployment.
**Network update:** Classical channel between UK and SG will be a **4G/5G router on each side** (public IPs likely behind CGNAT — confirm reachability strategy).

---

## 1. ETSI QKD 014 API — Implementation Specifics

ETSI 014 is a standard, but vendors implement different subsets. We need their exact behavior.

### 1.1 Endpoint & Transport
1. What is the **base URL format** of the KMS endpoint exposed by your ground terminal? (e.g., `https://<terminal-ip>:<port>/api/v1/keys/...`)
2. What **port** does the KMS listen on? (ETSI default examples vary: 443, 8443, 11400)
3. Is the API exposed over **HTTPS only**, or also plain HTTP for local network use?
4. Do you support both **GET** and **POST** patterns from the standard, or only one?
5. Is there an OpenAPI / Swagger spec available, even informal?

### 1.2 Key Request Endpoints — Exact Behavior
6. For `GET /api/v1/keys/{slave_SAE_ID}/status` — what is the **exact JSON response shape**? (We need to match against ETSI standard.)
7. For `GET /api/v1/keys/{slave_SAE_ID}/enc_keys?number=N&size=S` (master side):
   - Max `number` per request?
   - Allowed `size` values? (256-bit only? 128/256/512?)
   - Response shape for the `keys` array?
   - How is `key_ID` formatted? (UUID v4, hex, base64?)
   - How is the key material encoded? (base64, hex, raw bytes?)
8. For `POST /api/v1/keys/{master_SAE_ID}/dec_keys` (slave side):
   - How are `key_IDs` submitted in the request body?
   - What if a requested `key_ID` is not found, expired, or already consumed — error format?
   - Can we request multiple keys in one call, or one at a time?

### 1.3 SAE_ID Conventions
9. What format are SAE_IDs? (FQDN, UUID, custom string, integer?)
10. Are SAE_IDs assigned by SpeQtral, or do we choose them and register?
11. Is there a registration step required before the KMS will issue keys for our SAE_IDs?

### 1.4 Key Lifecycle
12. **Key TTL:** How long are keys retained in your KMS after generation? (Important for our pre-fetch strategy.)
13. After a key is consumed via `dec_keys`, is it immediately deleted, or marked-for-deletion with a grace period?
14. What happens if both UK and SG try to consume the same key_ID twice — error, or idempotent return?
15. Can we **reserve** a batch of keys ahead of time, or must we fetch on demand?

---

## 2. Authentication & Trust to the Terminal

16. How does our `qkd-kms` container **authenticate** to your terminal's API?
   - Mutual TLS (client certificate)?
   - API key in header?
   - HTTP Basic Auth?
   - IP allowlist only?
17. If mTLS — who issues the client certificate? Do you provide a CA cert, or do we generate a CSR for you to sign?
18. What is the terminal's TLS server certificate? Self-signed, internal CA, public CA? (Affects our pinning strategy.)
19. Are there per-SAE_ID credentials, or one credential per terminal?
20. Is there any rate-limiting on the API we need to respect?

---

## 3. Key Generation Rates & Satellite Pass Schedule

21. **Expected key rate** during a satellite overpass — kbps of usable key material? (After error correction and privacy amplification.)
22. **Pass duration** — typical overpass window in seconds?
23. **Pass cadence** — passes per day for UK and for Singapore? Are passes simultaneous (one overpass keys both ground stations) or independent?
24. **Total daily key budget** — typical 24-hour key pool we can plan around?
25. Is there a **schedule API** or notification we can subscribe to so our daemon knows when fresh keys arrived?
26. **Worst-case dry spell** — if weather/cloud blocks several passes, what's the longest gap we should plan for?
27. **Initial key pool** — at first activation, will the KMS already have keys, or do we wait for first pass?

---

## 4. Network & Connectivity (Critical with 4G/5G WAN)

### 4.1 Ground Terminal ↔ Our Server
28. What is the **physical interface** on the ground terminal? (Gigabit Ethernet, SFP, other?)
29. Does the terminal expect a **direct point-to-point** Ethernet connection to our server, or can it sit on a switched LAN?
30. **IP configuration** — does the terminal use DHCP, or does it have a fixed IP we must conform to?
31. What **subnet** does it expect? Can we choose `10.0.1.0/24` (our planned subnet)?
32. Are there **other ports/services** on the terminal we should expose/firewall? (Management UI, SNMP, syslog?)

### 4.2 Inter-Site Classical Channel (4G/5G Routers)
33. Does the terminal need its own **outbound internet** connection, or is everything local (only our `qkd-kms` reaches out)?
34. With 4G/5G routers behind CGNAT, neither side will have a public IP. **Are you OK with us using a small VPS as a relay** for the classical channel, or a Tailscale/ZeroTier/WireGuard mesh?
35. **Bandwidth needs** — what's the minimum sustained bandwidth required between the UK and SG terminals? (4G is typically 10-50 Mbps.)
36. **Latency tolerance** — does the QKD reconciliation/sifting between terminals need bounded RTT?
37. Does the **terminal-to-terminal classical traffic** flow through us (our `qkd-kms` relays it), or do the terminals talk to each other directly over the classical channel?

### 4.3 Firewall & Ports Required
38. Full list of **outbound ports** the terminal needs (TCP/UDP, destinations) — to other terminal? to a SpeQtral cloud service? to NTP?
39. Does the terminal phone home for updates, telemetry, or licensing?
40. Are there **time sync requirements** (NTP source, accuracy)?

---

## 5. Terminal Hardware & Site Requirements

41. **Form factor & weight** — what does the terminal look like physically? (Rack-mount, bench unit, outdoor mount?)
42. **Power requirements** — voltage, current, plug type (UK BS 1363 vs SG SS 145 vs IEC C13/C14)?
43. **Cooling/environment** — operating temp range, humidity, ventilation needs?
44. **Mounting** — does the optical head need roof access? Sky view requirements?
45. **Pointing/tracking** — is the satellite tracking automatic, or does it need calibration? Frequency of recalibration?
46. **Boot time** — from cold start to "ready to deliver keys"?
47. **Out-of-band management** — IPMI, console port, web UI for diagnostics?

---

## 6. Operational & Monitoring

48. What **health/status endpoint** does the terminal expose? Can we poll for "ready/degraded/down" state?
49. Are there metrics for: keys generated, keys consumed, current pool depth, last-pass timestamp, BER, QBER?
50. **Alarms / events** — does the terminal emit syslog, SNMP traps, webhook events on errors?
51. How do we tell programmatically that a satellite pass has just completed and new keys have landed?
52. What are the **failure modes** we need to handle in our daemon? (Terminal unreachable, no keys, key fetch timeout, malformed response?)
53. How do we **distinguish "no keys yet because no recent pass" vs "terminal broken"** from the API?

---

## 7. Failure Modes & Edge Cases

54. If the QKD ground link fails mid-pass, are partial keys still delivered, or discarded?
55. What happens if **only one of UK/SG terminals receives a pass** — do we have keys on one side that the other can't decrypt?
56. **Key drift** — how do you guarantee both terminals' KMS have the same keys with matching key_IDs?
57. If our `qkd-kms` consumes a key but never confirms, is the key permanently lost from the pool?
58. Is there a **manual key purge** API for testing/recovery?

---

## 8. Testing Without Simulator

59. Can you provide a **canned key dump** (a small CSV/JSON of pre-generated key_ID + key_material pairs from a real run) so we can hand-craft test fixtures matching your exact response format?
60. Can you give us **sample HTTP request/response captures** (curl + JSON) from a working installation?
61. Do you have a **staging/test terminal** at SpeQtral HQ we could hit remotely over VPN for integration testing **before shipping** the production terminals?
62. Can we get **read-only access to a customer's existing terminal** (under NDA) for one hour just to validate our client code?
63. Will you support a **paired "loopback" mode** at install where both terminals are co-located briefly so we can validate end-to-end before geographic separation?

---

## 9. Documentation Workarounds (Since None Available)

64. Can you have your engineering team do a **1-hour technical walkthrough call** (recorded) with us covering API, auth, and key lifecycle?
65. Can you share **scrubbed client implementation source code** from a previous integration (any language) as a reference?
66. Can you point us to **another customer / academic partner** who has integrated against your terminal, for a peer reference call?
67. Are there **public ETSI 014 reference implementations** you've validated against, that we could mirror?

---

## 10. Hardware Logistics & Install

68. **Lead time** to ship terminals to UK and SG?
69. **Install support** — do you ship engineers on-site, or is it self-install with remote support?
70. **Customs/import** — any special handling for satellite hardware import to UK and SG?
71. **Site survey** — do you need to survey our UK datacenter and SG mini-PC location before shipping?
72. **Decommissioning** — at end of pilot, do terminals return to SpeQtral, or are they ours?

---

## 11. Operational Support & SLA

73. **Support hours / SLA** during the pilot? (24x7, business hours, response time target?)
74. **Escalation contact** for terminal issues — phone, email, Slack/Teams?
75. **Firmware updates** — frequency, blackout windows, who pushes them?
76. **Spare/RMA** policy — if a terminal fails during demo prep, replacement turnaround?

---

## 12. Commercial / Compliance

77. **Per-key cost** — is there usage-based billing, or flat rental of the terminal?
78. **Export control** — any ITAR/EAR/dual-use restrictions on the QKD hardware affecting UK→SG transfer?
79. **Compliance certifications** — Common Criteria, FIPS 140-2/3, ETSI conformance test reports we can quote?
80. **NDAs/IP** — are there restrictions on what we can publish/demo about the integration?

---

## Priority Tiering for SpeQtral Response

| Tier | Questions | Why this tier |
|------|-----------|---------------|
| 🔴 **P0 — blocks build** | 1-15, 16-20, 28-32, 59-63 | Without these we cannot write the KMS client at all |
| 🟠 **P1 — blocks integration** | 21-27, 33-40, 48-58 | Without these we can't safely run the daemon in production |
| 🟡 **P2 — blocks deployment** | 41-47, 64-72 | Logistics and on-site readiness |
| 🟢 **P3 — risk management** | 73-80 | Important but workaroundable for the demo |

---

## Our Mitigation Plan While Waiting

While SpeQtral responds, we will:
1. **Build our existing simulator** to match the *strict ETSI 014 standard text* (not vendor-specific). When real responses arrive, we'll add an adapter shim if SpeQtral diverges from the spec.
2. **Mock the terminal** locally using a FastAPI service that mirrors the planned ETSI 014 endpoints — same simulator we have today, just renamed `terminal-mock`.
3. **Implement HTTPS + mTLS code paths early** so we can flip cert sources without rewrites.
4. **Design for graceful degradation** — if real terminals diverge from spec, the adapter layer absorbs the difference.

The biggest unknown that will hurt us late is the **exact JSON schema and authentication model** (Q1-Q20). Push hardest for these first.
