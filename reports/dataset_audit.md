# Dataset Audit: UNSW-NB15 vs. NSL-KDD

**Project:** Network Intrusion Triage  
**Access Date:** November 15, 2026  
**Source & Licence:** UNSW-NB15 public research benchmark dataset created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS), UNSW Canberra. Distributed for academic and non-commercial research use (Moustafa & Slay, 2015).

---

## 1. Benchmark Comparison: Why UNSW-NB15 Over NSL-KDD?

| Criterion | NSL-KDD (2009) | UNSW-NB15 (2015) — **Selected** |
| :--- | :--- | :--- |
| **Traffic Era** | Derived from 1998 DARPA/KDD99 | Modern synthetic + real normal/attack traffic (IXIA PerfectStorm) |
| **Attack Diversity** | 4 broad categories (DoS, Probe, R2L, U2R) | 9 modern categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) |
| **Feature Realism** | Legacy TCP/content features from 1990s | 34 modern flow, timing, jitter, TCP RTT, and connection-window features |
| **Suitability for SOC Triage** | Outdated traffic distribution | Much closer to contemporary enterprise/campus network telemetry |

---

## 2. Dataset Dimensions & Quality Audit (Observed on `.parquet` Files)

| Property | Training Split (`UNSW_NB15_training-set.parquet`) | Testing Split (`UNSW_NB15_testing-set.parquet`) |
| :--- | :---: | :---: |
| **Rows (Full)** | 175,341 | 82,332 |
| **Columns** | 36 (34 features + `attack_cat` + `label`) | 36 (34 features + `attack_cat` + `label`) |
| **Missing Values** | 0 | 0 |
| **Exact Duplicate Rows** | 78,519 | 32,361 |
| **Class Balance (`label`)** | `1` (Attack): 119,341 (68.06%) / `0` (Normal): 56,000 (31.94%) | `1` (Attack): 45,332 (55.06%) / `0` (Normal): 37,000 (44.94%) |

### Key Data Quality Findings
1. **No Missing Values:** Both splits contain `0` null entries across all 36 columns.
2. **Duplicate Rows:** Both splits contain substantial exact-row duplicates (78,519 in train; 32,361 in test) because repeated short flows (e.g., identical DNS/generic/DoS bursts) share identical aggregated flow statistics after IP/port identifiers are removed. We retain the natural flow distribution during stratified sampling (`20,000` train / `10,000` test in development; full dataset in final run) and document this as a known dataset characteristic.
3. **Class Distribution Shift:** Training is 68.1% attack while testing is 55.1% attack, which tests model generalization under shifted base rates.
4. **Target Leakage Guard:** `attack_cat` directly encodes the attack family (where `"Normal"` corresponds 1:1 with `label == 0`) and **must be dropped** from feature inputs.

---

## 3. Data Dictionary (All 36 Columns)

| # | Column | Type | Description |
| :---: | :--- | :--- | :--- |
| 1 | `dur` | `float32` | Total flow duration in seconds |
| 2 | `proto` | `category` | Transaction protocol (e.g., `tcp`, `udp`, `arp`) |
| 3 | `service` | `category` | Application service (`http`, `dns`, `ftp`, `smtp`, `-`, etc.) |
| 4 | `state` | `category` | Connection state (`FIN`, `INT`, `CON`, `REQ`, `RST`, etc.) |
| 5 | `spkts` | `int16` | Source-to-destination packet count |
| 6 | `dpkts` | `int16` | Destination-to-source packet count |
| 7 | `sbytes` | `int32` | Source-to-destination byte volume |
| 8 | `dbytes` | `int32` | Destination-to-source byte volume |
| 9 | `rate` | `float32` | Packets per second in flow |
| 10 | `sload` | `float32` | Source bits per second |
| 11 | `dload` | `float32` | Destination bits per second |
| 12 | `sloss` | `int16` | Source packets retransmitted or dropped |
| 13 | `dloss` | `int16` | Destination packets retransmitted or dropped |
| 14 | `sinpkt` | `float32` | Source inter-packet arrival time (ms) |
| 15 | `dinpkt` | `float32` | Destination inter-packet arrival time (ms) |
| 16 | `sjit` | `float32` | Source jitter (ms) |
| 17 | `djit` | `float32` | Destination jitter (ms) |
| 18 | `swin` | `int16` | Source TCP window advertisement value |
| 19 | `stcpb` | `int64` | Source TCP base sequence number |
| 20 | `dtcpb` | `int64` | Destination TCP base sequence number |
| 21 | `dwin` | `int16` | Destination TCP window advertisement value |
| 22 | `tcprtt` | `float32` | TCP connection setup round-trip time (`synack + ackdat`) |
| 23 | `synack` | `float32` | Time between SYN and SYN-ACK packets |
| 24 | `ackdat` | `float32` | Time between SYN-ACK and ACK packets |
| 25 | `smean` | `int16` | Mean packet size transmitted by source |
| 26 | `dmean` | `int16` | Mean packet size transmitted by destination |
| 27 | `trans_depth` | `int16` | HTTP pipelined request/response depth |
| 28 | `response_body_len` | `int32` | Uncompressed HTTP response body size |
| 29 | `ct_src_dport_ltm` | `int8` | Connections with same source IP and dest port in last 100 flows |
| 30 | `ct_dst_sport_ltm` | `int8` | Connections with same dest IP and source port in last 100 flows |
| 31 | `is_ftp_login` | `int8` | Binary indicator if FTP session used user/password |
| 32 | `ct_ftp_cmd` | `int8` | Number of flows with an FTP command |
| 33 | `ct_flw_http_mthd` | `int8` | Number of flows with HTTP methods (GET/POST) |
| 34 | `is_sm_ips_ports` | `int8` | Binary indicator if source and destination IP/port are equal |
| 35 | `attack_cat` | `category` | Attack family (`Normal`, `Generic`, `Exploits`, etc.) — **dropped to avoid leakage** |
| 36 | `label` | `int8` | **Target variable:** `0` = Normal, `1` = Suspicious/Attack |

---

## 4. Privacy, Bias & Limitations
- **Privacy:** This pre-processed Parquet release omits raw IP addresses (`srcip`, `dstip`), timestamps, and packet payloads, so no personally identifiable information (PII) is present.
- **Synthetic Bias:** Generated in a controlled cyber range (2015); traffic patterns do not reflect modern encrypted TLS 1.3 / QUIC traffic or post-2015 zero-day exploits.
- **Deployment Limitation:** Performance on UNSW-NB15 does not guarantee equivalent accuracy on a live campus network without local baseline calibration.
