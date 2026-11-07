# Hypotheses & Memory-Traffic Models

**สถานะ:** เป็น *candidate* ทั้งหมด ยังไม่ได้เลือก primary hypothesis ตาม Gate D เพราะยังไม่มีผลวัด baseline (เครื่องนี้ไม่มี NVIDIA GPU)
**กติกา:** ต้องเขียน hypothesis ก่อนลงมือ optimize ทุกครั้ง ตัวเลขในไฟล์นี้ทั้งหมดเป็น **ESTIMATE** จาก model ไม่ใช่ค่าวัด

Template ที่ใช้ (จาก CLAUDE.md):
`If we change X, then measured Y should improve on workload range Z, because data movement at tier T will change in way M; potential regressions are R.`

---

## Gate C — Traffic model ของ baseline B1 `copy_stride` (วัด smoke แล้ว 2026-11-04 ดู BASELINE_REPORT §10)

**Kernel:** `dst[i] = src[i*s]` ใช้ `uint32` โดยให้ `i < n_out = N/s` และ footprint ของ `src` ตายตัวที่ N elements (smoke: N = 16,777,216 = 64 MiB)

**สมมติฐานของ model** (ตรงกับ `benchmarks/scripts/bench_common.py`):
- A1: `src`/`dst` align กับ g (`cudaMalloc` ให้ 256 B)
- A2: flush L2 ก่อนทุก rep (ทำแบบ best effort ด้วยการอ่าน buffer ขนาด ≥ 4×L2)
- A3: chunk ขนาด g ที่มี element ที่ต้องใช้อยู่ จะถูกย้ายครั้งเดียวพอดี ไม่มี thrashing และไม่มี prefetch เกิน g
- A4: element ไม่คร่อม chunk
- A5: การเขียนเป็นแบบ contiguous และเต็ม chunk จึงไม่มี read-for-ownership

| ปริมาณ | สูตร | หมายเหตุ |
|---|---|---|
| Logical bytes | `8·N/s` | อ่าน 4 B + เขียน 4 B ต่อ element ที่ใช้ |
| Estimated read bytes (chunk g) | `min(4N, g·N/s)` | ถ้า `4s < g` ทุก chunk ใน buffer ถูกแตะ แต่ถ้า `4s ≥ g` จะแตะหนึ่ง chunk ต่อ element |
| Estimated write bytes | `4N/s` | contiguous |
| Amplification | `(read + write) / logical` | g=32: 1.0 (s=1), 1.5 (s=2), 2.5 (s=4), 4.5 (s≥8) |
| Arithmetic intensity | 0 FLOP/byte | คอขวดอื่นที่อาจแข่งกันคือ integer address math ซึ่ง SASS ใช้ 64-bit `IMAD.WIDE` และ `LEA` ประมาณ 10 คำสั่งต่อ element รวมถึง launch overhead และ memory-level parallelism |
| L1→L2 load sectors (32 B) | `min(N/8, N/s)` ซึ่งเท่ากับ N/8 เมื่อ s ≤ 8 และเท่ากับ 1 sector ต่อ element เมื่อ s ≥ 8 | ใช้เทียบกับ `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` ได้ โดยไม่ขึ้นกับ g ของ DRAM |

**ตารางทำนายสำหรับ smoke** (สร้างจาก `run_copy_stride.py --smoke --dry-run` เป็น ESTIMATE):

| stride | n_out | logical MiB | est MiB (g=32) | est MiB (g=64) | est MiB (g=128) |
|---:|---:|---:|---:|---:|---:|
| 1 | 16,777,216 | 128 | 128 | 128 | 128 |
| 2 | 8,388,608 | 64 | 96 | 96 | 96 |
| 8 | 2,097,152 | 16 | 72 | 72 | 72 |
| 16 | 1,048,576 | 8 | **36** | **68** | 68 |
| 32 | 524,288 | 4 | 18 | 34 | 66 |

**ทำไมแต่ละแบบอาจเร็วขึ้นหรือช้าลง (กลไกที่คาดไว้):**
- `copy_contig_u32` เทียบกับ `copy_contig_u32x4`: ทั้งสองแบบย้าย byte เท่ากัน แต่ u32x4 มี load ต่อ warp ขนาด 512 B ส่วน scalar มีแค่ 128 B
  ตาม Little's law (bytes in flight = bandwidth × latency) ถ้า bytes in flight ของ scalar ไม่พอจะปิด latency ของ DRAM แบบ scalar จะ**ช้ากว่า**แม้ byte เท่ากัน
  นี่คือตัวอย่างที่คอขวดเป็น memory-level parallelism ไม่ใช่ปริมาณ traffic
  ถ้าแบบ scalar วิ่งได้เท่ากับ `cudaMemcpy` แปลว่า occupancy อย่างเดียวก็ซ่อน latency ได้พอ
- `copy_strided_read_u32_s1` เทียบกับ `copy_contig_u32`: ควรได้ผลเท่ากันภายในระดับ noise เพราะต่างกันแค่การคูณ index ข้อนี้ใช้เป็น **sanity check** ของ harness
- stride ตั้งแต่ 2 ถึง g/4: ข้อมูลที่ใช้ลดลงแต่ DRAM ยังต้องอ่านทั้ง buffer เวลาจึงควร**แทบไม่ลด** (ลดลงเฉพาะส่วนที่เขียน)
- stride > g/4: แต่ละ element ใช้ chunk ของตัวเอง bytes จึงลดลงแบบ 1/s
  แต่ access ที่กระจายอาจทำให้ DRAM row-buffer locality แย่ลง เวลาจริงจึงอาจ**ช้ากว่า model** ทั้งที่ byte ตรงกับที่ทำนาย ต้องใช้ counter แยกสองกรณีนี้ออกจากกัน

---

---

## Gate C — Traffic models และคำทำนายของ B2–B4 (เขียนไว้ **ก่อน** วัด: 2026-11-04)

> **ผลวัดเทียบกับคำทำนาย (เพิ่มภายหลังวันเดียวกัน ไม่ได้แก้ข้อความคำทำนายด้านล่าง):** ดูรายละเอียดใน `reports/BASELINE_REPORT.md` §12
> - **ถูก:** P-B2-1..4, P-B3-1, P-B3-2, P-B4-1, P-B4-2 (สนับสนุน `lru_cliff` และตัด `proportional` ออก), P-B4-3, P-B4-4, P-B4-6
> - **ถูกบางส่วน:** P-B4-5 (ratio ลดเหลือ 1.42 แต่ไม่ถึง 1)
> - **ผิด:** P-B3-3 ส่วนที่บอกว่าเวลาจะต่ำกว่า no_reuse ÷ memcpy BW เพราะ random access ได้ BW ต่ำกว่า streaming มาก และ P-B3-4 (zipf scatter_add **เร็วกว่า** uniform)
> - **ไม่ได้ทำนายไว้:** dirty write-back ทำให้ consumer ช้าลง 21 % ที่ l2x = 1 และในโหมด steady ต้องเทียบ working set รวมกับ L2

ทุกตัวเลขในส่วนนี้เป็น **ESTIMATE** ใช้ g = 32 B ตามพฤติกรรมที่เห็นใน B1 บน RTX 3060
implement ไว้ใน `bench_common.py` (`model_transpose`, `model_gather_*`, `model_fusion_bytes`) และมี unit test ใน `test_models_b2_b4.py`
ผลวัดที่ออกมาภายหลังห้ามย้อนกลับมาแก้คำทำนายในส่วนนี้ ถ้าคำทำนายผิดให้บันทึกว่าผิด

### B2 transpose (`transpose.cu`): R×C `uint32`, N = R·C

| ปริมาณ | copy_2d | naive | tiled | tiled_pad |
|---|---|---|---|---|
| Logical bytes | 8N | 8N | 8N | 8N |
| Est. DRAM bytes (สมมติฐาน T2: L2 รวม partial sector ภายใน block ได้) | 8N | 8N | 8N | 8N |
| Est. L1→L2 store sectors | N/8 | **N** (1 sector ต่อ element) | N/8 | N/8 |
| Shared-memory bank conflicts | — | — | **32-way** ตอนอ่าน column | ไม่มี |

- **P-B2-1:** `tiled_pad` จะได้ ≥ 90 % ของ `copy_2d` ที่ 4096² Harris 2013 รายงานไว้ ≈ 95 % บน Kepler/Fermi
- **P-B2-2:** `naive` ช้ากว่า `copy_2d` อย่างชัดเจน **ทั้งที่ DRAM bytes เท่ากัน** กลไกที่คาดคือ L2 ต้องรับ store transaction มากขึ้น 8 เท่า ไม่ใช่ DRAM traffic เพิ่มขึ้น
  ถ้ามี counter จะเห็น `lts__t_sectors_op_write` สูงขึ้นราว 8× แต่ `dram__bytes_write` ใกล้เดิม **ถ้าไม่มี counter จะยืนยันกลไกนี้ไม่ได้ ได้แค่ timing**
- **P-B2-3:** `tiled` (ไม่มี pad) อยู่ระหว่าง `naive` กับ `tiled_pad` เพราะ bank conflict เพิ่มเวลาฝั่ง shared memory ไม่ใช่ DRAM
- **P-B2-4:** `cublas_geam` (ถ้า build ได้) ≈ `tiled_pad` ±10 %
- shape 1000×777 มีไว้ตรวจ boundary เป็นหลัก ขนาดเล็กมากจน launch overhead (≈ 3 µs) มีผลมาก

### B3 gather / scatter_add (`gather.cu`): T = 16M elements (64 MiB), M = 4M lookups

ไบนารีคำนวณสถิติจริงของ index stream ไว้ใน `meta.json`: distinct sectors, สัดส่วน lookup ที่ตกใน K = L2/32 sector ที่ร้อนที่สุด และจำนวนครั้งของ element ที่ร้อนที่สุด จากนั้น model ให้ค่า 3 ระดับ

| ค่า | ความหมาย |
|---|---|
| `compulsory` = 32 B × distinct sectors | **lower bound** ทุก sector ที่ถูกแตะต้องถูกดึงอย่างน้อย 1 ครั้ง |
| `hot_set` = 32 B × (M·(1 − share) + K) | point estimate โดยสมมติว่า L2 เก็บ K sector ที่ร้อนที่สุดไว้ได้พอดี (ideal LFU) |
| `no_reuse` = 32 B × M | **upper bound** ไม่มี L2 hit เลย |

- **P-B3-1:** `seq` gather ได้ useful GB/s ใกล้ memcpy (≥ 85 %)
- **P-B3-2:** `uniform` gather มี DRAM traffic ใกล้ `no_reuse` เพราะ L2 คิดเป็นแค่ ≈ 3.5 % ของ table useful GB/s จะตกลงเหลือ **≤ 0.3 เท่า** ของ memcpy (12 B ที่ต้องใช้ต่อ lookup จาก 4+32+4 = 40 B ที่ย้ายจริง) และอาจต่ำกว่านี้ถ้า random access ทำให้ DRAM efficiency ลดลง
- **P-B3-3:** `zipf1.0` gather เร็วกว่า `uniform` เพราะ hot set อยู่ใน L2 เวลาควรอยู่ระหว่าง time(`hot_set`) กับ time(`no_reuse`) เมื่อหารด้วย memcpy BW
- **P-B3-4 (competing bottleneck):** `scatter_add` กับ `zipf1.0` **อาจช้ากว่า** `uniform` แม้ DRAM bytes น้อยกว่า เพราะ atomic ที่ลง address เดียวกันถูก serialize ที่ L2
  `max_element_count` คือจำนวน atomic ที่ต้องเรียงคิวบน address ร้อนที่สุด ยังไม่รู้ throughput ต่อ address ของ hardware จึง**ไม่ทำนายตัวเลข** ทำนายเพียงทิศทาง
- **Falsifier:** ถ้า `compulsory` bytes ÷ เวลาที่วัดได้ เกิน DRAM peak (summary จะมีคอลัมน์ `compulsory_exceeds_peak`) แปลว่าสมมติฐาน 32-B sector ใช้กับ gather ไม่ได้

### B4 fusion (`fusion.cu`): B = 4n bytes ต่อ array, ขนาดเป็นเท่าของ L2 (l2x)

| ขนาด (l2x) | L2-tier bytes fused / mat | DRAM bytes fused | DRAM bytes mat ตาม `lru_cliff` | DRAM bytes mat ตาม `proportional` |
|---|---|---|---|---|
| 0.25 | 2B / 4B | 2B | 2B (t อยู่ใน L2 ได้) | 2B |
| 0.5 | 2B / 4B | 2B | 2B (2B = L2 พอดี จึงอยู่ที่ขอบ) | 2B |
| 1 | 2B / 4B | 2B | **4B** | **3B** |
| 4 / 16 | 2B / 4B | 2B | 4B | ≈ 3.9B |

- **P-B4-1 (กลไกหลักของ H2):** `stage2_warm / stage2_cold` < 1 อย่างชัดเจนที่ l2x = 0.25 (t ยังอยู่ใน L2) และ ≈ 1 ที่ l2x ≥ 4 ส่วน l2x = 0.5 เป็นจุดขอบที่ทั้งสอง model ยังทำนายไม่ชัด
  นี่คือ**หลักฐานเรื่อง tier ที่ได้จาก timing** โดยไม่ต้องใช้ counter
- **P-B4-2 (ชี้ขาดระหว่างสอง model):** ที่ l2x = 1 ถ้า `mat/fused` ≈ 2.0 จะสนับสนุน `lru_cliff` ถ้า ≈ 1.5 จะสนับสนุน `proportional` ถ้าเป็นค่าอื่นแปลว่าทั้งสอง model ผิด
- **P-B4-3:** ที่ l2x ≥ 4 และ K = 1 (DRAM-bound) `mat/fused` → 2.0 ± 0.15
- **P-B4-4:** ที่ l2x ≤ 0.5 (t อยู่ใน L2) `mat/fused` จะ **> 1 อยู่ดี** เพราะ (a) launch 2 ครั้งเทียบกับ 1 ครั้ง ซึ่ง B1 วัด floor ได้ ≈ 3 µs และ kernel ที่ขนาดนี้ใช้เวลาเพียงไม่กี่ µs (b) traffic ที่ L2 tier เป็น 2 เท่า
  เส้นโค้ง `mat/fused` เทียบกับขนาดจึง**อาจไม่ monotonic** คือสูงที่ขนาดเล็กเพราะ launch, ลดลงช่วงที่ทุกอย่างอยู่ใน L2 และขึ้นไปหา 2 ที่ขนาดใหญ่
  ข้อนี้อาจอธิบายข้อสังเกตของ Dodović 2026 ที่ว่า speedup ลดลงเมื่อ array ใหญ่ขึ้น ถ้าช่วงขนาดที่เขาวัดเริ่มจากช่วงที่ launch overhead เด่น **นี่เป็นการตีความ ยังไม่ได้ตรวจ**
- **P-B4-5:** ที่ K = 64 (AI_fused = 32 FLOP/B ขณะที่ ridge ที่ประเมินไว้ ≈ 38 FLOP/B) `mat/fused` จะลดลงเข้าหา 1 เพราะงานเริ่มติด compute bytes จึงไม่ใช่คอขวดอีกต่อไป
  นี่คือกรณีที่ fusion **ไม่ช่วย** แม้จะลด bytes ได้
- **P-B4-6:** ในโหมด `steady` ที่ขนาดเล็ก dirty lines ของ t และ y ถูกเขียนทับซ้ำใน L2 และ**อาจไม่ถูก write-back ลง DRAM เลย** ส่วนในโหมด `isolated` write-back ถูกเลื่อนไปอยู่นอกช่วงจับเวลา ทั้งสองโหมดจึงอาจให้ `mat/fused` ต่างกัน ความต่างนี้คือ "deferred write-back" ที่ H2 พูดถึง

---

---

## Gate C — B4 model v2 และ B3 fetch-granularity probe (pre-registered 2026-11-04 06:16 UTC ก่อนวัด)

> **ผล (เพิ่มภายหลัง ไม่ได้แก้คำทำนายด้านล่าง; ดู `reports/BASELINE_REPORT.md` §14):**
> - **P-B4v2-1:** ✅ v2 ไม่ถูก falsify LRU ได้ MAPE 4.2 % ส่วน random 6.4 % LRU จึงชนะ
> - **P-B4v2-2:** ✅ warm/cold สูงสุดที่ 0.625 แล้วลดลงตามรูปร่างของ LRU
> - **P-B4v2-3:** ✅ steady ≈ 2.0 ตามที่ LRU ทำนาย
> - **พลาด:** mat/fused iso ที่ 0.625 (LRU สูงไป 12 %) และช่วงต่ำกว่าหน้าผาตามข้อจำกัดที่ประกาศไว้
> - **P-B3g-1:** ✅
> - **P-B3g-2:** ❌ ผลไม่ตรงกับสามสถานการณ์ที่เขียนไว้ ค่าตั้งต้นทำงานเหมือน 32 แต่รายงานว่า 64 ส่วนค่าที่ตั้งเองถูกทำตาม

### B4 model v2: cache simulator + time model

**ทำไมต้องมี v2:** v1 (สูตรปิด `lru_cliff` / `proportional`) มองข้ามสองกลไกที่ smoke รอบที่ 2 แสดงให้เห็น ได้แก่ (1) ต้นทุน dirty write-back ไปตกที่ kernel ผู้บริโภค และ (2) ในโหมด steady ต้องเทียบ working set ทั้งหมดของแต่ละแบบกับ L2
**ข้อควรระวัง:** v2 ถูกออกแบบ*หลัง*เห็นผล smoke ขนาด l2x ∈ {0.25, 0.5, 1, 4, 16} ขนาดเหล่านี้จึงเป็น **in-sample** การทดสอบที่ใช้ได้จริงคือขนาดที่ยังไม่เคยวัด (**out-of-sample**) เท่านั้น

| ส่วน | รายละเอียด (`cache_sim.py`, `fusion_model_v2.py`) |
|---|---|
| Cache | fully-associative ขนาด = L2 attribute, line 128 B, write-allocate ที่ไม่ fetch, write-back ตอน evict |
| Replacement | **สองสมมติฐานที่แข่งกัน: `lru` กับ `random`** |
| ลำดับ access | stream ตาม index: สำหรับทุก line j ให้อ่าน src[j] แล้วเขียน dst[j] |
| Isolated | เริ่มจาก cache ว่าง (ผลของ read-flush) และนับเฉพาะ miss กับ dirty eviction ที่เกิด*ในช่วงจับเวลา* |
| Steady | รัน call sequence ซ้ำ แล้วรายงานรอบที่ 3 |
| Time model | t = k·τ + DRAM bytes / BW; BW = 333.4 GB/s (memcpy ที่วัดได้); τ_iso = 3.53 µs, τ_steady = 2.63 µs **calibrate จากจุด l2x = 4 เท่านั้น** |
| ข้อจำกัดที่รู้อยู่แล้ว | **ไม่มี term ของ L2 bandwidth** จึงทำนายต่ำกว่าจริงเมื่อ DRAM traffic น้อย (ขนาดเล็ก หรือ steady) ไม่คิด set conflict และ L2 slicing |

**In-sample (เพื่อความโปร่งใส ไม่ใช่การทดสอบ):**
- ที่ l2x ≥ 1, LRU คลาดเคลื่อน ≤ 5 % ทุก ratio (เช่น warm/cold ที่ l2x 1: วัดได้ 1.214 ทำนาย 1.250)
- ที่ l2x ≤ 0.5 คลาดสูงถึง 69 % ตามข้อจำกัดเรื่อง L2 bandwidth

**คำทำนาย (ESTIMATE จาก `research/preregistration/B4_v2_predictions.json`, sha256 ของ `cache_sim.py` ขึ้นต้นด้วย `ea8ad8ef`):**

| l2x | ชนิด | LRU mat/fused iso | random mat/fused iso | LRU warm/cold | random warm/cold | LRU mat/fused steady | random mat/fused steady |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.375 | OUT (ต่ำกว่าเกณฑ์) | 1.57 | 1.68 | 0.57 | 0.68 | 4.02 | 2.89 |
| **0.625** | **OUT** | **2.40** | **2.06** | **1.40** | **1.06** | 2.00 | 3.20 |
| **0.75** | **OUT** | **2.33** | **2.10** | **1.33** | **1.10** | 2.00 | 2.67 |
| **0.875** | **OUT** | **2.29** | **2.12** | **1.29** | **1.12** | 2.00 | 2.44 |
| 1.25 | OUT | 2.20 | 2.14 | 1.20 | 1.14 | 2.00 | 2.15 |
| 1.5 | OUT | 2.17 | 2.13 | 1.17 | 1.13 | 2.00 | 2.08 |
| 2 | OUT | 2.13 | 2.11 | 1.13 | 1.11 | 2.00 | 2.03 |

- **P-B4v2-1 (กฎตัดสินหลัก):** ใช้ MAPE ของ `mat/fused iso` และ `warm/cold` ที่ขนาด out-of-sample ที่ l2x ≥ 0.625 policy ที่ MAPE ต่ำกว่าถือว่าชนะ ถ้า**ทั้งสอง policy มี MAPE > 10 % ให้ถือว่า model v2 ถูก falsify** บน GPU นี้
- **P-B4v2-2 (รูปร่าง):** ถ้า L2 ทำงานคล้าย LRU, warm/cold จะมี**ค่าสูงสุดอยู่ใกล้ l2x 0.625** (≈ 1.4) แล้วลดลงเรื่อย ๆ ถ้าคล้าย random จะขึ้นอย่างค่อยเป็นค่อยไปและแตะสูงสุดราว 1.14 ที่ l2x 1.25
- **P-B4v2-3 (steady เป็นตัวรอง ไม่ใช้ตัดสิน):** LRU ทำนายว่า steady = 2.00 คงที่ตั้งแต่ l2x 0.625 ขึ้นไป ส่วน random ทำนายว่าจะลดจาก 3.2 ลงมา
- **วิธี score:** `python3 benchmarks/scripts/fusion_model_v2.py --score <run> --prereg research/preregistration/B4_v2_predictions.json` สำหรับ sweep 3 รอบใน `remote_followup.sh`

### B3 fetch-granularity probe

`gather_pair64` อ่านทั้ง `table[j]` และ `table[j^8]` ซึ่งเป็น sector 32 B อีกตัวใน chunk 64 B เดียวกัน ส่วน `gather_pair128` อ่าน `table[j]` กับ `table[j^16]` ซึ่งเป็นครึ่ง 64 B อีกครึ่งใน line 128 B เดียวกัน

| ถ้า DRAM ย้ายข้อมูลทีละ g ต่อ miss (ไม่มี reuse) | pair64 / gather | pair128 / gather |
|---|---:|---:|
| g = 32 B | 1.80 | 1.80 |
| g = 64 B | 1.00 | 1.89 |
| g = 128 B | 1.00 | 1.00 |

- **P-B3g-1:** สำหรับ `uniform` ใช้เกณฑ์ตัดสิน ratio ของ pair64 ดังนี้
  - ≤ 1.15 แปลว่า sector ข้างเคียง "ได้มาฟรี" (ย้ายข้อมูลจริง ≥ 64 B)
  - ≥ 1.5 แปลว่าดึงแยกทีละ 32 B
  - **ช่วง 1.15–1.5 ถือว่า ambiguous** เพราะการดึงทีละ 32 B ที่ได้ row-buffer hit ใน DRAM ก็ทำให้ ratio ต่ำกว่า 1.8 ได้
  - pair128 ใช้แยก g = 64 ออกจาก 128 ด้วยเกณฑ์เดียวกัน
- **P-B3g-2 (ablation ของ `cudaLimitMaxL2FetchGranularity` ซึ่งเป็นเพียง hint):** มีผลที่เป็นไปได้สามแบบ
  - ถ้า driver ไม่สนใจค่านี้ เวลาของ `gather` แบบ uniform ที่ตั้ง 32/64/128 จะต่างกันไม่เกิน 3 %
  - ถ้า driver ทำตาม และค่า default คือ 64 จริง การตั้ง 32 จะทำให้ `gather` แบบ uniform **เร็วขึ้น** (ถึงขั้นต่ำสุด (8+32)/(8+64) = 0.56×) และ pair64 ratio จะเปลี่ยนจาก ≈ 1.0 เป็น ≈ 1.8
  - ถ้าค่าที่ตั้งไม่เปลี่ยนเวลาเลย แต่ pair64 ≈ 1.0 แปลว่า hardware ดึง 64 B เสมอไม่ว่าจะตั้ง hint อย่างไร
- **ความสำคัญ:** ข้อนี้ตอบคำถามเปิดจาก smoke ว่า uniform gather ช้าเพราะ DRAM ดึงทีละ 32 B แต่ efficiency ต่ำ (135 GB/s) หรือเพราะดึงทีละ 64 B (244 GB/s)
  และยังบอกด้วยว่าค่า g ที่ H1 วัดได้ (32 B สำหรับ stride) ใช้กับ random access ได้หรือไม่

---

---

## RTX 4060 pre-registration (H2 generality test) — เขียน 2026-11-04 ก่อนแตะ RTX 4060 ใด ๆ

> **ผล (เพิ่มภายหลัง ไม่ได้แก้ข้อความด้านล่าง; ดู `reports/BASELINE_REPORT.md` §15):**
> - **F1 ✅ F2 ✅ F3 ✅ F4 ✅ D1 ✅ D2 ✅** ผ่านทั้งเมื่อใช้ median และในทุกรอบ
> - หน้าผาย้ายจาก 1.4 MiB (3060) ไปเป็น **15 MiB (4060) ซึ่งตรงกับ l2x 0.625 เดิมพอดี**
> - LRU MAPE 2.3 % ส่วน random 13.8 % ตกเกณฑ์
> - H1 รอง: T(16)/T(8) = 0.5015 แปลว่าค่าตั้งต้นทำงานแบบ 32 B

**คำถาม:** หน้าผาของ fusion payoff ที่เห็นบน RTX 3060 (L2 2.25 MiB) ผูกกับ**ขนาด L2** อย่างที่ H2 อ้างหรือไม่ หรือเป็นเพียงคุณสมบัติของ GPU ตัวนั้น RTX 4060 (Ada, cc 8.9) มี L2 ใหญ่กว่า ≈ 10.7 เท่า จึงเป็นการทดสอบที่ falsify H2 ได้ตรงที่สุดในงบที่มี

**ข้อมูลของ RTX 4060 ที่ใช้เป็นสมมติฐาน (ใช้กับตัวเลขคร่าว ๆ เท่านั้น):**
- spec สาธารณะ: L2 24 MB, GDDR6 128-bit ที่ 17 Gbps = 272 GB/s, 24 SMs (จาก review หลายแหล่ง เช่น Tom's Hardware และ TechSpot ยังไม่ได้ตรวจกับเครื่องจริง)
- vast.ai แสดง memory BW ของ offer 4060 ที่ 240–247 GB/s, VRAM 8 GB, PCIe 3.0 x8
- **ตอนให้คะแนนต้องใช้ค่าที่วัดได้เท่านั้น** ได้แก่ L2 จาก `cudaDevAttrL2CacheSize`, BW จาก memcpy ในรอบเดียวกัน และ τ จาก l2x = 4

### Protocol ที่ผูกมัด (ห้ามเปลี่ยนหลังเห็นผล)

script `benchmarks/scripts/remote_rtx4060.sh` (sha256 `bd858d58…`) และไฟล์ที่ล็อกไว้: `cache_sim.py` `ea8ad8ef…`, `fusion_model_v2.py` `e0b2df3f…`, `fusion.cu` `f4af4694…`, `common.cuh` `7a5ea8b8…`

1. build ด้วย `-arch=sm_89`
2. รัน B1 smoke ที่ footprint 256 MiB (≥ 10× L2) เพื่อเอา memcpy BW สำหรับ calibration
3. รัน B4 sweep 3 รอบอิสระ, K = 1, 30 reps ที่ l2x ∈ {0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 2, **4**} โดย 4 เป็นจุด calibrate
4. calibrate τ จาก **รอบที่ 1 ที่ l2x = 4 เท่านั้น** แล้วสร้างคำทำนายใหม่แบบ deterministic:
   `fusion_model_v2.py --calib-run <r1> --bw-run <B1> --calib-l2x 4 --in-sample-l2x 4 --sizes-l2x <ชุดเดียวกับ sweep> --prereg research/preregistration/B4_v2_rtx4060_calibrated.json`
5. ให้คะแนนด้วย `fusion_model_v2.py --score <r1> <r2> <r3> --prereg …calibrated.json` ใช้ median ของ 3 รอบ
6. B3 probe (default และ limit 64) เป็น **exploratory ไม่นำมาให้คะแนน**

### ตัวชี้ขาด (falsifiers) ของ H2 บน RTX 4060

| ID | คำทำนาย | H2 ถูก falsify ถ้า |
|---|---|---|
| **F1 หน้าผาเลื่อนตาม L2** | warm/cold (median 3 รอบ) **สูงสุดที่ l2x ∈ [0.5, 0.875] และมีค่า > 1.2** (3060: สูงสุด 1.35 ที่ 0.625) | ค่าสูงสุดอยู่ที่ l2x ≤ 0.125 (ขนาดเดียวกับหน้าผาของ 3060 ≈ 1.4–3 MiB) **หรือ**ไม่มีจุดไหนเกิน 1.1 |
| **F2 control ขนาดเท่ากัน** | ที่ l2x 0.0625 และ 0.125 (≈ 1.5 และ 3 MiB ซึ่งเป็นขนาดที่ 3060 *เกิด*หน้าผา) warm/cold **< 1.0** | warm/cold ≥ 1.1 ที่สองจุดนี้ |
| **F3 เชิงปริมาณ** | model v2 หลัง calibrate มี MAPE ของ ratio หลักที่ out-of-sample (l2x ≥ 0.625 ยกเว้น 4) **≤ 10 %** อย่างน้อยหนึ่ง policy | **ทั้งสอง policy > 10 %** แปลว่า v2 ใช้กับ Ada ไม่ได้ (H2 เชิงคุณภาพอาจยังรอด แต่ model ล้ม) |
| **F4 DRAM-bound** | mat/fused iso ที่ l2x 2 อยู่ใน **[1.9, 2.3]** | อยู่นอกช่วงนี้ |

**ไม่ได้ทำนายล่วงหน้าว่า policy ไหนจะชนะบน Ada** เพราะ L2 ของ Ada ใหญ่กว่าและอาจใช้ partition หรือ policy ต่างจาก Ampere ผลที่ได้ (LRU / random / ไม่ใช่ทั้งคู่) จะบันทึกตามจริง

### ตัวเลขคร่าว ๆ (INDICATIVE ไม่ใช่ตัวที่ใช้ให้คะแนน: `research/preregistration/B4_v2_rtx4060_indicative.json`, sha256 `5d876c3a…`)

สมมติ L2 = 24 MiB, BW = 245 GB/s, τ เท่ากับของ 3060 (3.53 / 2.63 µs) kernel บน 4060 ใหญ่กว่า 3060 ราว 10 เท่าที่ l2x เดียวกัน ผลของ τ จึงเจือจางลง และ ratio ใกล้ค่า byte ratio ล้วนมากขึ้น

| l2x | ≈ MiB/array | LRU mat/fused iso | random | **LRU warm/cold** | random | LRU steady | random steady |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.625 | 15 | 2.64 | 2.09 | **1.64** | 1.09 | 2.00 | 4.07 |
| 0.75 | 18 | 2.48 | 2.14 | 1.48 | 1.14 | 2.00 | 2.94 |
| 0.875 | 21 | 2.39 | 2.16 | 1.39 | 1.16 | 2.00 | 2.55 |
| 1 | 24 | 2.33 | 2.17 | 1.33 | 1.17 | 2.00 | 2.35 |
| 1.25 | 30 | 2.25 | 2.17 | 1.25 | 1.17 | 2.00 | 2.17 |
| 1.5 | 36 | 2.20 | 2.15 | 1.20 | 1.15 | 2.00 | 2.09 |
| 2 | 48 | 2.14 | 2.13 | 1.14 | 1.13 | 2.00 | 2.03 |

- **คำทำนายเชิงทิศทาง (D1):** ถ้า L2 ของ 4060 ทำงานคล้าย LRU ค่าสูงสุดของ warm/cold จะ**สูงกว่า 3060** (≈ 1.6 เทียบกับ 1.35) เพราะ τ เจือจางน้อยลง ถ้าคล้าย random ค่าสูงสุดจะอยู่ราว 1.1–1.2
- **ช่วงต่ำกว่าหน้าผา (l2x ≤ 0.5): ไม่ทำนายเป็นตัวเลข** model v2 ไม่มี term ของ L2 bandwidth ตัวเลขดิบที่ได้ (warm/cold 0.06–0.36, steady สูงถึง 60) จึงใช้ไม่ได้
  ทำนายเพียงทิศทาง (D2) ว่า warm/cold < 0.9 ที่ l2x 0.125–0.375
  จุดเหล่านี้ใช้**วัด L2 bandwidth ของ Ada** แทน ได้แก่ fused steady ที่ l2x ≤ 0.375 ซึ่ง working set อยู่ใน L2 ทั้งหมด
- **H1 รอง (B1, 256 MiB):** ถ้าค่าตั้งต้นของ Ada ทำงานแบบ 32 B เหมือน 3060 T(16)/T(8) จะได้ ≈ 0.50 ถ้าแบบ 64 B จะได้ ≈ 0.94 **ไม่มี prior ว่าแบบไหน**

---

---

## B5 pre-registration: workload ตระกูลที่สอง (multi-sweep 2D stencil) — เขียน 2026-11-04 ก่อนวัด

> **ผล (เพิ่มภายหลัง ไม่ได้แก้ข้อความด้านล่าง; ดู `reports/BASELINE_REPORT.md` §17):**
> - **RTX 3060:** G1 ✅ G3 ❌ G4 ✅ G5 ❌ D-B5-1 ✅
> - **RTX 4060:** G1 ✅ G2 ✅ G3 ✅ (9.7 % เฉียดเกณฑ์) G4 ✅ G5 ✅ (เฉียดเกณฑ์) D-B5-1 ✅
> - **ข้อสรุป:** กลไก L2 residency และ write-back (warm/cold) **ใช้ข้ามตระกูล workload ได้** บนทั้งสอง GPU (error ≤ 7.5 % และ G4 ผ่าน)
> - **ส่วนที่ล้มคือ fusion payoff ของ B5 บน 3060** เพราะ kernel 13 จุดไม่ได้ติด DRAM time model (τ คงที่ + BW ของ memcpy) จึงใช้ไม่ได้เมื่อ kernel ที่ fuse แล้วติดคอขวดอื่น

**คำถาม (RQ5):** model ที่อธิบาย fusion payoff ของ pointwise pipeline (B4) ใช้ได้โดย**ไม่ต้องแก้**กับ workload อีกตระกูลหรือไม่

**B5 (`stencil.cu`):**
- กริด H × 1024 `uint32` แบบ torus, stencil 5 จุดที่ใช้ coefficient ไม่สมมาตร (3, 5, 7, 11, 13), รัน S = 4 sweep
- **mat4** = 4 kernel แบบ ping-pong X→P→Q→P→Q
- **fused2** = 2 kernel ของ stencil 13 จุด (คือ self-convolution ของ stencil 5 จุด)
- ผลตรงกันทุก bit เพราะเป็นเลขจำนวนเต็ม mod 2³² บนกริด periodic ตรวจด้วย host reference และ `test_host.cpp` ยืนยันแล้วบน CPU รวมถึงกริดขนาด 1×4 และ 3×3
- `sweep_warm` / `sweep_cold` คือ sweep เดียว (P→Q) ที่รันหลัง X→P ที่ไม่ได้จับเวลา เทียบกับรันหลัง flush

**ต่างจาก B4 อย่างไร:**
- stencil 2D มีการ reuse ระหว่างแถว (ใช้ 3 และ 5 แถว)
- kernel 13 จุดมีงานคำนวณมากกว่า (38 registers, 13 load ต่อ output)
- ping-pong **นำ buffer กลับมาใช้ซ้ำ** ทำให้ working set หลัง sweep แรกเหลือ 2B
- มีมากกว่า 2 stage

**เหมือน B4 อย่างไร:** ที่ระดับ line ขนาด 128 B แต่ละ sweep ยังเป็น streaming ตามสมมติฐาน T5 ใน `cache_sim_b5.py` ดังนั้น **model ไม่ได้ถูกแก้เลย** ใช้ `cache_sim.py` ตัวที่ล็อกไว้ (sha256 `ea8ad8ef…`) และเพิ่มเพียง schedule ของ B5

**ข้อจำกัดของการทดสอบนี้ (ซื่อตรงไว้ก่อน):**
- B5 กับ B4 มีรูปแบบการเข้าถึงที่ระดับ line คล้ายกัน การ "ข้ามตระกูล" ครั้งนี้จึงเป็น**ระดับกลาง** ไม่ใช่การทดสอบที่ access pattern ต่างกันมาก เช่น gather
- warm/cold ของ B5 ถูกทำนายให้**เท่ากับ B4 ทุกค่า** (ตรวจไว้ใน `test_b5.py`) ส่วนที่ทดสอบความสามารถใหม่จริงคือ mat4/fused2 ซึ่งมี ping-pong และ kernel คนละแบบ

### Protocol ที่ผูกมัด

script `remote_b5.sh` (sha256 `0b10a97d…`; ตั้ง `CUDA_ARCH=sm_86` สำหรับ RTX 3060 หรือ `sm_89` สำหรับ RTX 4060)
ไฟล์ที่ล็อกไว้: `stencil.cu` `1b991f15…`, `stencil_host.hpp` `8e8a7d19…`, `common.cuh` `7a5ea8b8…`, `cache_sim_b5.py` `1b3e47d5…`, `b5_model.py` `b8f4cd4f…`

1. รัน test และ build
2. รัน B1 smoke ที่ 256 MiB เพื่อเอา memcpy BW
3. รัน B5 sweep 3 รอบ ที่ l2x ∈ {0.0625 … 2, **4**} เหมือนชุดของ RTX 4060 (4 = จุด calibrate)
4. รัน B4 sweep 1 รอบบน instance เดียวกัน ใช้สำหรับ G4
5. calibrate: `b5_model.py --calib-run <r1> --bw-run <B1> --prereg research/preregistration/B5_<gpu>_calibrated.json`
   τ แยกตาม (ชนิด kernel 5pt/13pt, โหมด) และใช้ค่าจาก r1 ที่ l2x 4 เท่านั้น
6. ให้คะแนน: `b5_model.py --score <r1> <r2> <r3> --prereg …calibrated.json` ใช้ median ของ 3 รอบ
7. G4: `b5_model.py --score <r1> <r2> <r3> --compare-b4 <B4 run>`

### ตัวชี้ขาด (ใช้กับทั้ง RTX 3060 และ RTX 4060 ยกเว้นที่ระบุ)

| ID | คำทำนาย | ถูก falsify ถ้า |
|---|---|---|
| **G1 รูปร่าง** | warm/cold ของ sweep (median) สูงสุดที่ l2x ∈ [0.5, 0.875] และมีค่า > 1.2 | จุดสูงสุดอยู่นอกช่วงนี้ หรือมีค่า ≤ 1.2 |
| **G2 control ขนาดเท่ากัน (4060 เท่านั้น)** | warm/cold < 1.0 ที่ l2x 0.0625 และ 0.125 (1.5 และ 3 MiB) | ≥ 1.0 ที่จุดใดจุดหนึ่ง |
| **G3 เชิงปริมาณ** | MAPE ของ ratio หลัก (`mat4/fused2_iso`, `warm/cold`) ที่ l2x ≥ 0.625 ยกเว้น 4 **≤ 10 %** อย่างน้อยหนึ่ง policy | ทั้งสอง policy > 10 % แปลว่า model ไม่ transfer มาที่ B5 บน GPU นั้น |
| **G4 ข้ามตระกูลแบบไม่ใช้ model** | warm/cold ของ B5 ต่างจาก warm/cold ของ B4 (instance เดียวกัน) ไม่เกิน ±10 % ทุกขนาดที่ l2x ≥ 0.625 | ต่างเกิน 10 % ที่จุดใดจุดหนึ่ง |
| **G5 bytes เป็นตัวกำหนด** | mat4/fused2 iso ที่ l2x 2 และ 4 อยู่ใน [1.8, 2.3] | อยู่นอกช่วง แปลว่า kernel 13 จุดไม่ได้ติด DRAM (เช่นติด compute หรือ L1) |
| **D-B5-1 (ทิศทาง จาก B4)** | LRU ได้ MAPE ต่ำกว่า random **บนทั้งสอง GPU** | random ชนะบน GPU ใดตัวหนึ่ง |

### ตัวเลขคร่าว ๆ (INDICATIVE: `B5_rtx3060_indicative.json` `90d42ff2…`, `B5_rtx4060_indicative.json` `7e37200e…`)

สมมติ τ ของ B4 สำหรับทั้งสองชนิด kernel ตอนให้คะแนนจริงจะ calibrate τ แยกตามชนิด kernel

| l2x | 3060 LRU m4/f2 iso | random | **3060 LRU warm/cold** | random | 4060 LRU m4/f2 iso | random | **4060 LRU warm/cold** | random |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.625 | 2.17 | 1.98 | **1.40** | 1.06 | 2.22 | 1.97 | **1.55** | 1.08 |
| 0.75 | 2.14 | 2.02 | 1.33 | 1.10 | 2.18 | 2.02 | 1.43 | 1.12 |
| 0.875 | 2.13 | 2.04 | 1.29 | 1.12 | 2.15 | 2.05 | 1.36 | 1.15 |
| 1 | 2.11 | 2.06 | 1.25 | 1.13 | 2.13 | 2.07 | 1.30 | 1.16 |
| 1.5 | 2.08 | 2.06 | 1.17 | 1.13 | 2.09 | 2.07 | 1.19 | 1.15 |
| 2 | 2.06 | 2.05 | 1.13 | 1.11 | 2.06 | 2.06 | 1.14 | 1.12 |

- ใน B5 ค่า mat4/fused2 ของ LRU กับ random ต่างกันน้อยกว่าใน B4 (2.17 เทียบกับ 1.98 ที่ 0.625) เพราะ ping-pong ทำให้ working set ลดลง ตัวที่ใช้แยกสอง policy ได้ชัดจึงยังเป็น **warm/cold**
- **ช่วงต่ำกว่าหน้าผาไม่ทำนายเป็นตัวเลข** ด้วยเหตุผลเดียวกับ B4 คือไม่มี term ของ L2 bandwidth ต่างจาก B4 ตรงที่ mat4 มี 4 launch เทียบกับ 2 ของ fused2 ช่วงนี้ ratio จึงน่าจะ ≈ 2 จาก launch และ L2-tier แม้ DRAM bytes จะเท่ากัน

---

---

## Model v3 + B5b pre-registration — เขียน 2026-11-04 ก่อนวัดขนาดใหม่

> **ผล (เพิ่มภายหลัง ไม่ได้แก้ข้อความด้านล่าง; ดู `reports/BASELINE_REPORT.md` §19):**
> - V3-1 ✅ (4/4 ชุด), V3-2 ✅ (LRU ชนะ 3/4), V3-3 ✅
> - P-B5b-1 ✅ (3060: 1.26 → 1.61×), P-B5b-2 ✅ ตามนิยาม BW ที่ fit ได้ (ถ้าวัดตรงจะต่าง 8–13 %), P-B5b-3 ✅
> - error สูงสุดอยู่ที่ขอบหน้าผา (0.5625) ซึ่ง L2 จริงเก็บข้อมูลได้มากกว่า LRU แท้
> - มีการแก้ crash หลังวัด 2 จุด ซึ่งไม่เปลี่ยนคำทำนายตัวใด (D-024)

**ที่มา:** B5 บน 3060 แสดงว่า time model ของ v2 (τ คงที่ต่อ kernel และใช้ BW ของ memcpy กับทุก kernel) ใช้ไม่ได้เมื่อ kernel ที่ fuse แล้วไม่ได้ติด DRAM (§17)
v3 และ B5b ออกแบบ**หลัง**เห็นผลนั้น ข้อมูลเดิมทั้งหมดจึงเป็น in-sample **การทดสอบจริงคือขนาดใหม่ที่ยังไม่เคยวัด**

### Model v3 (`model_v3.py`, sha256 `33c38569…`)

- ใช้ DRAM bytes จาก simulator เดิม**ที่ไม่ได้แก้** (`cache_sim.py` `ea8ad8ef…`, `cache_sim_b5.py` `1b3e47d5…`)
- time model: **t = k·τ[kt, mode] + bytes / BW[kt, mode]** โดย kt คือชนิด kernel (`pw_fused`, `pw_stage`, `5pt`, `13pt`, `13pt_c`)
- **calibrate:** least squares ที่ l2x = 2 และ 4 (ทั้งสองขนาดติด DRAM และ LRU กับ random ต่างกัน ≤ 1 %) โดยบังคับ **τ ≥ 0** ถ้า fit ได้ค่าติดลบให้ตั้ง τ = 0 แล้ว fit BW ใหม่
  เงื่อนไขนี้เพิ่ม**ก่อน**ล็อก เพราะการ fit B4 บน 4060 ได้ τ ติดลบ
- **post-hoc check (ไม่ใช่หลักฐาน):** ใช้ B5 เดิมบน 3060 calibrate จาก r1 ที่ l2x 2, 4 แล้วดูขนาดอื่น ได้ MAPE ของ LRU = 4.8 % (v2: 35.7 %) แต่ random ก็ได้ 4.8 % เท่ากัน **ข้อเสียคือ v3 ที่ fit แบบนี้แยก LRU กับ random ได้น้อยลง**

### B5b ablation: เปลี่ยนตัวแปรเดียว (`stencil.cu` sha256 `fb1bfb15…`)

- `fused2c` = `fused2` แต่ wrap ขอบแบบ conditional (`x<0 ? x+m : x>=m ? x-m : x`) แทน `%` ผลยังต้องตรงกันทุก bit
- **หลักฐาน static (SASS, ได้ค่าเดียวกันทั้ง sm_86 และ sm_89):** instruction ต่อ element ได้แก่ 5 จุด = 72, 13 จุดแบบ `%` = **216** (มีลำดับคำสั่งหาร I2F/RCP/F2I 9 คำสั่ง), 13 จุดแบบ conditional = **152** ทุกแบบมี LDG 13 ครั้ง
- **กลไกที่ทำนาย:** ถ้า kernel 13 จุดช้าเพราะติดที่ instruction issue เวลาจะลดลงตามจำนวน instruction (× 152/216) แต่ต้องไม่ต่ำกว่าเวลาที่ติด DRAM (BW ของ kernel 5 จุด)
  ข้อนี้ถูกเข้ารหัสไว้ในตัวเลข indicative เป็น `β_13c = max(β_5pt, β_13pt × 0.7037)`

| ID | คำทำนาย | ถูก falsify ถ้า |
|---|---|---|
| **P-B5b-1 (3060)** | BW ของ fused2c ที่ fit ได้ (r1, l2x 2 และ 4) อยู่ใน **[250, 320] GB/s** (indicative 288; fused2 เดิมได้ 203) และ mat4/fused2c iso ที่ l2x ≥ 1.75 อยู่ใน **[1.55, 1.95]** (indicative 1.71–1.74) | BW ≤ 1.1 × ของ fused2 **หรือ** mat4/fused2c ≤ 1.45 ซึ่งแปลว่า modulo **ไม่ใช่**ต้นเหตุ |
| **P-B5b-2 (4060)** | fused2c ติด DRAM: BW ต่างจาก 5 จุดไม่เกิน 5 % และ mat4/fused2c iso ที่ l2x ≥ 1.75 อยู่ใน **[1.85, 2.05]** (indicative 1.94–1.97) | อยู่นอกช่วงนี้ |
| **P-B5b-3** | fused2c ตรงกับ fused2 และ host reference ทุก bit | correctness gate ไม่ผ่าน |

### การทดสอบ v3 บนขนาดใหม่ (protocol `remote_v3.sh` sha256 `435b7de0…`)

1. ใช้ instance เดียวกันต่อ GPU: รัน B5 (mat4, fused2, fused2c, warm/cold) 3 รอบ และ B4 (K = 1) 3 รอบ
2. ขนาด l2x ∈ {0.125, **0.5625, 0.6875, 0.8125, 0.9375, 1.125, 1.375, 1.75, 2.5, 3**} ซึ่งเป็น**ขนาดใหม่ทั้งหมด** บวกกับ 2 และ 4 สำหรับ calibrate
3. ทำนาย: `model_v3.py predict --bench <stencil|fusion> --calib-run <r1 ของรอบใหม่> --sizes-l2x <ชุดเดียวกัน> --prereg research/preregistration/v3_<bench>_<gpu>_calibrated.json` โดย**ไม่ใช้** `--assume` ในขั้นนี้ เพราะ fused2c มีข้อมูลจริงแล้ว
4. ให้คะแนน: `model_v3.py score --bench … --runs <r1> <r2> <r3> --prereg …calibrated.json`
   ratio หลักคือ B5: `mat4/fused2_iso`, `mat4/fused2c_iso`, `sweep_warm/sweep_cold_iso` และ B4: `materialized/fused_iso`, `stage2_warm/stage2_cold_iso` ที่ l2x ≥ 0.5625 ยกเว้น 2 และ 4 ใช้ median ของ 3 รอบ

| ID | คำทำนาย | ถูก falsify ถ้า |
|---|---|---|
| **V3-1** | ในแต่ละชุด (2 GPU × 2 benchmark = 4 ชุด) มี policy อย่างน้อยหนึ่งตัวที่ MAPE ≤ 10 % | ชุดใดชุดหนึ่งทั้งสอง policy เกิน 10 % ให้ถือว่า v3 ล้มในชุดนั้น |
| **V3-2 (ทิศทาง)** | LRU ได้ MAPE ต่ำกว่า random อย่างน้อย 3 ใน 4 ชุด | random ชนะ ≥ 2 ชุด |
| **V3-3 (รูปร่าง ไม่ใช้ model)** | warm/cold ของทั้ง B4 และ B5 สูงสุดในช่วง l2x ∈ [0.5625, 0.8125] บนทั้งสอง GPU | สูงสุดที่ l2x ≥ 0.9375 |

**ตัวเลขคร่าว ๆ (INDICATIVE, calibrate จากข้อมูลเดิม; ไฟล์ `v3_{stencil,fusion}_{rtx3060,rtx4060}_indicative.json`; sha256 `5088f084…`, `fcc17220…`, `b7dbad4d…`, `0f7b38b1…`):**

| l2x | 3060 B5 m4/f2c (LRU/rnd) | 3060 B5 w/c | 3060 B4 mat/fused | 4060 B5 m4/f2c | 4060 B5 w/c | 4060 B4 mat/fused |
|---:|---|---|---|---|---|---|
| 0.5625 | 1.57 / 1.31 | 1.51 / 1.03 | 2.34 / 1.93 | 1.83 / 1.44 | 1.76 / 1.04 | 2.81 / 2.05 |
| 0.6875 | 1.60 / 1.44 | 1.41 / 1.09 | 2.28 / 2.00 | 1.86 / 1.63 | 1.55 / 1.11 | 2.58 / 2.13 |
| 0.9375 | 1.65 / 1.57 | 1.29 / 1.14 | 2.20 / 2.06 | 1.90 / 1.81 | 1.35 / 1.16 | 2.38 / 2.18 |
| 1.375 | 1.69 / 1.67 | 1.19 / 1.14 | 2.13 / 2.08 | 1.93 / 1.91 | 1.22 / 1.16 | 2.23 / 2.17 |
| 3 | 1.74 / 1.74 | 1.09 / 1.09 | 2.05 / 2.05 | 1.97 / 1.97 | 1.09 / 1.09 | 2.10 / 2.10 |

**ความเสี่ยงที่รู้อยู่แล้ว:**
- ขนาด 0.5625 อยู่ติดหน้าผาที่สุด ความแปรปรวนระหว่างรอบสูง (§14, §17)
- ที่ 0.5625 บน 4060, LRU ทำนาย B4 mat/fused = 2.81 ซึ่งสูงกว่าที่เคยวัดได้ทุกจุด ถ้าผิดจะเห็นชัดในจุดนี้

---

## RTX 3060 Laptop pre-registration: ขนาด L2 เปลี่ยน แต่ architecture เดิม (L-1..L-3) — เขียน 2026-11-04 12:34 UTC ก่อนรัน B4/B5 บน GPU นี้

> **ผล (เพิ่มภายหลัง ไม่ได้แก้คำทำนายด้านล่าง; ดู `reports/BASELINE_REPORT.md` §20):**
> - **L-1 ✅** B5 LRU 6.2 % / random 7.0 %, B4 LRU 4.1 % / random 6.0 %
> - **L-2:** B5 ✅ แต่ **B4 ❌ ตามกฎ** คือสูงสุดที่ 0.9375 (1.300) ส่วนที่ 0.6875 ได้ 1.294 ต่างกันไม่ถึง 1 tick ของ timer บน WSL2 ที่ละเอียด 1.024 µs กฎนี้ไม่ได้กำหนด tie tolerance ไว้ จึงต้องนับว่าล้ม
> - **L-3 ✅** ที่ 1.41 MiB ได้ B4 0.875 และ B5 0.923 ขณะที่ 3060 desktop ที่ขนาด byte เดียวกันได้ 1.13–1.38
> - D-L1 ✅, D-L2 ✅

**คำถาม:** หน้าผาของ fusion payoff ผูกกับ**ขนาด L2** จริงหรือไม่ เมื่อ**ไม่เปลี่ยน architecture**
การทดสอบข้าม GPU รอบก่อน (3060 desktop → 4060) เปลี่ยนสองอย่างพร้อมกัน คือ architecture (Ampere GA106 → Ada AD107) และ L2 (2.25 → 24 MiB) จึงแยกสองปัจจัยนี้ออกจากกันไม่ได้
GPU ในเครื่องนี้เป็น GA106 (cc 8.6) ตัวเดียวกับ 3060 desktop แต่ L2 = **3 MiB** (`cudaDevAttrL2CacheSize` = 3,145,728) ต่างกันเพียง **1.33 เท่า** การทดสอบนี้จึงละเอียดกว่าครั้งก่อนที่ต่างกัน 10.7 เท่ามาก

| | 3060 desktop (เช่า, machine 19544) | 3060 Laptop (เครื่องนี้) |
|---|---|---|
| L2 (attribute) | 2,359,296 B (2.25 MiB) | 3,145,728 B (3 MiB) |
| SMs | 28 | 30 |
| memcpy D2D (วัดจริง) | 333 GB/s | 270.9 GB/s (B1 smoke วันนี้, run `20261104-192459-smoke-laptop-b1-smoke`) |
| bus / mem clock (attribute) | 192-bit / 7501 MHz | 192-bit / 6001 MHz |

**สิ่งที่เห็นแล้วก่อนเขียนข้อนี้ (เปิดเผยตามจริง):** บน GPU นี้มีแค่ B1 smoke คือ T(16)/T(8) = 0.502, memcpy 270.9 GB/s และ empty-kernel floor 5.12 µs **ยังไม่ได้รัน B4/B5 เลย**
เกณฑ์ด้านล่างอ้างอิงค่าที่วัดไว้แล้วบน 3060 desktop ได้แก่ B4 warm/cold ≈ 0.76–0.89 ที่ l2x 0.375–0.5, 0.91–0.96 ที่ 0.5625, 1.13–1.38 ที่ 0.625 และ 1.37–1.42 ที่ 0.6875

**Protocol ที่ผูกมัด:** `benchmarks/scripts/local_v3_laptop.sh` (sha256 `eec610d5…`) เหมือน `remote_v3.sh` ทุกขั้น ต่างกันแค่ (1) เพิ่มขนาด **l2x 0.46875** (2) ไม่มีขั้น apt (3) บันทึก `nvidia-smi` ทุก 1 วินาทีตลอดการรัน
ไฟล์ที่ล็อก: `model_v3.py` `995f128d…`, `cache_sim.py` `ea8ad8ef…`, `cache_sim_b5.py` `1b3e47d5…`, `fusion.cu` `f4af4694…`, `stencil.cu` `fb1bfb15…`, `common.cuh` `7a5ea8b8…`, `stencil_host.hpp` `8e8a7d19…`
คำทำนายเชิงตัวเลขสร้างแบบ deterministic จาก r1 ที่ l2x 2 และ 4 ด้วย `model_v3.py predict` (`v3_{fusion,stencil}_rtx3060laptop_calibrated.json`) แล้วให้คะแนนด้วย `model_v3.py score` ใช้ median 3 รอบ เป็นวิธีเดียวกับ §19 ทุกอย่าง

| ID | คำทำนาย | ถูก falsify ถ้า |
|---|---|---|
| **L-1 (= V3-1)** | B4 และ B5 แต่ละชุดมี policy อย่างน้อยหนึ่งตัวที่ PRIMARY MAPE ≤ 10 % (l2x ≥ 0.5625 ยกเว้น 2 และ 4) | ชุดใดชุดหนึ่งเกิน 10 % ทั้งสอง policy |
| **L-2 (= V3-3, รูปร่าง)** | warm/cold (B4 `stage2`, B5 `sweep`) สูงสุดที่ l2x ∈ [0.5625, 0.8125] และมีค่า > 1.2 | จุดสูงสุดอยู่ที่ l2x ≥ 0.9375 หรือ ≤ 0.46875 หรือสูงสุดไม่เกิน 1.2 |
| **L-3 (control ขนาด byte เท่ากัน, architecture เดิม)** | ที่ **l2x 0.46875 = 1.40625 MiB ต่อ array** (เป็นขนาด byte ที่ 3060 desktop อยู่บนหน้าผา คือ l2x 0.625 ได้ warm/cold 1.13–1.38) warm/cold ของทั้ง B4 และ B5 **< 1.0** | ≥ 1.1 ใน benchmark ใดก็ได้ ถ้าอยู่ในช่วง [1.0, 1.1) ให้รายงานว่า ambiguous |
| D-L1 (ทิศทาง) | LRU มี MAPE ต่ำกว่า random ทั้ง B4 และ B5 | random ชนะชุดใดชุดหนึ่ง |
| D-L2 (ทิศทาง, สมดุล compute/BW) | kernel 13 จุดแบบใช้ `%` ไม่ติด DRAM บน laptop เช่นกัน ดูจาก mat4/fused2 iso ที่ l2x ≥ 1.75 < 1.8 และ mat4/fused2c > mat4/fused2 | mat4/fused2 ≥ 1.8 หรือ fused2c ไม่เร็วกว่า fused2 |

**ข้อควรระวังที่รู้อยู่แล้ว:**
- clock ของ laptop ไม่คงที่ ระหว่าง B1 เห็น SM 1282–1552 MHz (max 2100) และควบคุมไม่ได้เพราะไม่มีสิทธิ์ admin บน Windows host จึงต้องดู `gpu_monitor.csv` ประกอบ
- ratio ที่ l2x เดียวกันต่างจาก desktop ได้ด้วยเหตุผลที่ไม่เกี่ยวกับ L2 เช่น τ ถูกเจือจางต่างกัน (array ใหญ่ขึ้น 1.33 เท่า BW ต่ำลง) ดังนั้น L-3 ทดสอบ**ทิศทาง** (ต่ำกว่า 1 หรือมากกว่า 1) ไม่ใช่การเทียบค่าตรง ๆ

---

## Counter pre-registration (S-0, C-1..C-4) — เขียน 2026-11-04 ขณะที่ counter ยังถูกบล็อกบนเครื่องนี้

**คำถาม (PAPER_GAP_ANALYSIS §4 และความเสี่ยงข้อ 1):** DRAM bytes ที่ simulator ให้ ซึ่ง v2/v3 ใช้ทำนายเวลาได้คลาด 2–5 % ตรงกับ bytes ที่ GPU ย้ายจริงหรือไม่ หรือ timing ตรงเพราะ fit พารามิเตอร์ได้พอดีโดยบังเอิญ
**สถานะตอนเขียน:** `ncu` 2024.2.1 รันได้ แต่ได้ `ERR_NVGPUCTRPERM` เพราะ key `RmProfilingAdminOnly` ของ adapter NVIDIA ฝั่ง Windows ยังเป็น 1 (key `NVTweak` เป็น 0 แล้ว) **จึงยังไม่มี counter ใดถูกอ่าน** ตรวจได้จาก run `benchmarks/results/counters_20261104-123344/STATUS`

**Model:** `counter_prereg.py` (sha256 `64e1ad2d…`) ใช้ `cache_sim.Cache` ตัวที่ล็อกไว้ (`ea8ad8ef…`) โดย**ไม่แก้** จำลองทุก launch ของช่วงจับเวลาตามลำดับจริงที่ binary ส่ง (`--reps 1 --warmup 0 --steady-iters 3`) ผ่าน cache ตัวเดียว รวมถึง flush kernel ด้วย แล้วทำนาย read = read miss × 128 B และ write = dirty eviction × 128 B **ต่อ launch**
- **ต่างจาก v2/v3 (ประกาศไว้ก่อนวัด):**
  - หลัง flush, cache เต็มไปด้วย flush lines ที่ clean ไม่ได้ว่าง ถ้าเป็น LRU ผลจะเท่ากับ S5 ทุก window (มี unit test `test_counter_prereg.py` ยืนยัน) แต่ถ้าเป็น random line ใหม่ต้องแย่งที่กับ flush lines ทำให้ random เก็บข้อมูลได้น้อยกว่า "random" ของ v2/v3
  - write ของ flush ตัวแรกในแต่ละขนาดไม่นำมาให้คะแนน เพราะมาจากช่วงตรวจความถูกต้องที่ไม่ได้จำลอง
- **ไฟล์คำทำนาย:** `research/preregistration/counters_rtx3060laptop.json` (sha256 `e3196540…`) ที่ L2 = 3,145,728 B และ l2x ∈ {0.25, 0.46875, 0.5625, 0.75, 1, 2, 4}
- **Protocol:** `benchmarks/scripts/local_counters.sh` (sha256 `af024ea0…`) ใช้ `ncu --replay-mode application --cache-control none --clock-control none` กับ metric `dram__bytes_read.sum` และ `dram__bytes_write.sum` (ใช้ให้คะแนน) และ `lts__t_sectors_srcunit_tex_op_{read,write,read_lookup_miss}.sum` กับ `gpu__time_duration.sum` (บันทึกไว้ ไม่ใช้ให้คะแนน)
- **หน่วยของ error:** e = |วัดได้ − ทำนาย| / B โดย B = bytes ของ array หนึ่งตัว (4n) ใช้หน่วยนี้แทน % เพราะหลาย cell มีค่าทำนายเป็น 0 เช่น write ของ window ที่อยู่ใน L2 ทั้งหมด

| ID | คำทำนาย | ถูก falsify ถ้า |
|---|---|---|
| **S-0 (เงื่อนไขก่อนให้คะแนน)** | DRAM read ของ flush kernel ทุกตัวต่างจาก 16 MiB (ขนาด flush buffer) ไม่เกิน 5 % | ไม่ผ่าน ให้**หยุด**และไม่ตัดสิน C-* เพราะ counter ไม่ได้วัดสิ่งที่เราคิด |
| **C-1 (การนับ bytes, B4, ช่วงติด DRAM l2x 2 และ 4)** | อย่างน้อยหนึ่ง policy มี cell ที่ e ≤ 0.10 B ไม่ต่ำกว่า 90 % (cell = read ของทุก launch และ write ของทุก launch ยกเว้น flush ตัวแรก) | ทั้งสอง policy ต่ำกว่า 90 % แปลว่าความแม่นของ timing ใน C2 **อธิบายด้วยการนับ bytes ที่ถูกต้องไม่ได้** |
| **C-2 (deferred write-back, B4 ที่ l2x 0.25)** | DRAM write ของ window `fused` isolated < 0.5 B **และ** flush ที่ตามหลัง `stage2_cold` เขียน > 0.5 B (LRU ทำนาย 0.00 B และ 1.00 B) | ไม่เป็นตามนี้ แปลว่ากลไก "write-back ไปตกที่ kernel ถัดไป" ผิดที่ระดับ counter |
| C-3 (ทิศทาง, l2x 0.25–1) | LRU มี mean e ต่ำกว่า random ทั้ง B4 และ B5 cell ที่ชี้ขาดคือ read ของ `stage2_warm`/`sweep_warm` ที่ l2x 0.5625–1 ซึ่ง LRU ทำนาย 1.00 B ส่วน random ทำนาย 0.65–0.86 B | random ชนะ |
| **C-4 (= C-1 สำหรับ B5)** | เหมือน C-1 แต่ใช้กับ stencil | เหมือน C-1 |

**ข้อจำกัด / ภัยคุกคามต่อความถูกต้อง (ประกาศไว้ก่อน):**
- ncu รัน kernel ทีละตัวโดยมีช่วงว่างระหว่าง launch ถ้า L2 เขียน dirty line กลับตอน GPU ว่าง ผลจะต่างจากการรันจับเวลาจริง ข้อนี้คือสิ่งที่ C-2 ทดสอบด้วย
- application replay สร้าง process ใหม่ทุก pass ทำให้ physical address และ set mapping ของ L2 อาจเปลี่ยนระหว่าง pass
- การทดสอบนี้ใช้ GA106 (Ampere) เท่านั้น ผลจึง**ไม่ครอบคลุม Ada** (4060)
- counter บน WSL2 อาจมีพฤติกรรมต่างจาก Linux native ข้อนี้ยังไม่ได้ตรวจ

---

## H1 — Fetch-granularity amplification ของ strided read (baseline B1, characterization)

> **If** we increase the read stride s of a 4-byte copy from 1 to 64 at a fixed 64–256 MiB footprint (≫ L2), **then** measured DRAM read bytes (`dram__bytes_read.sum`) should follow `min(4N, g·N/s)` within ±10 %, and kernel time should follow the same curve divided by the memcpy-measured bandwidth within ±20 % for s ≤ g/4, **because** DRAM↔L2 traffic is quantized to g-byte chunks regardless of how many bytes the warp consumes; **potential deviations** are lower DRAM efficiency for scattered sectors (row-buffer misses), TLB misses at large stride, and launch/event overhead for small n_out.

- **สถานะ (2026-11-04): สนับสนุนที่ระดับ timing บน RTX 3060 ตัวเดียว (smoke 10 reps)**
  วัด T(16)/T(8) ได้ 0.507 ซึ่งตรงกับ model g=32 (0.500) ส่วน g=64 (0.944) ถูกตัดออกด้วย upper bound ทางกายภาพ เพราะต้องใช้ 622 GB/s ซึ่งเกิน peak 360 GB/s
  ต้องรู้ไว้ด้วยว่า driver รายงาน `cudaLimitMaxL2FetchGranularity` = 64 B **ยังไม่มีผลจาก counter** ดู `reports/BASELINE_REPORT.md` §10
- **คำทำนายย่อยที่ถูก falsify:** scalar contiguous copy **ไม่ได้**ช้ากว่า u32x4 อย่างมีนัยสำคัญ (ต่างกัน 0.76 %) Little's law จึงไม่ใช่คอขวดที่ 64 MiB บน GPU นี้
- **ข้อทดสอบที่แยก g ออกจากกันได้ชัด:** เวลาที่ s=16 หารด้วยเวลาที่ s=8 ถ้า g=32 model ทำนายว่า ≈ 0.50 ถ้า g=64 ทำนายว่า ≈ 0.94
  ablation คือตั้ง `--l2-fetch-granularity 32/64/128` แล้วดูว่าเวลาและ counter เปลี่ยนตามหรือไม่ (driver อาจไม่สนใจค่านี้ ตัวไบนารีจะอ่านค่ากลับมาบันทึกไว้)
- **Falsified ถ้า:** อัตราส่วน T(16)/T(8) ไม่ตกในช่วง [0.4, 0.65] และไม่ตกในช่วง [0.85, 1.1] หรือ counter ไม่ตรงกับ model ที่ g ใดเลย
- **Novelty risk: สูงมาก (ยืนยันแล้วจาก Gate A)** prior art ตรงคือ **Ding & Williams PMBS 2019** ซึ่งใช้ profiler counter วัด "stride walls" (stride-0/1/8) และ transactions ต่อ request บน V100 นอกจากนี้ CUDA Best Practices Guide 13.4 §10.2.1.4 ก็อธิบายผลของ stride ไว้แล้ว และ Bell & Garland SC'09 วัดผลของ stride ด้วย timing (ดู `LITERATURE_MATRIX.md` #21–23)
  **H1 ไม่ใช่ contribution** หน้าที่ของมันคือ *calibrate* วิธีวิจัยของเรา (model ↔ counter ↔ timing) บน hardware จริง ก่อนจะไปอ้างอะไรที่ซับซ้อนกว่านี้
  มีคำถามย่อยหนึ่งข้อที่อาจยังมีเอกสารน้อย คือผลของ `cudaLimitMaxL2FetchGranularity` บน consumer Ampere ข้อนี้ต้องค้นต่อ

## H2 — Fusion payoff ขึ้นกับ L2 residency ของ intermediate (กลไก L2 สนับสนุนบน 2 GPU × 2 ตระกูล workload; ด้วย model v3 (BW ต่อชนิด kernel) การทำนาย payoff ผ่านการทดสอบ out-of-sample ทั้ง 4 ชุด — §19)

> **If** we fuse a two-stage pointwise pipeline `t = f(x); y = g(t)` (fp32, n elements) into one kernel, **then** end-to-end time of the pair should improve by up to ≈2× when `4n ≫ L2 capacity`, but the gain should shrink toward launch-overhead-only when the intermediate `t` fits in L2, **because** materialization costs DRAM traffic of `write t + read t` (estimated 16n vs. 8n bytes) only when `t` is evicted before it is re-read; **potential regressions** are higher register pressure and lower occupancy when f and g are heavy, and losing a vendor-optimized implementation of either stage.

- **กลไกที่ต้องระวัง:** dirty line ของ `t` ใน L2 สุดท้ายก็ต้องถูก write-back ลง DRAM แม้จะไม่มีใครอ่าน `t` อีกแล้ว (ยกเว้น driver หรือ hardware จะทิ้งให้ซึ่งไม่น่าเป็นไปได้)
  write-back นี้อาจไปเกิดระหว่าง kernel ถัดไป การนับ DRAM bytes ทีละ kernel จึง**ทำให้เข้าใจผิดได้** ต้องวัด end-to-end ทั้งคู่
- **ช่วง shape ที่น่าสนใจ:** n·4 B ตั้งแต่ ¼×L2 ถึง 64×L2 crossover ที่ model ทำนายอยู่ใกล้ขนาด L2 ซึ่งต้องอ่านจาก `meta.json` ห้ามสมมติว่าเป็น 3 MB
- **Generalization (RQ5):** กลไกเดียวกันนี้ ("intermediate อยู่ใน L2 ได้หรือไม่") ใช้กับ stencil หลาย sweep (temporal blocking) ได้ ซึ่งเป็นตระกูล workload ที่ 2
- **Novelty risk: สูง** kernel fusion สำหรับงาน memory-bound มีงานก่อนหน้าแล้ว เช่น Wahib & Maruyama SC'14, AStitch ASPLOS'22, Ivanov et al. MLSys'21 และ compiler อย่าง XLA, TVM, Triton, `torch.compile` รวมถึง temporal blocking ใน AN5D CGO'20
  **อัปเดต 2026-11-04 (อ่าน full text แล้ว):** model ของ Wahib & Maruyama SC'14 คิดเฉพาะ DRAM bandwidth, shared memory และ registers **ไม่มี L2 term ไม่มีเรื่อง write-back และไม่มี sector granularity**
  paper ยังเขียนไว้ด้วยว่า "Varying the data arrays sizes at runtime has no impact on the effectiveness of kernel fusion" ซึ่งเป็นสมมติฐานที่ H2 ตั้งใจจะ falsify ตรง ๆ
  งานที่ใกล้ที่สุดเชิงประจักษ์คือ Dodović et al. (Electronics 2026) วัด fused vs. unfused ข้ามหลายขนาดบน RTX 3080 Ti/T4/A100 และอธิบายผลด้วย L2 แบบคุณภาพเท่านั้น ไม่มี analytical model
  ข้อที่ต้องระวังคือ Dodović รายงานว่า "As the arrays grow, speedup decreases slightly" ซึ่งดูสวนกับคำทำนายของเรา จึงเป็นจุดที่การวัดต้องอธิบายให้ได้ (ดู P-B4-4)
  **ยังไม่ได้อ่าน:** Qiao et al. CGO'19 ซึ่งผลค้นหาบอกว่ามี "analytic model that estimates locality gains" **ต้องอ่านก่อนอ้าง novelty**
  ความมั่นใจเรื่อง novelty ของส่วน "L2 residency + deferred write-back + granularity" ที่ subagent ประเมินคือ ≈ 0.6 จากงานที่อ่านแล้ว และ ≈ 0.3 ถ้าค้นกว้างกว่านี้ (ดู `LITERATURE_MATRIX.md` §4)
  ตอนนี้ยังไม่ได้ค้นว่า cost model ใน TVM/Ansor หรือ Halide autoscheduler ครอบคลุมเรื่องนี้ไปแล้วหรือไม่ **ต้องตรวจก่อนอ้าง**

## H3 — Gather ด้วย index แบบ skewed: L2 hit rate ทำนายได้จาก hot-set size (B3 smoke วัดแล้ว: ideal-LFU ทำนาย hit สูงเกินจริง ดู BASELINE_REPORT §12.3)

> **If** gather indices over a table ≫ L2 change from uniform to Zipf(α) skew, **then** measured DRAM bytes per gathered 4-byte element should fall below the ≈32 B/element of uniform random access by the fraction of accesses that hit the hottest rows fitting in L2, and a frequency-ordered layout (hot rows packed contiguously so one 32 B sector carries several hot elements) should reduce DRAM bytes further for α ≳ 1 but **not** for uniform indices or for rows ≥ 32 B, **because** L2→DRAM traffic is per-sector and packing raises useful bytes per sector; **potential regressions** are the one-time remap cost, an extra index-translation gather, and no benefit unless the same index distribution is reused many times.

- **คำทำนายที่ falsify ได้:** ถ้า row ≥ 32 B (เช่น embedding dim ≥ 8 fp32) การ pack layout ต้องลด DRAM bytes ได้ประมาณ 0 ถ้าลดได้มากกว่านั้น แปลว่า model ผิด
- **Novelty risk: สูง** การ reorder หรือ remap ตาม frequency มีงานก่อนหน้าทั้ง graph reordering, embedding cache ใน recommender systems และ SpMV format อย่าง Bell & Garland SC'09
  สิ่งที่อาจยังไม่มีคือ model ที่ผูก α, ขนาด row, L2 capacity และจำนวนครั้งที่ reuse เข้าด้วยกันแล้วตรวจกับ counter **ยังไม่ได้ตรวจ**

## H4 (ลำดับรอง) — `cp.async` ใน tiled transpose บน Ampere: คาดว่าจะเป็น negative result

> **If** a shared-memory tiled transpose uses `cp.async` (global→shared copy that bypasses registers, cc ≥ 8.0) instead of register-staged loads, **then** time should **not** improve materially (<5 %) for matrices ≫ L2, **because** a coalesced tiled transpose is already DRAM-bandwidth-bound and `cp.async` changes register/instruction pressure, not DRAM bytes; a gain is expected only where occupancy or latency hiding is the limiter; **potential regressions** are extra synchronization (`cp.async.wait_group` + `__syncthreads`) and more shared memory for multi-stage buffering.

- **Novelty risk: สูง** Luo et al. IPDPS'24 รายงานว่า AsyncPipe ได้ "+19.6% on A100" ที่ block 8×8 แต่ได้เพียง +1.7 % ที่ 32×32 ซึ่งเป็นตัวเลขของผู้เขียน และสอดคล้องกับคำทำนายของเรา
  สิ่งที่ยังไม่มีคือข้อมูลบน consumer Ampere (GA10x) พร้อมคำอธิบายกลไกด้วย counter คุณค่าจึงอยู่ที่การยืนยันซ้ำ (replication) และ negative result ไม่ใช่สิ่งใหม่
  CUDA Programming Guide 13.4 ระบุว่า LDGSTS ต้องใช้ cc ≥ 8.0 จึงทดสอบบน sm_86 ได้ ส่วน TMA ต้องใช้ cc ≥ 9.0 จึงอยู่นอก scope

---

## H5 (candidate ใหม่จาก §14.2) — L2 fetch-granularity hint สำหรับ gather ที่ใช้ sector ข้างเคียง

> **If** a gather touches neighbouring 32-B sectors of the same 64-B chunk (e.g. 8-byte pairs or row fragments spanning sectors), **then** setting `cudaLimitMaxL2FetchGranularity` = 64 should cut its time by up to ≈ 1.6× on RTX 3060 with ≤ 2 % cost for single-element gathers, **because** the default issues separate DRAM accesses per 32-B sector and random-access cost is dominated by access count rather than bytes; **potential regressions** are +34 % at 128 B for single-element gathers and unknown effects on streaming kernels and on other GPUs.

- **หลักฐานตอนนี้:** timing เท่านั้น, pattern สังเคราะห์, **เกิดซ้ำบน 2 GPU** (3060: 1.57×, 4060: 1.72× แบบ exploratory) ยังไม่ได้ทดสอบกับ workload จริง เช่น embedding lookup ที่ row 8–16 B
- **Novelty risk: ยังไม่ได้ตรวจ** เอกสาร CUDA ระบุว่าค่านี้เป็น "performance hint" อาจมีคนเขียนไว้แล้วใน blog, forum หรือ paper ต้องค้นก่อน
- เรื่องที่ค่าตั้งต้นรายงาน 64 แต่ทำงานแบบ 32 ต้องตรวจกับ driver และ GPU อื่น และอาจเป็นพฤติกรรมเฉพาะเวอร์ชัน driver (535.154.05)

---

## ลำดับที่เสนอ (ยังไม่ใช่การตัดสินใจ Gate D)

1. วัด **B1/H1** บน NVIDIA GPU จริงเพื่อ calibrate วิธีวิจัย ซึ่งถูกที่สุดและพร้อมรันแล้ว
2. Implement B2 (transpose), B3 (gather) และ B4 (fused vs. materialized) เป็น baseline เท่านั้น ยังไม่ optimize
3. ดูหลักฐานแล้วเลือก primary hypothesis ข้อเดียว ตอนนี้ **H2 มีศักยภาพทาง novelty สูงสุด** เพราะเชื่อมได้สองตระกูล workload แต่ต้องผ่านการอ่าน Wahib & Maruyama SC'14 ฉบับเต็มก่อน (D-005)
