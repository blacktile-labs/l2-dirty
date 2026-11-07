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

## B6 write-back probe pre-registration (W-0..W-5) — เขียน 2026-11-05 ก่อนวัด cell ใดใน grid

**คำถาม:** counter รอบ D-032 พบ DRAM write ใน window ของ B4 ที่ working set ทั้งหมดอยู่ใน L2 ได้ ซึ่ง model (cache_sim S3: write-back เฉพาะตอน evict) ทำนายว่าเป็น 0 ตัวอย่าง (หน่วย B = array หนึ่งตัว):
- fused ที่ l2x 0.46875 อ่าน DRAM เกือบ 0 (x ยังอยู่ใน L2) แต่เขียน 0.23–0.69 B ต่อ launch
- materialized steady ที่ l2x 0.25 เขียนรวม ≈ 0.7 B ต่อรอบ
- fused ที่ l2x 0.25 เขียน 0.000

ใน B4 footprint ส่วน clean (x) กับส่วน dirty (t, y) ขยับไปพร้อมกันเสมอ ข้อมูลชุดนี้จึงแยกไม่ได้ว่าตัวไหนเป็นตัวกำหนด

**Prior art (D-033):** Finn (Doubleword blog, 2026, RTX 4090) เสนอ model ว่า "เมื่อ store เข้า set ที่มี dirty ≥ 8 จาก 16 way ให้ clean dirty line ที่ถูก store นานที่สุด โดยเขียนกลับ DRAM แต่ยังเก็บไว้ใน cache" แต่ post ได้มาจากการ fit survival curve **ไม่ได้แยกทดลอง threshold และไม่ได้วัดด้วย DRAM counter** ฝั่ง CPU มี eager writeback (Lee/Tyson/Farrens MICRO 2000) ที่เขียน dirty line ใกล้ตำแหน่ง LRU กลับไปเมื่อ bus ว่าง

**Benchmark:** `benchmarks/src/wb_probe.cu` (B6, sha256 `05102e53…`, binary `e0c82a53…`)
- แต่ละ launch ของ `rw_kernel` อ่าน C (r·L2 B, `ld.global.cg`) และเขียน D (w·L2 B, ค่าใหม่ทุก launch) ใน pass เดียวกัน
- ลำดับต่อ cell: flush 16 MiB → warm → steady × 5 → readback (อ่าน D ทั้งหมด)
- grid: r, w ∈ {0, 1/8, …, 7/8} โดย 0 < r+w ≤ 7/8 (35 cell, write-back LRU เก็บได้ทั้งหมด) และ anchor 2:0, 0:2
- **Protocol:** `benchmarks/scripts/wb_probe.sh` (`da84291f…`)
  - 3 รอบ `ncu --replay-mode application` (counter ต่อ launch)
  - 3 รอบ `--replay-mode app-range` (steady 2..5 รันติดกัน ไม่มีช่วงว่างที่ ncu แทรกระหว่าง kernel)
- **Scorer:** `wb_probe_score.py` (`b165fd69…`, loader จาก `counter_prereg.py` `e7d21047…`) มี unit test ด้วยข้อมูลสังเคราะห์ (`test_wb_probe_score.py`)

**ปริมาณที่ใช้ (median ของ 3 รอบ × steady launch 2..5):**

| ชื่อ | ความหมาย |
|---|---|
| WB | DRAM write / B_w |
| RD | DRAM read / B_r |
| RB | read ของ readback / B_w |
| WARM | read ของ warm / B_r |
| WBR | เหมือน WB แต่มาจาก range |

active = WB ≥ 0.10, inactive = WB < 0.05

**สมมติฐานที่แข่งกัน:**

| ID | กลไก | ทำนาย |
|---|---|---|
| H-LRU | write-back ตอน evict เท่านั้น (model v2/v3) | ทุก cell ใน grid มี WB, RD < 0.05 |
| H-D | จำกัด dirty ต่อ set (แบบ Finn) | WB ขึ้นกับ **w อย่างเดียว** การเพิ่ม r ไม่เปลี่ยน WB |
| H-E | eager write-back ตาม occupancy | WB ขึ้นกับ **r+w** ถ้า w คงที่ การเพิ่ม r ทำให้ WB เพิ่ม |
| H-b | evict จริง (เช่น set conflict) | readback ต้อง miss: RB/WB ≥ 0.5 |
| H-G | write-back ตอน GPU ว่าง (ช่วงว่างจาก ncu) | WBR/WB ≤ 0.5 |

**กฎตัดสิน (ล็อกไว้ใน scorer):**

| ID | เกณฑ์ | ผล |
|---|---|---|
| **W-0 (ต้องผ่านก่อน)** | anchor 2:0 มี RD ∈ [0.9, 1.1], anchor 0:2 มี WB ∈ [0.9, 1.1] และ WARM ∈ [0.9, 1.1] ทุก cell ที่ r > 0 | ถ้าไม่ผ่าน ให้**หยุด** ไม่ตีความ W-1..W-5 |
| **W-1** | มี cell ใน grid อย่างน้อยหนึ่ง cell ที่ WB ≥ 0.10 หรือ RD ≥ 0.10 | H-LRU ถูก reject |
| **W-2 (ตัวชี้ขาดหลัก)** | Δ = WB(r, 1/8) − WB(0, 1/8) ที่ r = 0.625 และ 0.75 | ทั้งคู่ ≥ +0.10 → H-E ชนะ H-D · ทั้งคู่ \|Δ\| < 0.05 → H-D ชนะ H-E · นอกนั้น inconclusive |
| W-3 (รอง) | spread ของ WB ในแถว w = 1/8, 1/4, 3/8, 1/2 | ทุกแถว ≤ 0.15 → flat (H-D) · rise ≥ 0.20 อย่างน้อย 2 แถว → H-E |
| **W-4** | median RB/WB ของ cell ที่ active | < 0.25 → clean แต่ยังอยู่ใน cache (H-D/H-E) · ≥ 0.5 → evict (H-b) |
| **W-5** | median WBR/WB ของ cell ที่ active | ≥ 0.75 → ไม่ได้เกิดจากช่วงว่าง · ≤ 0.5 → เกิดจากช่วงว่าง (H-G) |

ใช้ W-2 แทนการดู WB(0.625, 1/8) เพียงค่าเดียว เพราะ cap ของ H-D บน GA106 ยังไม่รู้ค่า ถ้า cap ต่ำ WB ที่ w = 1/8 อาจไม่เป็น 0 ได้แม้ไม่มีผลจาก r การเทียบกับ r = 0 ในแถวเดียวกันจึงทนต่อค่า cap ที่ไม่รู้

**ส่วน exploratory (ไม่ให้คะแนน):**
- คอลัมน์ w = 0: RD บอก capacity สำหรับข้อมูลที่ clean
- คอลัมน์ r = 0: รูปร่างของ WB(w) เทียบกับ H-D แบบ mapping สม่ำเสมอ ซึ่งควรเป็นขั้นบันไดที่ w = 0.5
- counter `lts__*` และ gpu time

**ข้อจำกัด (ประกาศไว้ก่อน):**
- GPU ตัวเดียว (GA106 laptop บน WSL2) และไม่ได้ล็อก clock
- application replay สร้าง process ใหม่ทุก pass ทำให้ physical address และ set mapping อาจเปลี่ยน
- range mode ต้องใช้ 2 pass ต่อ range (rerun ทั้ง application)
- cell เป็น streaming ตามลำดับ index เท่านั้น ยังไม่ทดสอบลำดับอื่น
- โหมด smoke (cell 0.0625:0.0625, 1.5:0, 0:1.5) รันก่อนเขียนเอกสารนี้ เพื่อตรวจ format ของไฟล์ export ทั้งสาม cell ไม่อยู่ใน grid ที่ให้คะแนน

---

> **ผล (MEASURED 2026-11-05 04:24–04:26 UTC, `benchmarks/results/wb_probe_20261105-042441/`, D-034):**
>
> | กฎ | ผล |
> |---|---|
> | W-0 | ✅ PASS (anchor RD 1.000, WB 0.992, WARM 1.000–1.001; correctness 0 failure ทั้ง 6 รอบ) |
> | W-1 | ✅ **H-LRU ถูก reject** มี 11/35 cell ใน grid ที่ active และไม่มี cell ไหนมี RD ≥ 0.10 |
> | W-2 (ตัวหลัก) | ⚠️ **inconclusive** Δ = +0.001 ที่ r = 0.625 และ **+0.108** ที่ r = 0.75 ตัวหลังเกิน 0.10 ไป 0.008 |
> | W-3 | flat rows (เอียงไปทาง H-D) spread 0.045–0.108 |
> | W-4 | ✅ **clean แต่ยังอยู่ใน cache** RB = 0.000 ทุก cell ที่ active |
> | W-5 | ✅ **ไม่ได้เกิดจากช่วงว่าง** WBR/WB = 1.017 |
>
> **Exploratory (ไม่ได้ pre-register ไว้):**
> - **ผลหลักเป็นขั้นบันไดตาม w:** w ≤ 0.375 ได้ WB ≤ 0.004 · w = 0.5 ได้ 0.19–0.28 · w ≥ 0.625 ได้ 0.97–0.99 ทุกค่า r รูปร่างนี้ตรงกับ rule "dirty ≥ ครึ่งหนึ่งของ way ต่อ set" ถ้า set mapping กระจายเกือบสม่ำเสมอ เพราะพอ dirty ต่อ set เกิน cap ทุก store จะสั่ง clean หนึ่ง line ทำให้ WB → 1 แต่ยังไม่ได้วัด associativity ของ GA106
> - **r มีผลรองตรงขอบ w = 0.5:** ใน range mode WBR เพิ่มจาก 0.19 เป็น 0.27 เมื่อ r เพิ่มจาก 0 เป็น 0.375
> - **ผลรองที่ r+w = 0.875 เกิดเฉพาะตอนมีช่วงว่าง:** cell (0.75, 1/8), (0.625, 1/4), (0.5, 3/8) มี WB ต่อ launch กระโดดอยู่ระหว่าง 0–0.57 แต่ WBR = 0.00–0.05 cell (0.75, 1/8) ที่ทำให้ W-2 กลายเป็น inconclusive เป็นของผลรองนี้
>   - ถ้าคิด W-2 จาก WBR จะได้ Δ = 0.000 และ +0.041 ซึ่งเข้าเกณฑ์ H-D **แต่นี่เป็นการวิเคราะห์หลังเห็นผล จึงยืนยันอะไรไม่ได้**
> - **ไม่เห็นผลต่อเวลา:** steady kernel ใช้ ≈ 8 µs/MiB ทุก cell ที่ r = 0 ไม่มีขั้นบันไดตรง w = 0.625 ที่ขนาดนี้ (≤ 2.6 MiB ต่อ launch, 10–20 µs) จึงยังไม่ติด bandwidth ของ DRAM write
>
> **ความหมายต่อ H2 / model:**
> - assumption S3 ของ `cache_sim.py` ผิดในระดับ counter เมื่อ dirty footprint ≥ ≈ 0.5 L2 ข้อมูลที่ถูกเขียนยังอยู่ใน L2 (read ยังได้ประโยชน์) แต่ bytes ที่เขียนไปถึง DRAM ภายใน kernel ที่ผลิตมันเอง
> - C-2 ("write-back ไปตกที่ kernel ถัดไป") จึงใช้ได้เฉพาะเมื่อ dirty < ≈ 0.5 L2
> - ผลเรื่อง timing ของ model v3 ยังไม่ขัดกับเรื่องนี้ เพราะที่ขนาดนี้ไม่เห็นผลต่อเวลา

## B6b pre-registration (V-0..V-5) — เขียน 2026-11-05 ก่อนวัด cell ใดของ B6b

**คำถาม:** B6 (D-034) ยืนยันแล้วว่า L2 เขียน dirty line กลับโดยยังเก็บไว้ใน cache และผลนี้ไม่ได้เกิดจาก profiler แต่ตัวชี้ขาดหลัก W-2 (ขึ้นกับ w อย่างเดียว หรือขึ้นกับ r+w) ออกมา inconclusive รอบนี้มี 3 เป้าหมาย:
1. ยืนยัน W-2 ด้วย **cell ใหม่** และใช้ **WBR** (launch ติดกันแบบที่ pipeline จริงรัน) เป็นตัวหลัก เหตุผลคือ B6 พบว่าการรันทีละ kernel ทำให้เกิด write-back เพิ่มที่ r+w = 0.875
2. วัดว่าขั้นบันไดที่ w ≈ 0.5 คมแค่ไหน
3. cap นับเป็น **line** หรือเป็น **bytes**

**สิ่งที่เปลี่ยนจาก B6 (ทีละตัวแปร):**
- เพิ่ม `rw1s_kernel` / `readback1s_kernel` (flag `--one-sector`) ที่เขียนแค่ sector แรก (32 B) ของทุก line ขนาด 128 B จำนวน line ที่ครอบครองเท่าเดิม แต่ dirty bytes เหลือ ¼
- `rw_kernel`, `readback_kernel` และ flush ไม่ได้เปลี่ยน SASS ตรงกับ binary ของ B6 (`e0c82a53…`, เก็บไว้ที่ `benchmarks/results/wb_probe_20261105-042441/wb_probe.binary`) ทุก instruction
- scorer ของ B6 รองรับ one-sector ด้วย default เดิม ให้คะแนน B6 ซ้ำได้ `score.txt` และ `score.json` ตรงกันทุก byte

**ไฟล์ (sha256):**
- `wb_probe.cu` `7dbbd451…`, binary `0921da36…`
- protocol `wb_probe_b6b.sh` `0dc645e8…`
- scorer `wb_probe_b6b_score.py` `8f7092ba…`, `wb_probe_score.py` `e7751d87…`, `counter_prereg.py` `e7d21047…`
- test `test_wb_probe_b6b_score.py`

**Cell (ไม่เคยวัดมาก่อน ยกเว้น 0:0.5 และ 0:0.625 ที่ใช้ตรวจ replication):**
- **full:** แถว w = 3/16 กับ r ∈ {0, 0.5, 0.625, 0.6875}, แถว w = 5/16 กับ r ∈ {0, 0.375, 0.5, 0.5625}, คอลัมน์ r = 0 กับ w ∈ {0.4375, 0.5, 0.5625, 0.625, 0.6875} และ anchor 2:0, 0:2
- **sector:** r = 0 กับ w ∈ {0.375, 0.625, 0.75, 0.875} และ anchor 0:2 (ค่าทั้งหมดของ sector ทำให้เป็นสัดส่วนของ bytes ที่เขียนจริง)
- รันอย่างละ 3 รอบทั้งวัดทีละ kernel และวัดแบบรันติดกัน

**กฎตัดสิน (ล็อกไว้ใน scorer):**

| ID | เกณฑ์ | ผล |
|---|---|---|
| **V-0 (ต้องผ่านก่อน)** | full 2:0 มี RD และ RDR ∈ [0.9, 1.1]; full 0:2 และ sector 0:2 มี WB และ WBR ∈ [0.9, 1.1]; WARM ∈ [0.9, 1.1] | ถ้าไม่ผ่าน ให้หยุด |
| V-1 replication | \|WBR − ค่าของ B6\| ≤ 0.10 ที่ 0:0.5 (B6 = 0.190) และ 0:0.625 (B6 = 0.992) | ถ้าไม่ผ่าน แปลว่าผลขึ้นกับรอบหรือ process (เช่น physical address) |
| **V-2 (ตัวหลัก)** | Δ = WBR(r, w) − WBR(0, w) ใน 6 cell ของสองแถว | ทุก \|Δ\| < 0.05 → **H-D ได้รับการยืนยัน** · Δ ≥ 0.10 ทั้งสอง cell ที่ r+w = 0.875 → H-E · นอกนั้น inconclusive |
| V-3 | WBR(0, 0.4375) และ WBR(0, 0.5625) | < 0.05 และ ≥ 0.80 → ขั้นบันไดคม · ≥ 0.10 หรือ < 0.50 → ไม่คม · นอกนั้น partial |
| V-4 | sector: control WBR(0, 0.375) ต้อง < 0.05 แล้วดู median WBR ที่ w ∈ {0.625, 0.75, 0.875} | ≥ 0.80 → **นับเป็น line** · ≤ 0.05 → นับเป็น bytes/sector · นอกนั้น mixed |
| V-5 | median RB/WB ของ cell ที่ active (ทั้งสองชุด, วัดทีละ kernel) | < 0.25 → ยังอยู่ใน cache · ≥ 0.5 → ถูก evict |

**คำทำนายของผู้วิจัย (ตั้งไว้ก่อนวัด):** V-0 ผ่าน, V-1 ผ่าน, V-2 = H-D, V-3 = คม, V-4 = นับเป็น line, V-5 = ยังอยู่ใน cache
- ถ้า V-2 ออกมาเป็น H-D และ V-4 เป็น "นับเป็น line" จะสอดคล้องกับ rule "dirty line ≥ ครึ่งหนึ่งของ way ต่อ set" (Finn 2026) ที่วัดบน GA106 ด้วย DRAM counter
- ถ้า V-4 ออกมาเป็น "นับเป็น bytes" แปลว่า cap คิดจาก dirty bytes ไม่ใช่จาก line ข้อนี้จะขัดกับ rule ของ Finn ในรูปแบบที่ใช้ line

**ข้อจำกัด:**
- GPU ตัวเดียว ไม่ได้ล็อก clock
- ยังไม่รู้ associativity ของ GA106 จึงยังแยกไม่ได้ระหว่าง "ครึ่งหนึ่งของ way" กับ "ครึ่งหนึ่งของ capacity"
- V-2 ยังเป็นการทดสอบภายใต้ pattern เดียว (stream ตามลำดับ index)
- smoke (cell 0.0625:0.0625, 1.5:0, 0:1.5 ทั้งสองโหมด) รันก่อนเขียนเอกสารนี้เพื่อตรวจ format ทั้งสาม cell ไม่อยู่ใน grid ของ B6b

---

> **ผล (MEASURED 2026-11-05 06:38–06:41 UTC, `benchmarks/results/wb_probe_b6b_20261105-063839/`, D-036): ตรงกับคำทำนายทุกข้อ**
>
> | กฎ | ผล |
> |---|---|
> | V-0 | ✅ PASS |
> | V-1 | ✅ replicate ได้ WBR 0.194 (B6 0.190) และ 0.991 (B6 0.992) ต่างรอบและต่าง process |
> | **V-2** | ✅ **H-D ได้รับการยืนยัน** Δ ทั้ง 6 cell อยู่ระหว่าง 0.000 ถึง +0.003 (WB ที่วัดทีละ kernel ซึ่งไม่ใช้ตัดสินได้ ≤ +0.032) |
> | **V-3** | ✅ **ขั้นบันไดคม** WBR ที่ w = 0.4375 / 0.5 / 0.5625 / 0.625 / 0.6875 ได้ 0.014 / 0.194 / 0.886 / 0.991 / 1.000 |
> | **V-4** | ✅ **นับเป็น line** sector-mode median WBR = 0.994 (0.825, 0.994, 0.997) และเขียนกลับเฉพาะ sector ที่ dirty |
> | V-5 | ✅ ยังอยู่ใน cache RB/WB = 0.000 |
>
> **ข้อสังเกต (exploratory):**
> - control ของ V-4 ผ่านแบบเฉียด: WBR 0.042 (3 รอบได้ 0.041/0.058/0.042) ขณะที่ WB ที่วัดทีละ kernel ได้ 0.001 คำอธิบายที่น่าจะเป็น (หลังเห็นผล) คือ range mode มี DRAM traffic คงที่ราวสิบกว่า KB ต่อ range ซึ่งดูใหญ่เมื่อหารด้วย bytes ที่เขียนจริงในโหมด sector (smoke ก็มี RDR floor 0.02–0.09)
> - sector-mode ที่ w = 0.625 ได้ 0.825 ต่ำกว่า full line (0.991) ยังไม่ได้อธิบาย
>
> **ข้อสรุป (B6 + B6b):** บน RTX 3060 Laptop (GA106) เมื่อ **จำนวน line ที่ dirty** ใน L2 เกิน ≈ 0.5 ของ L2 ฮาร์ดแวร์จะเขียน sector ที่ dirty กลับ DRAM โดยยังเก็บ line ไว้ใน cache
> - ขนาดข้อมูล clean (สูงสุด 0.69 L2) ไม่มีผล
> - ไม่ได้เกิดจาก profiler และ replicate ได้
> - ตรงกับ rule ต่อ set ที่ Finn (2026) เสนอไว้บน RTX 4090 ตอนนี้วัดด้วย DRAM counter บน Ampere แล้ว
> - **ยังแยกไม่ได้**ระหว่าง "ครึ่งหนึ่งของ way ต่อ set" กับ "ครึ่งหนึ่งของ capacity ทั้งหมด" เพราะยังไม่ได้วัด associativity

## Model v4 counter pre-registration (K-0..K-3) — เขียน 2026-11-05 ก่อนอ่าน counter ที่ขนาดเหล่านี้

**คำถาม:** ถ้าเพิ่มกลไกที่ B6/B6b วัดได้ลงใน simulator จะทำนาย DRAM bytes ต่อ launch ของ workload จริงสองตระกูล (B4 fusion และ B5 stencil) ได้ดีกว่า v3 หรือไม่ ในช่วงที่สอง model ทำนายต่างกัน

**Model v4** (`cache_sim_v4.py` `6c68abc0…`) = LRU write-back ของ v3 บวก **D1:** ถ้ามี dirty line เกิน ½ ของ L2 ให้ clean dirty line ที่ถูก store นานที่สุด (เขียน 128 B กลับ DRAM แต่ line ยังอยู่ใน cache)
- **พารามิเตอร์ ½ มาจาก B6/B6b เท่านั้น** ไม่ได้ fit กับ counter ของ B4/B5 ผู้ช่วยวิจัย**ยังไม่ได้รัน v4 กับข้อมูล D-032** และจะรันหลังรอบนี้เท่านั้น โดยติดป้ายว่า explanatory
- unit test (`test_cache_sim_v4.py`) ยืนยันสองข้อ:
  - ถ้าปิด D1 (cap = capacity) ผลตรงกับ `cache_sim.Cache("lru")` ที่ล็อกไว้ทุกครั้ง
  - ถ้าเปิด D1 จะได้รูปแบบเดียวกับ B6 คือขั้นบันไดตาม w ไม่ขึ้นกับ r และ line ยังอยู่ใน cache

**การเลือกขนาด (ใช้ simulation อย่างเดียว ก่อนเห็นข้อมูล):** v3 กับ v4 ทำนายต่างกัน **เฉพาะที่ l2x ∈ (0.25, 0.5]** เลือก l2x **0.3125, 0.375, 0.4375, 0.5** (จุดที่ทำนายต่างกัน) บวก 0.1875 (ทำนายเหมือนกัน, อยู่ใน L2) และ 2, 3 (ทำนายเหมือนกัน, ติด DRAM) **ทุกขนาดยังไม่เคยวัดด้วย counter**
- cell ที่ใช้แยก v3/v4 (\|v4 − v3\| ≥ 0.25 B): fusion 36 cell, stencil 132 cell
- ส่วนใหญ่เป็น DRAM write ของ window materialized/mat4/fused2 ตัวอย่างที่ l2x 0.375: steady k1 W v3 0.33 vs v4 1.00, isolated k2 W 0.00 vs 0.67

**ไฟล์ (sha256):**
- คำทำนาย `research/preregistration/counters_v4_rtx3060laptop.json` `0481efef…`
- `counter_v4.py` `63d77de5…`, `local_counters_v4.sh` `fa38be87…`, `counter_prereg.py` `e7d21047…`, `cache_sim.py` `ea8ad8ef…` (ล็อกเดิม ไม่แก้)
- binary `fusion` `84f19ebd…` และ `stencil` `3dc27d89…` เป็นตัวเดียวกับที่ใช้ใน D-032 source ตรงกับ `LOCK_20261104_laptop.sha256`
- test `test_counter_v4.py`

**Protocol:** เหมือน D-026 (application replay, `--cache-control none --clock-control none`, `--reps 1 --warmup 0 --steady-iters 3`) แต่รัน **3 รอบ** ต่อ benchmark ค่าวัดของแต่ละ cell คือ median ของ 3 รอบ e = \|วัด − ทำนาย\| / B

| ID | เกณฑ์ (ใช้กับ fusion และ stencil แยกกัน) | ผล |
|---|---|---|
| **K-0 (ต้องผ่านก่อน)** | S-0: DRAM read ของ flush ต่างจาก 16 MiB ไม่เกิน 5 % | ถ้าไม่ผ่าน ให้หยุด |
| **K-1 (ตัวหลัก)** | สัดส่วนของ cell แยก model ที่ v4 ใกล้ค่าวัดมากกว่า v3 | ≥ 75 % → **v4 ดีกว่า** · ≤ 25 % → v3 ดีกว่า · นอกนั้น mixed |
| **K-2 (ความแม่น)** | mean e ของ v4 บน cell แยก model ≤ 0.20 B | ไม่ผ่าน แปลว่า v4 ดีขึ้นแต่ยังไม่แม่นพอจะใช้ |
| **K-3 (ไม่ถอยหลัง)** | mean e ของ v4 บนทุก cell ≤ mean e ของ v3 + 0.01 **และ** ที่ l2x 2, 3 มี cell ที่ e ≤ 0.10 B ไม่ต่ำกว่า 90 % | ไม่ผ่าน แปลว่า D1 ทำให้ที่อื่นแย่ลง |

**คำทำนายของผู้วิจัย:** K-0 ผ่าน, K-1 = v4 ดีกว่า, K-2 ผ่าน, K-3 ผ่าน ทั้งสอง benchmark

**ข้อจำกัด (ประกาศไว้ก่อน):**
1. ncu แบบ application replay รัน kernel ทีละตัว B6 พบว่าการรันแบบนี้เพิ่ม write-back เมื่อ footprint รวมใกล้เต็ม L2 (≥ 0.875) ซึ่งทั้ง v3 และ v4 ไม่มี error ของทั้งสอง model ใน window แบบนั้นจะสูงขึ้น
2. D1 เป็น cap ระดับทั้ง cache ขั้นบันไดจึงคมเกินจริง (B6b ที่ dirty 0.5 วัดได้ 0.19 แต่ v4 ทำนาย 0) window ที่ dirty อยู่ใกล้ 0.5 จะมี error ราว 0.2 B
3. แรงจูงใจทั้งหมดมาจากข้อมูล D-032 (l2x 0.25 และ 0.46875) ที่เห็นไปแล้ว รอบนี้จึงใช้เฉพาะขนาดใหม่
4. stencil ถูกจำลองเป็น stream src → dst เหมือน v3 (ไม่มี halo)
5. GPU ตัวเดียว

---

> **ผล (MEASURED 2026-11-05 06:58–07:01 UTC, `benchmarks/results/counters_v4_20261105-065851/`, D-038):**
>
> | เกณฑ์ | fusion (B4) | stencil (B5) |
> |---|---|---|
> | K-0 (S-0) | ✅ (err 0.1 %) | ✅ (err 0.4 %) |
> | **K-1** | ✅ **v4 ดีกว่า** 100 % ของ 36 cell | ✅ **v4 ดีกว่า** 78 % ของ 132 cell (เกณฑ์ 75 %) |
> | **K-2** | ✅ mean e v4 **0.102** (v3 0.691) | ❌ **FAIL** mean e v4 **0.237** > 0.20 (v3 0.617) |
> | K-3 | ✅ ทุก cell 0.040 vs 0.132; far 100 % | ✅ 0.075 vs 0.171; far 100 % |
>
> **v4 พลาดตรงไหน (exploratory):**
> - **fusion:** แม่น (e ≤ 0.11) ที่ l2x 0.375–0.5 ส่วนที่ l2x 0.3125 window materialized steady วัดได้ 0.72 แต่ v4 ทำนาย 1.00 ซึ่งอยู่ใกล้ขอบของขั้นบันไดตามข้อจำกัดข้อ 2
> - **stencil:** fused2/fused2c/sweep_cold แม่น (e 0.05–0.28) แต่ **mat4 (ping-pong 4 kernel) และ sweep_warm ที่ l2x 0.4375–0.5** วัดได้ 0.40–0.64 ขณะที่ v4 ทำนาย 0.86–1.00 (e 0.36–0.54) ทั้งหมดอยู่ในช่วงที่ footprint รวมเกิน L2
>   - **คำอธิบายที่เป็นไปได้ (หลังเห็นผล):** L2 จริงอาจเลือก evict line ที่ clean ก่อน dirty (Finn 2026 เสนอไว้) ทำให้ write-back น้อยกว่า LRU ล้วน หรืออาจเป็นข้อจำกัดข้อ 4 (ไม่ได้จำลอง halo) **ยังไม่ได้ทดสอบ**
> - **v4 กับข้อมูล D-032 (explanatory, รันหลังล็อกและวัดรอบนี้แล้ว):** fusion mean e ทุก cell 0.066 → 0.042, stencil 0.084 → 0.043 cell ที่แยก model ได้ (6 และ 24 cell, ที่ l2x 0.46875) v4 ใกล้กว่า v3 ทุก cell

## B6c pre-registration (X-0..X-4) — เขียน 2026-11-05 ก่อนวัด cell ใดของ B6c

**คำถาม:** เมื่อ working set เกิน L2 แล้ว L2 เลือก evict line ที่ clean ก่อน (ทำให้ dirty line อยู่ได้นานกว่า) หรือใช้ LRU ล้วนตามที่ v4 สมมติ
- **ที่มา (post hoc):** v4 ทำนาย write ของ B5 mat4/sweep_warm สูงเกินไป 0.4–0.5 B เมื่อ footprint เกิน L2 (D-038)
- **prior art:** Finn (2026, 4090) บรรยาย replacement ที่เลื่อน dirty victim ออกไปแล้วหา clean victim แทน

**Model ที่แข่งกัน (ทั้งคู่มี D1 = dirty cap ½):**
- **v4:** `cache_sim_v4.DirtyCapLRU` (`6c68abc0…`, ล็อกแล้วใน D-037) ใช้ victim ตาม LRU
- **cf:** `cache_sim_cf.CleanFirstDirtyCap` (`e6503819…`) ใช้ victim เป็น clean line ที่ LRU ที่สุดก่อน (D2)
  - unit test ยืนยันว่าเมื่อมีแต่ read หรือมีแต่ write (ไม่มีให้เลือก) cf ให้ผลเท่ากับ LRU ที่ล็อกไว้ทุกครั้ง

**Cell** (binary `wb_probe` `0921da36…` ตัวเดียวกับ B6b, full line):
- **mixed 12 cell:** w ∈ {1/8, 1/4, 3/8} (ต่ำกว่า dirty cap จึงไม่มี cleaning) ที่ footprint รวม r+w ∈ {1.125, 1.25, 1.5, 2}
- **control:** 0.5:0.25 (อยู่ใน cache), 2:0, 0:2
- **exploratory:** 0.625:0.5, 0.5:0.75
- **คำทำนาย** (`research/preregistration/b6c_predictions.json` `2e61093f…`) ใน mixed ทุก cell:
  - **v4:** WB = 1.00, RB 0.33–1.00, RD = 1
  - **cf:** WB = 0.00, RB = 0.00, RD = 1
- **Protocol:** `wb_probe_b6c.sh` (`66f55286…`) 3 รอบวัดทีละ kernel + 3 รอบวัดแบบรันติดกัน (เหมือน B6b) scorer `wb_probe_b6c_score.py` (`31fd60f5…`)

| ID | เกณฑ์ | ผล |
|---|---|---|
| **X-0 (ต้องผ่านก่อน)** | 2:0 มี RD และ RDR ∈ [0.9, 1.1]; 0:2 มี WB และ WBR ∈ [0.9, 1.1]; 0.5:0.25 มี WBR และ RDR < 0.05; WARM ∈ [0.9, 1.1] | ถ้าไม่ผ่าน ให้หยุด |
| **X-1 (ตัวหลัก)** | สัดส่วนของ 12 mixed cell ที่ WBR ใกล้ cf มากกว่าใกล้ v4 | ≥ 75 % → **clean-first ได้รับการสนับสนุน** · ≤ 25 % → LRU (v4) · นอกนั้น mixed |
| X-2 (ขนาดผล) | median WBR ของ mixed | ≤ 0.10 → ปกป้อง dirty เต็มที่ · ≥ 0.90 → ไม่ปกป้อง · นอกนั้นปกป้องบางส่วน |
| X-3 | median RB (readback miss ของ D, วัดทีละ kernel) | ≤ 0.25 → D ยังอยู่ใน cache · ≥ 0.75 → ถูก evict · นอกนั้น partial |
| X-4 (ตรวจทั้งสอง model) | mixed cell ที่ RDR ≥ 0.9 ไม่ต่ำกว่า 75 % | ถ้าไม่ผ่าน แปลว่า**ทั้งสอง model**ผิดเรื่อง clean miss |

**คำทำนายของผู้วิจัย:**
- X-0 ผ่าน
- **X-1 เอียงไปทาง clean-first แต่ไม่มั่นใจ** เพราะ stencil พลาดเพียง 0.4–0.5 B ไม่ใช่ 1 B ผู้วิจัยจึงคาดว่า X-2 จะออกมาเป็น "ปกป้องบางส่วน"
- X-4 ผ่าน

ถ้า X-1 ออกมาเป็น LRU ข้อผิดพลาดของ v4 ใน stencil ต้องมีสาเหตุอื่น (เช่น halo ของ stencil) และกลไก D2 จะตกไป

**ข้อจำกัด:**
- pattern เดียว (stream ตามลำดับ index, C กับ D สลับกันทุก element)
- GPU ตัวเดียว
- cf เป็น clean-first แบบสุดขั้ว (ปกป้อง dirty ได้ไม่จำกัด ภายใต้ D1) ส่วน hardware จริงอาจมี buffer จำกัด (Finn: write-back FIFO ต่อ set)

---

> **ผล (MEASURED 2026-11-05 07:12–07:15 UTC, `benchmarks/results/wb_probe_b6c_20261105-071251/`, D-040): NEGATIVE สำหรับ clean-first**
>
> | กฎ | ผล |
> |---|---|
> | X-0 | ✅ PASS |
> | **X-1** | **LRU victim (v4) ได้รับการสนับสนุน** WBR ใกล้ cf **0 %** ของ 12 cell |
> | X-2 | ไม่ปกป้อง dirty: median WBR **1.001** (ทุก cell 0.83–1.02) |
> | X-3 | ถูก evict: median RB 0.940 |
> | X-4 | ✅ RDR ≈ 1.000 ทุก cell |
>
> **ข้อสังเกต (exploratory):** ที่ footprint รวม 1.125–1.25 ค่า RB ต่ำกว่าที่ LRU ทำนายมาก ตัวอย่าง (1, 1/8): RB 0.085 แต่ LRU ทำนาย 1.00; (0.75, 3/8): 0.032 แต่ LRU ทำนาย 0.33 ทั้งที่ WBR ≈ 1
> - แปลว่าใต้แรงบีบ L2 **เขียน dirty line กลับทั้งที่ line ยังอยู่ใน cache** คล้าย eager write-back ฝั่ง CPU
> - และ replacement ตรงขอบ capacity ไม่ใช่ LRU ล้วน (line บางส่วนรอด cyclic thrash)
> - **ไม่กระทบ bytes ที่ write** ซึ่ง v4 ทำนายถูก แต่กระทบ residency
>
> **ความหมาย:**
> - D2 (clean-first) **ตกไป** ไม่ใช้ใน model
> - ความคลาดของ v4 ใน B5 mat4/sweep_warm (D-038) **ยังไม่มีคำอธิบาย** ผู้ต้องสงสัยที่เหลือคือ access pattern ของ stencil (halo/ลำดับแบบ 2D) ที่ simulator ทำให้ง่ายลงเป็น stream และ replacement ที่ไม่ใช่ LRU ล้วนตรงขอบ capacity ตามที่เห็นใน RB ข้างบน

## B7 pre-registration (R-0..R-3, T-1) — เขียน 2026-11-05 ก่อนวัดที่ขนาดเหล่านี้

**คำถาม:** model ที่สร้างและตรวจกับ kernel สังเคราะห์ของเรา (v3, v4) ทำนาย DRAM bytes ต่อ launch ของ **kernel จริงของ PyTorch** ได้หรือไม่ ทั้ง eager elementwise ของ ATen และ Triton kernel ที่ TorchInductor สร้าง และอัตราเร่งจาก fusion ในช่วงที่ติด DRAM ตรงกับอัตราส่วน bytes ที่ model ทำนายหรือไม่

**Workload:** `y = gelu(x + b) + r` fp32, x, r, y: [M, 4096], b: [4096] (epilogue ของ MLP ใน transformer)
- **eager:** 3 kernel (add แบบ broadcast, GELU แบบ erf, add) ที่ caching allocator ของ PyTorch จัด address ให้ (t1 ถูกคืนก่อน y ถูกจองตามพฤติกรรมของ `gelu(x + b) + r`)
- **compiled:** `torch.compile` ได้ Triton kernel เดียว (`triton_poi_fused_add_gelu_0`)
- `torch==2.7.1` (cu126), `triton==3.3.1` ใน venv แยก (`benchmarks/requirements-torch.txt` `9eb6c66c…`)
- correctness: compiled เทียบ eager (rtol/atol 1e-5) และ eager เทียบ float64 บน CPU

**Model (ไม่มีพารามิเตอร์ที่ fit):** v3 = `cache_sim.Cache("lru")` (`ea8ad8ef…`), v4 = `cache_sim_v4.DirtyCapLRU` (`6c68abc0…`) replay address จริงทุก launch (รวม flush) ตามลำดับ
- คำทำนายต่อ launch ล็อกใน `research/preregistration/b7_predictions.json` (`4505b589…`) สร้างจาก plan ของ run ที่ไม่มี ncu
- address ต่างกันระหว่าง process แต่ smoke แสดงว่า pattern การ reuse เหมือนกัน และคำทำนายตรงกันทุกค่า scorer จะตรวจซ้ำทุก run (`plan_consistent`)

**ขนาด (เลือกจาก simulation อย่างเดียว, ยังไม่เคยวัด):** l2x (bytes ของ tensor หนึ่งตัว / L2) = 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 1, 2, 3
- v3/v4 ทำนายต่างกันที่ 0.1875–0.5 (32 cell)
- smoke ใช้ 0.0625 และ 1.5 ซึ่งดูเฉพาะการจับคู่ launch และความสม่ำเสมอของ plan **ไม่ได้ดูคะแนน**

**Protocol:** `b7_opchain.sh` (`5ed8ea4b…`), harness `torch_opchain.py` (`630aea7f…`; แก้ bug recompile limit แล้วเป็น `a79a3487…` ตาม D-044 คำทำนายไม่เปลี่ยน), scorer `counter_b7.py` (`d3f5e6d5…`)
- ncu `--replay-mode application --target-processes application-only --cache-control none --clock-control none` 3 รอบ
- ใช้ median ของ 3 รอบ ลำดับต่อขนาด: flush → eager | flush → compiled | flush → eager × 3 | flush → compiled × 3
- **เวลา:** 3 รอบที่ไม่มี profiler ใช้ CUDA graph ที่บรรจุ 50 chain (steady) หรือ 20 × (flush + chain) ลบเวลาของ flush (isolated) เพื่อเฉลี่ยต้นทุน replay บน WSL (~50 µs ต่อครั้ง) ออก และวัด e2e (Python loop) แยก

| ID | เกณฑ์ | ผล |
|---|---|---|
| **R-0 (ต้องผ่านก่อน)** | DRAM read ของ flush ทุกตัวต่างจาก 16 MiB ไม่เกิน 5 % | ถ้าไม่ผ่าน ให้หยุด |
| **R-1 (ตัวหลัก)** | cell ทั้งหมด (read ทุก launch ที่ไม่ใช่ flush, write ทุก launch ยกเว้น flush ตัวแรก) ที่ e ≤ 0.10 B ไม่ต่ำกว่า **80 %** สำหรับ v4 | ไม่ผ่าน แปลว่า model ที่ตรวจกับ kernel สังเคราะห์**ใช้กับ kernel จริงไม่ได้**ในระดับนี้ |
| R-2 | cell ที่ \|v4 − v3\| ≥ 0.25 B: v4 ใกล้กว่าใน ≥ 75 % → v4 ดีกว่า · ≤ 25 % → v3 ดีกว่า · นอกนั้น mixed | |
| R-3 | l2x ≥ 2: v4 ภายใน 0.10 B ≥ 90 % | |
| **T-1 (เวลา)** | graph_steady speedup (eager/compiled) ที่ l2x 2 และ 3 อยู่ใน **[1.9, 2.8]** (อัตราส่วน DRAM bytes ที่ v4 ทำนาย = 7.00/3.00 = 2.33 ± ≈ 20 %) | ไม่ผ่าน แปลว่าเวลาไม่ได้ตาม bytes ที่ขนาดนี้ (เช่น kernel ของ ATen กับของ Triton ได้ bandwidth ต่างกันมาก) |

**ส่วนที่รายงานแบบ descriptive (ไม่ตัดสิน):**
- speedup ที่ขนาดเล็ก ซึ่งยังไม่มี time model สำหรับ kernel เหล่านี้ และ launch overhead มีผลมาก
- เวลา e2e

**คำทำนายของผู้วิจัย:** R-0 ✅, R-1 ✅, R-2 = v4 ดีกว่า, R-3 ✅, T-1 ✅

**ข้อจำกัด:**
- op chain เดียว, fp32, H = 4096, GPU ตัวเดียว (WSL2)
- เวลาเป็น GPU-side ภายใน CUDA graph ส่วน e2e รวม overhead ของ Python/guard ซึ่งขึ้นกับเครื่อง
- ATen kernel แบบ broadcast (`x + b`) ไม่ได้ vectorize อาจมี pattern ของ sector ต่างจาก stream ล้วน
- ไม่ได้ทดสอบ bf16 และ op chain ที่มี reduction (เช่น LayerNorm)

---

> **ผล (MEASURED 2026-11-05 09:01–09:07 UTC, `benchmarks/results/b7_opchain_20261105-090133/`, D-045; run แรกใช้ไม่ได้ตาม D-044):**
>
> | กฎ | ผล |
> |---|---|
> | R-0 | ✅ (flush err 0.18 %) |
> | **R-1 (ตัวหลัก)** | ❌ **FAIL**: v4 อยู่ภายใน 0.1 B **73.1 %** ของ 323 cell (เกณฑ์ ≥ 80 %); v3 ได้ 72.1 % |
> | R-2 | ✅ v4 ดีกว่า: 81 % ของ 32 cell ที่แยก model ได้ (mean e 0.24 vs 0.64) |
> | R-3 | ✅ 100 % ของ 72 cell ที่ l2x ≥ 2 |
> | **T-1** | ✅ graph_steady speedup **2.36× (l2x 2), 2.34× (l2x 3)** ขณะที่ v4 ทำนายอัตราส่วน bytes ไว้ 2.33 |
>
> - plan ของทั้ง 3 รอบให้คำทำนายตรงกับไฟล์ที่ล็อกไว้ และ correctness 0 failure ทุก run
>
> **ความคลาดอยู่ตรงไหน (exploratory):**
> - 87 cell ที่ e > 0.1 อยู่ที่ l2x 0.1875–1 ทั้งหมด (ช่วงขอบ capacity) ส่วนใหญ่เป็น write ของ eager_add2, flush และ gelu
> - **พลาดสองทิศ:**
>   - (ก) L2 จริงเก็บ t2 ที่เพิ่งเขียนไว้ได้ ทำให้ eager_add2 อ่าน DRAM 1.00 B ขณะที่ model ทำนาย 1.33–2.00
>   - (ข) ตอนใกล้เต็ม fused ที่ 0.3125 (footprint 0.94 L2) อ่าน DRAM 1.53 B ขณะที่ model ทำนาย 0
> - flush เขียนกลับมากกว่าที่ทำนาย 0.3–0.45 B ที่ขนาดเล็ก สาเหตุยังไม่รู้
> - รวมกับ D-038 (stencil) และ B6c (RB) แล้ว **replacement ตรงขอบ capacity ที่ไม่ใช่ LRU** เป็นแหล่ง error หลักที่เหลือของทั้ง v3 และ v4
>
> **เวลา (descriptive, median ของ 3 รอบ):**
> - **graph_steady:** speedup 2.3–2.6× ที่ l2x ≤ 0.25 สูงสุด **3.66×** ที่ 0.3125 (intermediate ของ eager ล้น L2 แต่ fused ยังพอดี) ลดเหลือ 1.8–1.9× ที่ 0.375–0.5 (fused เริ่มล้น) แล้วกลับเป็น 2.3× ที่ ≥ 1 **ขนาดที่ fusion คุ้มที่สุดจึงอยู่ตรงช่วงที่ footprint ของสองแบบอยู่คนละฝั่งของ L2**
> - **e2e (Python loop ไม่ใช้ CUDA graph):** compiled **ช้ากว่า** eager ที่ l2x ≤ 0.5 (0.44–0.72×) เพราะ wrapper ของ torch.compile ใช้ ~70 µs ต่อ call บน WSL ขณะที่ eager 3 op ใช้ ~43 µs fusion เริ่มคุ้มแบบ e2e ที่ l2x 1 (1.32×) และ ≥ 2 (2.0–2.3×) ตัวเลข overhead นี้ขึ้นกับเครื่องและ WSL

## B8 pre-registration (Z-0..Z-3) — เขียน 2026-11-05 ก่อนวัด cell ใดของ B8

**คำถาม:** ตรงขอบ capacity L2 ของ GA106 จัดการ dirty victim และ capacity ที่ใช้ได้จริงอย่างไร ทั้ง v3 และ v4 พลาดในช่วงนี้ทั้งกับ stencil (D-038) และกับ kernel จริงของ PyTorch (D-045)

**ที่มาของสมมติฐาน (post hoc, D-046):**
- B6c เห็นว่าใต้แรงบีบ D ถูกเขียนกลับทุก launch (WB ≈ 1) แต่ส่วนใหญ่**ยังอยู่ใน cache** (RB 0.03–0.31 ที่ footprint 1.125) ขณะที่ C ถูกไล่ออกหมด
- ตรงกับคำบรรยายของ Finn (2026): dirty victim ถูกเขียนกลับและ "stays readable" แล้ว scan ต่อไปหา clean victim
- ผู้ช่วยวิจัย fit ค่า "อายุใน write-back FIFO" กับข้อมูล B6c (เห็นแล้ว) ได้ life = **0.125 × capacity** (MAE 0.026 เทียบกับ LRU 0.068) **ค่านี้ fit จากข้อมูลที่เห็นแล้ว** การทดสอบรอบนี้จึงใช้เฉพาะ cell ใหม่

**Model:** `cache_sim_v5cand.CandidateL2` (`b7306c78…`):
- **v4:** policy "lru" (unit test ยืนยันว่าเท่ากับ `cache_sim_v4.DirtyCapLRU` ทุกครั้ง)
- **dvsc:** policy "dvsc_life" ที่ life 0.125 (unit test ยืนยันว่าเท่ากับ LRU เมื่อไม่มีการเขียน)
- คำทำนายอยู่ใน `research/preregistration/b8_predictions.json` (`8aafe4da…`)

**Cell** (binary `wb_probe` `0921da36…` ตัวเดิม, correctness 21/21 ผ่าน):
- **M (mixed 8):** r+w ∈ {1.0625, 1.1875, 1.3125, 1.4375}, w ∈ {3/16, 5/16}
- **W (write-only 3):** w ∈ {1.0625, 1.125, 1.25}
- **R (read-only 6):** r ∈ {0.90625, 0.9375, 0.96875, 1.0, 1.03125, 1.0625}
- **control:** 0.5:0.25, 2:0, 0:2 และ cell replication 1:0.125 ของ B6c
- **Protocol:** `wb_probe_b8.sh` (`96049e73…`) 3 รอบวัดทีละ kernel + 3 รอบวัดแบบรันติดกัน scorer `wb_probe_b8_score.py` (`dcb32d92…`) พร้อม unit test

| ID | เกณฑ์ | ผล |
|---|---|---|
| **Z-0 (ต้องผ่านก่อน)** | anchor/control เหมือน X-0; replication: \|RB(1, 1/8) − 0.085\| ≤ 0.10 | ถ้า sanity ไม่ผ่าน ให้หยุด |
| **Z-1 (ตัวหลัก)** | ใน cell ของ M+W ที่ \|RB_v4 − RB_dvsc\| ≥ 0.15 (7 cell) RB ที่วัดได้ใกล้ dvsc มากกว่าใน ≥ 75 % **และ** MAE ของ RB ของ dvsc บน M+W ≤ 0.15 | dvsc ได้รับการสนับสนุน · ใกล้ v4 ≥ 75 % → LRU · นอกนั้น mixed |
| **Z-2** | RDR ที่ r = 0.9375 (read-only) ทั้งสอง model ทำนาย 0 | ≥ 0.10 → **capacity ที่ใช้ได้จริง < 0.94 L2** (อาจอธิบาย fused ที่ 0.94 ใน B7 ได้) |
| Z-2b | RDR ที่ r = 1.03125 ทั้งสอง model ทำนาย 1 | ≤ 0.90 → read ถูกเก็บไว้บางส่วนแม้เกิน capacity (ไม่ใช่ LRU cliff ที่คม) |
| Z-3 | median WBR ของ M+W (ทั้งสอง model ทำนาย 1) | รายงานผลเท่านั้น |

**คำทำนายของผู้วิจัย:**
- Z-0 ผ่าน
- **Z-1 = dvsc ได้รับการสนับสนุน** (ไม่มั่นใจมากนัก เพราะ model แบบ fully associative ให้รอยต่อที่คมกว่าของจริง)
- Z-2: ผู้วิจัยเดาว่า capacity ที่ใช้ได้จริง < 0.94 (จาก B7) แต่ B6 พบว่า read-only ถึง 0.875 ไม่มี miss จึงไม่มั่นใจ
- Z-2b: ไม่มีคำทำนาย

**ข้อจำกัด:**
- life เป็นค่าที่ fit มาหนึ่งค่า และ model เป็น fully associative (ของจริงเป็น set-associative แบบ hash)
- RB วัดได้เฉพาะในโหมดทีละ kernel
- GPU ตัวเดียว

---

> **ผล (MEASURED 2026-11-05 09:32–09:35 UTC, `benchmarks/results/wb_probe_b8_20261105-093216/`, D-047):**
>
> | กฎ | ผล |
> |---|---|
> | Z-0 | ✅ sanity; replication RB(1, 1/8) = 0.138 (B6c 0.085) ✅ |
> | **Z-1** | ✅ **dvsc ได้รับการสนับสนุน**: ใกล้ dvsc 100 % ของ 7 cell; MAE RB 0.103 vs v4 0.266 |
> | Z-2 | capacity ที่ใช้ได้จริง ≥ 0.94 L2: RDR(0.9375) = 0.005 (**ผู้วิจัยเดาผิด**) |
> | Z-2b | read cliff ไม่คมแบบ LRU: RDR ที่ r = 0.906 / 0.938 / 0.969 / 1.0 / 1.031 / 1.063 ได้ 0.002 / 0.005 / 0.009 / 0.073 / **0.539** / 0.998 (cliff กว้าง ≈ 0.06 L2 รอบ 1.03) |
> | Z-3 | median WBR ของ M+W = 0.993 (แต่ที่ F = 1.0625 ได้ 0.73–0.80) |
>
> **v5 candidate (v4 + dvsc_life 0.125) กับข้อมูล kernel จริงที่เห็นแล้ว (explanatory):**
> - B7: ภายใน 0.1 B **74.3 %** (v4 73.1 %); mean e 0.086 (v4 0.093); edge 0.115 (v4 0.126)
> - B4/B5 (D-038): **ไม่เปลี่ยน** (0.040 / 0.075)
>
> **กลไกนี้มีจริงแต่ไม่ใช่แหล่งหลักของ error ที่เหลือในงานจริง**

## B7r pre-registration (Y-0..Y-3) — เขียน 2026-11-05 ก่อนวัด B7 แบบ range ที่ขนาดเหล่านี้

**คำถาม:** error ของ model v4 กับ kernel จริงตรงขอบ capacity (B7 R-1 ภายใน 0.1 B ได้ 73.1 %) เกิดจาก**วิธีวัด** (ncu วัดทีละ kernel และมีช่วงว่างระหว่าง kernel) หรือจาก **model** เอง (ข้อ ก ใน D-047)

**สิ่งที่เห็นแล้วก่อนเขียนกฎนี้ (exploratory, ข้อมูล B7 แบบทีละ kernel):**
- เมื่อรวม byte ทั้ง segment (การเรียก chain หลัง flush หนึ่งครั้ง) error ยังอยู่ ขนาดใกล้เดิม (เช่น ที่ 0.3125 รวม 0.131 เทียบกับรายตัว 0.142 B/launch) แปลว่า model ผิดที่**ยอดรวม** ไม่ใช่แค่ระบุผิดว่า byte เกิดใน launch ไหน
- error ที่ใหญ่สุดคือ **read**: ที่ 0.5 eager model ทำนายว่าอ่าน 3.0 B/chain แต่วัดได้ 2.0 (t1 และ t2 hit ใน L2 ทั้งหมด) ส่วน write วัดได้**น้อยกว่า** model ในช่วง 0.3125–0.5
- smoke (ขนาด 0.0625 และ 1.5 เท่านั้น ไม่ใช่ขนาดที่ใช้ตัดสิน):
  - ช่วง range ที่ว่างเปล่ามี DRAM read จากเบื้องหลังราว 12 KB/ms แบบเป็นช่วง ๆ (60 ช่วง) และ**ไม่มี write เลย**
  - เวลาส่งคำสั่งของ segment ที่ยาวที่สุด 954 µs น้อยกว่า spin kernel ที่ 1.95 ms

**Design (เปลี่ยนตัวแปรเดียว: replay mode):**
- ใช้ harness `torch_opchain.py` (`9cca5b05…`) ขนาดและ launch เดียวกับ B7: 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 1, 2, 3
- **kernel mode:** `--replay-mode application` ได้ byte ราย launch
- **range mode:** `--replay-mode app-range` ร่วมกับ `--profiler-ranges` ใช้ 1 range ต่อ segment:
  - ใน range ขึ้นต้นด้วย spin kernel 2 M cycle (`torch.cuda._sleep` ไม่แตะหน่วยความจำ) เพื่อให้ CPU ส่งคำสั่งทุก launch ของ segment ไปรอไว้ก่อน GPU จึงรัน kernel ติดกันจริง
  - control range (มีแต่ spin) หนึ่งช่วงต่อขนาด ใช้วัด traffic เบื้องหลัง
- วัดอย่างละ 5 รอบ สลับกัน (K1 G1 K2 G2 …) protocol `b7r_opchain.sh` (`86a4c726…`) scorer `counter_b7r.py` (`b290db3a…`) unit test `test_counter_b7r.py` (`12828dbf…`)
- **คำทำนาย:** ใช้ของ B7 เดิมทุกค่า `b7_predictions.json` (`4505b589…`, model v4) ไม่สร้างใหม่ และ scorer ตรวจว่า plan ของทุกรอบให้คำทำนายเท่ากับที่ล็อกไว้

**นิยาม** (segment s มี n launch, B = byte ของ tensor [M, 4096] หนึ่งตัว):
- P = ผลรวมคำทำนาย v4
- K = median ของผลรวมราย launch ในโหมดทีละ kernel
- G = median ของ (byte ของ range − median ของ control range ในรอบเดียวกัน)
- k = (K − P)/(B n), g = (G − P)/(B n), d = (G − K)/(B n)
- **ขอบ (edge):** l2x ∈ {0.25, 0.3125, 0.375, 0.5, 1} มี 20 segment · **ไกล (far):** l2x ∈ {2, 3}

| ID | เกณฑ์ | ผล |
|---|---|---|
| **Y-0 (ต้องผ่านก่อน)** | (a) plan ทุกรอบให้คำทำนายเท่ากับที่ล็อก + correctness 0 failure (b) median read ของ control ≤ 16 KB และ write สูงสุด ≤ 4 KB (c) เวลาส่งคำสั่งสูงสุด < 0.8 × เวลา spin (d) ที่ far ทุก segment \|d\| ≤ 0.05 ทั้ง read และ write | ไม่ผ่าน → หยุด หาสาเหตุ ไม่ตีความต่อ |
| Y-1 | สัดส่วน segment ที่ edge ที่ \|d_read\| ≤ 0.05 | ≥ 80 % → read ไม่ขึ้นกับ replay mode |
| Y-2 | ρ_w = mean\|g_write\| / mean\|k_write\| ที่ edge | ≤ 0.5 → error ของ write ส่วนใหญ่มาจากวิธีวัด · ≥ 0.8 → ไม่ได้มาจากวิธีวัด · นอกนั้น บางส่วน |
| **Y-3 (ตัวตัดสิน)** | ρ = mean\|g\| / mean\|k\| ที่ edge รวม read และ write (40 ค่า) | ≤ 0.5 → **MEASUREMENT** · ≥ 0.8 → **MODEL** · นอกนั้น **PARTIAL** |
| (descriptive) | สัดส่วน segment-dir ที่ \|g\| ≤ 0.1 เทียบกับ \|k\| ≤ 0.1 | รายงานผลเท่านั้น |

**คำทำนายของผู้วิจัย:**
- Y-0 ผ่าน
- **Y-1: read ไม่ขึ้นกับ replay mode** เพราะช่วงว่างทำให้ line ที่ถูกไล่ออกไปแล้วกลับมา hit ไม่ได้
- **Y-2: ไม่ได้มาจากวิธีวัด** (ρ_w ราว 0.8–1.2) อาจมี write เพิ่มเล็กน้อยใน range mode ถ้า L2 ทยอยเขียนกลับตอน GPU ว่าง
- **Y-3: MODEL**

**ผลที่ตามมา:**
- MEASUREMENT → v4 ใช้ได้ ใช้ range mode เป็นวิธีวัดหลักต่อไป รวมถึงบนเครื่องเช่า
- MODEL → ต้องทำข้อ 2 (set-associativity และลำดับ access จริง) ก่อนจะสรุปว่า model ใช้กับ kernel จริงตรงขอบ capacity ได้
- PARTIAL → รายงานทั้งสองส่วน

**ข้อจำกัด:**
- read ใน range mode มี traffic เบื้องหลังที่ลบออกได้เพียงประมาณ ส่วน write ไม่มี
- ncu ยังหยุดคั่นที่ขอบ range
- spin kernel ทำให้ไม่มีช่วงว่างจาก CPU เลย ซึ่งต่างจาก loop eager จริงที่มีช่วงว่างจากการส่งคำสั่ง ข้อนี้ตั้งใจ เพราะต้องการทดสอบวิธีวัด
- GPU ตัวเดียว

---

> **ผล (MEASURED 2026-11-05 13:04–13:16 UTC, `benchmarks/results/b7r_opchain_20261105-130359/`, D-049):**
>
> - **Y-0 ❌ ไม่ผ่าน (c back-to-back)**
>   - เวลาส่งคำสั่งของ segment ใน run เต็มยาวกว่าใน smoke มาก (median 1.1 ms, สูงสุด 4.2 ms) และมี 6 จาก 180 range ที่ส่งคำสั่งนานกว่า 0.8 × spin (5 ใน 6 เป็น compiled steady)
>   - ช่วงเหล่านั้นจึงอาจมีช่วงว่าง
>   - ข้อ a, b, d ผ่าน: plan ตรง 10/10, background median 10.4 KB, write ของ control = 0, far \|d\| สูงสุด 0.026
> - **ตามกฎจึงหยุดและไม่ตีความ Y-1..Y-3** ตัวเลขที่ scorer คำนวณไว้ (ไม่นับเป็นผล) คือ Y-1 90 %, ρ_w 0.56, ρ 0.76 ซึ่งผู้วิจัยได้เห็นแล้ว
> - **สาเหตุ:** spin มีความยาวคงที่ ขณะที่เวลาส่งคำสั่งของ CPU บน WSL2 ภายใต้ ncu แกว่งมาก
> - **แก้:** เปลี่ยนเป็น gate แบบ flag (B7r2)

---

## B7r2 pre-registration (Y-0..Y-3) — เขียน 2026-11-05 หลัง B7r ไม่ผ่าน Y-0 ก่อนรัน B7r2

**เหมือน B7r ทุกอย่าง** (ขนาด, launch, คำทำนาย `b7_predictions.json` `4505b589…`, นิยาม K/G/P, edge/far, เกณฑ์และค่าคงที่ของ Y-0a, Y-0b, Y-0d, Y-1, Y-2, Y-3) **ยกเว้น:**
- **วิธีกั้น:** ใช้ `--range-gate flag` แทน spin
  - range ขึ้นต้นด้วย gate kernel ที่วนอ่าน flag word จนกว่า CPU จะส่งคำสั่งครบทั้ง segment แล้วปล่อย (fill 1 ค่าบน side stream)
  - kernel จึงรันติดกันจริงไม่ว่า CPU จะช้าแค่ไหน
  - flag word เป็น view ใน flush buffer จึงไม่มีการจองหน่วยความจำใหม่ และ address/plan ไม่เปลี่ยน (scorer ตรวจ)
- **control range:** ถือ gate ไว้ 1.5 ms (≈ median เวลาส่งคำสั่งของ B7r) แล้วปล่อย
- **Y-0c ใหม่:** ไม่มี range ใดที่ gate หมดเวลา (gate_iters < 2,000,000 ≈ 0.3 s)
- **ไฟล์:**
  - harness `torch_opchain.py` (`f89c221b…`)
  - protocol `b7r2_opchain.sh` (`eed125e0…`)
  - scorer `counter_b7r.py` (`e44a01d7…`) เพิ่มเฉพาะทางของ flag gate และยืนยันแล้วว่าให้ผลของ B7r เหมือนเดิมทุกบรรทัด
  - unit test `test_counter_b7r.py` (`a25169f3…`)
- smoke (ขนาด 0.0625/1.5) ด้วย gate: ไม่มี timeout (iters สูงสุด ≈ 20k) และ write ของ control = 0

**คำทำนายของผู้วิจัย (ได้เห็นตัวเลขของ B7r ที่ไม่ผ่าน Y-0 แล้ว รอบนี้จึงเป็นการวัดซ้ำด้วยวิธีที่แก้แล้ว ไม่ใช่การทดสอบแบบไม่รู้ผลมาก่อน):**
- Y-0 ผ่าน
- Y-1 read ไม่ขึ้นกับ replay mode
- Y-2 บางส่วน (ρ_w ราว 0.5–0.6)
- Y-3 PARTIAL (ρ ราว 0.75)

---

> **ผล (MEASURED 2026-11-05 13:27–13:41 UTC, `benchmarks/results/b7r2_opchain_20261105-132701/`, D-050):**
>
> - **Y-0 ❌ ไม่ผ่านอีกครั้ง แต่คนละข้อกับ B7r: ข้อ b background** median read ของ control = 17,152 B เกินเกณฑ์ 16,000 B
>   - ข้อ c ผ่านแล้ว: gate ไม่ timeout เลย (iters สูงสุด 39k จาก 2M) แม้ส่งคำสั่งนานถึง 6.7 ms
>   - ข้อ a, d ผ่าน, write ของ control = 0
> - **ตามกฎจึงไม่ตีความ Y-1..Y-3 เป็นผลทางการ**
> - **Scorer bug (แก้หลังรัน ไม่กระทบเกณฑ์):** การตรวจ layout ไม่ได้ตัด `gate_iters` ซึ่งต่างกันทุกรอบ จึง crash ก่อนให้คะแนน (`score_crashed_layout_check.txt`) แก้บรรทัดเดียว (`e44a01d7` → `333ed46b`) และ B7r ให้ผลเหมือนเดิม
>
> **ตัวเลขที่ scorer คำนวณ (exploratory ไม่ใช่ผลทางการ) สอดคล้องกับ B7r ที่ไม่ผ่านด้วยสาเหตุอื่น:**
>
> | | B7r (ไม่ผ่าน c) | B7r2 (ไม่ผ่าน b) |
> |---|---|---|
> | Y-1 read ที่ \|G − K\| ≤ 0.05 | 90 % | 100 % |
> | ρ_w (write) | 0.56 | 0.53 |
> | ρ (รวม) | 0.76 | 0.76 |
> | eager write ที่ edge mean \|k\| → \|g\| | — | 0.190 → 0.059 |
> | compiled write ที่ edge mean \|k\| → \|g\| | — | 0.086 → 0.087 |
>
> - **อ่านแบบ exploratory:**
>   - error ของ **read** ตรงขอบไม่ได้มาจากวิธีวัด เช่น ที่ 0.5 eager model ทำนาย 3.0 B/chain แต่วัดได้ 2.0 ทั้งสองโหมด จึงเป็น error ของ model
>   - error ของ **write ของ eager** ส่วนใหญ่หายไปเมื่อ kernel รันติดกัน (ที่ 0.375 steady: K 4.20, G 7.43, P 7.46) ncu แบบทีละ kernel จึงนับ write-back ของ chain eager ได้ไม่ครบในช่วงนี้
>   - write ของ compiled ไม่เปลี่ยน
> - ข้อนี้ยังเป็นสมมติฐานที่ต้องทดสอบแบบ pre-register ใหม่ (เช่น probe วัด write-back ตอน GPU ว่างโดยตรง)

---

## B9 pre-registration (I-0..I-3) — เขียน 2026-11-05 ก่อนวัด cell ใดของ B9

**คำถาม:** L2 ของ GA106 เขียน dirty line กลับลง DRAM ระหว่างที่ GPU ว่าง (ไม่มี kernel รัน) หรือไม่ ถ้าใช่ จะอธิบายได้ว่าทำไม ncu แบบทีละ kernel นับ write ของ chain eager ได้น้อยกว่าแบบรันติดกัน (exploratory จาก B7r/B7r2, D-050) เพราะ write-back ที่เกิดในช่วงว่างระหว่าง kernel ไม่ถูกนับให้ kernel ใด

**Design** (`idle_wb_probe.py` `2d695fc5…`, Triton + PyTorch venv):
- cell = (w, idle, T) โดย w ∈ {0.125, 0.25, 0.375} × L2 ซึ่งต่ำกว่า dirty cap ½ ทั้งหมด
  - idle "host": CPU sleep และไม่มี kernel รัน เหมือนช่วงว่างของ ncu ใช้ T ∈ {0, 0.1, 1, 10} ms
  - idle "spin": spin kernel ราว 10 ms ที่ไม่แตะหน่วยความจำ
- ทุก repetition: flush (อยู่นอก range) แล้ว **range A** = write kernel (full line) → sync → idle T จากนั้น **range B** = flush ซึ่งไล่ทุก line ออก line ที่ยัง dirty จึงถูกเขียนกลับในช่วงนี้
- ตรวจ D ทุก repetition (นอก range) control range = sleep 10 ms หลัง flush (2 ช่วงต่อรอบ)
- 5 repetition × 3 รอบ ncu app-range protocol `idle_wb_probe.sh` (`21df26d6…`) scorer `idle_wb_score.py` (`26fb10c6…`) unit test `test_idle_wb_score.py` (`eaeb53b8…`)
- **วัดเฉพาะ write** เพราะ traffic เบื้องหลังเป็น read ล้วน (D-048)

**ตัววัดหลัก:** IWB(w, idle, T) = (B(w, host, 0) − B(w, idle, T)) / byte ที่เขียน คือสัดส่วนที่ถูกเขียนกลับก่อน flush
- ใช้ B แทน A เพราะทนต่อสองเรื่อง:
  - smoke (w = 0.1875, T = 0 เท่านั้น) เห็น flush range นับ write เกินราว 5 % แบบคงที่
  - อาจไม่แน่ใจว่า range นับ traffic ช่วงที่ไม่มี kernel ได้ครบ (control ในช่วง sleep เห็น read เบื้องหลัง 8–41 KB จึงน่าจะนับได้)
- smoke: correctness ผ่าน, A(T = 0) = 0, A+B = 1.05, write ของ control = 0

| ID | เกณฑ์ | ผล |
|---|---|---|
| **I-0 (ต้องผ่านก่อน)** | (a) correctness 0 failure (b) write ของ control สูงสุด ≤ 4 KB (c) ที่ T = 0 ทุก w: 0.95 ≤ A+B ≤ 1.15 | ไม่ผ่าน → หยุด |
| I-1 | A(w, host, 0) ≤ 0.05 ทุก w | ไม่มี write-back ระหว่างเขียนใต้ cap (ทำซ้ำ B6) |
| **I-2 (ตัวหลัก)** | IWB(w, host, 10 ms) | ≥ 0.5 ทุก w → **มี idle write-back** · ≤ 0.1 ทุก w → **ไม่มี** · นอกนั้น บางส่วน/ขึ้นกับขนาด |
| I-3 | เส้น IWB ตาม T, spin เทียบ host, A เทียบ IWB | รายงานผลเท่านั้น |

**คำทำนายของผู้วิจัย:**
- I-0 ผ่าน
- I-1 ผ่าน
- **I-2 = มี idle write-back** ความมั่นใจปานกลาง จากรูปแบบใน B7r/B7r2 และผลที่ขึ้นกับช่วงว่างใน B6 W-5
- ไม่มีคำทำนายเรื่องค่าคงที่เวลา หรือเรื่อง spin เทียบ host

**ถ้า I-2 = ไม่มี:** สมมติฐานว่า ncu นับ write ไม่ครบเพราะช่วงว่างจะตกไป ความต่างระหว่าง K และ G ใน B7r/B7r2 ต้องมีสาเหตุอื่น เช่น ncu ทำบางอย่างกับ cache ระหว่าง kernel

**ข้อจำกัด:**
- GPU ตัวเดียว (laptop WSL2)
- "ว่าง" ของ host sleep อาจต่างจากช่วงว่างจริงของ ncu เช่น clock อาจลดลง

---

> **ผล (MEASURED 2026-11-05 13:57–14:00 UTC, `benchmarks/results/idle_wb_20261105-135723/`, D-052):**
>
> | กฎ | ผล |
> |---|---|
> | I-0 | ✅ correctness 0 failure, write ของ control = 0, A+B ที่ T = 0 = 1.020 / 1.033 / 1.009 |
> | I-1 | ✅ A(T = 0) = 0.000 ทุก w: ไม่มี write-back ระหว่างเขียนใต้ cap (ทำซ้ำ B6) |
> | **I-2** | ❌ **ไม่มี idle write-back** IWB(10 ms) = −0.129 / −0.026 / −0.034 (**ผู้วิจัยทำนายผิด**) |
>
> **I-3 (descriptive):**
> - dirty line อยู่ใน L2 ครบหลังว่าง 0.1, 1 และ 10 ms ทั้งแบบ host idle และแบบ spin (A ≈ 0, B ≈ 1)
> - IWB ติดลบเล็กน้อยและโตตาม T: flush ที่ตามหลังช่วงว่างนานเขียนเพิ่ม 20–85 KB (แบบ spin มากกว่า)
>   - ข้อสันนิษฐาน (ยังไม่ทดสอบ): client อื่นบน GPU (ฝั่ง Windows) เขียนข้อมูลค้างไว้ใน L2 ระหว่างที่เราว่าง แล้ว flush ของเราไล่ออก
>
> **ผลต่อคำอธิบายของ B7r/B7r2:**
> - การที่ ncu แบบทีละ kernel นับ write ของ eager น้อยกว่าแบบรันติดกัน (exploratory) **ไม่ได้เกิดจากการเขียน dirty line ที่ค้างอยู่ตอนว่าง**
> - คำอธิบายที่เหลือ (ยังไม่ทดสอบ): write-back ที่**ค้างคิวอยู่ตอน kernel จบ** (เช่น dirty victim ใน write-back buffer ตาม dvsc ของ B8) ถูกเขียนหลัง kernel จบ จึงไม่ถูกนับให้ kernel ใดในโหมดทีละ kernel
> - B9 ทดสอบเฉพาะ dirty line ที่อยู่นิ่งใต้ cap ไม่ได้ทดสอบคิวที่เกิดจากแรงบีบ

---

## B9b pre-registration (Q-0..Q-3) — เขียน 2026-11-05 ก่อนวัด cell ใดของ B9b

**คำถาม:** ncu แบบทีละ kernel ทำให้ write-back บางส่วน "หาย" (ไม่ถูกนับให้ kernel ใด) เมื่อข้อมูลเกิน dirty cap หรือไม่ ข้อนี้เป็นคำอธิบายที่เหลือของรูปแบบใน B7r/B7r2 หลังจาก B9 ตัดเรื่อง idle write-back ทิ้งไปแล้ว

**Design** (`queued_wb_probe.py` `b130d1c0…`):
- ใช้ write kernel และ flush ตัวเดียวกับ B9 ที่ w ∈ {0.375 (control ใต้ cap), 0.75, 1.0, 1.5} × L2
- ทุก repetition ทำคู่ (write, flush) หลัง flush สองครั้ง
  - **range mode:** split = range A [write] + range B [flush], joint = range C [write; flush] รันติดกันในช่วงเดียว ใช้เป็นค่าอ้างอิง
  - **kernel mode:** ทั้งสองคู่วัดทีละ kernel ได้ W (write kernel) และ F (flush ถัดไป)
- 5 repetition × 3 รอบต่อโหมด สลับกัน protocol `queued_wb_probe.sh` (`aaab7fe3…`) scorer `queued_wb_score.py` (`69fd4749…`) unit test `test_queued_wb_score.py` (`ac07a21a…`)
- **คำทำนายของ v4** (`cache_sim_v4`, ไม่มีพารามิเตอร์ใหม่): A_v4 = 0 / 0.333 / 0.5 / 0.667 ที่ w = 0.375 / 0.75 / 1 / 1.5
- smoke ที่ w = 0.5625 (ไม่ใช่ cell ที่ให้คะแนน):
  - A = 0.11–0.13 และ W = 0.12 (v4 0.111)
  - S_c 1.00–1.02, S_r 0.97–0.98, S_k 0.94–1.01 ต่อคู่ (F แกว่ง ±0.04)

**นิยาม** (normalize ด้วย byte ที่เขียน ใช้ median ของทุก repetition ทุกรอบ): S_k = W + F, S_r = A + B, S_c = C

| ID | เกณฑ์ | ผล |
|---|---|---|
| **Q-0 (ต้องผ่านก่อน)** | correctness 0 failure, write ของ control ≤ 4 KB, ที่ w = 0.375: S_k, S_r, S_c ∈ [0.95, 1.15] | ไม่ผ่าน → หยุด |
| **Q-1 (ตัวหลัก)** | L = S_c − S_k ที่ w ∈ {0.75, 1, 1.5} | ≥ 0.05 ทุก w → **โหมดทีละ kernel ทำ write-back หาย** · ≤ 0.02 ทุก w → ไม่หาย · นอกนั้น mixed |
| Q-2 | A − W ที่ w เหนือ cap | ≥ 0.05 ทุก w → หน้าต่างของ kernel นับ write-back ของตัวเองไม่ครบ · ≤ 0.02 → นับครบ · นอกนั้น mixed |
| Q-3 | \|A − A_v4\| ≤ 0.10 ทุก w เหนือ cap | v4 ทำนาย write-back ของ write kernel หนึ่งตัวได้ |

**คำทำนายของผู้วิจัย:**
- Q-0 ผ่าน
- **Q-1 = หาย** ความมั่นใจต่ำ: smoke ที่ 0.5625 ไม่เห็นชัด แต่ B7r/B7r2 เห็นที่ footprint ใหญ่กว่า
- Q-2 = นับครบ ตาม smoke
- Q-3 ผ่าน

**ถ้า Q-1 = ไม่หาย:** ความต่างของ write ระหว่างสองโหมดใน B7r/B7r2 ไม่ได้มาจาก write kernel เดี่ยว แต่ต้องมาจากปฏิสัมพันธ์ระหว่าง kernel ใน chain เช่น การ read ของ kernel ถัดไปไปทำให้ dirty line ถูกไล่ออกต่างกัน ซึ่งจะเป็นงานของ model ไม่ใช่ของวิธีวัด

---

> **ผล (MEASURED 2026-11-05 14:06–14:10 UTC, `benchmarks/results/queued_wb_20261105-140557/`, D-054):**
>
> |  w | W (ทีละ kernel) | A (range) | v4 | F | B | S_k | S_r | S_c |
> |---|---|---|---|---|---|---|---|---|
> | 0.375 | 0.000 | 0.000 | 0.000 | 1.020 | 1.011 | 1.020 | 1.011 | 1.003 |
> | 0.75 | **0.248** | **0.328** | 0.333 | 0.670 | 0.664 | **0.916** | 0.999 | 1.005 |
> | 1.0 | **0.367** | **0.420** | 0.500 | 0.575 | 0.577 | **0.936** | 1.000 | 1.003 |
> | 1.5 | 0.686 | 0.685 | 0.667 | 0.313 | 0.313 | 0.999 | 1.000 | 1.001 |
>
> | กฎ | ผล |
> |---|---|
> | Q-0 | ✅ |
> | **Q-1** | **mixed:** S_c − S_k = +0.089 / +0.067 / +0.001 ที่ w = 0.75 / 1 / 1.5 คือหายที่ 0.75 และ 1 แต่ไม่หายที่ 1.5 (ผู้วิจัยทำนายว่าหายทุก w จึงถูกบางส่วน) |
> | **Q-2** | **mixed:** A − W = +0.080 / +0.053 / −0.001 ส่วนที่หายอยู่ในหน้าต่างของ write kernel เอง (ผู้วิจัยทำนายว่านับครบ จึงผิด) |
> | Q-3 | ✅ \|A − A_v4\| ≤ 0.1 (0.328 vs 0.333, 0.420 vs 0.500, 0.685 vs 0.667) |
>
> **สิ่งที่สรุปได้ (GA106):**
> - เมื่อ kernel หนึ่งตัวดัน dirty footprint ข้าม cap ไป write-back ส่วนหนึ่ง (5–9 % ของ byte ที่เขียน ที่ w = 0.75–1) **เสร็จหลัง kernel จบ**
> - ncu แบบทีละ kernel จึงไม่นับ byte เหล่านี้ให้ kernel ใดเลย
>   - F ของ flush ถัดไปเท่ากับ B ของ range mode พอดี แปลว่า byte เหล่านี้ไม่ได้ไปโผล่ใน kernel ถัดไป
>   - แบบ range นับครบ (S_r ≈ S_c ≈ 1.00)
> - ที่ w = 1.5 ไม่มีส่วนที่หาย
> - **ข้อควรระวังเชิงวิธีวัด:** ผลรวม `dram__bytes_write.sum` ราย kernel ของ ncu ต่ำกว่าจริงสำหรับ kernel ที่ข้าม dirty cap ซึ่งเข้ากับรูปแบบ exploratory ของ eager chain ใน B7r/B7r2
> - model v4 ทำนาย byte ของ write kernel เดี่ยวได้ภายใน 0.1 (ที่ w = 1 ต่างที่สุด 0.08 เข้ากับ cap จริงที่ราว 0.5–0.56 จาก B6)

---

## B10 pre-registration (P-0, P-1, T-6b, T-8, T-9b) — เขียน 2026-11-06 ก่อนเช่าเครื่องและก่อนวัด cell ใดบน GA102

**คำถาม:**
1. กลไกที่วัดได้บน GA106 ถ่ายทอดไปยัง GA102 (RTX 3090, L2 6 MiB, bus 384-bit) หรือไม่ ได้แก่
   - dirty cap ที่ ½ L2 ซึ่งนับเป็น line
   - dvsc (dirty victim ที่ถูกเขียนกลับยังอ่านได้)
   - read cliff ที่ไม่คม
   - write-back ที่หลุดจากหน้าต่าง ncu แบบทีละ kernel
2. VM บน vast.ai อ่าน hardware counter ได้หรือไม่ (container บน vast.ai ได้ `ERR_NVGPUCTRPERM` ทุกครั้ง)

**ทำไมต้องเป็น GA102:**
- ใช้ architecture เดิม (Ampere, sm_86) และคาดว่ามี L2 ต่อ memory controller เท่ากัน (512 KiB × 12 เทียบกับ × 6 บน GA106)
- ถ้าผลต่างจาก GA106 แปลว่ากลไกผูกกับชิปตัวนั้น ไม่ใช่ของ Ampere ทั้งรุ่น
- เป็นเครื่อง VM ราคาถูกที่สุดที่เช่าได้ (มี offer แค่ 2 รายการ, D-056)

**สิ่งที่เหมือน GA106 ทุกประการ** (sha256 ตรงกับที่บันทึกไว้ใน `environment.txt` ของรอบ GA106):

| ส่วน | ไฟล์ (sha256) |
|---|---|
| source | `wb_probe.cu` `7dbbd451…`, `common.cuh` `7a5ea8b8…`, `dmk_host.hpp` `5bcd6f86…` |
| protocol | `wb_probe_b6b.sh` `0dc645e8…`, `wb_probe_b8.sh` `96049e73…`, `queued_wb_probe.sh` `aaab7fe3…`, `queued_wb_probe.py` `b130d1c0…` |
| scorer | `wb_probe_b6b_score.py` `8f7092ba…`, `wb_probe_b8_score.py` `dcb32d92…`, `queued_wb_score.py` `69fd4749…` |
| คำทำนายของ model | `b8_predictions.json` `8aafe4da…` |

- cell ทุกตัวเป็นสัดส่วนของ L2 และ probe อ่านขนาด L2 จาก device เอง (`cudaDeviceProp` / `torch`)
- toolchain เป็น nvcc 12.5 ที่ใช้ flag เดิม และ ncu ของ CUDA 12.5 (laptop ใช้ V12.5.82 และ ncu 2024.2.1)
- ต้อง build บน VM เพราะ binary ของ laptop ต้องใช้ GLIBC 2.38 (D-055)

**คำทำนายของ B8 ใช้กับ L2 6 MiB ได้โดยไม่ต้องคำนวณใหม่:**
- ทำ check แบบ exploratory ในเครื่องก่อนเขียนเอกสารนี้: simulate v4 และ dvsc ใหม่ด้วย `L2_BYTES = 6291456`
- ได้ max |Δ| เทียบกับไฟล์ที่ล็อกไว้ = **0.0002** ใน 21 cell
- scorer ของ B8 จึงใช้ไฟล์เดิม ส่วน B9b อ่าน `l2_bytes` จาก meta อยู่แล้ว

**สิ่งที่เปลี่ยนพร้อมกัน (ไม่ใช่ ablation ตัวแปรเดียว):**
- die (GA102 / GA106)
- ขนาด L2 (6 / 3 MiB)
- memory (GDDR6X 936 GB/s / GDDR6 laptop)
- OS (Linux ใน VM / WSL2)
- driver (610.57 ของ host / 591.74)
- **ถ้า verdict ต่าง จะระบุสาเหตุเป็นตัวแปรเดียวไม่ได้**

**ตัว wrapper:**
- wrapper `remote_vm_pilot.sh` แก้ก่อนล็อกเพื่อให้รันได้แม้ไม่ใช่ root (เรียก ncu ผ่าน sudo), ประหยัด disk และ dump SASS ของ binary B6 จาก laptop
- ไม่ได้แตะ protocol, scorer หรือ source

| ID | เกณฑ์ | คำทำนายของผู้วิจัย |
|---|---|---|
| **P-0 (ต้องผ่านก่อน, เชิงปฏิบัติการ)** | ncu บน VM ไม่ได้ `ERR_NVGPUCTRPERM` และ `correctness_failures` = 0 ทุกรอบของ B6b/B8 | **ผ่าน**: ใน VM ระบบ guest คุม driver เองและรัน ncu เป็น root ได้ ถ้าไม่ผ่าน: บันทึกว่า "VM ไม่ช่วยเรื่อง counter" แล้วหยุด และไม่ลอง host อื่นในงบนี้โดยไม่ขอ PI |
| P-1 (ตัวตนของโค้ด) | SASS ของ `rw_kernel`, `readback_kernel`, `flush_l2_read_kernel` ใน build ของ VM ตรงกับ binary B6 ของ laptop (`e0c82a53…`) ทุก instruction (ไม่นับ address) | **ตรง** (nvcc 12.5 และ flag เดิม) ถ้าไม่ตรง: รายงานผลว่า "source เดียวกัน แต่ SASS ต่าง" |
| **T-6b** | ให้คะแนนด้วย scorer ของ B6b โดยไม่แก้อะไร | V-0 ผ่าน · **V-2 = H-D · V-3 = คม · V-4 = นับเป็น line · V-5 = ยังอยู่ใน cache** (ความมั่นใจปานกลางถึงสูง เพราะ Finn 2026 รายงาน rule ครึ่งหนึ่งของ way บน AD102 ซึ่งเป็นอีกรุ่น) · V-1 (ค่าสัมบูรณ์ของ B6 บน GA106): **ความมั่นใจต่ำ** เพราะ w = 0.5 อยู่บนขั้นบันไดพอดี (GA106 = 0.194) ถ้าตำแหน่ง cap ของ GA102 ต่างไปราว 0.03 L2 ก็ไม่ผ่านได้ |
| **T-8** | ให้คะแนนด้วย scorer ของ B8 และ `b8_predictions.json` เดิม | Z-0 ผ่าน · **Z-1 = dvsc ได้รับการสนับสนุน** (ความมั่นใจต่ำถึงปานกลาง: life 0.125 fit มาจาก GA106 ถ้าคิว write-back มีขนาดต่อ slice เท่าเดิม สัดส่วนจะเท่าเดิม) · **Z-2 = capacity ≥ 0.94 L2** · **Z-2b = cliff ไม่คม** (RDR(1.031) ≤ 0.90) |
| **T-9b** | ให้คะแนนด้วย scorer ของ B9b | Q-0 ผ่าน · **Q-1 = mixed รูปแบบเดิม** (L ≥ 0.05 ที่ w = 0.75 และ 1, ≤ 0.02 ที่ 1.5) · Q-2 = mixed · **Q-3 ผ่าน** ความมั่นใจต่ำ เพราะขนาดของส่วนที่หายขึ้นกับความลึกของคิวเทียบกับความเร็วที่ DRAM ระบายออก ซึ่งต่างกันบน GA102 |

**Verdict รวมของ B10 (ล็อก):** นับ verdict ของ 9 กฎหลัก {V-2, V-3, V-4, V-5, Z-1, Z-2, Z-2b, Q-1, Q-3} ที่ตรงกับ GA106 (V-1 รายงานแยกเป็นการ replicate ค่าสัมบูรณ์)
- **ตรงทั้ง 9** → กลไกถ่ายทอดไปยัง GA102
- **V-2, V-3 หรือ V-4 ข้อใดข้อหนึ่งไม่ตรง** → rule ของ dirty cap ผูกกับ GA106 (negative ที่สำคัญ ต้องแก้ claim C5)
- นอกนั้น → ถ่ายทอดได้บางส่วน รายงานรายกฎ
- **คำทำนายของผู้วิจัย:** ตรงทั้ง 9 แต่ไม่มั่นใจที่ Z-1 และ Q-1

**เครื่องและงบ (D-051, D-056):**
- offer 59353127 (RTX 3090 VM, Japan, $0.241/ชม., disk 28 GB, ไม่คิดค่า bandwidth)
- `HARD_CAP_S` = 5400 s จึงจ่ายไม่เกินราว $0.37
- destroy ทันทีหลังดาวน์โหลดผลแล้วยืนยันว่า destroy จริง
- รายงานค่าใช้จ่ายจากผลต่างของเครดิต
- image ระบุเอง (`docker.io/vastai/kvm:ubuntu_cli_22.04-2025-05-16`) ไม่ใช้ template

**ข้อจำกัด:**
- การ์ดเดียว host เดียว ไม่ได้ล็อก clock
- virtualization อาจกระทบเวลา แต่ไม่ควรกระทบจำนวน byte
- ไม่รู้ associativity และ hash ของ slice บน GA102
- Linux native จะไม่มี DRAM read เบื้องหลังจาก Windows คาดว่า floor ของ range mode จะต่ำกว่า laptop (เป็นข้อสังเกต ไม่ใช่กฎ)
- ยังไม่ได้รัน smoke บน GA102 ก่อนล็อก เพราะยังไม่มีเครื่อง

---

> **ผล (MEASURED 2026-11-06 03:53–03:58 UTC, RTX 3090 VM บน vast.ai, `benchmarks/results/b10_rtx3090_vm_20261106-0353/`, D-057):**
>
> **เครื่อง:**
> - GA102, 82 SM, L2 6,291,456 B, 384-bit
> - guest driver 575.51.03, Ubuntu 22.04.5
> - nvcc V12.5.82, ncu 2024.2.1
>
> **ความน่าเชื่อถือของการให้คะแนน:** ให้คะแนนบน VM แล้วให้คะแนนซ้ำในเครื่องด้วย scorer เดิม ได้ผลตรงกันทุกบรรทัด
>
> | กฎ | GA106 (อ้างอิง) | GA102 (B10) | ตรงกัน? |
> |---|---|---|---|
> | **P-0** | – | ✅ ncu อ่าน counter ได้ในฐานะ root แม้ `RmProfilingAdminOnly: 1` และ correctness 0 failure ทุกรอบ | ทำนายถูก |
> | **P-1** | – | ✅ SASS ของ `rw_kernel` / `readback_kernel` / `flush_l2_read_kernel` (216 / 176 / 48 instruction) ตรงกับ binary B6 ของ laptop ทั้งข้อความและ encoding | ทำนายถูก |
> | V-0 | ✅ | ✅ | ✓ |
> | V-1 | ✅ 0.194 / 0.991 | ✅ 0.149 / 0.974 | ✓ |
> | **V-2** | H-D | **H-D** (Δ −0.031 ถึง +0.036) | ✓ |
> | **V-3** | คม (0.014 / 0.886) | **partial** (0.009 / **0.725**) | **✗** |
> | **V-4** | นับเป็น line (0.994) | **ไม่ตีความ: control ไม่ผ่าน** WBR(0:0.375) = 0.143 | **✗** |
> | **V-5** | ยังอยู่ใน cache | ยังอยู่ใน cache (RB/WB 0.001) | ✓ |
> | Z-0 | sanity ✅ และ replication 0.138 ✅ | sanity ✅ แต่ replication ❌ RB(1, 1/8) = **0.452** | sanity ✓ |
> | **Z-1** | dvsc | **dvsc** (ใกล้ dvsc 86 % ของ cell, MAE 0.126 เทียบกับ v4 0.257) | ✓ |
> | **Z-2** | ≥ 0.94 | ≥ 0.94 (RDR 0.009) | ✓ |
> | **Z-2b** | ไม่คม (0.539) | ไม่คม (0.542) | ✓ |
> | Z-3 | 0.993 | 0.963 | (รายงานผลเท่านั้น) |
> | Q-0 | ✅ | ✅ | ✓ |
> | **Q-1** | mixed (+0.089 / +0.067 / +0.001) | **mixed** (+0.144 / +0.092 / **+0.035**) | verdict ✓ แต่รูปแบบที่ทำนายไว้ (≤ 0.02 ที่ w = 1.5) ผิด |
> | Q-2 | mixed | mixed (+0.061 / +0.007 / +0.000) | ✓ |
> | **Q-3** | ✅ | ✅ (A 0.323 / 0.441 / 0.680 เทียบกับ v4 0.333 / 0.500 / 0.667) | ✓ |
>
> **Verdict รวมตามกฎที่ล็อก: ตรง 7 จาก 9 ข้อหลัก**
> - V-3 และ V-4 ไม่ตรง จึงเข้ากรณี **"rule ของ dirty cap ผูกกับ GA106"** และต้องแก้ claim C5
> - ผู้วิจัยทำนายว่าตรงทั้ง 9 ข้อจึงผิด และผิดในข้อที่มั่นใจ (V-3, V-4) ส่วนข้อที่ไม่มั่นใจ (Z-1, Q-1) กลับตรง
>
> **ถ่ายทอดไปยัง GA102 ได้:**
> - write-back ขึ้นกับ dirty footprint อย่างเดียว (V-2)
> - line ที่ถูกเขียนกลับยังอยู่ใน cache (V-5)
> - dvsc (Z-1)
> - read cliff รอบ 1.03 L2: RDR ที่ r = 0.906…1.063 ได้ 0.005 / 0.009 / 0.025 / 0.053 / 0.542 / 0.995 (GA106 0.002 / 0.005 / 0.009 / 0.073 / 0.539 / 0.998)
> - v4 ทำนาย write-back ของ write kernel เดี่ยวได้ (Q-3)
> - ncu แบบทีละ kernel นับ write-back ไม่ครบ (Q-1, Q-2)
>
> **ไม่ถ่ายทอดตามเกณฑ์:**
> - ความคมของขั้นบันได: WBR ที่ w = 0.4375 / 0.5 / 0.5625 / 0.625 / 0.6875 ได้ 0.009 / 0.149 / 0.725 / 0.974 / 0.979 (GA106 0.014 / 0.194 / 0.886 / 0.991 / 1.000)
> - ขั้นยังอยู่ในช่วง 0.44–0.63 L2 แต่ที่ 0.5625 ต่ำกว่าเกณฑ์ 0.80
>
> **ทดสอบไม่ได้:**
> - คำถามว่า cap นับเป็น line หรือเป็น byte ทดสอบไม่ได้ เพราะ control ของ sector ไม่ผ่าน
> - median ของ cell ที่ active ได้ 0.954 แต่ไม่ตีความตามกฎ
>
> **ข้อสังเกต (exploratory หลังเห็นผล ไม่เปลี่ยน verdict):**
> 1. **cell inactive ที่ WBR สูงผิดปกติเป็น cell แรกของ process ทั้งคู่:**
>    - full 0:0.1875 ได้ 0.031 และ sector 0:0.375 ได้ 0.143 (≈ 82 KiB ต่อ range) ขณะที่ WB แบบทีละ kernel ≈ 0 (0.000, 0.005)
>    - cell inactive อื่นได้ ≤ 0.004
>    - control ของ V-4 บน GA106 ที่ผ่านแบบเฉียด (0.042) ก็เป็น cell แรกเช่นกัน
>    - **สมมติฐาน:** write-back จากช่วงเตรียม buffer ค้างคิวแล้วไปเสร็จใน range แรก ทดสอบได้ด้วยการใส่ cell หลอกไว้หน้าสุด
> 2. **B9b บน GA102 range แยกก็นับไม่ครบ:**
>    - S_c − S_r = +0.082 / +0.088 / +0.039 ที่ w = 0.75 / 1 / 1.5 (GA106 ≈ 0)
>    - write-back ส่วนหนึ่งเสร็จในช่วงว่างระหว่างสอง range
>    - ข้อแนะนำ "ใช้ app-range" จึงต้องเข้มขึ้นเป็น "range ต้องครอบ kernel ถัดไปด้วย"
> 3. **RB(1, 0.125) = 0.452** (GA106 0.138, B6c 0.085) แปลว่า dvsc ที่ cell นี้อ่อนกว่าบน GA106
> 4. **VM:** เครดิตลดลงเท่ากับช่วงที่ VM อยู่ในสถานะ running (≈ 10 นาที) จึงดูเหมือนว่าช่วง loading ≈ 15 นาทีไม่ถูกคิดเงิน (อนุมานจากเครดิต)

---

## B11 pre-registration: ข้าม architecture (Turing, Ada, Blackwell) + B10b first-cell — เขียน 2026-11-06 ก่อนเช่าเครื่องใด

**คำถาม (PI, D-059):** การวัดที่ไม่ข้าม architecture จะเป็นงานวิจัยได้จริงไหม? ผลที่วัดด้วย counter ทั้งหมดตอนนี้มาจาก Ampere (GA106, GA102) รอบนี้ถามว่า
1. dirty cap ที่ ≈ ½ L2 (C5 ฉบับแคบ) เกิดบน architecture อื่นไหม
2. ncu ราย kernel นับ write-back ขาด (C6) เกิดบน architecture อื่นไหม
3. control ของ V-4 ที่ไม่ผ่านบน GA102 เป็นเพราะ "cell แรกของ process" หรือไม่ (B10b)

**GPU** (VM บน vast.ai PI อนุมัติ GPU และงบ $1 แล้ว):

| Architecture | GPU | die (คาด) | L2 (คาด, probe อ่านค่าจริงเอง) |
|---|---|---|---|
| Turing (sm_75) | RTX 2060S | TU106 | 4 MiB |
| Ada (sm_89) | RTX 4060 Ti หรือ RTX 4080S (ตัวที่ถูกกว่าตอนเช่า) | AD106 / AD103 | 32 / 64 MiB |
| Blackwell (sm_120) | RTX 5060 Ti | GB206 | 32 MiB |

**สิ่งที่เหมือน B10 ทุกประการ:**
- source, protocol, scorer และ `b8_predictions.json` ใช้ hash เดียวกับใน `LOCK_20261106_b10.sha256` และ cell ทุกตัวเป็นสัดส่วนของ L2
- เช็ค scale ซ้ำ: simulate คำทำนาย B8 ที่ L2 = 4 / 32 / 64 MiB ได้ max |Δ| เทียบกับไฟล์ที่ล็อก = 0.0001 / 0.0003 / 0.0003 จึงใช้ไฟล์เดิม

**สิ่งที่เปลี่ยน (นอกจาก architecture):**
- **ขนาด L2 และชนิด memory:** เปลี่ยนตาม GPU
- **Blackwell ต้องใช้ toolchain ใหม่:** CUDA 12.9 (nvcc ของ 12.9 และ ncu 2025.2.x จากแพ็กเกจ `cuda-nsight-compute-12-9`) และ torch 2.7.1+cu128 เพราะ CUDA 12.5 build sm_120 ไม่ได้
- GPU อื่นใช้ CUDA 12.5, ncu 2024.2.1 และ torch 2.7.1+cu126 เหมือน B10
- ถ้า Blackwell ให้ผลต่าง จะแยกผลของ architecture ออกจากผลของ ncu ไม่ได้
- **B10b เพิ่มเข้ามา:** `wb_probe_firstcell.sh` คือ sector set ของ B6b ที่ใส่ cell หลอก 0:0.25 ไว้หน้าสุด เปลี่ยนตัวแปรเดียว ใช้ binary และ flag เดิม
  - scorer: `wb_probe_firstcell_score.py` (กฎ F-0..F-2, V-4')
- **wrapper:** `remote_vm_pilot.sh` เลือก toolchain ตาม `CUDA_ARCH` และรัน B10b หลัง B8

**กฎตัดสิน** (ต่อ GPU ล็อกไว้ใน `b11_summary.py`, `wb_probe_firstcell_score.py` และ scorer เดิม):

| ID | เกณฑ์ |
|---|---|
| **P-0 (ต้องผ่านก่อน)** | อ่าน counter ได้ (B6b และ B8 ให้ score) และ `correctness_failures` = 0 ทุกรอบ ถ้าไม่ผ่าน GPU นั้นไม่นับใน verdict ข้ามรุ่น |
| match | จำนวน verdict หลัก 9 ข้อ (V-2, V-3, V-4, V-5, Z-1, Z-2, Z-2b, Q-1, Q-3) ที่ตรงกับ GA106 (รายงานผล) |
| **C5n** | B6b full set (range): WBR(0:0.4375) ≤ 0.10 **และ** WBR(0:0.625) ≥ 0.80 → "มี cap (เปลี่ยนในช่วง 0.44–0.63 L2)" · WBR(0:0.6875) < 0.50 → "ไม่มี cap ถึง 0.69 L2" · นอกนั้น other |
| **C6** | B9b: L = S_c − S_k ที่ w = 0.75 ≥ 0.05 → มี · ≤ 0.02 → ไม่มี · นอกนั้น inconclusive |
| F-0 | B10b: correctness 0 และ sector 0:2 WB และ WBR ∈ [0.9, 1.1] |
| F-1 | WBR(0:0.375) เมื่อเป็น cell ที่สอง: < 0.05 → สนับสนุน artifact · ≥ 0.10 → control active เอง · นอกนั้น inconclusive |
| F-2 | WBR ของ cell หลอก (ซึ่งเป็น cell แรก) ≥ 0.05 → มี inflation ที่ cell แรก (ใช้สนับสนุน ไม่ใช้ตัดสิน) |
| **FC** | รวมค่า c1 (control เป็น cell แรก, จาก B6b) กับ c2 (เป็น cell ที่สอง, จาก F-1): c1 < 0.05 → ไม่มีอะไรให้ทดสอบ · c1 ≥ 0.05 และ c2 < 0.05 → **artifact ยืนยัน** · c2 ≥ 0.10 → ไม่ใช่ artifact · นอกนั้น inconclusive |
| V-4' | กฎ V-4 ของ B6b ใช้กับข้อมูล B10b |

**Verdict ข้ามรุ่น (ล็อก):** นับเฉพาะ GPU ใหม่ที่ผ่าน P-0 ส่วน GA106 และ GA102 ไม่นับ
- **C5n-X:** มี cap บน ≥ 2 GPU → **generalizes** (ข้าม ≥ 3 architecture รวม Ampere)
- C5n-X: มีบน 0 GPU (จากที่ valid ≥ 2) → **Ampere-specific**
- C5n-X: นอกนั้น mixed และถ้า valid น้อยกว่า 2 → insufficient
- **C6-X:** ใช้เกณฑ์เดียวกันกับ C6

**ที่มาของเกณฑ์:** C5n คือ claim C5 ฉบับแคบตาม D-057 เกณฑ์ตั้งหลังเห็นข้อมูล GA106/GA102 (0.014 / 0.991 และ 0.009 / 0.974) แต่ตั้งก่อนเห็นข้อมูลของ GPU ใหม่ใด ๆ

**คำทำนายของผู้วิจัย:**

| กฎ | Turing (2060S) | Ada | Blackwell (5060 Ti) |
|---|---|---|---|
| P-0 | ผ่าน | ผ่าน | ผ่าน (ความมั่นใจปานกลาง: ncu 2025.2 กับ GB206) |
| C5n | มี (**ต่ำ**: ไม่มีหลักฐานภายนอกบน Turing) | มี (ปานกลาง–สูง: Finn 2026 บน AD102) | มี (ปานกลาง) |
| C6 | มี (ต่ำ) | มี (ปานกลาง) | มี (ปานกลาง) |
| V-2 / V-5 | H-D / ยังอยู่ใน cache (ต่ำ) | H-D / ยังอยู่ใน cache (ปานกลาง) | H-D / ยังอยู่ใน cache (ต่ำ–ปานกลาง) |
| V-3 | ไม่ทำนาย | ไม่ทำนาย | ไม่ทำนาย |
| V-4' | นับเป็น line (ต่ำ) | นับเป็น line (ปานกลาง: rule ของ Finn นับ line) | นับเป็น line (ต่ำ) |
| Z-1 | ไม่ทำนาย | dvsc (ปานกลาง: Finn บอกว่า "stays readable") | ไม่ทำนาย |
| Z-2 / Z-2b | ≥ 0.94 / ไม่คม (ต่ำ) | ≥ 0.94 / ไม่คม (ปานกลาง) | ≥ 0.94 / ไม่คม (ต่ำ) |
| Q-3 | ผ่านถ้ามี cap | ผ่าน | ผ่าน |
| FC | ถ้า control ของ B6b ไม่ผ่าน → artifact ยืนยัน (ต่ำ–ปานกลาง) | เช่นเดียวกัน | เช่นเดียวกัน |
| **verdict ข้ามรุ่น** | **C5n-X = generalizes, C6-X = generalizes** | | |

**ถ้า C5n-X = Ampere-specific:**
- เป็น negative ที่สำคัญ: rule ½ L2 ที่ Finn รายงานบน Ada จะขัดกับผลของเรา ต้องตรวจวิธีวัดก่อนสรุป
- งานของเราจะกลายเป็น characterization ของ Ampere

**เครื่องและงบ:**
- เช่า 3 เครื่องพร้อมกัน แต่ละเครื่องใช้ `HARD_CAP_S` = 5400 s
- ใช้ offer ที่ verified, `vms_enabled=true` และ ≤ $0.35/ชม.
- ถ้าช่วงโหลดก็ถูกคิดเงิน กรณีแย่สุดราว $0.75 ยังอยู่ในเพดาน $1
- **ไม่เปลี่ยน GPU เอง:** ถ้ารุ่นที่อนุมัติไม่มี offer ให้ข้ามรุ่นนั้นแล้วรายงาน PI ยกเว้นฝั่ง Ada ที่ได้รับอนุมัติไว้ 2 รุ่น
- image ระบุเองและไม่ใช้ template

**ข้อจำกัด:**
- architecture ละ 1 การ์ด, consumer เท่านั้น, ไม่มี Hopper/A100 (ไม่มี offer แบบ VM)
- ไม่ได้ล็อก clock
- ขนาด L2 เปลี่ยนไปพร้อมกับ architecture
- ncu ของ Blackwell เป็นรุ่นอื่น
- ยังไม่ได้รัน smoke บนเครื่องใด

---

> **ผล (MEASURED 2026-11-06 04:57–05:30 UTC, `benchmarks/results/b11_*_20261106/`, D-060):**
>
> **ความน่าเชื่อถือ:**
> - Turing และ Ada รันด้วย wrapper ตัวที่ล็อกไว้ (`cc46ed02…`)
> - Blackwell ต้องรัน 5 รอบเพราะปัญหา driver ดู D-060
> - ตัวสรุปรวมผลอยู่ที่ `benchmarks/results/b11_cross_arch_summary.json`
>
> | GPU (architecture) | L2 | P-0 | C5n: WBR(0:0.4375) / WBR(0:0.625) | C6: L(0.75) | ตรงกับ GA106 | FC (cell แรก) |
> |---|---|---|---|---|---|---|
> | *GA106 (Ampere, อ้างอิง)* | 3 MiB | ✅ | มี: 0.014 / 0.991 | 0.089 | 9/9 | – |
> | *GA102 (Ampere, B10)* | 6 MiB | ✅ | มี: 0.009 / 0.974 | 0.144 | 7/9 | – |
> | **TU106 RTX 2060S (Turing)** | 4 MiB | ✅ | **มี:** 0.016 / 0.976 (V-3 คม) | **0.147** | 5/9 (V-2, V-4, Q-1, Q-3) | **artifact ยืนยัน:** c1 0.181 → c2 0.001 และ V-4' = นับเป็น line |
> | **AD103 RTX 4080S (Ada)** | 64 MiB | ✅ | **มี:** 0.071 / 0.933 (V-3 partial) | **0.058** | 7/9 (V-3, Q-3) | ไม่มีให้ทดสอบ (c1 0.001) และ V-4 = นับเป็น line |
> | GB206 RTX 5060 Ti (Blackwell) | 32 MiB | **❌** `dram__bytes_*` = n/a | – | – | – | – |
>
> **Verdict ข้ามรุ่นตามกฎที่ล็อก:**
> - **C5n-X = generalizes (มี cap บน 2/2 GPU ใหม่)**
> - **C6-X = generalizes (ncu ราย kernel นับขาดบน 2/2)**
> - เมื่อรวม Ampere แล้ว กลไกเกิดบน 3 architecture (Turing, Ampere, Ada) และ 4 die
> - Blackwell ไม่ผ่าน P-0 จึงไม่นับ
>
> **คำทำนายของผู้วิจัยเทียบกับผล:**
> - verdict ข้ามรุ่นทั้งสองข้อ: ทำนายถูก
> - C5n บน Turing (ความมั่นใจต่ำ): ถูก
> - P-0 บน Blackwell: **ผิด**
> - V-4' (นับเป็น line): ถูกทั้ง Turing และ Ada
> - Z-1 บน Ada (dvsc): ถูก
>
> **ผลรายกฎอื่น:**
> - Z-1 = dvsc ทั้ง Turing และ Ada
> - Z-2 ≥ 0.94 ทั้งคู่
> - V-5 ยังอยู่ใน cache ทุกตัว
> - Z-2b ไม่คมทุกตัว
> - **Q-3 ไม่ผ่านทั้ง Turing และ Ada:** v4 ทำนาย write-back ของ write kernel เดี่ยวผิดเกิน 0.1
>   - Turing ที่ w = 1: 0.372 เทียบกับที่ทำนาย 0.5
>   - Ada ทุก w: 0.189 / 0.317 / 0.543 เทียบกับที่ทำนาย 0.333 / 0.5 / 0.667
> - V-2 ของ Turing = inconclusive: Δ +0.049 / +0.053 ที่ r + w = 0.875 คือ clean footprint มีผลเล็กน้อยใกล้ capacity
> - Q-1 ของ Turing = หายทุก w (0.147 / 0.169 / 0.057)
>
> **B10b: artifact "cell แรกของ process" ยืนยันบน Turing**
> - control ของ V-4 ได้ 0.181 เมื่อเป็น cell แรก และ 0.001 เมื่อเป็น cell ที่สอง
> - เมื่อแก้แล้ว V-4' = นับเป็น line
> - ข้อนี้อธิบายได้ว่าทำไม V-4 บน GA102 ตีความไม่ได้ (ยังไม่ได้วัดซ้ำบน GA102)
> - **ผลสะสม:** cap นับเป็น line บน GA106, TU106 และ AD103
>
> **ข้อสังเกต (exploratory หลังเห็นผล):**
> 1. read cliff ที่ r = 1.031 L2 ได้ RDR **0.539 / 0.542 / 0.531 / 0.516** (GA106 / GA102 / TU106 / AD103) คือเกือบคงที่ข้าม 3 architecture และขนาด L2 3–64 MiB
> 2. **กราฟขั้นบันไดของ Ada ต่างรูป:** WBR ที่ w = 0.4375…0.6875 ได้ 0.071 / **0.558** / 0.614 / 0.933 / 0.939 คือเริ่มเร็วกว่าและมีช่วงราบ
>    - ใน B9b ของ Ada ที่เขียน kernel เดียวต่อจาก L2 สะอาด cap ที่วัดได้จริง ≈ w·(1 − A) = 0.61 / 0.68 / 0.69 L2
>    - ข้อนี้อธิบาย Q-3 ที่ไม่ผ่านได้: การเขียนครั้งเดียวกับการเขียนแบบ steady state อาจมี cap ต่างกัน
> 3. Turing ใน B9b มี cap ≈ 0.50 / 0.63 / 0.51 L2
> 4. RB(1, 0.125) ไม่ผ่าน replication บนทุก GPU ใหม่ (0.526 / 0.354 เทียบกับ B6c 0.085) dvsc ที่ cell นี้อ่อนกว่า GA106 ทุกตัว
>
> **ข้อจำกัดเพิ่มเติม:**
> - Blackwell ใช้ driver คนละรุ่น (575.57.08 open) และ ncu 2025.2.1
> - สาเหตุที่ DRAM metric เป็น n/a บน GB206 ยังไม่ได้พิสูจน์ อาจเป็นข้อจำกัดของ ncu กับ GeForce Blackwell, เป็นผลของ VM หรือเป็นการตั้งค่า


---

## B12 pre-registration: ความเพี้ยนของ `do_bench` จาก dirty L2 กับ GEMV ตอน LLM decode (G-0..G-2) — เขียน 2026-11-06 ก่อนวัดบน GPU ใด

**ที่มา (PI, D-061):** "อยากให้ไป merge กับด้าน AI มากขึ้น"

**กลไก:**
- `triton.testing.do_bench` ล้าง L2 ก่อนจับเวลาทุกครั้งด้วย `cache.zero_()` บน buffer int32 ขนาด 256 MiB ซึ่งเป็นการ**เขียน** (Triton ทุก tag v2.0.0–v3.9.0 ทำแบบนี้ และ benchmarker ของ Inductor ก็เช่นกัน)
- ตามผล C5n (B6b/B9b/B11) หลังการล้างแบบนี้ L2 จะยังมี **dirty residue ≈ R × L2**
- kernel ที่ถูกจับเวลาต้องเขียน residue นี้กลับ DRAM ตอนไล่ line ออก
- Triton #12554 วัดได้บน A100 ว่าผลอิ่มที่ราว ½ L2 แต่ยังอธิบายไม่ได้

**Design** (`dobench_flush_probe.py`, `dobench_flush_probe.sh`, `dobench_flush_score.py` พร้อม test, `remote_b12.sh`):
- **kernel ที่จับเวลา:** `F.linear(x, W)` แบบ fp16 x = [1, 4096] คือ linear layer ของ LLM ตอน decode ที่ batch 1 ติด memory และอ่าน W ครั้งเดียว
- **ขนาด W:** w_read × L2 โดย w_read ∈ {2, 4, 8} จึงไล่ทุก line ใน L2 ออก
- **เปลี่ยนตัวแปรเดียว:** วิธีล้างก่อนเรียก คือ `zero` (เหมือน do_bench) เทียบกับ `read` (`cache.view(float32).sum()`)
- **range mode** (ncu app-range, 5 รอบ × 3 run): นับ DRAM bytes ของ range ที่มีแค่ GEMV และมี control range ว่าง 2 ช่วง
- **timing mode** (ไม่มี profiler, 40 รอบ × 3 run, สลับ variant): ใช้ CUDA event เหมือน do_bench โดยไม่ sync ระหว่าง flush กับ start
- **exploratory:** W ขนาดคงที่ 4096 × 4096 (32 MiB) และ `triton.testing.do_bench` จริงแบบ default เทียบกับแบบ patch `clear_cache` ให้อ่าน
- ตรวจ correctness กับ fp32 ทุกขนาดก่อนวัด
- **GPU** (การ์ดรุ่นเดิมที่ PI อนุมัติแล้ว ใช้งบที่เหลือของ B11): RTX 2060S (TU106, 4 MiB), RTX 3090 (GA102, 6 MiB), RTX 4080S (AD103, 64 MiB)

**คำทำนายเชิงปริมาณ:**
- R_gpu = 1.5 × (1 − A(1.5)) จาก B9b ของแต่ละ GPU ซึ่งเป็นค่าที่วัดไว้แล้ว ไม่ได้ fit ใหม่: GA102 **0.480**, TU106 **0.506**, AD103 **0.686**
- v4 ทำนาย 0.5 ทุกตัว

| ID | เกณฑ์ |
|---|---|
| **G-0 (ต้องผ่านก่อน)** | correctness 0; control range เขียน ≤ 64 KiB; หลัง `read` flush GEMV เขียน ≤ 0.05 L2 และอ่าน 0.9–1.2 × W |
| **G-1 (ตัวหลัก)** | X(w) = [write ของ GEMV หลัง `zero` − หลัง `read`] / L2 · \|X − R_gpu\| ≤ 0.15 ทุก w → residue ตรงกับที่ B9b ทำนาย · ไม่ผ่านสักขนาด → not · นอกนั้น mixed (รายงานว่าใกล้ R_gpu หรือใกล้ v4 = 0.5 มากกว่า) |
| G-1b | max X − min X ≤ 0.10 → residue ไม่ขึ้นกับขนาดของ GEMV |
| **G-2** | Δt = median t(zero) − median t(read) · ทำนายจาก byte ที่อัตรา DRAM ของ run สะอาด: Δt_pred = X·L2 / ((R_read + W_read) / t(read)) · Δt > 0 ทุก w **และ** \|Δt − Δt_pred\| ≤ 0.30 Δt_pred ที่ w = 2 → ความเพี้ยนของเวลา = เวลาที่ใช้เขียน residue กลับที่อัตรา DRAM เดิม |
| รายงาน | bias = Δt / t(read) (ทำนาย ≈ X / w_read เช่นที่ w = 2: TU106 ≈ 25 %, GA102 ≈ 24 %, AD103 ≈ 34 %) และค่า exploratory |

**คำทำนายของผู้วิจัย:**
- G-0 ผ่าน
- **G-1 ผ่านทั้ง 3 GPU** (TU106 และ GA102 มั่นใจปานกลาง–สูง; AD103 มั่นใจปานกลาง เพราะ 256 MiB = 4 L2 เท่านั้น ขณะที่ R มาจาก w = 1.5)
- AD103 จะใกล้ R_gpu มากกว่า v4
- G-1b ผ่าน
- G-2 ผ่าน (ปานกลาง: Triton #12554 เห็นอัตรา DRAM เท่าเดิมบน A100)
- W 4096 × 4096 บน AD103 เล็กกว่า L2 จึงไล่ residue ออกไม่หมด ไม่ทำนายข้อนี้

**ความหมายถ้าผ่าน:**
- ใช้กฎ ½ L2 (และ R ที่วัดได้ของแต่ละรุ่น) ทำนายได้ว่า do_bench จะวัดเวลา kernel ที่ติด memory สูงเกินจริงเท่าไรบนการ์ดใดก็ได้: ≈ R·L2 / bytes ที่ kernel อ่าน
- ทำให้รู้ว่าการเลือก config ด้วย autotune อาจเอนเอียงไปทางไหน

**ข้อจำกัด:**
- GEMV แบบเดียว (cuBLAS ผ่าน torch), fp16 และ batch 1
- flush ขนาด 256 MiB คงที่เหมือน do_bench
- VM และไม่ได้ล็อก clock
- ไม่มี datacenter GPU

---

> **ผล (MEASURED 2026-11-06 06:22–06:37 UTC, `benchmarks/results/b12_*_20261106/`, D-062):**
>
> **ความน่าเชื่อถือ:**
> - 3090 และ 2060S ให้คะแนนซ้ำในเครื่องได้ผลตรงกันทุกบรรทัด
> - 4080S ดาวน์โหลดได้ไม่ครบ (ขาด CSV ของ range run 2) ใช้คะแนนที่ VM คำนวณจากครบ 3 run และให้คะแนนซ้ำในเครื่องจาก run 1 + 3 ได้ verdict และตัวเลขเดียวกัน
>
> | GPU | L2 | X ที่ w = 2 / 4 / 8 (ทำนาย R) | G-1 | G-1b | G-2 | bias ของเวลา w = 2 / 4 / 8 | bias ที่ W 4096 × 4096 (event / do_bench จริง) |
> |---|---|---|---|---|---|---|---|
> | TU106 RTX 2060S | 4 MiB | 0.043 / 0.046 / 0.050 (0.506) | **not** | ไม่ขึ้นกับขนาด | **อธิบายด้วย byte ไม่ได้** (Δt 1.9–4.1 µs เทียบกับที่ทำนาย 0.5) | 7.5 / 4.8 / 4.9 % | 4.9 % / 6.0 % |
> | GA102 RTX 3090 | 6 MiB | **0.074** / 0.458 / 0.456 (0.480) | **mixed** | ขึ้นกับขนาด | ✅ อธิบายด้วย byte ได้ | 4.8 / 13.9 / 7.8 % | 10.6 % / 8.7 % |
> | AD103 RTX 4080S | 64 MiB | 0.241 / 0.242 / 0.243 (0.686) | **not** (ใกล้ v4 = 0.5 กว่า แต่ก็ห่าง) | ไม่ขึ้นกับขนาด | ✅ อธิบายด้วย byte ได้ | 13.6 / 7.1 / 3.7 % | **38.5 % / 38.7 %** |
>
> G-0 ผ่านทุก GPU
>
> **คำทำนายของผู้วิจัย:** G-1 ผิดทั้ง 3 GPU · G-1b ถูก 2/3 · G-2 ถูก 2/3
>
> **สิ่งที่สรุปได้:**
>
> **1. do_bench วัดเวลา GEMV ตอน decode ที่ batch 1 สูงเกินจริงบนทุก GPU (MEASURED)**
> - เทียบกับการล้าง L2 แบบอ่าน วัดเกินไป 4–14 % เมื่อ W เป็น 2–8 เท่าของ L2
> - ที่ W = 4096 × 4096 fp16 (linear layer ของ LLM ขนาดราว 7B) วัดเกิน **39 % บน RTX 4080S**, 9–11 % บน 3090 และ 5–6 % บน 2060S
> - `triton.testing.do_bench` จริงให้ตัวเลขใกล้กับการวัดด้วย event
>
> **2. dirty residue หลัง `zero_()` ขนาด 256 MiB ไม่ได้เท่ากับ ½ L2 และ model ของเราทำนายไม่ได้ (negative)**
> - residue ≈ 0.05 / 0.46 / 0.24 L2 บน Turing / Ampere / Ada
> - R จาก B9b มาจากการเขียนครั้งเดียวขนาด 1.5 L2 แต่การเขียนยาว 4–64 เท่าของ L2 ทิ้ง residue ต่างออกไป
> - **กฎ ½ L2 (C5n) จึงใช้กับ steady state และการเขียนสั้นเท่านั้น ใช้กับ stream ยาวไม่ได้**
>
> **3. บน GA102 และ AD103 bias ของเวลาเท่ากับ residue ÷ อัตรา DRAM (G-2)**
> - ถ้าวัด residue ของ GPU หนึ่งครั้ง จะทำนาย bias ของ kernel ที่ติด memory ได้ ≈ X·L2 / byte ที่ kernel ย้าย
> - ส่วนบน Turing มี bias 5–7 % ที่ไม่ได้มาจาก byte (ยังไม่รู้กลไก)
>
> **ข้อสังเกต (exploratory):**
> 1. **GA102:** GEMV ที่อ่าน 2 × L2 ไล่ dirty residue ออกได้แค่ราว 15 % (0.074 / 0.46) แต่ที่ 4 × L2 ไล่ออกหมด เข้ากับสมมติฐานเดิมว่า L2 ต้านการอ่านแบบ stream (HANDOFF §7.2)
> 2. **AD103:** W 4096 × 4096 = 0.5 L2 ซึ่งเล็กกว่า L2 แต่ bias ยังสูงที่สุด เพราะ byte ที่ต้องเขียนกลับ (16 MB) เทียบกับ byte ที่ GEMV อ่าน (32 MB) มีสัดส่วนสูง
> 3. **2060S:** รอบแรกรัน timing ไปก่อน range จะล้ม (ไม่มี ncu) ผลส่วนนั้นเก็บไว้ใน `pilot_out/attempt1` และไม่นำมาให้คะแนน


---

## B13 pre-registration: bias ของ do_bench ทำให้ autotune เลือก kernel ของ LLM decode ผิดหรือไม่ (T-0..T-2) — เขียน 2026-11-06 ก่อนวัดบน GPU ใด

**ที่มา (PI, D-063):** "ทำให้เสร็จก่อน" ต่อจาก B12 ที่วัดได้ว่า do_bench วัดเวลา GEMV ตอน decode เกินจริง 5–39 % คำถามคือ bias นี้**เปลี่ยนการตัดสินใจ**หรือไม่

**Design** (`autotune_bias_probe.py`, `autotune_bias_probe.sh`, `autotune_bias_score.py` พร้อม test, `remote_b13.sh`):
- **ตัวเลือก:** Triton GEMV fp16 ตัวเดียว 36 config (BLOCK_N ∈ {8, 16, 32, 64} × BLOCK_K ∈ {128, 256, 512} × num_warps ∈ {2, 4, 8}) รวมกับ cuBLAS (`F.linear`) เป็นการตัดสินใจแบบเดียวกับที่ autotuner ของ Triton และ max-autotune ของ Inductor ทำ
- **shape (N, K):** 4096×4096, 14336×4096 และ 4096×14336 คือ projection ของ attention, MLP up และ MLP down ตอน batch 1
- **การจับเวลา:** ใช้ **`triton.testing.do_bench` จริง** (median, warmup 10 ms, rep 50 ms) สองโหมด สลับลำดับทุกตัวเลือก
  - `zero`: ตามที่ Triton ship มา
  - `read`: patch `clear_cache` ให้เป็นการอ่าน
- **รอบ:** 3 run และสลับลำดับตัวเลือกในแต่ละ run (seed 1, 2, 3)
- **ความถูกต้อง:** ตรวจทุกตัวเลือกกับ fp32 ก่อนจับเวลา ตัวที่ไม่ผ่านหรือ compile ไม่ได้จะถูกตัดออก
- **ไม่มี profiler**
- **GPU:** 2060S, 3090 และ 4080S เป็นรุ่นเดิม ใช้งบที่เหลือจากเพดาน $1 (ราว $0.76)

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **T-0 (ต้องผ่านก่อน)** | 3 run, ตัวเลือก ≥ 80 % ของ 37 ใช้ได้ทุก shape และ cuBLAS ใช้ได้ |
| **T-1 (ตัวหลัก)** | regret = t_read(ตัวที่ zero เลือก) / t_read(ตัวที่ read เลือก) − 1 · ค่าสูงสุดข้าม shape ≤ 0.02 → harmless · ≥ 0.05 → harmful · นอกนั้น minor |
| T-2 | การตัดสินใจว่า cuBLAS หรือ Triton ที่ดีที่สุดเร็วกว่า เปลี่ยนระหว่างสองโหมดใน shape ใดก็ได้ → flips · ไม่มี → no flip |
| รายงาน | Spearman ระหว่าง t_zero กับ t_read, bias ของตัวที่ถูกเลือก |

**คำทำนายของผู้วิจัย** (ความมั่นใจต่ำ เพราะยังไม่เคยวัดว่า bias นี้ต่างกันระหว่างตัวเลือกแค่ไหน):
- T-0 ผ่าน
- **T-1:** TU106 และ GA102 = harmless (bias ใกล้เคียงกันเพราะทุกตัวอ่าน byte เท่ากัน); AD103 = minor (bias สูงและ W 4096×4096 เล็กกว่า L2 ตัวเลือกที่อ่านต่างลำดับอาจไล่ residue ออกต่างกัน)
- **T-2:** TU106 และ GA102 = no flip; AD103 = flips (Triton #12554 พบว่าการเทียบกับ cuBLAS เปลี่ยน 9/96 config บน A100)

**ข้อจำกัด:**
- kernel Triton ที่เขียนเองแบบเรียบง่าย ไม่ใช่ kernel ที่ใช้ใน production
- fp16 อย่างเดียว ไม่มี quantization
- timing อย่างเดียว ไม่มี counter
- ความแปรปรวนของเวลาราว 1 % อาจทำให้ regret ต่ำ ๆ เป็นแค่ noise

---

> **ผล (MEASURED 2026-11-06 06:57–06:59 UTC, `benchmarks/results/b13_*_20261106/`, D-064):**
> - T-0 ผ่านทุก GPU (ตัวเลือกใช้ได้ครบ 37 ตัวทุก shape)
> - ให้คะแนนซ้ำในเครื่องได้ผลตรงกันทุกบรรทัด
>
> | GPU | T-1 (regret สูงสุด) | T-2 | shape ที่ตัวเลือกต่างกัน | bias ของตัวที่ถูกเลือก | Spearman |
> |---|---|---|---|---|---|
> | TU106 RTX 2060S | **harmless** (+1.4 %) | **flips** | 4096×4096: zero เลือก cuBLAS แต่ read เลือก Triton bn16_bk128_w8 | +2.5 ถึง +5.9 % | 0.88–0.97 |
> | GA102 RTX 3090 | **harmless** (0.0 %) | no flip | 2 shape เลือก config ต่างกัน แต่เร็วเท่ากันเมื่อวัดแบบ read | +3.0 ถึง +9.8 % | 0.84–0.92 |
> | AD103 RTX 4080S | **harmless** (+0.6 %) | **flips** | MLP up/down (112 MiB): zero เลือก cuBLAS แต่ read เลือก Triton | +16 ถึง +40 % | **0.44**–0.89 |
>
> **คำทำนายของผู้วิจัย:**
> - T-1: ถูก 2/3 (AD103 ทำนายว่า minor แต่ได้ harmless)
> - T-2: ถูก 2/3 (TU106 ทำนายว่า no flip แต่ได้ flips)
>
> **สิ่งที่สรุปได้ (GEMV fp16 batch 1 ตัวเลือกที่อ่าน byte เท่ากัน):**
> - bias ของ do_bench (B12) **บวกเวลาเพิ่มเกือบเท่ากันทุกตัวเลือก**
> - การตัดสินใจจึงเปลี่ยนได้ (cuBLAS หรือ Triton เปลี่ยนบน 2/3 GPU และลำดับสลับกันมากบน Ada) แต่**ตัวที่ถูกเลือกผิดช้ากว่าจริงไม่เกิน 1.4 %** เพราะตัวเลือกที่แข่งกันเร็วใกล้กันมาก
> - **ผลเสียหลักของ bias อยู่ที่ "ตัวเลขเวลาและ speedup ที่รายงาน"** (เกินจริง 3–40 %) **ไม่ใช่การเลือก config** ในกรณีนี้
> - ยังไม่ได้ทดสอบตัวเลือกที่อ่าน byte ต่างกัน (เช่น weight แบบ int8 เทียบกับ fp16) ซึ่ง Triton #12554 รายงานว่ามีการสลับลำดับ ในกรณีนั้น bias อาจไม่ได้บวกเพิ่มเท่ากัน


---

## B14 pre-registration: bias ของ do_bench บิด speedup ของ GEMV แบบ quantized หรือไม่ (U-0..U-2) — เขียน 2026-11-06 ก่อนวัดบน GPU ใด

**ที่มา (PI, D-065):** "ลองเลย" ต่อจาก B13 ซึ่งพบว่า bias บวกเวลาเพิ่มเกือบเท่ากันในตัวเลือกที่อ่าน byte เท่ากัน รอบนี้ถามต่อว่า เมื่อตัวเลือก**อ่าน byte ต่างกัน** (weight แบบ fp16 / int8 / int4 ตอน LLM decode) speedup ของการ quantize ที่วัดด้วย do_bench จะบิดไปหรือไม่

**Design** (`quant_bias_probe.py`, `quant_bias_probe.sh`, `quant_bias_score.py` พร้อม test, `remote_b14.sh`):
- **ตัวเลือก:** Triton GEMV 3 รูปแบบ แบบละ 12 config (BLOCK_N {8, 16, 32} × BLOCK_K {128, 256} × num_warps {4, 8}) รวม cuBLAS ฝั่ง fp16 เป็น 37 ตัวเลือกต่อ shape
  - fp16 (2 B ต่อ weight)
  - W8A16: int8 + scale ต่อแถว (1 B ต่อ weight)
  - W4A16: int4 pack 2 ค่าต่อ byte ค่าคือ nibble − 8 + scale ต่อแถว (0.5 B ต่อ weight)
- **ตรวจก่อนวัด:** kernel ทั้ง 3 แบบผ่านบน CPU ด้วย Triton interpreter แล้ว รวมถึง shape ที่หารด้วย block ไม่ลงตัว (rel err ≤ 5e-4) และการ pack int4 ถอดกลับได้ตรง
  - บน GPU ทุกตัวเลือกยังตรวจกับ fp32 ของ weight ที่ dequantize แล้วอีกครั้ง
- **shape:** 3 แบบเดิมของ B13
- **การจับเวลา:** `triton.testing.do_bench` จริงสองโหมด สลับลำดับ, 3 run, ไม่มี profiler
- **GPU:** 2060S, 3090 และ 4080S (รุ่นเดิม) ใช้งบที่เหลือจากเพดาน $1 (ราว $0.72)

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **U-0 (ต้องผ่านก่อน)** | 3 run, ตัวเลือก ≥ 80 % ใช้ได้ทุก shape และทุกรูปแบบมี config ที่ใช้ได้ |
| **U-1 (ตัวหลัก)** | S_q,m = t(fp16 ที่ดีที่สุด) / t(q ที่ดีที่สุด) และ D_q = S_q,zero / S_q,read − 1 · max \|D\| ≤ 0.03 → faithful · ≥ 0.10 → distorted · นอกนั้น mildly distorted · รายงานทิศทาง (ต่ำเกินจริงหรือสูงเกินจริง) |
| U-2 | รูปแบบที่เร็วที่สุดเปลี่ยนระหว่างสองโหมดใน shape ใดก็ได้ → flips · ไม่มี → no flip |
| รายงาน | regret ของตัวที่ถูกเลือก (แบบ B13 T-1) |

**คำทำนายของผู้วิจัย** (ความมั่นใจปานกลาง):
- **กลไก:** B13 พบว่า bias บวกเพิ่มเกือบเท่ากัน ถ้า Δ ยังบวกเพิ่มเท่ากันแม้ตัวเลือกอ่าน byte น้อยกว่า kernel ที่เร็วกว่าจะโดน bias **สัดส่วนสูงกว่า** speedup ของ int8/int4 ที่วัดด้วย do_bench จึง**ต่ำกว่าจริง**
- **U-1 = distorted ทั้ง 3 GPU ทิศทาง "speedup ต่ำกว่าจริง"** ประเมินคร่าว ๆ จาก Δ ของ B12:
  - 4080S ที่ 4096×4096: S_w4 อาจต่ำกว่าจริงราว 40 %
  - 3090 ราว 20 %
  - 2060S ราว 10 %
- **ทางเลือกที่ทำให้คำทำนายผิด:** ถ้า kernel ที่อ่านน้อยไล่ dirty residue ออกน้อยกว่า (Triton #12554 เสนอว่า "evicts fewer dirty lines") Δ จะลดตาม byte และ D จะใกล้ 0 (faithful)
- **U-2 = no flip:** ถ้า Δ บวกเพิ่มเท่ากัน ลำดับจะไม่เปลี่ยน
- U-0 ผ่าน

**ข้อจำกัด:**
- kernel quantized ที่เขียนเองแบบเรียบง่าย ไม่ใช่ kernel ที่ใช้ใน production เช่น Marlin หรือ AWQ
- scale ต่อแถวเท่านั้น
- batch 1
- timing อย่างเดียว

---

> **ผล (MEASURED 2026-11-06 07:19–07:28 UTC, `benchmarks/results/b14_*_20261106/`, D-066):**
> - U-0 ผ่านทุก GPU (ตัวเลือกใช้ได้ครบ 37 ตัว)
> - ให้คะแนนซ้ำในเครื่องได้ผลตรงกันทุกบรรทัด
>
> | GPU | U-1 (D ที่แย่ที่สุด) | ทิศทาง | U-2 | regret สูงสุด | S_w4 ที่ 14336×4096 (zero / read) |
> |---|---|---|---|---|---|
> | TU106 RTX 2060S | **mildly** (−7.2 %) | ปนกัน | no flip | 0 % | 2.12 / 2.09 |
> | GA102 RTX 3090 | **mildly** (−6.5 %) | ปนกัน | no flip | 0 % | 2.92 / 2.89 |
> | AD103 RTX 4080S | **distorted** (−17.0 %) | **ต่ำกว่าจริงทุกค่า** | no flip | **+22.6 %** (W4 ที่ 4096×14336) | **3.08 / 3.71** |
>
> **คำทำนายของผู้วิจัย:**
> - U-1 (distorted ทุก GPU และต่ำกว่าจริง): ถูกเฉพาะ AD103 ส่วน TU106 และ GA102 ได้ mildly ทิศทางปนกัน
> - U-2 (no flip): ถูกทุก GPU
>
> **สิ่งที่สรุปได้:**
> - **การ์ด L2 ใหญ่ (4080S, 64 MiB):** do_bench รายงาน speedup ของ int8/int4 เทียบกับ fp16 **ต่ำกว่าจริง 14–17 %** ใน shape ของ MLP (112 MiB) และเลือก config ของ W4 ที่**ช้ากว่าตัวที่ดีที่สุดจริง 23 %**
>   - ผลนี้คงที่ทั้ง 3 run: do_bench วัด 73.7 กับ 75.8 µs แต่เวลาจริงคือ 66.6 กับ 54.3 µs
> - **การ์ด L2 เล็ก (2060S, 3090):** บิดไม่เกิน 7 % และเลือกถูก
> - ที่ 4096×4096 บน 4080S (weight เล็กกว่า L2) bias แทบเป็นสัดส่วนกับ byte (+20.5 / +11.3 / +7.2 µs สำหรับ fp16 / int8 / int4) speedup จึงแทบไม่บิด (−0.5 / −2.6 %) ข้อนี้เข้ากับคำอธิบายของ Triton #12554 ว่า kernel ที่อ่านน้อยไล่ dirty line ออกน้อยกว่า
>
> **ข้อสังเกต (exploratory หลังเห็นผล):**
> - bias ต่าง config กันได้มากใน format เดียวกัน: บน 4080S ที่ 4096×14336 config W4 ตัวหนึ่งโดน +7 µs อีกตัวโดน +21.5 µs
> - **สมมติฐาน:** kernel ที่ใช้ DRAM bandwidth ไม่เต็มมีช่องว่างพอจะซ่อนการเขียน residue กลับ ส่วน kernel ที่ใช้เต็มต้องจ่ายเต็ม do_bench จึงมีแนวโน้มให้คะแนน kernel ที่ใช้ bandwidth ไม่เต็มสูงเกินจริง
> - ข้อนี้อธิบายได้ว่าทำไม regret ของ B13 ต่ำ (ตัวเลือกใกล้กัน) แต่ B14 บน Ada สูง ยังไม่ได้ทดสอบโดยตรง


---

## B15 pre-registration: penalty ต่อ config อธิบายด้วย bandwidth slack หรือไม่ (K-0..K-3) + ทำซ้ำ B14 บน RTX 4090 และ L40S — เขียน 2026-11-06 ก่อนวัดบน GPU ใด

**ที่มา (PI, D-067):** "ทำเลย" ทั้งสองข้อที่เสนอใน D-066 ได้แก่ (1) พิสูจน์ด้วย counter และ (2) ทำซ้ำบนการ์ด L2 ใหญ่ตัวอื่น ผู้ช่วยวิจัยตีความว่าเป็นการอนุมัติ RTX 4090 และ L40S ภายในงบที่เหลือของเพดาน $1 (ราว $0.62)

**สมมติฐาน (post hoc จาก B14 บน 4080S ที่ W4 4096×14336: config หนึ่ง +7 µs อีก config +21.5 µs):**
- dirty residue ทำให้มี DRAM write-back เป็น**จำนวน byte เท่ากัน**ทุก config
- แต่ config ที่ใช้ DRAM bandwidth ไม่เต็มซ่อนการเขียนนั้นใน slack ได้ ส่วน config ที่ใช้เต็มต้องจ่าย
- **4080S เป็นที่มาของสมมติฐานจึงเป็น in-sample บางส่วน** (เป็นการวัดใหม่ด้วย counter แต่ GPU เดิม) ส่วน 4090 และ L40S เป็นการทดสอบ out-of-sample

**Design** (`absorb_probe.py`, `absorb_probe.sh`, `absorb_score.py` พร้อม test, `remote_b15.sh`):
- **kernel:** fp16 / W8A16 / W4A16 ของ B14 แบบละ 12 config (ไม่มี cuBLAS)
- **shape:** 4096×14336 และ 14336×4096
- **range mode** (ncu app-range, 3 รอบต่อ config ต่อโหมด): นับ DRAM read/write ของ kernel หลัง `zero_()` และหลังการล้างด้วยการอ่าน
- **timing mode:** `triton.testing.do_bench` จริงสองโหมด 3 run
- **GPU:** 4080S (AD103, 64 MiB), **RTX 4090 (AD102, 72 MiB)** และ **L40S (AD102, 96 MiB การ์ด datacenter)**
- บน 4090 และ L40S รัน protocol ของ B14 (`quant_bias_probe.sh`) ซ้ำโดยไม่แก้อะไร ส่วน 4080S ข้ามเพราะวัดไปแล้วใน B14

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **K-0 (ต้องผ่านก่อน)** | ทุก range มี counter, config ≥ 80 % ใช้ได้, median write หลังการล้างแบบอ่าน ≤ 0.05 L2 และ timing ≥ 3 run |
| **K-1 (ตัวหลัก)** | Spearman ρ(Δt, U) ทุก shape ≥ 0.5 → **penalty โตตามการใช้ bandwidth** · ≤ 0 ทุก shape → no · นอกนั้น weak (U = (R/t) / ค่าสูงสุดในบรรดา config) |
| **K-2** | CV ของ E (write ที่เพิ่มขึ้น) ข้าม config ≤ 0.20 ทุก shape → **byte ที่เขียนกลับเท่ากัน** · นอกนั้น bytes differ |
| K-3 (รายงาน) | slack model: Δt_pred = max(0, E/BWmax − (t − R/BWmax)) และ R² |
| U-0..U-2 | กฎของ B14 ไม่เปลี่ยน ใช้กับ 4090 และ L40S |

**คำทำนายของผู้วิจัย:**
- K-0 ผ่าน
- **K-1 = penalty โตตามการใช้ bandwidth ทั้ง 3 GPU** (ปานกลาง และต่ำกว่าบน 4090/L40S)
- **K-2 = byte ที่เขียนกลับเท่ากัน** (ปานกลาง)
- K-3: R² ≥ 0.5 (ต่ำ)
- **B14 บน 4090 และ L40S:** U-1 = distorted ทิศทางต่ำกว่าจริง (ปานกลาง), U-2 = no flip, regret ≥ 5 % อย่างน้อยหนึ่ง shape (ต่ำ)

**งบและการหยุด:**
- คาดว่าใช้ราว $0.4
- `SSH_WAIT_S` 900, `RUN_WAIT_S` 1500 และ `HARD_CAP_S` 2700
- ผู้ช่วยวิจัยจะคอยดูและหยุดเครื่องที่ค้าง ถ้าค่าใช้จ่ายสะสมจะเกิน $0.62

**ข้อจำกัด:**
- การ์ด Ada ทั้งหมด (สมมติฐานยังไม่ได้ทดสอบข้าม architecture)
- kernel ที่เขียนเอง
- U คิดจาก byte ที่อ่านต่อเวลา ไม่ใช่ counter ของ bandwidth จริง

---

> **ผล (MEASURED 2026-11-06 07:42–07:46 UTC, `benchmarks/results/b15_*_20261106/`, D-068):**
> - K-0 และ U-0 ผ่านทุก GPU
> - ให้คะแนนซ้ำในเครื่องได้ผลตรงกันทุกบรรทัด
>
> | GPU | L2 | K-1: ρ(Δt, U) (2 shape) | K-2: CV(E) | E median / L2 | B14 ซ้ำ: U-1 (D แย่สุด) | U-2 | regret |
> |---|---|---|---|---|---|---|---|
> | AD103 RTX 4080S | 64 MiB | **ผ่าน** (+0.81 / +0.88) | **bytes differ** (0.43 / 0.45) | 0.35 / 0.33 | (B14: −17 %) | – | (B14: +23 %) |
> | AD102 RTX 4090 | 72 MiB | **ผ่าน** (+0.75 / +0.85) | **bytes differ** (0.40 / 0.45) | 0.32 / 0.30 | **distorted** (−20.2 %, ต่ำกว่าจริง) | no flip | 0 % |
> | AD102 L40S | 96 MiB | **ผ่าน** (+0.52 / +0.78) | **bytes differ** (0.30 / 0.40) | 0.30 / 0.27 | **distorted** (−12.8 %, ต่ำกว่าจริง) | no flip | 0 % |
>
> **คำทำนายของผู้วิจัย:**
> - K-1: ถูกตามกฎทั้ง 3
> - K-2: ผิดทั้ง 3
> - U-1 / U-2 บน 4090 และ L40S: ถูก
> - "regret ≥ 5 % อย่างน้อยหนึ่ง shape": ผิด (0 % ทั้งคู่)
>
> **ข้อแก้ไขสำคัญ (exploratory หลังเห็นผล แต่เปลี่ยนการตีความ K-1):**
> - กฎ K-1 ไม่ได้แยกตามรูปแบบ weight แต่ U ปนกับรูปแบบ
>   - config ของ fp16 อ่าน byte ต่อเวลาสูงสุด (U ≈ 1)
>   - และยังทำให้ต้องเขียนกลับมากที่สุด: E ≈ 0.48–0.65 L2 สำหรับ fp16, 0.26–0.35 สำหรับ w8 และ 0.18–0.30 สำหรับ w4
> - **เมื่อคิดภายในรูปแบบเดียวกัน ρ(Δt, U) เป็นลบหรือใกล้ 0 ใน 16 จาก 18 กรณี** (เช่น w4 บน 4090 = −0.80, บน L40S = −0.83)
>   - *นับใหม่ (D-073):* ≤ 0 จริง 15/18 ค่าบวก 3 ค่าคือ +0.06, +0.31, +0.36 จึงต้องอ้างเป็น "≤ 0 ใน 15/18" ไม่ใช่ 16/18
> - **สมมติฐานว่า "ใช้ bandwidth ไม่เต็มจะซ่อน write-back ได้" จึงไม่ได้รับการสนับสนุน**
> - การที่ K-1 ผ่านเป็นผลของตัวแปรกวน (format)
> - **สิ่งที่ข้อมูลชี้:**
>   - penalty ตามจำนวน byte ที่ config นั้นทำให้ต้องเขียนกลับ: ρ(Δt, E) ภายในรูปแบบเดียวกันเป็นบวกใน 15/18 กรณี (สูงสุด +0.88)
>   - E ต่างกันระหว่าง config (K-2) และภายในรูปแบบเดียวกันมักสัมพันธ์**ลบ**กับ U คือ config ที่ช้ากว่าทำให้ต้องเขียนกลับมากกว่า (กลไกยังไม่รู้)
>   - kernel ที่อ่าน byte มากไล่ dirty residue ออกมากกว่า ซึ่งตรงกับ Triton #12554
>   - แต่ penalty ของ kernel quantized ไม่ได้ลดตามสัดส่วนของ byte ที่อ่าน ทำให้ speedup ที่วัดได้ต่ำกว่าจริง
> - **บทเรียนเชิงวิธี:** กฎที่ล็อกไว้ต้องควบคุมตัวแปรกวนที่รู้อยู่แล้ว (ในที่นี้คือ format)
>
> **ข้อสรุปของงาน AI ที่แข็งขึ้นจาก B15:**
> - บนการ์ด Ada L2 ใหญ่ 3 ตัว (64 / 72 / 96 MiB) do_bench รายงาน speedup ของ int8/int4 ต่ำกว่าจริง 13–20 % (ซ้ำได้)
> - การเลือก config ผิดเกิดได้แต่ไม่ทุกครั้ง: +23 % บน 4080S ใน B14 แต่ 0 % บน 4090 และ L40S


---

## B16 pre-registration: bias ของ do_bench กับ kernel 4-bit ที่ใช้จริงใน production (P-0, V-1..V-3) — เขียน 2026-11-06 ก่อนวัดบน GPU ใด

**ที่มา (PI, D-070):** "เอาข้อ 1-2 ก่อน" คือ (1) ร่าง technical report และ (2) วัดกับ kernel ที่ใช้จริงใน production เพื่อให้คนดูแล Triton และ PyTorch เชื่อว่าผลของ B14/B15 ไม่ได้เกิดแค่กับ kernel ที่เราเขียนเอง

**ตัวเลือก** (`prod_quant_probe.py`, `prod_quant_probe.sh`, `prod_quant_score.py` พร้อม test, `remote_b16.sh`):
- `cublas_fp16`: `F.linear` แบบ fp16 (baseline)
- `tinygemm_k{2,4,8}`: `torch._weight_int4pack_mm` ของ PyTorch 2.7.1 ซึ่ง gpt-fast และ torchao ใช้กับ int4 weight-only
  - activation แบบ bf16, group 128, innerKTiles 2/4/8 (ให้ autotune เลือก 3 แบบ)
  - quantize ด้วยขั้นตอนเดียวกับ `_group_quantize_tensor` ของ PyTorch ตรวจแล้วว่าตรงทุก byte (packed และ scale/zero)
- `bnb_nf4`: `bitsandbytes.matmul_4bit` (0.50.2) กับ weight แบบ NF4 blocksize 64 ซึ่งเป็น kernel ที่ HuggingFace `load_in_4bit` ใช้ ตอน batch 1 จะเรียก GEMV แบบ 4-bit

**การวัด:**
- **shape:** 3 แบบเดิม
- **correctness:** เทียบกับ fp32 ของ weight ที่ dequantize แล้ว (≤ 2e-2)
- **การจับเวลา:** `triton.testing.do_bench` จริงสองโหมด 3 run
- **GPU:** RTX 3090 (6 MiB, ควบคุม L2 เล็ก), 4080S (64 MiB) และ 4090 (72 MiB) ส่วน tinygemm ต้องใช้ sm_80 ขึ้นไปจึงไม่มี Turing

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **P-0** | 3 run; ทุก shape มี cublas_fp16, tinygemm อย่างน้อย 1 config และ bnb_nf4 ที่ผ่าน correctness |
| **V-1 (ตัวหลัก)** | D_q = S_q,zero / S_q,read − 1 สำหรับ q ∈ {tinygemm (ตัวดีที่สุดของโหมดนั้น), bnb_nf4} · max \|D\| ≤ 0.03 → faithful · ≥ 0.10 → distorted · นอกนั้น mildly · รายงานทิศทาง |
| V-2 | kernel 4-bit ที่เร็วกว่า (tinygemm หรือ bnb) เปลี่ยนระหว่างสองโหมด → flips |
| V-3 (รายงาน) | regret ของการเลือก innerKTiles ของ tinygemm |

**คำทำนายของผู้วิจัย:**
- P-0 ผ่าน (ความเสี่ยง: การติดตั้ง bitsandbytes หรือ convention ของ tinygemm อาจล้ม)
- **V-1:**
  - 4080S และ 4090 = **distorted, speedup ต่ำกว่าจริง** (ปานกลาง: B14/B15 ได้ 13–20 % กับ kernel ที่เราเขียน แต่ kernel ของ production อาจไล่ dirty residue ออกต่างไป)
  - 3090 = mildly หรือ faithful (≤ 7 % แบบ B14)
- V-2 = no flip (ต่ำ–ปานกลาง)
- V-3: ไม่ทำนาย

**ความหมาย:**
- **ถ้า V-1 = distorted บน L2 ใหญ่:** speedup ของ int4 ที่คนวัดด้วย do_bench สำหรับ kernel ที่ใช้จริง (tinygemm / bitsandbytes) ก็ต่ำกว่าจริง ซึ่งเป็นหลักฐานที่ใช้ประกอบ report และการแจ้ง upstream (ต้องให้ PI อนุมัติก่อน)
- **ถ้า faithful:** ผลของ B14/B15 อาจเฉพาะกับ kernel ที่เราเขียน ต้องลดน้ำหนักของ C7(b)

**งบ:** ส่วนที่เหลือของเพดาน $1 (ราว $0.47) คาดว่าใช้ราว $0.15–0.2

---

> **ผล (MEASURED 2026-11-06 08:20–08:25 UTC, `benchmarks/results/b16_*_20261106/`, D-071):**
> - P-0 ผ่านทุก GPU: ทุก candidate ผ่าน correctness ทุก shape (`invalid` ว่าง) และ bitsandbytes 0.50.2 ติดตั้งได้ทั้ง 3 เครื่อง
> - ให้คะแนนซ้ำในเครื่องได้ผลตรงกันทุกบรรทัด
>
> | GPU | L2 | V-1 (D แย่สุด) | D ของ tinygemm (4096² / 4096×14336 / 14336×4096) | D ของ bnb_nf4 | V-2 | V-3 regret |
> |---|---|---|---|---|---|---|
> | GA102 RTX 3090 | 6 MiB | **mildly** (−7.6 %, mixed) | −5.6 / −7.6 / −6.0 % | −0.5 / −2.5 / +1.6 % | no flip | 2.4 % |
> | AD103 RTX 4080S | 64 MiB | **distorted** (−17.0 %, ต่ำกว่าจริงทุกตัว) | −2.2 / −16.8 / −17.0 % | −8.8 / −10.1 / −15.9 % | no flip | 0 % |
> | AD102 RTX 4090 | 72 MiB | **distorted** (−16.9 %, mixed) | **+3.2** / −16.9 / −16.9 % | −10.0 / −12.0 / −14.4 % | no flip | 0 % |
>
> ตัวอย่างที่อ่านง่าย (4080S, MLP up-proj 14336×4096): do_bench แบบ shipped รายงานว่า tinygemm int4 เร็วกว่า cuBLAS fp16 **2.95×** แต่เมื่อล้างด้วยการอ่านได้ **3.56×**; bitsandbytes NF4 2.47× เทียบกับ 2.93×
>
> **bias สัมบูรณ์ (zero / read − 1):**
> - 3090: cuBLAS +3–6 %, tinygemm +7–19 %, bnb +2–7 %
> - 4080S / 4090: cuBLAS +16–41 %, tinygemm +36–44 %, bnb +29–56 %
>
> **คำทำนายของผู้วิจัย:**
> - P-0 ผ่าน: ถูก
> - V-1 บน 4080S / 4090 = distorted: ถูกทั้งคู่
> - ทิศ "ต่ำกว่าจริง": ถูกบน 4080S แต่**ผิดตามกฎบน 4090** (mixed) เพราะ D ของ tinygemm ที่ 4096² = +3.2 % ซึ่งมาจากต่างกัน 1 tick ของตัวจับเวลา (ดูข้อจำกัด)
> - 3090 = mildly หรือ faithful: ถูก (mildly)
> - V-2 = no flip: ถูกทั้ง 3
>
> **ข้อจำกัดที่พบหลังเห็นผล (ไม่เปลี่ยน verdict ของกฎที่ล็อกไว้):**
> - **ความละเอียดของเวลา:** median ของ do_bench 240/270 ค่าอยู่บนตาราง ≈ 1.024 µs (อีก 30 ค่าอยู่กึ่งกลางสองช่องจากการ interpolate)
>   - ที่ 4096² บน Ada, tinygemm ใช้ 14–25 µs ดังนั้น 1 tick ≈ 4–7 % ซึ่งหยาบกว่าเกณฑ์ faithful (3 %)
>   - D ที่ shape ใหญ่ (38–200 µs) จึงเชื่อถือได้กว่า และ verdict "distorted" ทั้งสองตัวมาจาก shape ใหญ่ (ห่างกันราว 15–20 tick)
>   - regret 2.4 % บน 3090 ก็คือ 1 tick (44.03 vs 43.01 µs) ไม่ควรตีความ
> - **pattern เชิงสำรวจ (exploratory):** penalty เป็นไมโครวินาทีบน 4080S
>   - ที่ shape ใหญ่ ใกล้กันทุก kernel: cuBLAS +28 µs, tinygemm +19.5 µs, bnb +22 µs จึงบิด speedup ของ kernel ที่เร็วกว่ามากที่สุด
>   - ที่ 4096² ต่างกันมาก: cuBLAS +21 µs, tinygemm +7 µs, bnb +13 µs ซึ่งสอดคล้องกับ B15 ว่า kernel ที่อ่าน byte น้อยไล่ dirty residue ออกน้อย
>   - D ของ tinygemm ที่ 4096² จึงเล็ก (−2 %) ทั้งที่ bias สัมบูรณ์ใหญ่ (+44 %)
>   - ยังไม่ได้วัดด้วย counter ใน B16
> - ใช้ bf16 activation กับ tinygemm แต่ fp16 กับ cuBLAS และ bnb (convention ของแต่ละ API) speedup จึงเทียบข้ามชนิด activation
>
> **ข้อสรุป:**
> - ผลของ B14/B15 **ไม่ได้เกิดเฉพาะกับ kernel ที่เราเขียน**
> - บน Ada ที่ L2 64–72 MiB, `do_bench` แบบ shipped รายงาน speedup ของ kernel int4 ที่ใช้จริง (PyTorch tinygemm ที่ gpt-fast/torchao ใช้ และ bitsandbytes NF4 ที่ HuggingFace `load_in_4bit` ใช้) ต่ำกว่าจริงสูงสุด 17 % ที่ shape MLP ของ LLM ขนาด 7–8B
> - บน 3090 (L2 6 MiB) ผลเล็กกว่ามาก (≤ 7.6 %)
> - ไม่พบการตัดสินใจผิด (ไม่มี flip ระหว่าง tinygemm/bnb และไม่มี regret ที่เกิน 1 tick)

---

## B17 pre-registration: ขยายผล do_bench ไปยัง GPU ที่คนใช้จริง (A100/H100, Ada ไล่ขนาด L2, Blackwell) เทียบกับวิธีแก้ของ Triton PR #12556 (E-0, F-1..F-6) — เขียน 2026-11-06 ก่อนวัดบน GPU ใด

**ที่มา (PI, D-074):**
- PI: "อยากได้แค่ technical report ... อยากได้สิ่งที่มันช่วยคนได้จริงครับ และมี impact"
- PI: "ขยายผลได้ไหม ให้หลักฐานมันหนาแน่นขึ้น"
- PI เลือกแบบ "ครบรวม H100/A100" งบราว $1.0–1.5

**ช่องโหว่ที่ B17 ตั้งใจปิด:**
1. ยังไม่มี GPU ที่คน benchmark kernel ของ LLM ใช้มากที่สุด (A100, H100)
2. ขนาด L2 ปนกับรุ่นการ์ด เพราะการ์ด L2 ใหญ่ทุกตัวเป็น Ada
3. baseline ที่ใช้คือ `torch.sum` ไม่ใช่วิธีแก้จริงของ PR #12556
4. kernel ที่วัดมีแค่ GEMV
5. แต่ละรุ่นวัดบนเครื่องเดียว

**การวัด** (`flush_bias_probe.py`, `flush_bias_probe.sh`, `flush_bias_score.py` + test, `remote_b17.sh`):
- **จับเวลาด้วย `triton.testing.do_bench` จริง (median) แบบ timing อย่างเดียว ไม่มี ncu** จึงใช้ docker container ได้
  - image ที่ระบุเอง: `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (official, ไม่ใช้ template)
  - torch 2.7.1 + triton 3.3.1 (cu126; Blackwell ใช้ cu128)
  - bitsandbytes 0.50.2 และ liger-kernel 0.6.2 (`--no-deps`)
- **วิธีล้าง L2 (patch `clear_cache` ของ do_bench):**

  | โหมด | วิธี |
  |---|---|
  | `zero` | แบบ shipped |
  | `pr` | kernel ของ PR #12556 head `ba27327b` คัดลอกตรงตัว: masked volatile load ไม่มี store |
  | `sum` | `torch.sum` ที่ใช้ใน B12–B16 (เพื่อเชื่อมกับข้อมูลเดิม) |
  | `none` | ไม่ล้าง (เฉพาะ case ควบคุม cuBLAS GEMV 4096²) |

  สลับลำดับโหมดต่อ case และต่อ run
- **kernel** (ทุกตัวต้องผ่าน correctness เทียบ fp32 ≤ 2e-2 ก่อน):

  | กลุ่ม | ขนาด | candidate |
  |---|---|---|
  | `gemv` | ชุดเดิมของ B16, 3 shape | `cublas_fp16`, `tinygemm_k{2,4,8}`, `bnb_nf4` |
  | `attn` | decode ด้วย `F.scaled_dot_product_attention` (GQA 32/8 heads, d 128), KV cache 16 / 64 / 256 MiB | `sdpa_bf16` |
  | `rmsnorm` | prefill [T, 4096], T = 2048 / 8192 | `liger_rmsnorm`, `torch_rmsnorm` |
  | `swiglu` | prefill [T, 14336], T = 512 / 2048 | `liger_swiglu`, `torch_swiglu` |

  - บันทึกชื่อ CUDA kernel ที่แต่ละ candidate เรียกจริงด้วย `torch.profiler`
  - ตรวจ PTX ของ flush ของ PR
- **จำนวนรอบ:** 5 run ต่อ GPU
- **GPU (PI อนุมัติ):**

  | สถาปัตยกรรม | GPU |
  |---|---|
  | Ada | RTX 4060 (24 MiB), 4060 Ti (32), 4080 SUPER **เครื่องอื่นที่ไม่ใช่ machine 44860 ของ B11–B16** (64), 4090 (72), L40S (96) |
  | Blackwell | RTX 5060 Ti (32), 5090 (96) |
  | Ampere | A100 (40) |
  | Hopper | H100 (50, NVL หรือ SXM) |

- **ลำดับการรัน:**
  1. pilot บน 4060 Ti ก่อน (ราคาถูกที่สุด) เพื่อตรวจ pipeline
  2. การ์ด consumer และ A100 พร้อมกัน
  3. H100 และ L40S หลังรู้ค่าใช้จ่ายสะสม
- **งบ:** เพดาน $1.5 สำหรับ B17 ทั้งหมด
  - แต่ละ instance มี `HARD_CAP_S` (H100 20 นาที, ตัวอื่น 30 นาที)
  - ถ้าค่าใช้จ่ายสะสมบวกกรณีแย่สุดของ H100 จะเกิน $1.5 ให้ตัด L40S ก่อน

**นิยาม:**
- B = t_zero / t_pr − 1
- B_ref = B ของ `cublas_fp16` ที่ GEMV 14336×4096 (weight 112 MiB ใหญ่กว่า L2 ทุกตัว)

**กฎ (ต่อ GPU):**

| ID | เกณฑ์ |
|---|---|
| **E-0 (ต้องผ่านก่อน)** | ≥ 4 run · PTX ของ flush ของ PR มี volatile global load ≥ 1 และไม่มี global store และ buffer ไม่เปลี่ยน · GPU ที่ L2 ≥ 64 MiB: t_pr / t_none ≥ 1.2 ที่ cuBLAS GEMV 4096² (weight 32 MiB ซึ่งอยู่ใน L2 ได้ถ้าไม่ล้าง) แปลว่า flush ของ PR ไล่ข้อมูลออกจริง |
| **F-1 (ตัวหลัก)** | B_ref ≥ 0.05 → affected · ≤ 0.02 → not affected · นอกนั้น marginal |
| F-3 | D_q = S_q,zero / S_q,pr − 1 (q = tinygemm ตัวดีสุดของแต่ละโหมด, bnb_nf4) · worst \|D\| ≥ 0.10 → distorted · ≤ 0.03 → faithful · นอกนั้น mildly |
| F-4 | ต่อกลุ่ม attn / rmsnorm / swiglu ที่ขนาดใหญ่สุด, max B ข้าม candidate: ≥ 0.05 affected · ≤ 0.02 not affected |
| F-6 (รายงาน) | R = t_sum / t_pr − 1 ที่ GEMV ที่ t_pr ≥ 38 µs: median และ max \|R\| (baseline เดิมของเราต่างจากวิธีแก้ของ PR เท่าไร) |

**กฎ (ข้าม GPU):**

| ID | เกณฑ์ |
|---|---|
| **F-2** | Spearman ρ(L2, B_ref) บน Ada ที่ผ่าน E-0 (ต้อง ≥ 4 ตัว): ≥ 0.8 → scales with L2 within Ada · ≤ 0.4 → not explained by L2 size · นอกนั้น mixed |
| F-5 | 4080 SUPER เครื่องที่สอง ใช้ baseline `sum` แบบ B16: \|B_sum(cuBLAS 14336×4096) − 0.1661\| ≤ 0.05 และ \|D_sum(tinygemm 14336×4096) − (−0.1696)\| ≤ 0.05 → replicates |

**คำทำนายของผู้วิจัย:**
- **E-0:** ผ่านทุก GPU (ปานกลาง; ความเสี่ยงคือ toolchain ของ Blackwell, bnb บน sm_120 และ container)
- **F-1:** ใช้ model หยาบ bias ≈ residue / byte ที่ kernel อ่าน และ residue ≈ 0.27 × L2 (จาก B16 บน 4080S/4090)
  - 4060 ≈ 0.06 (affected หรือ marginal)
  - 4060 Ti / 5060 Ti ≈ 0.08
  - 4080S ≈ 0.16, 4090 ≈ 0.17
  - L40S / 5090 ≥ 0.10
  - A100 / H100 = affected (ความมั่นใจต่ำเรื่องขนาด เพราะ L2 แบ่ง partition และใช้ HBM)
  - **สรุป:** affected ทุกตัว ยกเว้น 4060 ที่อาจเป็น marginal
- **F-2:** scales with L2 within Ada (ปานกลาง; ความเสี่ยงคือ L40S ที่ใน B15 บิดน้อยกว่า 4090)
- **F-3:**
  - distorted บน 4080S / 4090 / L40S / 5090
  - mildly หรือ distorted บน A100 / H100 (ต่ำ)
  - mildly บน 4060 / 4060 Ti / 5060 Ti
- **F-4:** attn, rmsnorm และ swiglu = affected บน GPU ที่ L2 ≥ 64 MiB · ส่วน L2 ≤ 32 MiB เป็น marginal หรือ not affected (ปานกลาง)
- **F-5:** replicates (ปานกลาง–สูง)
- **F-6:** \|R\| ≤ 0.05 สำหรับ GEMV ที่ยาว ≥ 38 µs คือ baseline เดิมของเราใกล้กับวิธีแก้ของ PR (ปานกลาง)

**ความหมาย:**
- **ถ้า F-1 = affected บน A100/H100:** คำแนะนำใน report ใช้กับ GPU ที่คนส่วนใหญ่ใช้ benchmark จริง
- **ถ้า F-2 ผ่าน:** ข้อสรุปว่า "L2 ใหญ่ ความเพี้ยนมาก" แยกออกจากผลของรุ่นการ์ดได้
- **ถ้า F-6 เล็ก:** ตัวเลข B12–B16 ใช้ได้กับวิธีแก้ของ PR · ถ้าใหญ่ ต้องรายงานตัวเลขใหม่ด้วย baseline ของ PR

**ข้อจำกัดที่รู้ล่วงหน้า:**
- จับเวลาอย่างเดียว ไม่มี counter
- container / VM ที่ไม่ล็อก clock
- เครื่องเดียวต่อรุ่น ยกเว้น 4080S
- ตัวจับเวลาละเอียดราว 1 µs

**B17a amendment (เขียน 2026-11-06 ราว 09:10 UTC ก่อนรันชุดใหญ่ ระหว่างที่ pilot บน 4060 Ti กำลังตรวจ pipeline):**
- **ที่มา:** ผลสำรวจ source code ของเครื่องมือ benchmark (D-075) พบว่า PyTorch Inductor (benchmarker ตั้งต้นของ `torch.compile` autotune), nvbench และ CUTLASS `tune` ล้าง L2 ด้วยการ `zero_()` / memset **buffer ขนาดเท่า L2 พอดี** ไม่ใช่ 256 MiB
- **เพิ่มโหมด `l2w`:** patch `clear_cache` ให้ `zero_()` buffer int32 ขนาด L2 (จัดสรรแบบเดียวกับ Inductor) ภายใน do_bench เดิม
  - ใช้ median ของ do_bench ทั้งหมด ไม่ได้เลียนแบบสถิติของ Inductor
  - สลับลำดับ 4 โหมดต่อ case และต่อ run
- **กฎ F-7:** B_l2 = t_l2w / t_pr − 1 ที่ case อ้างอิง (GEMV 14336×4096)
  - B_l2 − B_ref ≥ 0.02 → "L2-sized write flush biases more than do_bench"
  - ≤ −0.02 → "less"
  - นอกนั้น similar
  - รายงาน B_l2 ทุก case
- **คำทำนาย:** **more** บน Ada (ปานกลาง)
  - การเขียนครั้งเดียวขนาด 1 × L2 ทิ้ง dirty ไว้ราว cap ≈ 0.5 L2 (B9b: w = 1 L2 เขียนกลับระหว่าง kernel แค่ราว 0.4)
  - ส่วนการเขียน 256 MiB ทิ้งไว้ราว 0.24–0.29 L2 (B12, B16)
  - ดังนั้น B_l2 ≈ 1.5–2 × B_ref
  - บน A100/H100 ไม่ทำนาย
- **ข้อมูลของ pilot (4060 Ti, probe ก่อน amendment) ใช้ตรวจ pipeline เท่านั้น** ชุดใหญ่จะรัน 4060 Ti ใหม่ด้วย probe ที่มีโหมด l2w
- กฎ E-0, F-1..F-6 และคำทำนายเดิมไม่เปลี่ยน

**B17b amendment (เขียน 2026-11-06 ราว 09:20 UTC):**
- **ข้อมูลที่เห็นก่อนเขียน:** ผลของ 4080 SUPER เครื่องที่สองเพียงเครื่องเดียว ส่วน 4060 / 4060 Ti / 4090 / 5060 Ti รันเสร็จแล้ว**แต่ยังไม่ได้เปิดดูผล** และ 5090 / A100 ยังรันอยู่
- **สิ่งที่พบบน 4080S:** E-0 ข้อ (c) ไม่ผ่าน (t_pr / t_none = 1.02)
  - แต่ t_none มีสองค่า: 33.8 µs ใน 2/5 run (weight อยู่ใน L2) และ 52.1–52.2 µs ใน 3/5 run (เท่ากับอ่านจาก DRAM)
  - แปลว่า**แม้ไม่ล้างเลย L2 ก็ไม่เก็บ weight 32 MiB ไว้ทุกครั้ง** สมมติฐานของ case ควบคุมจึงผิด
  - ใน run ที่ weight อยู่ใน L2 ได้ t_pr = 53.25 µs ไม่ใช่ 33.8 และ t_pr = t_sum ทุก run แปลว่า flush ของ PR ไล่ข้อมูลออกจริง
- **E-0' (แทนข้อ c):**
  - ข้อ (a) และ (b) เหมือนเดิม
  - (c') flush ของ PR ทำให้ cuBLAS GEMV 4096² "เย็น" อย่างน้อยเท่า `torch.sum` flush ซึ่งยืนยันด้วย DRAM counter แล้วใน B12 G-0: t_pr ≥ t_sum − max(0.03 t_sum, 1.1 µs)
  - ใช้กับทุก GPU
- **การรายงาน:** ให้คะแนนทุกกฎทั้งภายใต้ E-0 (ล็อกเดิม) และ E-0' (amendment) แล้วรายงานทั้งสองแบบ
  - ภายใต้ E-0 เดิม 4080S ถูกตัดออก
  - โหมด `none` ยังบันทึกต่อ แต่ใช้เป็นข้อมูลเชิงสำรวจเท่านั้น

---

> **ผล (MEASURED 2026-11-06 09:12–09:30 UTC, `benchmarks/results/b17_*_20261106/`, D-078):**
> - ทุก GPU รันครบ 5 run, correctness ผ่านทุก candidate (`invalid` ว่าง), Liger ติดตั้งได้ทุกเครื่อง
> - PTX ของ flush ของ PR มี `ld.volatile.global` และไม่มี store
> - ให้คะแนนซ้ำในเครื่องได้ผลตรงกับบนเครื่องเช่าทุกบรรทัด
> - ค่าใช้จ่ายรวม $1.012 (pilot $0.02) ทุกเครื่อง destroy แล้ว
>
> | GPU | L2 | E-0 / E-0' | **F-1 B_ref** | F-7 B_l2 | F-3 (D แย่สุด, ทิศ) | F-4 max B: attn / rmsnorm / swiglu | F-6 max \|R\| |
> |---|---|---|---|---|---|---|---|
> | RTX 4060 (Ada) | 24 MiB | ✅ / ✅ | **+0.067** affected | +0.064 | distorted (−0.240, ต่ำกว่าจริง) | 0.07 / 0.16 / 0.11 | 0.010 |
> | RTX 4060 Ti (Ada) | 32 | ✅ / ✅ | **+0.086** | +0.087 | distorted (−0.229, ต่ำกว่าจริง) | 0.09 / 0.23 / 0.17 | 0.004 |
> | RTX 5060 Ti (Blackwell) | 32 | ✅ / ✅ | **+0.066** | +0.073 | mildly (−0.075, mixed) | 0.04 / 0.11 / 0.09 | 0.023 |
> | A100 PCIe (Ampere) | 40 | ✅ / ✅ | **+0.160** | +0.160 | distorted (+0.308, mixed) | 0.08 / 0.16 / 0.13 | 0.018 |
> | H100 NVL (Hopper) | 60 | ✅ / ✅ | **+0.231** | +0.229 | distorted (+0.220, **สูงกว่าจริง**ทุกตัว) | 0.08 / 0.14 / 0.13 | 0.011 |
> | RTX 4080 SUPER #2 (Ada) | 64 | ❌ / ✅ | **+0.160** | +0.161 | distorted (−0.170, ต่ำกว่าจริง) | 0.18 / 0.55 / 0.31 | 0.018 |
> | RTX 4090 (Ada) | 72 | ❌ / ✅ | **+0.174** | +0.177 | distorted (−0.178, ต่ำกว่าจริง) | 0.20 / 0.57 / 0.32 | 0.019 |
> | L40S (Ada) | 96 | ❌ / ✅ | **+0.253** | +0.256 | distorted (−0.168, mixed) | 0.28 / **0.88** / 0.48 | 0.021 |
> | RTX 5090 (Blackwell) | 96 | ❌ / ✅ | **+0.108** | +0.101 | distorted (+0.371, mixed) | 0.06 / 0.26 / 0.24 | 0.002 |
>
> **ข้ามการ์ด:**
> - **F-2:**
>   - ภายใต้ E-0 (ล็อกเดิม) เหลือ Ada แค่ 2 ตัว จึงตัดสินไม่ได้ (not enough)
>   - ภายใต้ E-0' (B17b) **scales with L2 within Ada, ρ = 1.0** บน 5 ตัว (24 → 96 MiB: 0.067 → 0.253)
> - **F-5: replicates** บน 4080S เครื่องที่สอง (B_sum 0.162 เทียบกับ 0.166; D_sum −0.168 เทียบกับ −0.170)
> - **residue ÷ L2 (B_ref × 112 MiB / L2):** Ada 0.27–0.31 คงที่ทั้ง 5 ตัว · A100 0.45, H100 0.43 (ใกล้ ½ L2 ตรงกับที่ Triton #12554 เห็นบน A100) · Blackwell 0.13 (5090) / 0.23 (5060 Ti)
>
> **คำทำนายของผู้วิจัย:**
> - **ถูก:**
>   - E-0' ผ่านทุกตัว
>   - F-1 affected ทุกตัว รวม 4060 และ A100/H100
>   - F-2 (ภายใต้ E-0')
>   - F-3 บน L2 ≥ 64 MiB และ A100/H100
>   - F-4 บน L2 ≥ 64 MiB
>   - F-5
>   - F-6 (\|R\| ≤ 0.023)
> - **ผิด:**
>   - **E-0 ตามที่ล็อก** ไม่ผ่านบนการ์ด L2 ≥ 64 MiB ทั้ง 4 ตัว เพราะ case ควบคุมผิด (B17b)
>   - F-3 บน 4060 / 4060 Ti ทำนาย mildly แต่ได้ distorted (−23 ถึง −24 %)
>   - F-4 บนการ์ด L2 ≤ 32 MiB ทำนาย marginal แต่ได้ affected เกือบทุกกลุ่ม
>   - **F-7 ทำนาย "more" แต่ได้ similar ทั้ง 9 ตัว:** การล้างแบบเขียนขนาด L2 ของ Inductor / nvbench / CUTLASS บิดพอ ๆ กับ do_bench
>
> **ข้อสังเกตเชิงสำรวจ (ไม่ใช่กฎที่ล็อก):**
> - **ทิศของการบิด speedup ของ int4 ขึ้นกับ GPU:**
>   - Ada consumer ต่ำกว่าจริง
>   - H100 สูงกว่าจริงทุก shape
>   - A100 / 5090 / L40S สองทิศ
>   - เพราะ penalty ของ cuBLAS fp16 กับ kernel int4 ไม่ได้สัดส่วนกันในแต่ละการ์ด
> - **โหมดไม่ล้าง (`none`):** บน 4080S / 4090 / L40S มีสองค่า ส่วนบน 5090 ช้ากว่าการล้างด้วย PR (55–80 เทียบกับ 26.6 µs) ยังอธิบายไม่ได้
> - **5090 + โหมด l2w:** bnb_nf4 ช้าลง 4–5 เท่า (79–96 µs เทียบกับ 14–48 µs) ยังอธิบายไม่ได้
> - **kernel ของ Liger ได้รับผลมากกว่า PyTorch eager ในงานเดียวกัน:** RMSNorm +0.88 เทียบกับ +0.06 บน L40S เพราะ eager แตกเป็นหลาย kernel และ kernel แรกรับ residue ไปคนเดียว
> - **ตัวจับเวลา:** H100 ไม่มีตาราง 1.024 µs (tick fraction 0.01) ส่วนการ์ดอื่น 0.28–0.83
> - **(เพิ่มหลังการตรวจแบบอิสระ D-079) kernel int4 ตัวที่เร็วกว่าสลับกันบน 2/9 GPU:**
>   - บน 4060 และ 4060 Ti ที่ 14336×4096 แบบ zero ดูเหมือน bnb เร็วกว่า (2.98× เทียบกับ 2.79× บน 4060)
>   - แต่แบบ pr ได้ tinygemm เร็วกว่า 11–13 % (3.67× เทียบกับ 3.29×)
>   - ส่วน GPU และ shape อื่นไม่สลับ
> - **(D-079) โหมด l2w กับ kernel สั้น:** Liger RMSNorm [2048, 4096] (16 MiB)
>   - บิดหนักกว่า zero มากบน A100 (+352 % เทียบกับ +48 %), H100 (+248 % เทียบกับ +16 %), 5090 (+537 % เทียบกับ +39 %) และ 4090 (+164 % เทียบกับ +82 %)
>   - บน GPU อื่นอีก 5 ตัวเท่ากัน (±3 จุด)
>   - กฎ F-7 ดูเฉพาะ GEMV อ้างอิง จึงยังเป็น "similar" ตามที่ล็อกไว้
> - **(D-079) kernel เล็กโดนหนักกว่า:** Liger RMSNorm 16 MiB ภายใต้ zero ได้ +16 % ถึง +110 % เทียบกับ +11 % ถึง +88 % ที่ 64 MiB ซึ่งสอดคล้องกับ penalty แบบคงที่

---

## B18 pre-registration: วิธีล้าง L2 แบบไหนใช้ได้ และใครจ่ายค่าเขียนของ kernel เอง (H-0..H-4) — เขียน 2026-11-06 ก่อนวัด B18 ใด ๆ

**ที่มา (PI, D-080):** "ทั้งหมดทั้งมวลนี้มันไม่ได้เสนอวิธีการแก้เลยนี่ มีแต่บอกว่า err % เท่าไหร่เฉย ๆ" และ "ทำเลย ตอนนี้มี 3060-laptop ละ"

**สิ่งที่มีอยู่แล้ว (ไม่วัดซ้ำเป็นคำถามหลัก):**
- B12 (G-0) ยืนยันด้วย counter บน VM 3 ตัวแล้วว่าหลังล้างด้วยการอ่าน (`sum`) GEMV เขียน DRAM ≤ 0.05 L2
- B17 วัดเวลาของวิธีล้าง zero / pr / sum / l2w บน 9 GPU แล้ว

**คำถามใหม่สามข้อ (วิธีแก้ที่คนอาจเลือก ซึ่งยังไม่มีใครวัด):**
1. **รอหลังล้างแบบเขียน** (sync + sleep) หรือ**ล้างแบบเขียนแล้วตามด้วยการอ่าน** ลบ residue ได้ไหม
2. **ล้างแบบอ่านด้วย buffer ขนาดเท่า L2** (patch ขั้นต่ำของ Inductor ถ้าเปลี่ยนแค่ `zero_()` เป็นการอ่านบน buffer เดิม) ไล่ข้อมูลของ kernel ออกพอไหม
3. **ใครจ่ายค่าเขียนของ kernel เอง:** ภายใต้การล้างแบบอ่าน เทียบกับการหมุน input (FlashInfer path CUDA graph, QuACK autotuner)

**Design** (`flush_fix_probe.py`, `flush_fix_probe.sh`, `flush_fix_score.py`, `test_flush_fix_score.py`; sha256 ใน `research/preregistration/LOCK_20261106_b18.sha256`):
- **GPU:** RTX 3060 Laptop (GA106, L2 3,145,728 B) ในเครื่อง counter เปิดแล้ว (D-031) ไม่ต้องเช่า
- **kernel ที่จับเวลา:**
  - `gemv4`: `F.linear(x[1, 4096], W)` fp16, W = 4 × L2 (12 MiB), output 3 KB
  - `e_tiny`: `torch.add(x, 1, out=y)` float32, x และ y ขนาด L2/128 (24 KiB) footprint L2/64
  - `e_small`: เหมือนกัน x และ y ขนาด L2/8 (384 KiB) footprint 0.25 L2 output ต่ำกว่าเพดาน dirty
  - `e_large`: เหมือนกัน x และ y ขนาด 2 × L2 (6 MiB)
- **โหมด (เปลี่ยนตัวแปรเดียวคือสิ่งที่เกิดก่อนการเรียกที่ถูกวัด):** `none`, `zero256` (แบบ do_bench), `zero256_wait` (zero256 + sync + sleep 10 ms), `zero256_read` (zero256 แล้วตามด้วย read flush ของ PR บน buffer เดียวกัน), `zeroL2` (แบบ Inductor / nvbench / CUTLASS), `pr256` (อ้างอิง: kernel ของ PR #12556 head `ba27327b`), `sum256`, `prL2` (read flush ของ PR บน buffer ขนาด L2), `pr2L2` (2 × L2), `rotate` (ไม่ล้าง หมุนสำเนา input/output จน footprint รวม ≥ 4 × L2)
- **range mode** (ncu `--replay-mode app-range --cache-control none --clock-control none`, 5 รอบต่อ cell × 3 run, สลับลำดับโหมดทุกรอบด้วย seed 1818):
  - โหมดทั่วไป: เรียก kernel หนึ่งครั้ง (สถานะเดียวกับ loop ของ do_bench) → ล้าง → sync → **range: เรียก kernel → sync**
  - `rotate`: เรียกหมุนสำเนา copies + 1 ครั้ง → sync → **range: เรียกสำเนาถัดไป → sync**
  - control: range ว่าง 2 ช่วงหลัง pr256
- **timing mode** (ไม่มี profiler, เฉพาะ gemv4 และ e_large): CUDA event ต่อการเรียก ล้างก่อนทุกครั้งโดยไม่ sync แบบ do_bench, block ละ 5 ครั้งต่อโหมด ทิ้ง 2 ครั้งแรก, 20 รอบ × 3 run **ใช้รายงานเท่านั้น** เพราะบน L2 3 MiB ผลต่อเวลาเล็กและตัวจับเวลาหยาบ (≈ 1.02 µs)
- **ตรวจก่อนวัด:** ทุกสำเนาของทุก kernel เทียบกับ reference (GEMV เทียบ fp32 tol 2 %, add ต้องเท่ากันทุกบิต); flush ของ PR ต้องไม่เปลี่ยน buffer และ PTX ไม่มี `st.global`
- **pipeline check ก่อนล็อก:** `--mode check` ผ่าน (correctness 0, PTX มี `ld.volatile.global` 2 ตัว ไม่มี `st.global`, buffer ไม่เปลี่ยน) และ smoke ของ ncu (`benchmarks/results/flush_fix_smoke_20261106-143005/`) ตรวจแค่จำนวนแถว (42 range ตรงกับ event) **ยังไม่ได้เปิดดูค่าวัดของ smoke**

**ตัววัด:** ต่อ (kernel k, โหมด m) W, R = median ของ byte ที่เขียน / อ่าน DRAM ใน range ของการเรียกที่ถูกวัด
- w = W / L2 · r = R / byte ของ input · ow = W / byte ของ output

| ID | เกณฑ์ |
|---|---|
| **H-0 (ต้องผ่านก่อน)** | correctness 0 (รวม flush ของ PR); control เขียน ≤ 64 KiB; gemv4 หลัง pr256 เขียน ≤ 0.05 L2 และอ่าน 0.9–1.2 × W; e_large หลัง pr256 อ่าน 0.9–1.2 × input · ไม่ผ่าน → หยุด ไม่ให้คะแนน |
| H-0e | X = w(gemv4, zero256) − w(gemv4, pr256) ≥ 0.05 (มี residue จากการล้างแบบเขียน) · ไม่ผ่าน → H-1, H-2 ให้คะแนนไม่ได้ |
| **H-1 (รอ)** | [W(zero256_wait) − W(pr256)] / [W(zero256) − W(pr256)] บน gemv4 · ≥ 0.7 → **รอไม่ช่วย** · ≤ 0.3 → รอแล้วหาย · นอกนั้น partial |
| **H-2 (เขียนแล้วอ่าน)** | w(zero256_read) − w(pr256) บน gemv4 · ≤ 0.02 → **ลบ residue ได้** · ≥ 0.05 → ไม่ได้ · นอกนั้น partial |
| **H-3 (ขนาดของ read flush)** | ต้องมี r(e_tiny, pr256) ≥ 0.9 และ r(e_small, pr256) ≥ 0.9 ก่อน · **a:** r(e_tiny, prL2) ≤ 0.6 → **buffer ขนาด L2 ปล่อย kernel จิ๋วให้อุ่น** · ≥ 0.9 → พอ · **b:** r(e_small, prL2) ≥ 0.9 → พอที่ footprint 0.25 L2 · ≤ 0.6 → ไม่พอ · **c:** min r(e_tiny, e_small ภายใต้ pr2L2) ≥ 0.9 → **2 × L2 พอ** |
| **H-4 (ใครจ่ายค่าเขียน)** | **a:** ow(e_small, pr256) ≤ 0.1 → **การล้างแบบอ่านไม่นับค่าเขียนกลับของ kernel ที่ output < ½ L2 ไว้ในหน้าต่างของมัน** · ≥ 0.5 → นับ · **b:** ow(e_small, rotate) ≥ 0.7 → **การหมุนคิดค่าเขียนราว 1 output ต่อครั้ง** · ≤ 0.3 → ไม่คิด · **c:** ow(e_large, pr256) ≥ 0.6 → **kernel ที่ output > ½ L2 จ่ายค่าเขียนของตัวเองเกือบทั้งหมด** · ≤ 0.3 → ไม่จ่าย |
| รายงาน | w / r / ow ทุก cell, residue ของ zeroL2 และ sum256 เทียบ pr256, timing (median, mean, bias เทียบ pr256) |

**คำทำนายของผู้วิจัย (เขียนก่อนเห็นค่าวัดใด ๆ ของ B18):**
- **H-0 ผ่าน** และ **H-0e ผ่าน** (ปานกลาง: GA102 ซึ่งเป็น Ampere เหมือนกันมี X = 0.46 ใน B12 แต่ TU106 มีแค่ 0.05 จึงไม่แน่นอน)
- **H-1 = รอไม่ช่วย** (ปานกลาง–สูง): B9 วัดแล้วว่า GA106 ไม่เขียน dirty line กลับตอนว่าง ข้อที่อาจทำให้ ratio ต่ำกว่า 1 คือ write-back ที่ค้างคิวตอน flush จบ (B9b) ซึ่งน่าจะเสร็จระหว่างรอ
- **H-2 = ลบ residue ได้** (สูง): การอ่าน 256 MiB ไล่ dirty line ทั้งหมดออกและเขียนกลับระหว่างการอ่าน ซึ่งอยู่นอกหน้าต่างวัด
- **H-3a = buffer ขนาด L2 ปล่อย kernel จิ๋วให้อุ่น** (ต่ำ–ปานกลาง) ค่าที่ทำนาย r ≈ 0.31
  - ที่มา: ระยะ reuse ของ x คือ footprint ของ kernel + buffer = 1/64 + 1 = 1.016 L2 แล้วอ่านค่าจาก cliff ของ B8 (RDR 0.073 ที่ 1.0, 0.539 ที่ 1.031)
  - สมมติว่า cliff ของการอ่านวนซ้ำใน B8 ใช้กับลำดับ "kernel → flush → kernel" ได้ ซึ่งยังไม่เคยทดสอบ
- **H-3b = พอ** (ปานกลาง–สูง): ระยะ reuse 1.25 L2 เลย cliff (RDR(1.063) = 0.998)
- **H-3c = 2 × L2 พอ** (สูง)
- **H-4a = ไม่นับค่าเขียนของ kernel เล็กไว้ในหน้าต่าง** (สูง): output 0.125 L2 ต่ำกว่าเพดาน dirty (B6b: เขียน 0.4375 L2 → ลง DRAM 1.4 %)
- **H-4b = การหมุนคิดราว 1 output ต่อครั้ง** (ปานกลาง–สูง): ใน steady state ข้อมูลที่เขียนทุก byte ต้องถูกเขียนกลับ และไม่มี flush มารับแทน ค่าที่ทำนาย ≈ 0.85–1.0 (หักส่วนที่ range นับไม่ถึงแบบ C6)
- **H-4c = จ่ายเกือบทั้งหมด** (ปานกลาง): เหนือเพดาน dirty ข้อมูลถูกเขียนกลับระหว่าง kernel (WBR ≈ 1) แต่การเขียนยาวทิ้ง residue ไว้ 0.05–0.46 L2 (B12) ค่าที่ทำนาย ≈ 0.75–0.95

**ความหมายถ้าเป็นตามที่ทำนาย:**
1. การ sync หรือรอหลังล้างแบบเขียนไม่ใช่วิธีแก้
2. เครื่องมือที่ใช้ memset (nvbench, CUTLASS) แก้ได้ด้วยการเพิ่มรอบอ่าน แต่ล้างแบบอ่านอย่างเดียวถูกกว่า
3. ถ้า Inductor เปลี่ยนแค่ `zero_()` เป็นการอ่านบน buffer ขนาด L2 เดิม kernel จิ๋วจะยังอุ่นอยู่ buffer ต้องใหญ่กว่า L2 (≥ 2 × L2) (ยืนยันบน GA106 เท่านั้น)
4. **การล้างแบบอ่านเองก็ไม่ใช่ "เวลาจริง":** kernel ที่ output < ½ L2 ไม่ต้องจ่ายค่าเขียนกลับของตัวเองในหน้าต่างจับเวลา ขณะที่การหมุน input คิดค่านี้ราว 1 output ต่อครั้ง วิธีล้างจึงเป็นการเลือกว่าใครจ่ายค่าเขียน ควรรายงานทั้งสองแบบหรือเลือกตามบริบทการใช้งานจริง

**ข้อจำกัด:**
- GPU ตัวเดียว (GA106, L2 3 MiB) บน WSL2 ซึ่งมี traffic เบื้องหลังจาก Windows (ส่วนใหญ่เป็นการอ่าน) นโยบายของ L2 บน Ada / Hopper (L2 แบ่ง partition) อาจต่างออกไป
- range แบบ app-range มี sync ก่อนเริ่ม จึงไม่ใช่ path เดียวกับ do_bench ทุกประการ และอาจนับ write-back ที่เสร็จหลัง kernel จบไม่ครบ (C6)
- kernel แค่ GEMV และ elementwise add ผลต่อเวลาเป็นข้อมูลรายงานเท่านั้น

---

> **ผล (MEASURED 2026-11-06 14:32–14:40 UTC, `benchmarks/results/flush_fix_20261106-143212/`, D-081):**
>
> | กฎ | ค่า | ผล | คำทำนาย |
> |---|---|---|---|
> | H-0 | correctness 0, control เขียน 0, gemv4 หลัง pr256 เขียน 0.003 L2 อ่าน 1.00 × W, e_large อ่าน 1.00 × input | ✅ ผ่าน | ✓ |
> | H-0e | X = 0.499 L2 | ✅ มี residue | ✓ |
> | **H-1** | ratio 0.946 | **รอไม่ช่วย** | ✓ |
> | **H-2** | 0.006 L2 | **เขียนแล้วอ่านลบ residue ได้** | ✓ |
> | **H-3a** | r(e_tiny, prL2) = 1.005 | **พอ** (kernel จิ๋วเย็นสนิท) | ❌ **ผิด** (ทำนาย 0.31) |
> | H-3b | r(e_small, prL2) = 1.000 | พอ | ✓ |
> | H-3c | min r = 1.000 | 2 × L2 พอ | ✓ |
> | **H-4a** | ow(e_small, pr256) = 0.000 | **การล้างแบบอ่านไม่นับค่าเขียนของ kernel เล็กไว้ในหน้าต่าง** | ✓ |
> | **H-4b** | ow(e_small, rotate) = 0.987 | **การหมุนคิดราว 1 output ต่อครั้ง** | ✓ |
> | H-4c | ow(e_large, pr256) = 0.750 | kernel ที่ output > ½ L2 จ่ายเกือบทั้งหมด | ✓ (ทำนาย 0.75–0.95) |
>
> **คำทำนายถูก 9 จาก 10 กฎ** (นับ H-0e) ข้อที่ผิดคือ H-3a: cliff ของการอ่านวนซ้ำใน B8 ใช้กับลำดับ kernel → flush → kernel ไม่ได้ (คำอธิบายหลังเห็นผล ยังไม่ทดสอบ: line ที่เคย hit ซ้ำถูกปกป้อง แต่ input ที่ถูกล้างทุกครั้งไม่เคย hit)
>
> **descriptive (counter):**
> - residue หลัง `zero_()` 256 MiB บน GA106 = **0.50 L2** ตรงกับค่าที่ B9b ทำนาย (0.472) ภายใน 0.03 (B12 บน GPU อื่นไม่ตรง)
> - zeroL2 ทิ้ง residue 0.585 L2 (มากกว่า 256 MiB เล็กน้อย)
> - **การล้างแบบอ่านเลื่อนค่าเขียนกลับของ kernel เองออกนอกหน้าต่างได้ราว ½ L2:** e_small (output 0.125 L2) ไม่ถูกคิดเลย, e_large (output 2 L2) ถูกคิด 1.5 L2 ส่วนแบบ none / rotate / zero256 คิดราว 2.0 L2
> - **read flush ขนาด L2 ไล่ input ออกหมด แต่ทิ้ง output ส่วนท้ายของรอบก่อนไว้เป็น dirty:** e_small เขียน 0.056 L2 ในหน้าต่าง (0.45 ของ output) ส่วน 2 × L2 สะอาด (0.0001 L2)
> - sum256 ≈ pr256 ทุก kernel
>
> **descriptive (timing, ไม่ได้ล็อกเป็นกฎ):**
> - gemv4: zero256 +10.7 % เทียบ pr256 (ทำนายจาก byte 0.5 / 4 = 12.5 %) · e_large zero256 +13 %
> - **zero256_wait แย่ลงมาก:** gemv4 +381 %, e_large +250 % ทั้งที่ byte เท่ากับ zero256 (สาเหตุที่เป็นไปได้: clock ลดระหว่าง sleep 10 ms และ queue ว่างหลัง sync ยังแยกไม่ได้ เพราะ monitor 1 Hz ละเอียดไม่พอ)
> - **flush ที่สั้นหรือไม่ล้างเลยทำให้เวลาสูงขึ้นทั้งที่ byte เท่ากัน:** gemv4 none +37 %, rotate +37 %, prL2 +70 %, pr2L2 +68 %, zeroL2 +66 % · e_large none / rotate / pr2L2 +13 %, prL2 +46 % (counter: r = 1.00 และ w ใกล้ pr256 ทุกตัว)
>   - **สมมติฐานหลังเห็นผล (ยังไม่ทดสอบ):** flush 256 MiB ใช้เวลาราว 0.9 ms บนเครื่องนี้ จึงทำให้ queue ของ GPU ไม่ว่าง ถ้า flush สั้นกว่า overhead ของ Python ต่อรอบ queue จะว่าง event เริ่มต้นจึงถูกบันทึกก่อน kernel มาถึง เวลาที่วัดได้รวม launch latency ของ host
>   - ถ้าจริง จะอธิบายข้อผิดปกติสองข้อของ B17 ได้: โหมด l2w กับ kernel สั้นบน GPU เร็ว (A100 / H100 / 5090 / 4090 ซึ่ง flush ขนาด L2 ใช้เวลาเพียงราว 16–70 µs) และการไม่ล้างที่ช้ากว่าการล้างบน H100 / 5090

---

## B18b pre-registration: flush ที่สั้นทำให้ event timing รวม launch latency ของ host หรือไม่ ("queue drain", Q-0..Q-4) — เขียน 2026-11-06 หลังเห็นผล B18 ก่อนวัด B18b ใด ๆ

**ที่มา:** B18 (descriptive) เห็นว่า flush ที่สั้นหรือไม่ล้างเลยทำให้เวลาสูงขึ้น 13–70 % ทั้งที่ counter บอกว่า byte เท่ากัน (D-081) PI: "ทำต่อเรื่อย ๆ จนกว่าจะได้ผลที่ตีพิมพ์ technical report แล้วเท่"

**สมมติฐาน (เขียนหลังเห็น B18):**
- loop แบบ do_bench คือ flush → `start.record()` → `fn()` → `end.record()` โดยไม่ sync
- ถ้า GPU ทำ flush เสร็จก่อนที่ host จะส่ง kernel ของ `fn` มาถึง queue จะว่าง event เริ่มต้นจึงถูก timestamp ทันที และช่วงเวลาที่วัดได้รวมเวลาที่ host ใช้ dispatch `fn` ด้วย
- flush 256 MiB ใช้เวลาราว 0.9 ms บน GA106 จึงทำให้ queue ไม่ว่าง ส่วน flush ขนาด L2 (ราว 10 µs) ทำให้ queue ว่าง
- **ถ้าจริง:** อธิบายข้อผิดปกติของ B17 ได้สองข้อ ได้แก่ l2w กับ kernel สั้นบน A100 / H100 / 5090 / 4090 และการไม่ล้างที่ช้ากว่าการล้างบน H100 / 5090

**Design** (`queue_drain_probe.py`, `queue_drain_probe.sh`, `queue_drain_score.py`, `test_queue_drain_score.py`; sha256 ใน `research/preregistration/LOCK_20261106_b18b.sha256`):
- **GPU:** RTX 3060 Laptop ไม่มี profiler (timing ล้วน)
- **ตัวแปรที่เปลี่ยน:** cond `plain` (แบบ do_bench) เทียบกับ `pad` = `torch.cuda._sleep` ราว 300 µs บน GPU ทันทีก่อน `start.record()` เพื่อให้ GPU ยังไม่ว่างขณะที่ host ส่ง start, fn, end · pad รันก่อน event เริ่มต้นจึงไม่ถูกจับเวลา
- **flush:** pr256, prL2, pr2L2, zero256, zeroL2, zero256_wait
- **kernel:** gemv4 (cuBLAS, W = 4 × L2), e_large (`torch.add` 2 × L2), t_small (Triton kernel y = 2x, x และ y ขนาด L2/2 ซึ่งสั้นและ dispatch จาก Python แบบเดียวกับ Liger ใน B17)
- block ละ 6 ครั้งต่อ (kernel, cond, flush) ทิ้ง 2 ครั้งแรก 20 รอบ × 3 run สลับลำดับด้วย seed 1819 · T = median ของทุก sample
- **รายงานเพิ่ม:** เวลา GPU ของ flush แต่ละแบบ, เวลาที่ host ใช้ส่ง start + fn + end ขณะ GPU ไม่ว่าง, `triton.testing.do_bench` จริงที่ patch `clear_cache` (exploratory)
- **pipeline check ก่อนล็อก:** check mode ผ่าน (correctness 0, pad วัดได้ 304 µs) และ smoke 1 run ตรวจเฉพาะจำนวน cell (36) **ยังไม่ได้เปิดดูค่าเวลาของ smoke**
- **อ้างอิงจาก B18 (counter):** gemv4 และ e_large อ่านและเขียน byte เท่ากันภายใต้ pr256, prL2, pr2L2 · e_small (ใกล้ t_small) ภายใต้ pr2L2 สะอาด ความต่างของเวลาระหว่างคู่เหล่านี้จึงไม่ได้มาจาก DRAM traffic

**ตัววัด:** I(k, cond, a, b) = T(k, cond, a) / T(k, cond, b) − 1

| ID | เกณฑ์ |
|---|---|
| **Q-0 (ต้องผ่านก่อน)** | correctness 0; pad ≥ 200 µs ทุก run; plain I(gemv4, prL2, pr256) ≥ 0.20 (ทำซ้ำ B18) |
| **Q-1 (ตัวหลัก)** | padded I ของ gemv4 (prL2 เทียบ pr256), e_large (prL2 เทียบ pr256), t_small (pr2L2 เทียบ pr256) · max \|I\| ≤ 0.05 → **queue drain อธิบายส่วนที่เกินได้** · min I ≥ 0.15 → ไม่ใช่ queue drain · นอกนั้น partial |
| Q-2 | max ของ \|T(pad, pr256) / T(plain, pr256) − 1\| ≤ 0.05 → flush 256 MiB ทำให้ queue ไม่ว่างอยู่แล้ว |
| **Q-3 (เทียบข้อผิดปกติ l2w ของ B17)** | t_small: P = plain I(zeroL2, zero256), Q = padded I(zeroL2, zero256) · P < 0.30 → ไม่มีข้อผิดปกติบน GPU นี้ · ไม่เช่นนั้น Q ≤ 0.15 → **ข้อผิดปกติคือ queue drain** · Q ≥ 0.30 → ไม่ใช่ · นอกนั้น partial |
| Q-4 | gemv4: padded I(zero256_wait, zero256) ≤ 0.10 → การรอช้าเพราะ queue drain · ≥ 0.30 → เพราะอย่างอื่น (เช่น clock) · นอกนั้น partial |

**คำทำนายของผู้วิจัย:**
- **Q-0 ผ่าน** (สูง: B18 เห็น +70 %)
- **Q-1 = queue drain อธิบายได้** (ปานกลาง: อาจมีผลของ clock ปนอยู่บ้าง เพราะ GPU ว่างนานกว่าเมื่อ flush สั้น)
- **Q-2 ผ่าน** (ปานกลาง–สูง)
- **Q-3 = ข้อผิดปกติคือ queue drain** (ปานกลาง): zeroL2 ราว 10 µs สั้นกว่าเวลา dispatch ของ Triton จาก Python ส่วนเมื่อ pad แล้ว residue ต่างกันแค่ 0.585 กับ 0.50 L2 บน kernel ที่ย้ายราว 1 L2 จึงคาดว่า Q ≈ 0.08
- **Q-4 = อย่างอื่น** (ต่ำ): หลังว่าง 10 ms clock น่าจะลดลง และ pad 300 µs ไม่น่าพอให้ clock กลับมา

**ความหมายถ้าเป็นตามที่ทำนาย:**
- flush 256 MiB ของ do_bench ทำหน้าที่ซ่อน launch latency ไปด้วยโดยไม่ได้ตั้งใจ การลดขนาด flush ให้เท่า L2 (Inductor) หรือไม่ล้างเลยจะเปิดเผย latency นี้ใน kernel สั้น
- เครื่องมือที่ถูกต้องต้องทำให้ GPU ไม่ว่างก่อน event เริ่มต้นเสมอ ไม่ว่า flush จะใหญ่แค่ไหน
- ยังต้องยืนยันบน GPU เร็ว (A100 / H100) ว่าอธิบายตัวเลข +164–537 % ของ B17 ได้จริง ซึ่งต้องเช่าเครื่อง

**ข้อจำกัด:**
- GPU ตัวเดียว, WSL2 (overhead ของ host อาจต่างจาก Linux แท้)
- `torch.cuda._sleep` เป็น API ภายในของ PyTorch
- ไม่ได้วัด clock ระดับ µs (monitor ทุก 200 ms)

> **ผล (MEASURED 2026-11-06 14:44–14:52 UTC, `benchmarks/results/queue_drain_20261106-144419/`, D-083):**
>
> | กฎ | ค่า | ผล | คำทำนาย |
> |---|---|---|---|
> | Q-0 | plain gemv4 prL2 +65.8 % · pad 304 µs · correctness 0 | ✅ ผ่าน | ✓ |
> | **Q-1** | padded I: gemv4 0.000, t_small 0.000, e_large **+0.067** (plain: +0.658, +1.667, +0.434) | **partial** (e_large เกินเกณฑ์ 0.05) | ✗ (ทำนาย "อธิบายได้") |
> | Q-2 | pad/plain − 1 ที่ pr256: +0.018, −0.022, 0.000 | ✅ flush 256 MiB ทำให้ queue ไม่ว่างอยู่แล้ว | ✓ |
> | **Q-3** | t_small: plain zeroL2 เทียบ zero256 **+187 %**, padded **0.0 %** | **ข้อผิดปกติแบบ l2w คือ queue drain** | ✓ |
> | Q-4 | padded zero256_wait เทียบ zero256 +2.4 % (plain +348 %) | **การรอช้าเพราะ queue drain** ไม่ใช่ clock | ✗ (ทำนาย "อย่างอื่น") |
>
> **คำทำนายถูก 3 จาก 5**
>
> **คำอธิบายของส่วนที่เหลือใน Q-1 (หลังเห็นผล ใช้ข้อมูล counter ที่ล็อกไว้แล้วจาก B18):** e_large ภายใต้ prL2 เขียน DRAM ในหน้าต่าง 1.762 L2 เทียบกับ 1.499 L2 ภายใต้ pr256 (+0.263 L2 เพราะ read flush ขนาด L2 ทิ้ง output ส่วนท้ายของรอบก่อนไว้เป็น dirty) บน kernel ที่ย้าย 4 L2 คิดเป็นเวลา ≈ +6.6 % ซึ่งตรงกับ +6.7 % ที่วัดได้ ส่วนนี้จึงเป็น byte จริง ไม่ใช่ queue drain
>
> **descriptive:**
> - เวลา GPU ของ flush: 256 MiB ≈ 1,045 µs, prL2 12 µs, pr2L2 32 µs, zeroL2 18 µs · เวลาที่ host ส่ง start + fn + end บน WSL2: gemv4 139 µs, e_large 72 µs, t_small 78 µs
> - **`triton.testing.do_bench` จริง (exploratory, 3 run):** t_small: shipped 16.4 µs, zeroL2 43–46 µs, pr256 11.3 µs, prL2 41–47 µs, **zeroL2 + pad 15.4 µs, prL2 + pad 13.3 µs** · gemv4: shipped 62.5, pr256 55.3–56.3, prL2 88–94, **prL2 + pad 55.3–56.3 µs** → การเติม `torch.cuda._sleep` ใน `clear_cache` บรรทัดเดียวแก้ queue drain ใน do_bench ได้
> - prL2 + pad ของ t_small ยังสูงกว่า pr256 ราว 2 µs (+17 %) สอดคล้องกับ output ส่วนท้ายที่ค้าง dirty (B18 e_small prL2)

---

## B18c pre-registration: เครื่องมือ `cleanbench` (read flush + เติมงานบน GPU + โหมด steady) ใช้ได้จริงไหม (T-0..T-2) — เขียน 2026-11-06 หลัง B18 / B18b ก่อนวัด B18c ใด ๆ

**ที่มา:** PI ต้องการวิธีแก้ที่เป็นของเรา (D-080, D-082) · `benchmarks/scripts/cleanbench.py` รวมสิ่งที่วัดได้จาก B18 / B18b ไว้ในเครื่องมือเดียว:
1. ล้าง L2 ด้วยการอ่าน (kernel ของ PR #12556) บน buffer ≥ max(256 MiB, 4 × L2)
2. เติมงานบน GPU (`torch.cuda._sleep`) ก่อน event เริ่มต้นทุกครั้ง ขนาด = max(100 µs, 3 × เวลา dispatch ของ host ที่วัดได้) เพื่อไม่ให้ queue ว่าง
3. `mode="steady"`: ไม่ล้าง แต่หมุนสำเนา input / output (รวม ≥ 4 × L2) ให้แต่ละครั้งจ่ายค่าเขียนกลับของครั้งก่อน
4. รายงาน median, IQR, p95 และเตือนเมื่อ kernel สั้นจนตัวจับเวลา (≈ 1.024 µs) แยกไม่ได้

**Design** (`cleanbench.py`, `cleanbench_validate.py`, `cleanbench_validate.sh`, `cleanbench_score.py`, `test_cleanbench_score.py`; sha256 ใน `research/preregistration/LOCK_20261106_b18c.sha256`):
- **kernel:** gemv4 (2 สำเนา), e_large (`torch.add` 2 × L2, 2 สำเนา), **e_mid** (`torch.add` x และ y ขนาด 0.4 L2 ซึ่ง output ต่ำกว่าเพดาน dirty, 5 สำเนา), t_small (Triton y = 2x ขนาด L2/2, 4 สำเนา)
- **config:** cold_2L2 (`bench(..., flush_bytes=2 × L2)`), cold_256 (256 MiB), steady (หมุนสำเนา)
- ต่อ run: ทุก (kernel, config) เรียก `bench()` 5 ครั้ง ครั้งละ 100 rep สลับลำดับด้วย seed 1820 · 3 run
- **ค่าของ cell** M = median ข้าม `bench()` ของค่าเฉลี่ยต่อครั้ง (ใช้ค่าเฉลี่ยเพราะ median ของ kernel สั้นติดตาราง 1.024 µs)
- **descriptive:** `triton.testing.do_bench` แบบ shipped เทียบกับ cold_256, เวลา dispatch และ pad ที่เครื่องมือเลือก, IQR
- **pipeline check ก่อนล็อก:** smoke ครั้งแรก compile ไม่ผ่าน (`NameError: tl` เพราะประกาศ kernel ใน function) แก้โดยย้าย kernel ไปที่ระดับ module และลบ `from __future__ import annotations` แล้ว smoke ผ่าน (12 cell, correctness 0) **ยังไม่ได้เปิดดูค่าเวลาของ smoke**

| ID | เกณฑ์ |
|---|---|
| **T-0 (ต้องผ่านก่อน)** | correctness 0 ทุก run และครบทุก (kernel, config) |
| **T-1 (ไม่ขึ้นกับขนาด flush)** | max_k \|M(cold_2L2) / M(cold_256) − 1\| ≤ 0.03 → **เวลาแบบ cold ไม่ขึ้นกับขนาด flush** · มีตัวใด ≥ 0.10 → ขึ้น · นอกนั้น partial (เทียบ: แบบ do_bench ไม่เติมงาน B18b วัดได้ +43 ถึง +167 %) |
| **T-2 (โหมด steady คิดค่าเขียน)** | S(k) = M(steady) / M(cold_256) − 1: \|S(gemv4)\| ≤ 0.05 (output เล็ก ไม่มีค่าเขียนให้เลื่อน), 0.30 ≤ S(e_mid) ≤ 1.00 (cold ไม่คิดค่าเขียน 0.4 L2 เลย steady คิดเต็ม), 0.05 ≤ S(e_large) ≤ 0.20 (cold เลื่อนไป ½ L2 จาก 4 L2) → **คิดค่าเขียนตามที่ counter ทำนาย** · นอกนั้น "not as predicted" |

**คำทำนายของผู้วิจัย:**
- **T-0 ผ่าน**
- **T-1 = ไม่ขึ้นกับขนาด flush** (ปานกลาง–สูง): B18 counter ของ e_large ภายใต้ pr2L2 เขียนเกิน pr256 แค่ 0.06 L2 (≈ +1.6 %) และ e_small ภายใต้ pr2L2 สะอาด
- **T-2 = ตามที่ทำนาย** (ปานกลาง): ค่าที่คาด S(gemv4) ≈ 0, S(e_large) ≈ +0.12, S(e_mid) ≈ +0.6 (ไม่แน่นอนที่สุด เพราะขึ้นกับ overhead คงที่ของ kernel ราว 7 µs)

**ความหมายถ้าเป็นตามที่ทำนาย:** มีเครื่องมือที่แก้ปัญหาทั้งสามที่เราวัดได้ และให้ตัวเลขสองแบบที่ตีความได้ชัด คือ "cold" (ไม่มีค่าเขียนจาก flush ไม่มี latency ของ host) กับ "steady" (จ่ายค่าเขียนของตัวเองเหมือนรันต่อเนื่อง)

**ข้อจำกัด:** GPU ตัวเดียว, `torch.cuda._sleep` เป็น API ภายใน, steady ต้องให้ผู้ใช้สร้างสำเนาเอง, kernel ที่สั้นกว่า ≈ 30 µs ยังติดความละเอียดของตัวจับเวลา

> **ผล (MEASURED 2026-11-06 14:50–15:05 UTC, `benchmarks/results/cleanbench_20261106-144941/`, D-085):**
>
> | กฎ | ค่า | ผล | คำทำนาย |
> |---|---|---|---|
> | T-0 | correctness 0, ครบ 12 cell × 15 `bench()` | ✅ ผ่าน | ✓ |
> | **T-1** | M(cold_2L2) / M(cold_256) − 1: gemv4 −0.05 %, e_large +1.6 %, **e_mid +5.6 %, t_small +9.4 %** | **partial** | ✗ (ทำนายไม่ขึ้นกับขนาด) |
> | **T-2** | S: gemv4 +0.2 %, **e_large +14.0 %, e_mid +38.1 %**, (t_small +45.5 %, ไม่อยู่ในกฎ) | ✅ **steady คิดค่าเขียนตามที่ counter ทำนาย** | ✓ |
>
> **คำทำนายถูก 2 จาก 3**
>
> **ข้อสังเกต:**
> - **T-1:** ตัวที่ต่างคือ kernel สั้นที่มี output 0.4–0.5 L2 ต่างกันราว 0.5–1 µs คำอธิบายหลังเห็นผล (ยังไม่ทดสอบ): read flush 2 × L2 ไล่ output dirty ของรอบก่อนออกไม่หมด แบบเดียวกับที่ B18 เห็นกับ prL2 แต่ไม่เห็นกับ e_small (output 0.125 L2) ภายใต้ pr2L2 · **ค่าตั้งต้นของ cleanbench (≥ max(256 MiB, 4 × L2)) จึงจำเป็น ไม่ควรลดเหลือ 2 × L2**
> - **T-2 ในหน่วยเวลา:** e_large steady − cold = +6.4 µs เทียบกับเวลาเขียน 0.5 L2 (1.5 MiB) ที่อัตรา DRAM ของ cold run (≈ 276 GB/s) = 5.7 µs · e_mid +3.7 µs เทียบกับ 1.2 MiB = 4.6 µs → ส่วนต่างระหว่าง cold กับ steady คือค่าเขียนกลับของ kernel เองตามที่ counter ของ B18 บอก
> - **do_bench แบบ shipped เทียบกับ cleanbench cold (descriptive, median เทียบค่าเฉลี่ย):** gemv4 +12 %, e_large +15 %, e_mid +36 %, t_small +41 % บน L2 แค่ 3 MiB
> - pad ที่เครื่องมือเลือก 217–351 µs จากเวลา dispatch 72–117 µs บน WSL2 · IQR ≈ 2 % สำหรับ kernel ยาวและ ≈ 7–11 % สำหรับ kernel ≈ 10 µs (ตาราง 1.024 µs)

---

## B19 pre-registration: บน GPU เร็วที่ L2 ใหญ่ ข้อผิดปกติ l2w ของ B17 คือ queue drain หรือไม่ และ cold / steady ต่างกันเท่าไร (R-0..R-4) — เขียน 2026-11-06 ก่อนเช่า GPU ใด ๆ ของ B19

**ที่มา:** B18b สร้างข้อผิดปกติแบบ l2w ซ้ำได้บน laptop และหายเมื่อเติมงานบน GPU (D-083) · B18 / B18c พบว่าการล้างแบบอ่านเลื่อนค่าเขียนกลับของ kernel เองออกนอกหน้าต่างได้ราว residue (D-081, D-085) ทั้งหมดวัดบน GA106 ตัวเดียว ต้องยืนยันบน GPU ที่ B17 เห็นข้อผิดปกติจริง

**Design** (`b19_probe.py`, `b19_probe.sh`, `b19_score.py`, `test_b19_score.py`, `remote_b19.sh`, ใช้ `cleanbench.py` ตัวที่ล็อกใน B18c; sha256 ใน `research/preregistration/LOCK_20261106_b19.sha256`):
- **GPU (ขออนุมัติ PI):** A100 40GB, H100 (NVL ถ้ามี), RTX 5090, L40S · docker container, image `nvidia/cuda:12.8.1-runtime-ubuntu22.04`, ไม่ใช้ template, timing อย่างเดียว · torch 2.7.1, triton 3.3.1, liger-kernel 0.6.2 แบบ B17
- **kernel:** rms16 (Liger RMSNorm bf16 [2048, 4096] ตัวเดียวกับข้อผิดปกติของ B17), rms64 ([8192, 4096]), gemv (112 MiB ตัวอ้างอิงของ B17), gemv16 (W [4096, 2048] = 16 MiB ต่ำกว่าครึ่ง L2 ทุกตัว), e_q (`torch.add` x และ y ขนาด 0.25 L2)
- **Part A** (แบบ B18b): flush zero256 / zeroL2 / pr256 / none × cond plain / pad, pad = max(300 µs, 3 × เวลา dispatch ของ kernel), block 6 ทิ้ง 2, 20 รอบ, kernel rms16 / gemv16 / gemv
- **Part B:** `cleanbench.bench` cold (flush ค่าตั้งต้น ≥ max(256 MiB, 4 × L2)) และ steady (สำเนารวม ≥ 4 × L2) 5 ครั้ง × 100 rep สำหรับ rms64 / rms16 / gemv / e_q
- **Part C:** `triton.testing.do_bench` แบบ shipped (descriptive)
- 3 run ต่อ GPU · smoke ในเครื่อง (laptop) ผ่าน: 5 kernel, 24 loop cell, 8 bench cell, correctness 0 (ครั้งแรก Liger import ไม่ได้เพราะ venv ไม่มี `packaging` เติมแล้ว) **ไม่ได้เปิดดูค่าเวลาของ smoke**

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **R-0 (ต้องผ่านก่อน)** | correctness 0, มี Liger (rms16 และ rms64), pad ≥ 200 µs |
| R-1 | plain I(rms16, zeroL2, zero256) ≥ 0.5 → ข้อผิดปกติของ B17 เกิดซ้ำ · ไม่เช่นนั้น "no anomaly" |
| **R-2 (ตัวหลัก)** | ถ้า R-1 เกิดซ้ำ: padded I(rms16, zeroL2, zero256) ≤ 0.15 → **queue drain** · ≥ 0.5 → ไม่ใช่ · นอกนั้น partial |
| R-3 | padded T(gemv16, none) ≤ 0.85 × padded T(gemv16, pr256) → เมื่อ queue ไม่ว่าง การไม่ล้างเร็วกว่าตามที่ควร (ข้อผิดปกติ E-0 ของ B17 หายไป) |
| **R-4a** | S(e_q) ∈ [0.30, 1.20] และ \|S(gemv)\| ≤ 0.05 → steady คิดค่าเขียนกลับของ output ที่ cold ไม่คิด |
| R-4b | \|S(rms64) − P\| ≤ 0.10 โดย P = r·L2 / (128 MiB − r·L2) และ r = residue โดยนัยของ GPU นั้นจาก B17 → residue ทำนายช่องว่าง cold / steady ได้ |
| ข้าม GPU | R-2: queue drain อธิบายได้กี่ตัวจากตัวที่เกิดซ้ำ · R-4b ผ่านถ้า ≥ 3 จาก 4 |

**คำทำนายของผู้วิจัย:**
- **R-0 ผ่านทุกตัว**
- **R-1:** เกิดซ้ำบน A100, H100, 5090 และไม่เกิดบน L40S ตาม B17 (ปานกลาง: CPU ของเครื่องเช่าต่างจากเดิม เวลา dispatch จึงต่างได้)
- **R-2 = queue drain ทุกตัวที่เกิดซ้ำ** (ปานกลาง)
- **R-3 = เร็วกว่าทุกตัว** (ปานกลาง)
- **R-4a ผ่านทุกตัว** (ปานกลาง–สูง)
- **R-4b ผ่าน ≥ 3 / 4** (ต่ำ–ปานกลาง): ค่าที่ทำนาย A100 0.163, H100 NVL 0.254 (ถ้าได้ H100 ที่ L2 50 MiB = 0.203), L40S 0.284, 5090 0.104 · อาศัยว่าบน GA106 residue ของ output ตัวเอง (0.50 L2) เท่ากับ residue หลัง zero_ (0.499 L2) ซึ่งยังไม่เคยทดสอบบน GPU อื่น

**ความหมายถ้าเป็นตามที่ทำนาย:**
- ตัวเลข +164–537 % ของ B17 ไม่ใช่ปัญหา L2 แต่เป็น launch latency ของ host ที่ถูกจับเวลาเพราะ flush ขนาด L2 สั้นเกินไปบน GPU เร็ว
- การล้างแบบอ่านเพียงอย่างเดียวยังไม่ครบ ต้องเติมงานบน GPU และรายงาน steady คู่กับ cold เพราะบน L2 ใหญ่ ค่าเขียนที่ถูกเลื่อนออกไปมีขนาดหลายสิบ MiB

**ข้อจำกัด:** เครื่องเดียวต่อรุ่น, container ไม่มี counter, clock ไม่ได้ล็อก, residue ของ B17 เป็นค่าโดยนัย

> **ผล (MEASURED 2026-11-06 14:59–15:03 UTC, `benchmarks/results/b19_{ga100_a100sxm,gb202_5090,ad102_l40s}_20261106/`, สรุป `b19_summary_20261106/`, D-089):**
>
> | GPU (host) | R-1: plain I(rms16, zeroL2, zero256) | R-2: padded | R-3: none / pr256 padded (plain) | R-4a: S(e_q), S(gemv) | R-4b: S(rms64) เทียบที่ทำนาย |
> |---|---|---|---|---|---|
> | A100-SXM4-40GB | **+203 %** เกิดซ้ำ | **−3.1 % → queue drain** | **0.76** (1.50) warm | +38.6 %, −1.1 % ✅ | **0.160** เทียบ 0.163 ✅ |
> | RTX 5090 | 0.0 % ไม่เกิด | n/a | 0.875 (0.92) ไม่ผ่าน | +67.1 %, +0.2 % ✅ | **0.098** เทียบ 0.104 ✅ |
> | L40S | 0.0 % ไม่เกิด | n/a | 0.927 (0.929) ไม่ผ่าน | +108.9 %, −0.3 % ✅ | **0.721** เทียบ 0.284 ❌ |
>
> **ข้าม GPU (กฎที่ล็อก):** R-0 ผ่าน 3/3 · **R-2: queue drain อธิบายข้อผิดปกติได้ 1/1 GPU ที่เกิดซ้ำ** · R-3 1/3 · R-4a 3/3 · R-4b **not** (2/3 และไม่ได้วัด H100)
>
> **คำทำนาย:** R-0 ✓ · R-1 ถูก 2/3 (5090 ไม่เกิดซ้ำ) · R-2 ✓ · R-3 ถูก 1/3 · R-4a ✓ · R-4b ✗ ตามกฎ (ต่อ GPU ถูก 2/3)
>
> **descriptive:**
> - **ข้อผิดปกติขึ้นกับ host (หลังเห็นผล):** เวลา dispatch ของ Liger RMSNorm จาก Python: A100 140 µs, L40S 102 µs, 5090 43 µs · เวลาโดยประมาณของ flush ขนาด L2 (L2 ÷ อัตราของ GEMV 112 MiB แบบ cold): A100 ≈ 37 µs, 5090 ≈ 63 µs, L40S ≈ 138 µs → ข้อผิดปกติเกิดเฉพาะเครื่องที่ dispatch นานกว่า flush (A100) ตรงกับ laptop (dispatch 72–139 µs, flush 12–18 µs) ส่วน host ของ 5090 ใน B17 น่าจะช้ากว่า host รอบนี้ (ไม่ได้บันทึกเวลา dispatch ใน B17)
> - **ข้อผิดปกติ E-0 ของ B17 ("ไม่ล้างช้ากว่าล้าง") เกิดซ้ำบน A100** (gemv16: none 33.8 µs เทียบ pr256 22.5 µs) และกลับทิศเมื่อเติมงาน (16.4 เทียบ 21.5 µs)
> - **ทำซ้ำ B17 บน host ใหม่ด้วย harness ต่างกัน:** do_bench shipped เทียบ cleanbench cold: GEMV A100 +15.6 % (B17 +16.0 %), 5090 +10.6 % (+10.8 %), L40S +25.5 % (+25.3 %) · RMSNorm 64 MiB A100 +19 % (+15.6 %), 5090 +29 % (+26.5 %), L40S +90 % (+88 %)
> - **R-4b บน L40S:** steady − cold ของ rms64 = +78 µs ≈ 57 MiB ที่อัตรา DRAM เกือบเท่ากับ output ทั้งก้อน (64 MiB = 0.67 L2) แสดงว่าภายใต้ cold, L40S เลื่อนค่าเขียนกลับออกไปเกือบทั้งหมด ซึ่งมากกว่า residue หลัง stream ยาวที่ B17 คำนวณได้ (0.295 L2) แบบเดียวกับที่ B12 พบว่า residue ของ stream ยาวต่างจากของการเขียนครั้งเดียว
> - rms16 steady เร็วกว่า cold บน 5090 (−14 %) และ L40S (−4 %) ยังอธิบายไม่ได้

---

## B20 pre-registration: Nsight Compute รุ่นล่าสุด (2026.4.0) ยังนับ write-back ราย kernel ขาดหรือไม่ (N-0..N-2) — เขียน 2026-11-06 ก่อนรัน B20

**ที่มา (PI):** "ncu latest version ยัง" · C6 (ncu ราย kernel นับ write ขาด) วัดด้วย ncu 2024.2.1 เท่านั้น ครั้งเดียวที่ลอง 2025.2.1 (GB206) อ่าน DRAM metric ไม่ได้ ถ้าจะแจ้ง NVIDIA หรืออ้างใน report ต้องรู้ว่ารุ่นล่าสุดยังเป็นอยู่ไหม

**Design** (`ncu_version_probe.sh`, `ncu_version_score.py`, `test_ncu_version_score.py`; protocol B9b เดิมไม่แก้: `queued_wb_probe.sh`, `queued_wb_probe.py`, `queued_wb_score.py`; sha256 ใน `research/preregistration/LOCK_20261106_b20.sha256`):
- **ncu ใหม่:** Nsight Compute 2026.4.0 (build 39012641) จาก NVIDIA apt repo (`ubuntu2404`, sha256 ของ .deb ตรงกับ index) แตกด้วย `dpkg-deb -x` ไว้ที่ `~/.cache/ncu-2026.4.0/` ไม่ได้ติดตั้งลงระบบ · `--query-metrics` บน GA106 มี `dram__bytes_read` / `dram__bytes_write`
- **ncu เดิม:** 2024.2.1 (`/usr/local/cuda-12.5/bin`) · driver 591.74 · RTX 3060 Laptop
- **ลำดับ:** protocol B9b ทั้งชุดด้วย ncu ใหม่ก่อน แล้วด้วย ncu เดิม ต่อกันใน session เดียว (แต่ละชุด: 3 × kernel mode + 3 × range mode)
- loss(w) = S_c − S_k แบบ B9b (สัดส่วนของ byte ที่เขียนซึ่งการวัดทีละ kernel นับไม่ถึง) · ค่าเดิมของ B9b (2026-11-05): 0.089 / 0.067 / 0.001 ที่ w = 0.75 / 1 / 1.5 L2

| ID | เกณฑ์ |
|---|---|
| **N-0 (ต้องผ่านก่อน)** | กฎ Q-0 ของ B9b ผ่านทั้งสองรุ่น |
| **N-1 (ตัวหลัก)** | ncu ใหม่ ที่ w ∈ {0.75, 1}: loss ≤ 0.02 ทั้งคู่ → **แก้แล้วในรุ่นใหม่** · ≥ 0.05 ทั้งคู่ → **ยังเป็นอยู่** · นอกนั้น partial |
| N-2 | \|loss_new − loss_old\| ≤ 0.03 ทั้งสอง w → ไม่ต่างระหว่างรุ่น · นอกนั้น "changed" |

**คำทำนายของผู้วิจัย:**
- **N-0 ผ่าน**
- **N-1 = ยังเป็นอยู่** (ปานกลาง): การนับที่ขาดน่าจะมาจาก write-back ที่เสร็จหลังหน้าต่างของ kernel ซึ่งเป็นเรื่องของ hardware / วิธีเก็บ counter มากกว่าเวอร์ชันของเครื่องมือ แต่เครื่องมือรุ่นใหม่อาจเปลี่ยนวิธีปิดหน้าต่าง
- **N-2 = ไม่ต่างระหว่างรุ่น** (ปานกลาง)

**ข้อจำกัด:** GPU ตัวเดียว (GA106) บน WSL2, ไม่ได้ทดสอบรุ่นระหว่างกลาง (2025.x), ลำดับใหม่ก่อนเดิมคงที่ (ไม่สลับ)

> **ผล (MEASURED 2026-11-06 15:52–16:10 UTC, `benchmarks/results/ncu_version_20261106-155257/` → `queued_wb_20261106-155257/` (2026.4.0) และ `queued_wb_20261106-155343/` (2024.2.1), D-094):**
>
> | กฎ | ค่า | ผล | คำทำนาย |
> |---|---|---|---|
> | N-0 | Q-0 ของ B9b ผ่านทั้งสองรุ่น | ✅ | ✓ |
> | **N-1** | ncu 2026.4.0: loss **0.081 / 0.054** / 0.002 ที่ w = 0.75 / 1 / 1.5 | **ยังเป็นอยู่ในรุ่นใหม่** | ✓ |
> | N-2 | ncu 2024.2.1 ใน session เดียวกัน: 0.086 / 0.058 / 0.002 (ต่าง ≤ 0.005) | ไม่ต่างระหว่างรุ่น | ✓ |
>
> **คำทำนายถูก 3 จาก 3** · ค่าของ B9b เดิม (2026-11-05): 0.089 / 0.067 / 0.001
>
> **deviation (ไม่เปลี่ยนการวัดหรือกฎ):**
> 1. ncu 2026.4.0 บันทึก report เป็น `.ncu-repz` แทน `.ncu-rep` ขั้น `--import` ใน `queued_wb_probe.sh` จึงได้ CSV ว่างและ scorer ล้ม · แปลงใหม่จาก `.ncu-repz` ด้วยคำสั่ง import เดิมของ ncu 2026.4.0 แล้วให้คะแนนด้วย `queued_wb_score.py` ตัวเดิม
> 2. `ncu_version_score.py` อ่าน key ระดับบนสุด แต่ไฟล์ score ของ B9b ห่อผลไว้ใต้ `"verdict"` (unit test ใช้รูปแบบแบนจึงไม่เจอ) · แก้ให้แกะ `"verdict"` ก่อน และเพิ่ม unit test · เกณฑ์ไม่เปลี่ยน

---

## B21 pre-registration: เวลาราย kernel แบบไหนทำนายเวลาของ decode step จริงได้ (U-0..U-3) — เขียน 2026-11-06 ก่อนวัด B21 ใด ๆ

**ที่มา (PI):** "เอาเลย ๆ อยากได้หลักฐานแน่น ๆ" · จุดอ่อนใหญ่ที่สุดของ report คือไม่มี end-to-end: ยังไม่รู้ว่าตัวเลขราย kernel แบบไหน (ล้างแบบเขียน, cold, steady) ตรงกับเวลาตอนใช้งานจริง

**Design** (`e2e_probe.py`, `e2e_probe.sh`, `e2e_score.py`, `test_e2e_score.py`, `remote_b21.sh`, ใช้ `cleanbench.py` ตัวที่ล็อกใน B18c; sha256 ใน `research/preregistration/LOCK_20261106_b21.sha256`):
- **decode proxy:** 4 layer ของ shape Llama-2-7B (H = 4096, FFN = 11008) batch 1 ใช้ GEMV 4 ตัวของแต่ละ layer ตามลำดับ: QKV [3H, H], O [H, H], gate+up [2 FFN, H], down [H, FFN] · ทุก layer มี weight ของตัวเอง (แบบ model จริง)
- **variant:** fp16 (cuBLAS), int4 (tinygemm, group 128, innerKTiles 8), nf4 (bitsandbytes ถ้ามี)
- **"ความจริง":** step ทั้งหมดอยู่ใน CUDA graph เดียว replay ต่อกัน (แบบ decode ที่ใช้ CUDA graph) จับเวลาด้วย event รอบแต่ละ replay median ของ 300 ครั้ง
- **ตัวทำนาย:** เวลาราย kernel ต่อ shape × 4 layer จาก `do_bench` แบบ shipped (write), cleanbench cold, cleanbench steady (หมุนบน weight ของ 4 layer) และ l2w (descriptive)
- 3 run ต่อ GPU · smoke ในเครื่อง: 3 variant จับเป็น graph ได้, correctness 0 **ไม่ได้ดูค่าเวลา**
- **ข้อจำกัดของ proxy:** มีแต่ GEMV (ไม่มี attention, norm, activation) ซึ่งเป็นส่วนที่ใช้เวลามากที่สุดของ decode ที่ batch 1

**ตัววัด:** P_h(v) = 4 × ผลรวมเวลาราย kernel ของ harness h · E_h(v) = P_h(v) / T_step(v) − 1 · S_e2e = T(fp16) / T(v) · D_h = S_h / S_e2e − 1

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **U-0 (ต้องผ่านก่อน)** | correctness 0, จับ graph ของ fp16 และ int4 ได้, IQR / median ของ step ≤ 0.05 |
| **U-1 (ตัวหลัก)** | \|E_cold(fp16)\| ≤ 0.10 และ \|E_cold(int4)\| ≤ 0.10 → **cold ทำนายเวลาของ step จริงได้** |
| **U-2** | เฉพาะ L2 ≥ 40 MiB: E_write(fp16) − E_cold(fp16) ≥ 0.05 → **การล้างแบบเขียนทำนายเวลาเกินจริง** · L2 เล็กกว่า: n/a |
| **U-3** | \|D_cold(int4)\| ≤ 0.05 และ \|D_cold(int4)\| < \|D_write(int4)\| → **cold ทำนาย speedup ของ int4 ได้ดีกว่าการล้างแบบเขียน** |
| ข้าม GPU | นับจำนวน GPU ที่ผ่านแต่ละข้อ · descriptive: E ของ steady / l2w, ลำดับของ int4 กับ nf4 ตามแต่ละ harness เทียบกับ step จริง |

**คำทำนายของผู้วิจัย:**
- **U-0 ผ่านทุก GPU**
- **U-1 ผ่านทุก GPU** (ปานกลาง): ใน decode จริง activation เล็กมาก L2 ก่อน GEMV แต่ละตัวมีแต่ weight ที่ clean ของ kernel ก่อนหน้า ซึ่งตรงกับสถานะหลังการล้างแบบอ่าน · ความเสี่ยง: clock ตอน replay ต่อเนื่องอาจต่างจากตอนจับเวลาทีละตัว
- **U-2 ผ่านทุก GPU ที่ L2 ≥ 40 MiB** (สูง)
- **U-3 ผ่านทุก GPU ที่ L2 ≥ 40 MiB** (ปานกลาง) · บน laptop (L2 3 MiB) ไม่ทำนาย เพราะ bias เล็กจนสองค่าใกล้กัน

**ความหมายถ้าเป็นตามที่ทำนาย:** ตัวเลขของ `do_bench` แบบ shipped ทำนายเวลาของ decode step จริงเกินไปและบิด speedup ของ int4 ส่วน cold (read flush + busy-wait) ทำนายได้ตรง ซึ่งเป็นหลักฐาน end-to-end ข้อแรกของ report

**GPU:** laptop (ฟรี) + เครื่องเช่าตามที่ PI อนุมัติ (D-095)

> **ผล (MEASURED 2026-11-06 16:00–16:47 UTC, `benchmarks/results/e2e_20261106-1612*` (laptop) และ `b21_{ad107_4060,ga100_a100sxm,gh100_h100nvl,ad102_l40s}_20261106/`, สรุป `b21_summary_20261106/`, D-097):**
>
> | GPU | L2 | E fp16: write / cold | E int4: write / cold | D int4: write / cold | U-1 | U-2 | U-3 |
> |---|---|---|---|---|---|---|---|
> | RTX 3060 Laptop | 3 MiB | +5.2 / +2.0 % | +8.6 / +1.8 % | −3.1 / +0.2 % | ✅ | n/a | ✅ |
> | RTX 4060 | 24 MiB | +8.2 / +0.6 % | +43.6 / +2.0 % | −24.6 / −1.4 % | ✅ | n/a | ✅ |
> | A100-SXM4-40GB | 40 MiB | +26.3 / +5.2 % | +46.2 / **+17.5 %** | −13.6 / −10.4 % | ❌ | ✅ | ❌ |
> | H100 NVL | 60 MiB | +32.4 / +8.4 % | +22.6 / **+17.2 %** | +8.0 / −7.5 % | ❌ | ✅ | ❌ |
> | L40S | 96 MiB | +28.1 / +2.0 % | +52.5 / +7.1 % | −16.1 / −4.8 % | ✅ | ✅ | ✅ |
>
> **ข้าม GPU:** U-0 5/5 · U-1 3/5 · **U-2 3/3** · U-3 3/5
>
> **คำทำนาย:** U-0 ✓ · U-1 ผิดบน A100 และ H100 · U-2 ✓ ทั้ง 3 · U-3 ทำนายว่าผ่านบน GPU ที่ L2 ≥ 40 MiB ถูกแค่ L40S (ผิดบน A100, H100)
>
> **descriptive:**
> - **การเลือก kernel int4 บน RTX 4060:** ผลรวมแบบ write บอกว่า NF4 เร็วกว่า tinygemm แต่ step จริง tinygemm เร็วกว่า 15 % ส่วนผลรวมแบบ cold เรียงถูก (GPU อื่นทุก harness เรียงถูก)
> - **speedup ของ NF4 เทียบ fp16 ทั้ง step:** H100 จริง 1.14× (write 1.32×, cold 1.09×) · 4060 จริง 3.20× (write 2.85×, cold 3.17×) · L40S จริง 3.64× (write 2.27×, cold 2.73×) · laptop จริง 2.45× (write 2.75×, cold 2.71×)
> - **คำอธิบายหลังเห็นผล (ยังไม่ทดสอบ):** บน A100 / H100 GEMV ของ int4 สั้น (step 457 / 286 µs ต่อ 16 kernel ≈ 18–28 µs ต่อตัว) การจับเวลาทีละ kernel มีต้นทุนคงที่ต่อ kernel ที่ CUDA graph ไม่มี · NF4 ของ bnb บน L40S คลาดมาก (cold +36 %) อาจเพราะ op เดียวเรียกหลาย kernel
> - steady ของ int4 บน L40S ต่ำไป 18 % เพราะสำเนา weight ของ 4 layer รวมกันไม่ถึง 4 × L2 (ผิดเงื่อนไขของโหมด steady เอง)
> - **A100 ครั้งแรก** (PCIe, California) ติดตั้งไม่เสร็จจนถึง hard cap ไม่มีผล · ผลข้างบนมาจาก A100 SXM4 (Sweden) ที่ลองใหม่ภายในงบเดิม
> - ค่าใช้จ่าย B21: เครดิต $5.517 → $5.0576 = **$0.459** (เพดาน $2.0) ทุกเครื่อง destroy แล้วและตรวจแล้ว

---

## B22 pre-registration: จับเวลา kernel ใน CUDA graph แบบหมุนสำเนา weight ทำนาย decode step จริงได้ภายใน 1 % หรือไม่ (V-0..V-3) — เขียน 2026-11-06 หลังเห็นผล B21 ก่อนวัด B22 ใด ๆ

**ที่มา (PI):** "เป็นไปได้ไหมที่จะให้คลาดเคลื่อนน้อย ≤ 1 %" · อนุมัติ laptop + A100 + H100 เพดาน $1.2 (D-098)

**สมมติฐาน (หลังเห็น B21):** ส่วนที่ cold เกิน decode step จริงคือต้นทุนคงที่ราว 2–5 µs ต่อ kernel ของการจับเวลาทีละ kernel ด้วย event (B21: cold เกิน 2.1–5.0 µs ต่อ kernel ทุก GPU ยกเว้น laptop fp16 7.3 µs) ไม่ใช่ cache ถ้าจับเวลาใน CUDA graph แบบที่ model รัน ต้นทุนนี้จะหาย

**Design** (`graph_time_probe.py`, `graph_time_probe.sh`, `graph_time_score.py`, `test_graph_time_score.py`, `remote_b22.sh`; ใช้ `e2e_probe.py` และ `cleanbench.py` ที่ล็อกไว้แล้วโดยไม่แก้; sha256 ใน `research/preregistration/LOCK_20261106_b22.sha256`):
- decode proxy เดียวกับ B21 · truth = step ใน CUDA graph, median ของ **1,000** replay (B21 ใช้ 300)
- **harness ใหม่ "graph":** ต่อ shape สร้างสำเนา weight N = max(4, ⌈4 × L2 / ขนาด weight⌉) (4 สำเนาแรกคือ 4 layer ของ step) แล้วจับเป็น CUDA graph ที่เรียก kernel ครั้งละสำเนาวนไป replay ต่อกัน 100 ครั้ง · เวลาต่อครั้ง = median ÷ N · แต่ละครั้งอ่าน weight ที่ไม่ได้แตะมา ≥ 4 × L2 แบบใน step และไม่มีต้นทุนของ event / host ต่อ kernel
- reference ใน run เดียวกัน: cold (cleanbench) และ write (do_bench shipped) แบบ B21
- 3 run ต่อ GPU · smoke บน laptop: 3 variant, correctness 0 **ไม่ได้ดูค่าเวลา**

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **V-0 (ต้องผ่านก่อน)** | correctness 0, มี step ของ fp16 และ int4, IQR / median ของ step ≤ 0.03 |
| **V-1 (ตัวหลัก)** | m = max(\|E_graph(fp16)\|, \|E_graph(int4)\|): ≤ 0.01 → **ภายใน 1 %** · ≤ 0.02 → ภายใน 2 % · นอกนั้น not |
| V-2 | \|D_graph(int4)\| ≤ 0.01 → speedup ภายใน 1 % · ≤ 0.02 → ภายใน 2 % · นอกนั้น not |
| V-3 | \|E_graph\| < \|E_cold\| ทั้ง fp16 และ int4 → graph ดีกว่า cold |

**คำทำนายของผู้วิจัย:**
- **V-0 ผ่านทุกเครื่อง** (H100 int4 ใน B21 มี IQR 2.1 % จึงเสี่ยง)
- **V-1:** laptop "ภายใน 1 %" (ปานกลาง) · A100 "ภายใน 2 %" (ปานกลาง) · H100 "ภายใน 2 %" (ต่ำ–ปานกลาง) · สิ่งที่อาจทำให้เกิน: kernel ใน step สลับชนิดกัน ส่วน harness เรียกชนิดเดียวซ้ำ ซึ่งการซ้อนช่วงท้ายของ kernel อาจต่างกัน
- **V-2:** ภายใน 2 % ทุกเครื่อง (ปานกลาง)
- **V-3 ผ่านทุกเครื่อง** (สูงบน A100 / H100, ปานกลางบน laptop เพราะ cold คลาดแค่ ~2 %)

**ความหมาย:** ถ้า V-1 ≤ 1 % → เราจับเวลาราย kernel ให้ทำนายเวลาใน model ได้ภายใน 1 % ด้วยการจับเวลาใน graph แบบหมุนสำเนา และส่วนที่เหลือของ cold คือต้นทุนต่อ kernel ไม่ใช่ cache · ถ้าไม่ถึง → ต้องวัด graph ของ model จริง (in-situ)

**ข้อจำกัด:** proxy เป็น GEMV ล้วน; ไม่ได้ทดสอบ kernel ที่ขึ้นกับ kernel ข้างเคียง

> **ผล (MEASURED 2026-11-06 16:58–17:20 UTC, `benchmarks/results/b22_{ga100_a100sxm,gh100_h100nvl}_20261106/`, สรุป `b22_summary_20261106/`, D-099):**
>
> | GPU (host) | step fp16 / int4 (µs) | E graph fp16 / int4 | E cold fp16 / int4 | E write fp16 / int4 | D int4 graph / cold / write | V-1 | V-2 | V-3 |
> |---|---|---|---|---|---|---|---|---|
> | A100-SXM4-40GB (Sweden) | 1,275 / 457 | **+0.96 % / +1.58 %** | +6.0 / +17.5 % | +26.9 / +47.1 % | **−0.6** / −9.8 / −13.7 % | ภายใน 2 % | **ภายใน 1 %** | ✅ |
> | H100 NVL (Czechia) | 616 / 364 | **−12.2 % / −19.5 %** | −4.5 / −6.4 % | +16.8 / +0.5 % | +9.1 / +2.0 / +16.2 % | not | not | not |
> | RTX 3060 Laptop | — | — | — | — | — | V-0 ไม่ผ่าน (step แกว่ง 61–146 %) | | |
>
> **คำทำนาย:** A100 V-1 ✓ (ทำนาย ≤ 2 %), V-2 ✓ (ทำนาย ≤ 2 % ได้ ≤ 1 %), V-3 ✓ · H100 V-1 ✗, V-2 ✗, V-3 ✗ · laptop V-0 ✗
>
> **descriptive:**
> - **A100:** graph แบบหมุนสำเนาทำนาย step ได้ fp16 +0.96 %, int4 +1.58 %, NF4 +0.21 % และ speedup ของ int4 −0.6 %, NF4 +0.75 % · ส่วนที่ cold เกินใน B21 จึงเป็นต้นทุนต่อ kernel จริง
> - **H100 NVL (host ใหม่):** graph **ต่ำ**กว่า step 12–20 % คือ kernel ที่ถูกเรียกซ้ำชนิดเดียวใน graph เร็วกว่าตอนอยู่ใน step ที่สลับชนิดกัน (หลังเห็นผล: เวลาของ kernel ใน step ขึ้นกับ kernel ข้างเคียง ยังไม่ทดสอบ) · cold ก็เปลี่ยนจาก +8 / +17 % (B21, host Virginia) เป็น −4.5 / −6.4 %
> - **H100 NVL สอง host ต่างกันมาก:** step fp16 532 µs (B21, Virginia) เทียบกับ 616 µs (B22, Czechia) +16 % และ int4 286 µs เทียบกับ 364 µs +27 % · ตัวเลขเวลาสัมบูรณ์บน GPU รุ่นเดียวกันจึงต่างกันได้ระดับนี้ระหว่าง host
> - **laptop ใช้ไม่ได้:** รอบแรก step แกว่ง 61–146 % (GPU อยู่ P3, mem 5,500 MHz, 70 °C) รอบซ้ำ (deviation: รันซ้ำหนึ่งครั้ง) ก็ไม่ผ่าน V-0 · ตรวจภายหลังพบว่า GPU ถูกโปรแกรมอื่นฝั่ง Windows ใช้ราว 41 % ทั้งที่ไม่มีงานของเรา
> - ค่าใช้จ่าย B22: เครดิต $5.0299 → $4.6399 = **$0.39** (เพดาน $1.2) ทุกเครื่อง destroy แล้วและตรวจแล้ว
> - **วิเคราะห์เพิ่มหลังเห็นผล (D-100, 2026-11-07):** เวลา cold ของ kernel เดี่ยวบน H100 NVL สอง host ตรงกัน −0.8 ถึง +4.3 % (ครบ 12 แถว) แต่ step ต่างกัน +16 / +27 / +9 % (fp16 / int4 / NF4) · ผลรวม graph แบบหมุนสำเนาของ host B22 ตรงกับ step ของ host B21 +1.7 / +2.4 / −2.2 % → สิ่งที่เปลี่ยนคือ step บน host B22 ไม่ใช่ kernel และไม่ใช่ตัวชิป คำอธิบาย "ขึ้นกับ kernel ข้างเคียง" ข้างบนจึงอ่อนลง · สอง host ต่างกันที่ power limit 400 W (B21) เทียบ 300 W (B22), driver 580.178 เทียบ 615.71 และ VM pass-through เทียบไม่ระบุ · ไม่ได้บันทึก clock หรือ throttle reason ระหว่างวัด step จึงยังระบุสาเหตุไม่ได้ · A100 ของ B21 และ B22 เป็น host เดียวกัน (Sweden, bus 00000000:44:00.0, driver 535.154.05, step fp16 1274.88 µs เท่ากันทั้งสองรอบ) ผล 1 % ของ A100 จึงมาจาก host เดียว


---

## B23 pre-registration: benchmark kernel เดี่ยวจับเวลาได้ภายใน ±1 % ของเวลา kernel ตัวเดียวกันใน model หรือไม่ (W-0..W-3) — เขียน 2026-11-07 หลังเห็นผล B21 / B22 ก่อนวัด B23 ใด ๆ

**ที่มา (PI, 2026-11-07):** "ผมอยากได้วิธีที่ทำให้คลาดเคลื่อน +/- 1%" · "ความคลาดเคลื่อนในการวัด kernel ดิ" · "ทำเลย" (D-101)

**สมมติฐาน (หลังเห็น B21 / B22):**
- การจับเวลา kernel เดี่ยวด้วย CUDA event รวมต้นทุนคงที่ไว้ด้วย: B22 cleanbench cold เกินเวลาต่อครั้งแบบ graph 1.9–6.1 µs ต่อ kernel (A100 / H100) ซึ่งคือ +3 % สำหรับ kernel 140 µs แต่ +40 % สำหรับ kernel 8–18 µs
- ถ้าใช้เวลาเริ่ม / จบของ kernel เองจาก GPU (CUPTI activity record) ต้นทุนนี้จะหายไป
- ถ้าเรียก kernel ซ้ำบนสำเนา weight ≥ 4 × L2 ใน CUDA graph หลังอุ่นเครื่อง ≥ 1 s สภาพ cache (weight ไม่อยู่ใน L2, activation อยู่) และ clock จะเหมือนใน model → เวลาแต่ละ kernel ควรตรงกับเวลาใน model ภายใน 1 %

**ไม่ใช่ของใหม่:** FlashInfer มี `bench_gpu_time_with_cupti` (จับเวลาด้วย CUPTI และมีตัวเลือกล้าง L2) · nvbench มีการวัดแบบ batch / cold · สิ่งที่ B23 เพิ่มคือการตรวจกับ **เวลาของ kernel ตัวเดียวกันใน model** ราย kernel ที่ระดับ 1 %

**Design** (`kernel_time_probe.py`, `cupti_records.py`, `kernel_time_probe.sh`, `kernel_time_score.py`, `test_kernel_time_score.py`, `test_cupti_records.py`, `remote_b23.sh`; ใช้ `graph_time_probe.py`, `e2e_probe.py`, `cleanbench.py` ที่ล็อกไว้แล้วโดยไม่แก้; sha256 ใน `research/preregistration/LOCK_20261107_b23.sha256`):
- decode proxy และสำเนา weight เดียวกับ B22 (`build_copies`) · fp16 (cuBLAS), int4 (tinygemm), NF4 (bitsandbytes ถ้ามี) × 4 shape = **12 แถว**
- **เวลา kernel** = ผลรวม (end − start) ของ CUPTI record ของ kernel ทุกตัวที่ op หนึ่งปล่อย อ่านผ่าน `torch.profiler` (ไม่ใช้ counter จึงไม่ต้องมีสิทธิ์ profiling) · ไม่นับช่องว่างระหว่าง kernel
- **segment** (หน่วยการวัด): replay แบบไม่ trace ≥ 1 s (ส่งเป็นชุดละราว 20 ms ของงาน GPU และรอชุดก่อนหน้าเสร็จ จึงมีงานค้างในคิวไม่เกินราว 2 ชุด และ GPU ยังทำงานอยู่ตอนเริ่ม profiler) → เริ่ม profiler → replay แบบ trace 0.05 s → kernel marker → replay ที่วัด (trace + CUDA event) → ปิด profiler → replay 0.05 s → replay จำนวนเท่ากันแบบไม่ trace (CUDA event) · เวลาอุ่นเครื่องที่ทำจริงบันทึกใน meta
- **ต่อ variant วัด 3 รอบ (round) สลับกัน:** แต่ละรอบ = segment ของ step แล้ว segment ของ rotation ทีละ shape → truth กับ harness หลักถูกวัดสลับกันในเวลาใกล้กัน การลอยของ clock / อุณหภูมิจึงกระทบทั้งคู่
  - **truth:** step 16 op ใน CUDA graph · เวลาของแต่ละ op ใน step (4 layer × 100 replay × 3 รอบ)
  - `cupti_rot` **(ตัวหลัก):** graph ที่เรียก op บนสำเนา N ตัว (แบบ B22) · CUPTI ต่อครั้ง (N × 40 replay × 3 รอบ)
  - `event_graph`: ครึ่งที่ไม่ trace ของ segment rotation · event รอบ replay ÷ N (harness ของ B22)
- **หลังครบรอบ ต่อ shape:**
  - `event_cold`: cleanbench cold แบบเดิม (CUDA event) แบบ B21 / B22
  - `cupti_cold`: loop cold ของ cleanbench (read flush ≥ max(256 MiB, 4 × L2), busy-wait, เรียก) 200 ครั้ง · CUPTI
  - `cupti_rot_idle`: graph rotation หลัง GPU ว่าง 1 s ไม่อุ่น (ผลของการอุ่นเครื่อง)
- **record check:** ทุก segment ต้องมีจำนวน record และชื่อ kernel ตรงตามคาดทุกตัว (helper แยกใน `cupti_records.py` มี unit test)
- **log เชิงพรรณนา (ไม่ใช้ตัดสิน):** nvidia-smi ทุก 1 s (clock, power, power limit, clock-event reasons) และ counter สะสมของ `nvidia-smi -q -d PERFORMANCE` (เช่น SW power capping) ก่อน / หลังแต่ละ run · ผลของ trace ต่อ graph rotation · ช่วงของ median truth ต่อรอบ
- 3 run ต่อ GPU · แต่ละ run จำกัด 900 s · บนเครื่องเช่า ถ้า smoke ไม่ผ่าน (rc ≠ 0 หรือ record check ไม่ครบ) จะไม่รัน B23
- smoke บน laptop: record check 39/39, correctness 0 **ไม่ได้ดูค่าเวลา** · ก่อนหน้านั้น smoke ของรุ่นที่อุ่นเครื่องด้วยเวลา CPU ผ่านแค่ 9/39 (CPU ส่ง replay เข้าคิวเร็วกว่า GPU ทำหลายวินาที) จึงเปลี่ยนเป็นส่งเป็นชุด
- ตรวจโค้ดแบบอิสระก่อน lock: ไม่พบ bug ที่ทำให้ตัวเลขผิด · แก้ตามความเสี่ยงที่พบ: วัดสลับเป็นรอบ, อุ่นเครื่องตามเวลาจริงและบันทึก, gate ต้องมี 3 run เต็ม, timeout, หยุดถ้า smoke ไม่ผ่าน, log ทุก 1 s, test เพิ่ม

| ID | เกณฑ์ (ต่อ GPU) |
|---|---|
| **W-0 (ต้องผ่านก่อน)** | 3 run เต็ม (ไม่ใช่ smoke) ที่มีแถวชุดเดียวกัน · correctness 0 · record check ผ่านทุก segment ทุก run · มี truth ครบทุกแถวของ fp16 และ int4 · **tracing:** \|median step ที่เปิด trace / median step ที่ไม่เปิด − 1\| ≤ 0.005 ทุก variant (CUDA event, ครึ่งคู่กันใน segment เดียวกัน, รวมทุก run) · truth นิ่ง: IQR / median ของ truth รวมทุก run ≤ 0.05 ทุกแถว |
| **W-1 (ตัวหลัก)** | m = max ทุกแถวของ \|E_cupti_rot\|: ≤ 0.01 → **ทุก kernel ภายใน 1 %** · ≤ 0.02 → ทุก kernel ภายใน 2 % · นอกนั้น not · รายงานจำนวนแถวที่อยู่ใน 1 % ทุกกรณี |
| W-2 | median \|E_cupti_cold\| < median \|E_event_cold\| → CUPTI ใกล้กว่า event |
| W-3 (พรรณนา) | w = median ของ cupti_rot_idle / cupti_rot − 1: \|w\| ≥ 0.01 → การอุ่นเครื่องเปลี่ยนเวลา ≥ 1 % · นอกนั้น < 1 % |

E_h = H / T − 1 โดย T และ H เป็น median ของค่าต่อ run ทั้ง 3 run · ส่วนพรรณนา: จำนวนแถวที่อยู่ใน 1 % ของทุก harness และสัดส่วนเวลา kernel ใน step (= 4 × Σ T / เวลา step) ซึ่งที่เหลือคือช่องว่างระหว่าง kernel

**คำทำนายของผู้วิจัย:**
- **W-0:** ผ่านบนเครื่องเช่า (ปานกลาง–สูง) · ความเสี่ยง: trace อาจเพิ่มเวลา step เกิน 0.5 % (ต่ำ–ปานกลาง) · laptop เสี่ยงเพราะมีโปรแกรมอื่นใช้ GPU (B22 laptop ไม่ผ่าน V-0)
- **W-1:** A100 "ภายใน 2 %" (ปานกลาง), "ภายใน 1 %" (ต่ำ–ปานกลาง) · RTX 4060 / 4090 หรือ 5090 "ภายใน 2 %" (ปานกลาง) · H100 NVL "ภายใน 2 %" (ต่ำ–ปานกลาง) · laptop ไม่ทำนาย
  - ความเสี่ยงหลัก: ใน step โค้ดของ kernel ถูกไล่ออกจาก L2 ระหว่างการเรียกแต่ละครั้ง (weight 400 MB–1.6 GB ต่อ step) แต่ใน rotation เรียก kernel เดิมซ้ำ (โค้ดอยู่ใน cache) · ถ้าต่างกันคงที่ 0.5–1 µs แถวสั้นที่สุด (O-proj int4 / NF4 ราว 8–18 µs) จะเกิน 1 %
  - ความเสี่ยงอื่น: (ก) step อ่าน weight ทั้ง step (fp16 ราว 1.6 GB) แต่ rotation อ่านแค่ ≥ 4 × L2 ของ shape เดียว TLB และ DRAM page ที่ใช้จึงต่างกัน · (ข) kernel ก่อนหน้าใน step เป็นคนละ shape / คนละโค้ด และกำลังไฟของงานผสมต่างจาก GEMV ตัวเดียวซ้ำ · (ค) truth ใช้สำเนา 0–3 ส่วน rotation ใช้สำเนาทั้ง N ตัว
- **W-2:** ผ่านทุกเครื่อง (สูง)
- **W-3:** < 1 % บนเกือบทุกเครื่อง · ≥ 1 % ได้บนเครื่องที่ power limit ต่ำ (ต่ำ)

**ความหมาย:**
- W-1 ภายใน 1 % → benchmark kernel เดี่ยวด้วย CUPTI + สำเนา weight + อุ่นเครื่อง ให้เวลาราย kernel ภายใน 1 % ของเวลาใน model บนเครื่องนั้น
- ไม่ถึง → แถวที่เกินและทิศทางบอกว่าเหลือผลของ context อะไร (โค้ด / cache / clock)
- ไม่ว่าผลใด ใช้ได้เฉพาะเครื่องที่วัด (B21 / B22: H100 NVL สอง host ต่างกัน 9–27 %)

**ข้อจำกัด:** proxy เป็น GEMV ล้วน · truth คือ CUPTI ใน graph ของ proxy ไม่ใช่ model เต็ม · CUPTI อาจรบกวนการวัด (จึงมี gate tracing) · ไม่นับช่องว่างระหว่าง kernel ซึ่งเป็นต้นทุนของ graph ไม่ใช่ของ kernel

**เครื่อง:** laptop (ฟรี เมื่อ GPU ว่าง) · เครื่องเช่ารอ PI อนุมัติ GPU และงบ

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
