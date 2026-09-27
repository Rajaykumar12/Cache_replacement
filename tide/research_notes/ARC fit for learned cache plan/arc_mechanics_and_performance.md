# ARC (Adaptive Replacement Cache): Mechanics, Descendants, Modern Performance, Weaknesses (as of Sept 2026)

Source-handling note: I extracted the PDFs locally with `pdftotext` because the WebFetch summaries of several PDFs were wrong. For example, one summary said LeCaR weights "ARC and LRU", but it actually weights LRU and LFU. The math glyphs in the ARC FAST'03 PDF don't extract cleanly, so the pseudocode below comes from the paper's prose (cases A/B, B.1–B.3) together with libCacheSim's ARC.c, which states that it follows the paper. Items marked **[recalled, unverified]** are not confirmed by a source fetched in this session.

## 1. Exact ARC algorithm (lists, p, adaptation, REPLACE, cost, metadata)

### Takeaway
ARC keeps two LRU lists of cached pages: T1 (seen once recently) and T2 (seen at least twice). It also keeps two LRU "ghost" lists of evicted keys only: B1 (evicted from T1) and B2 (evicted from T2). A single real-valued target `p` (the desired size of T1) moves up on B1 ghost hits and down on B2 ghost hits. The step is `max(1, |other ghost| / |this ghost|)`. Each request costs O(1). The directory holds c resident pages plus up to c ghost keys, so 2c entries in total.

### Cited Findings
- Authors and venue: Nimrod Megiddo and Dharmendra S. Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache", USENIX FAST 2003 (IBM Almaden) — [paper PDF](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf); [conference page](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache)
- The model is demand paging with uniformly sized pages ("managed in units of uniformly sized items"). This matches the target simulator (integer keys, uniform sizes) exactly — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- Directory size: "Together the two lists remember exactly twice the number of pages that would fit in the cache", so ARC maintains a cache directory of 2c pages: c resident and c history — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- Complexity: "ARC is simple-to-implement and, like LRU, has constant complexity per request. In comparison, policies LRU-2 and LRFU both require logarithmic time complexity in the cache size" — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- Scan resistance: "ARC is scan-resistant: it allows one-time sequential requests to pass through without polluting the cache" — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- Relation to FRC: "If the two 'ADAPTATION' steps are removed, and the parameter p is a priori fixed to a given value, then the resulting algorithm is exactly FRC_p". FRC_p is the fixed-split policy, so ARC is FRC_p plus online adaptation of p — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- Invariants (paper, Sec. III–IV): L1 = T1 ∪ B1 and L2 = T2 ∪ B2 satisfy |L1| ≤ c and |L1| + |L2| ≤ 2c, with |T1| + |T2| ≤ c and 0 ≤ p ≤ c — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf); the same bounds are enforced in [libCacheSim ARC.c](https://github.com/1a1a11a/libCacheSim/blob/develop/libCacheSim/cache/eviction/ARC.c)
- Adaptation, verbatim from libCacheSim, which follows the paper and uses float p and delta:
  - B1 hit: `delta = MAX(L2_ghost_size / L1_ghost_size, 1); p = MIN(p + delta, cache_size);`
  - B2 hit: `delta = MAX(L1_ghost_size / L2_ghost_size, 1); p = MAX(p - delta, 0);`

  — [libCacheSim ARC.c](https://github.com/1a1a11a/libCacheSim/blob/develop/libCacheSim/cache/eviction/ARC.c)
- REPLACE, verbatim from libCacheSim: evict the T1 LRU into B1 if `(|T1|>0 && (|T1|>p || (|T1|==p && x∈B2))) || |T2|==0`; otherwise evict the T2 LRU into B2. The paper's REPLACE has no `|T2|==0` clause; libCacheSim added it as a guard for its variable-size mode — [libCacheSim ARC.c](https://github.com/1a1a11a/libCacheSim/blob/develop/libCacheSim/cache/eviction/ARC.c)
- libCacheSim's header says it was "cross checked with https://github.com/trauzti/cache/blob/master/ARC.py". It also notes that "one thing not clear in the paper is whether delta and p is int or float", and libCacheSim uses float — [libCacheSim ARC.c](https://github.com/1a1a11a/libCacheSim/blob/develop/libCacheSim/cache/eviction/ARC.c)
- Pseudocode for ARC(c), Fig. 4 of the paper, reconstructed from its prose and cross-checked against libCacheSim — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf), [libCacheSim ARC.c](https://github.com/1a1a11a/libCacheSim/blob/develop/libCacheSim/cache/eviction/ARC.c):
  ```
  INIT: T1=T2=B1=B2=∅, p=0
  on request x:
   Case I   x ∈ T1∪T2 (hit):  move x to MRU of T2.
   Case II  x ∈ B1 (ghost miss): p ← min(c, p + max(1, |B2|/|B1|)); REPLACE(x,p); move x B1→MRU(T2); fetch.
   Case III x ∈ B2 (ghost miss): p ← max(0, p − max(1, |B1|/|B2|)); REPLACE(x,p); move x B2→MRU(T2); fetch.
   Case IV  x ∉ T1∪B1∪T2∪B2 (cold miss):
     A: |T1|+|B1| = c:
          if |T1| < c: drop LRU(B1); REPLACE(x,p)
          else (B1 empty): drop LRU(T1) from cache entirely (no ghost)
     B: |T1|+|B1| < c:
          if |T1|+|T2|+|B1|+|B2| ≥ c:
              if total = 2c: drop LRU(B2)
              REPLACE(x,p)
     insert x at MRU(T1); fetch.
  REPLACE(x,p):
   if T1≠∅ and (|T1| > p or (x ∈ B2 and |T1| = p)): move LRU(T1) → MRU(B1)
   else: move LRU(T2) → MRU(B2)
  ```
- Headline results from the paper's tables (parsed from a glyph-garbled PDF, so column alignment was inferred). Workload P6 (workstation disk trace, c=32768 pages, 16 MB): LRU 4.24% hit ratio, ARC 23.84%, offline-best fixed FRC_p 22.62%. SPC1-like benchmark at 4 GB (c=1,048,576): LRU 9.19%, MQ 14.96%, 2Q 17.88%, ARC 20.00%. The abstract states the SPC1-like 4 GB comparison in LRU-vs-ARC form — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- Traces used in the paper: CODASYL, 14 workstation disk drives, a commercial ERP system, an SPC1-like benchmark, and web search requests. ARC is compared against LRU, LFU, FBR, LRU-2, 2Q, LRFU, LIRS and MQ, which "require user-defined parameters" while ARC has none that the user sets — [ARC FAST'03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- Per-object metadata in a modern simulator is about 17 B for ARC, against 16 B for LRU/FIFO. Lines of code for hit/eviction/insertion are 64/108/20 for ARC, against 5/4/3 for LRU (libCacheSim-based count) — [SIEVE, NSDI'24, Table 2](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
- A companion magazine article, N. Megiddo and D. S. Modha, "Outperforming LRU with an Adaptive Replacement Cache Algorithm", IEEE Computer 37(4), 2004, pp. 58–65 **[recalled, unverified; not fetched]**.

Reference Python implementation. I wrote it and tested it in this session: it asserts |T1|+|T2| ≤ c, |T1|+|B1| ≤ c, total ≤ 2c and 0 ≤ p ≤ c over 200k mixed Zipf-plus-scan requests at c=100, and all checks passed.
```python
from collections import OrderedDict as OD
class ARC:  # FAST'03 Fig.4, uniform sizes; OrderedDict first=LRU, last=MRU
    def __init__(s, c): s.c, s.p = c, 0.0; s.T1, s.T2, s.B1, s.B2 = OD(), OD(), OD(), OD()
    def _replace(s, x_in_B2):
        if s.T1 and (len(s.T1) > s.p or (x_in_B2 and len(s.T1) == s.p)):
            k, _ = s.T1.popitem(last=False); s.B1[k] = None
        else:
            k, _ = s.T2.popitem(last=False); s.B2[k] = None
    def access(s, x):  # True on hit
        if x in s.T1: del s.T1[x]; s.T2[x] = None; return True
        if x in s.T2: s.T2.move_to_end(x); return True
        if x in s.B1:
            s.p = min(s.c, s.p + max(1, len(s.B2) / len(s.B1)))
            s._replace(False); del s.B1[x]; s.T2[x] = None; return False
        if x in s.B2:
            s.p = max(0, s.p - max(1, len(s.B1) / len(s.B2)))
            s._replace(True); del s.B2[x]; s.T2[x] = None; return False
        L1 = len(s.T1) + len(s.B1)
        if L1 == s.c:
            if len(s.T1) < s.c: s.B1.popitem(last=False); s._replace(False)
            else: s.T1.popitem(last=False)
        else:
            tot = L1 + len(s.T2) + len(s.B2)
            if tot >= s.c:
                if tot == 2 * s.c: s.B2.popitem(last=False)
                s._replace(False)
        s.T1[x] = None; return False
```

### Inferences
- ARC's `p` is an online learner with one parameter. Its "gradient signal" is a ghost hit (a would-have-hit under a different split), and its step size is the ratio of the ghost-list sizes. It is not a per-object predictor: every decision within T1 and within T2 is pure LRU.
- B1 and B2 are cheap for integer keys (key only, O(c) extra). They already give "shadow" information that a learned policy could use: for example, ghost-hit labels, or an ARC-style fallback in place of a shadow-LRU.
- The uniform-size assumption in the paper matches the project's simulator exactly, so the original algorithm can be used as-is. The variable-size deviations (ZFS, libCacheSim's byte mode) are unnecessary here.

### Gaps
- I couldn't get a clean text rendering of Fig. 4 (glyphs garbled). The pseudocode above follows the paper's prose plus libCacheSim, and matches the widely reproduced version, but I haven't compared it character by character against the figure.
- The paper does not specify whether p should be an integer or a float. libCacheSim uses float and flags the ambiguity.

## 2. Descendants and variants: CAR/CART, ZFS ARC, LIRS, 2Q, LeCaR/CACHEUS, patent status

### Takeaway
CAR and CART (FAST 2004) keep ARC's adaptive target but replace the LRU lists with CLOCKs, which removes the global lock on hits. ZFS started from ARC, and OpenZFS 2.2 replaced the scalar `arc_p` with fraction-based balancing driven by ghost hits. LeCaR and CACHEUS generalize ARC's "ghost hit means I made the wrong choice" signal into regret-minimizing expert weights; CACHEUS even uses ARC as one of its experts. IBM's core ARC patent US6996676 is listed as **expired (Feb 22, 2024)** on Google Patents. A continuation patent may run to Feb 2027 (not verified directly).

### Cited Findings
- **CAR/CART.** Sorav Bansal (Stanford) and Dharmendra S. Modha (IBM), "CAR: Clock with Adaptive Replacement", USENIX FAST 2004 — [PDF](https://www.usenix.org/legacy/publications/library/proceedings/fast04/tech/full_papers/bansal/bansal.pdf)
  - Motivation: "On every cache hit, the policy LRU needs to move the accessed item to the most recently used position, at which point ... it serializes cache hits behind a single global lock. CLOCK eliminates this lock contention." CAR "inherits virtually all advantages of ARC including its high performance, but does not serialize cache hits behind a single global lock" — [CAR FAST'04](https://www.usenix.org/legacy/publications/library/proceedings/fast04/tech/full_papers/bansal/bansal.pdf)
  - Adaptation is the same as ARC: "Whenever a hit in B1 is observed, the target size of T1 is incremented; similarly, whenever a hit in B2 is observed, the target size of T1 is decremented". T1 and T2 become CLOCKs with reference bits, while B1 and B2 remain history lists — [CAR FAST'04](https://www.usenix.org/legacy/publications/library/proceedings/fast04/tech/full_papers/bansal/bansal.pdf)
  - CART's motivation: "A limitation of ARC and CAR is that two consecutive hits are used as a test to promote a page ... not a guarantee of long-term utility". CART adds a per-page filter bit, "L" (long-term) or "S" (short-term). Every page in T2 and B2 must be "L", every page in B1 must be "S", and T1 may hold either — [CAR FAST'04](https://www.usenix.org/legacy/publications/library/proceedings/fast04/tech/full_papers/bansal/bansal.pdf)
- **ZFS ARC.** The OpenZFS arc.c header calls it "DVA-based Adjustable Replacement Cache". It lists three deviations from "the Megiddo and Modha model": (1) not every page is evictable, since referenced buffers are pinned; (2) the cache size is variable and shrinks under OS memory pressure; (3) block sizes vary (512 B–128 KB), so eviction must free approximately the space of the incoming block — [OpenZFS arc.c](https://github.com/openzfs/zfs/blob/master/module/zfs/arc.c)
  - State lists are anon, mru, mru_ghost, mfu, mfu_ghost, plus an uncached state. A hit on an MRU buffer promotes it to MFU only if more than `ARC_MINTIME` (hz>>4, about 62 ms) has passed since the previous access, so back-to-back accesses don't count as "frequent". An mru_ghost hit moves the buffer to MFU, or to MRU if the access was a prefetch — [OpenZFS arc.c](https://github.com/openzfs/zfs/blob/master/module/zfs/arc.c)
  - The current balancing is not ARC's `p`. `arc_evict_adj(frac, total, up, down, balance)` adjusts 32-bit fixed-point fractions (`arc_pd` for data, `arc_pm` for metadata, and `arc_meta`) from the ghost-hit bytes accumulated since the last eviction. Adjustment speed is rate-limited: hits are rescaled if `up+down >= total/16`. Ghost lists are trimmed to half the size of the other three states: "excessive ghost size may result in false ghost hits (too far back)". `zfs_arc_meta_balance = 500` biases toward metadata — [OpenZFS arc.c](https://github.com/openzfs/zfs/blob/master/module/zfs/arc.c)
  - This rework is "More adaptive ARC eviction" by Alexander Motin, PR #14359, commit a8d83e2. It removed `arc_p` and shipped in OpenZFS 2.2 — [PR #14359](https://github.com/openzfs/zfs/pull/14359); [commit a8d83e2](https://github.com/openzfs/zfs/commit/a8d83e2a24de6419dc58d2a7b8f38904985726cb). The version attribution comes from search snippets and the [OpenZFS module-parameters docs](https://openzfs.github.io/openzfs-docs/Performance%20and%20Tuning/Module%20Parameters.html); I did not read the PR page itself.
- Other deployments: IBM DS6000/DS8000 controllers, ZFS/OpenZFS (plus the L2ARC SSD tier), and VMware vSAN. PostgreSQL used ARC only in 8.0.0 and "quickly replaced it with another algorithm, citing concerns over an IBM patent on ARC" — [Wikipedia: Adaptive replacement cache](https://en.wikipedia.org/wiki/Adaptive_replacement_cache). The PostgreSQL replacements were 2Q in 8.0.2 and clock-sweep in 8.1 **[recalled, unverified]**.
- **LIRS / CLOCK-Pro.** LIRS (Jiang & Zhang, SIGMETRICS 2002) is based on reuse distance and routes one-time accesses through a short filter list Q that is fixed at 1% of the cache. CACHEUS notes that "LIRS's ability to adapt is compromised because of its use of a fixed-length Q", and that DLIRS makes Q adaptive but "does not perform as well as LIRS in practice" — [CACHEUS FAST'21](https://www.usenix.org/system/files/fast21-rodriguez.pdf). CLOCK-Pro (Jiang, Chen & Zhang, USENIX ATC 2005) is LIRS's CLOCK analogue, as CAR is ARC's; it is cited in [S3-FIFO refs](https://dl.acm.org/doi/10.1145/3600006.3613147).
- **2Q.** Johnson & Shasha, VLDB 1994, is cited in [CAR refs](https://www.usenix.org/legacy/publications/library/proceedings/fast04/tech/full_papers/bansal/bansal.pdf). 2Q is essentially ARC with a static split: "TwoQ ... reserves a fixed 25% of the cache space for new objects, preventing overly aggressive demotion" — [SIEVE NSDI'24](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf). S3-FIFO describes ARC and 2Q as sharing one idea: "They separate new and frequent objects into two queues (S and M) so that popular objects are not affected by scan requests" — [S3-FIFO SOSP'23](https://dl.acm.org/doi/10.1145/3600006.3613147)
- **LeCaR** (Vietri et al., HotStorage 2018) "uses reinforcement learning and regret minimization to control its dynamic use of two cache replacement policies, LRU and LFU", and "was shown to outperform ARC for small cache sizes". It has two hand-set hyperparameters, a learning rate and a discount rate — [CACHEUS FAST'21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- **CACHEUS** (Rodriguez, Yusuf, Lyons, Paz, Rangaswami, Liu, Zhao, Narasimhan, USENIX FAST 2021). It classifies workloads into four "primitives": LRU-friendly, LFU-friendly, scan and churn. It removes LeCaR's discount rate and adapts the learning rate. Its variants include C1 = CACHEUS(ARC, LFU), with SR-LRU and CR-LFU as new experts. CACHEUS keeps about 2N metadata, "equivalent to state-of-the-art algorithms such as ARC and LIRS" — [CACHEUS FAST'21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- **Patent.** US6996676B2, "System and method for implementing an adaptive replacement cache policy", inventors Megiddo and Modha. Filed Nov 14, 2002; granted Feb 7, 2006; original assignee IBM; current assignee Tahoe Research Ltd (recorded Aug 15, 2022). Status: **"Expired – Lifetime", adjusted expiration Feb 22, 2024** — [Google Patents US6996676B2](https://patents.google.com/patent/US6996676B2/en)
  - Related patent US7469320B2 ("Adaptive replacement cache"): a search snippet shows expiration Feb 13, 2027 — [Google Patents US7469320B2](https://patents.google.com/patent/US7469320B2/en) **[snippet only, page not fetched]**. US7058766, "adaptive replacement cache with temporal filtering" (CART-like) — [Google Patents US7058766](https://patents.google.com/patent/US7058766) **[status not checked]**.

### Inferences
- For the project (single-threaded Python simulator), CAR/CART's lock-free advantage doesn't matter; plain ARC is the right baseline. CART's "S/L" filter bit is conceptually close to a binary reuse label: a learned reuse probability generalizes it.
- ZFS's move from a unit step on `p` to rate-limited, hit-volume-proportional adjustment of a fraction is an admission by practitioners that ARC's raw adaptation rule over-reacts or under-reacts. The S3-FIFO and CACHEUS findings in Section 4 say the same.
- Patent risk for a research or hackathon simulator is negligible now that the core patent has expired. If ARC ships in a commercial product, check US7469320's claims before 2027 (not legal advice).

### Gaps
- I didn't extract CAR/CART's quantitative hit-ratio tables because the tables did not come through in text form. The only CAR-vs-ARC performance claim I have is the paper's own "inherits virtually all advantages of ARC".
- I didn't verify the claims or status of US7469320 and US7058766.

## 3. ARC's performance in large modern evaluations

### Takeaway
In web, KV and CDN-scale studies from 2023–2025, ARC is a solid "upper-middle" heuristic. It beats LRU by about 6–7% mean miss-ratio reduction, but it is consistently beaten by S3-FIFO and SIEVE and never ranks as the best algorithm on most datasets. On block/storage traces (MSR), ARC is often the best of the classics (CACHEUS). Learned policies (HALP, 3L-Cache, CACHEUS) beat ARC on their own benchmarks.

### Cited Findings
- **SIEVE** (Zhang, Yang, Yue, Vigfusson, Rashmi, USENIX NSDI 2024), evaluated on 1559 traces (247B requests, 14.85B objects) from Twitter, Meta KV/CDN, Wikimedia, TencentPhoto and 2 proprietary CDNs, simulated with libCacheSim — [SIEVE PDF](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
  - "Compared to ARC, SIEVE reduces miss ratio by up to 63.2% with a mean of 1.5%." ARC reduces LRU's miss ratio "by up to 33.7% with a mean of 6.7%" (intro), but the body says "ARC only reduces LRU's miss ratio by 6.3% on average". **The paper gives both 6.7% and 6.3%** — [SIEVE](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
  - SIEVE is best on more than 45% of traces; the "runner-up algorithm, TwoQ, only outperforms other algorithms on 15%". ARC is not the runner-up — [SIEVE](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
  - "TwoQ and ARC achieve efficiency close to SIEVE". Replacing ARC's T2 LRU with SIEVE (ARC-SIEVE) "achieves the best efficiency among all compared algorithms" and reduces ARC's miss ratio by 3.7% on average, up to 62.5% — [SIEVE](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
  - On synthetic Zipf α=1.0, LFU "performs similarly to ARC and is visibly worse than SIEVE", and ARC and SIEVE "can quickly remove new and potentially unpopular objects" — [SIEVE](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
- **S3-FIFO** (Yang, Zhang, Qiu, Yue, Rashmi, ACM SOSP 2023), "FIFO Queues are All You Need for Cache Eviction", evaluated on 6594 traces from 14 datasets (856B requests, 2007–2023) against 12 algorithms — [ACM DL](https://dl.acm.org/doi/10.1145/3600006.3613147)
  - S3-FIFO is the best algorithm on 10/14 datasets at the large cache size and 7/14 at the small size; "no other algorithm is the best on more than 3 datasets". "The next best algorithm (LIRS) obtains the highest efficiency on only 2 datasets", so ARC is not the next best — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147)
  - "ARC is less efficient than S3-FIFO because the adaptive algorithm is not sufficient"; "using two LRU queues, such as in ARC, is worse than S3-FIFO most of the time" — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147)
  - Table 2 / Fig. 10 miss ratios (ARC vs LRU), from the figure descriptions in the ACM PDF text: Twitter large cache 0.0483 vs 0.0488; Twitter small 0.1941 vs 0.2005; MSR large 0.2891 vs 0.3188; MSR small 0.4899 vs 0.5263 — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147)
  - Throughput: S3-FIFO achieves more than 6× the throughput of optimized LRU at 16 cores in CacheLib, because FIFO permits lock-free operation. ARC was not benchmarked for throughput — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147)
- **3L-Cache** (Zhou, Niu, Xiong, Fang, Wang, USENIX FAST 2025) compares 12 policies including ARC, SIEVE, S3-FIFO, TinyLFU, LHD, GDSF, LeCaR, CACHEUS, GL-Cache, LRB and HALP on 4855 traces from 8 datasets — [3L-Cache](https://www.usenix.org/conference/fast25/presentation/zhou-wenbin)
  - "Some policies, such as LHD, ARC, TinyLFU, and GL-Cache, occasionally fail to reduce LRU's byte miss ratio" — [3L-Cache](https://www.usenix.org/conference/fast25/presentation/zhou-wenbin)
  - With each policy tuned for byte miss ratio, 3L-Cache has a lower object miss ratio than ARC. With each tuned for object miss ratio, 3L-Cache's byte miss ratio is "worse than ARC, SIEVE, S3-FIFO, LeCaR, CACHEUS, and HALP" — [3L-Cache](https://www.usenix.org/conference/fast25/presentation/zhou-wenbin)
- **HALP** (Song et al., Google; USENIX NSDI 2023) on YouTube CDN traces, compared with LRU, FIFO and ARC: "HALP achieves a strictly better performance than all other algorithms on 92.6% of traces, ... the same ... on 7% ... worse than the best algorithm on only 0.4%" (P95 byte miss ratio) — [HALP](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu)
- **LRB** (Song, Berger, Li, Lloyd, NSDI 2020) cites ARC only as a reference. In the extracted text, ARC appears only in the bibliography, not in the evaluation — [LRB](https://www.usenix.org/conference/nsdi20/presentation/song)
- **CACHEUS** (FAST 2021) uses FIU, MSR, CloudPhysics, CloudVPS and CloudCache block traces — [CACHEUS](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
  - "No algorithm is a clear winner ... ARC outperforms the rest of the competitors for a majority of the MSR workloads". For example, "ARC has the highest hit-rate in 34% of the workloads for MSR at the 0.05% cache size" — [CACHEUS](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
  - FIU webmail (day 16), cache at 10% of footprint: total hit rates are ARC 30.08%, LIRS 40.71%, LeCaR 42.08%, CACHEUS C3 43.95%. The best-case C3 improvement is 38.32% over ARC (CloudPhysics, 10% cache) — [CACHEUS](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- W-TinyLFU: S3-FIFO calls TinyLFU "the closest competitor" (to S3-FIFO), but notes that TinyLFU is worse than FIFO on about 20% of traces (about 50% at small cache sizes) — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147). I found no direct ARC-vs-W-TinyLFU mean in the text I extracted.

### Inferences
- Rough ranking on web/KV/CDN traces by miss ratio: S3-FIFO ≈ SIEVE ≳ TinyLFU (on its good traces) / 2Q / ARC > LRU. On block/storage traces ARC is more competitive, and scan resistance matters more there.
- For a Belady-imitation learner with uniform sizes, ARC is a good "strong classical" baseline to report alongside LRU, S3-FIFO and SIEVE. Beating LRU only is a weak result, since ARC already gets about 6–7% over LRU at zero training cost.

### Gaps
- None of the papers gives exact ARC percentiles for the S3-FIFO Fig. 6/7 bars; I only have text-level statements.
- I found no head-to-head numbers for ARC vs LRB, and no ARC throughput numbers in modern multi-threaded benchmarks.

## 4. Known weaknesses

### Takeaway
ARC's adaptation is a unit-ish step driven by ghost hits, and it misjudges the split both ways. It can collapse T1 to almost nothing: about 0.01% of the cache on Twitter traces, causing premature eviction of new popular objects. It can also adapt too slowly after a scan followed by a phase change (steps of 1, B2 stays empty). T2 is LRU, so ARC cannot represent full frequency distributions and does badly on LFU-friendly and churn patterns. It can exhibit Belady's anomaly. Its LRU hit path needs a global lock, which is why CAR exists.

### Cited Findings
- **Split too small or too large.** ARC "can identify the correct direction to adjust the size, but the size it finds is often too large or too small ... ARC chooses a very small S on the Twitter trace ... around 0.01% of cache size". The cause: objects evicted from M are requested again very soon, and constantly generated new popular objects "suffer a miss before being inserted in M", which gives low precision — [S3-FIFO SOSP'23](https://dl.acm.org/doi/10.1145/3600006.3613147)
- **Arbitrary step and over-reaction.** "ARC moves one slot upon a hit on the ghost. But the question remains why one slot instead of half or two?" Also, "small perturbations in the workload often cause the adaptive algorithm to overreact", and adaptive schemes that follow gradients "implicitly assume that the miss ratio curve is convex", although scan-heavy workload curves often are not convex — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147)
- **Small caches.** "ARC's adaptive algorithm sometimes shrinks the recency queue to very small and yields a high miss ratio" at small cache sizes, whereas 2Q's fixed 25% avoids this. Also, "although ARC has no explicit parameters, its adaptive algorithm uses implicit parameters in deciding when and how much space to move" — [SIEVE NSDI'24](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
- **Slow adaptation after a scan followed by churn.** "ARC protects T2 ... p close to 0 ... Right after the scan finishes ... ARC starts to increase the size of T1 ... However, the increments in p grow T1 slowly in steps of 1 ... ARC maintains its shadow list B2 empty by avoiding evictions from T2, even during churn." Also: "when a scan phase is followed by a churn phase, ARC continues to evict from T1 and behaves similar to LRU" — [CACHEUS FAST'21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- **Frequency blindness and churn.** "Since ARC uses an LRU list for T2, it is unable to capture the full frequency distribution of the workload and perform well for LFU-friendly workloads ... for churn workloads, ARC's inability to distinguish between items that are equally important leads to continuous cache replacement" — [CACHEUS](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- **One-hit wonders.** Scan-resistant algorithms including ARC "cannot guarantee the minimum and maximum time one-hit wonders stay in the cache ... sometimes evict too fast or too slowly". S3-FIFO measured a median one-hit-wonder ratio of 26% across 6594 traces — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147)
- **Belady's anomaly.** "Previous work reports that both LIRS and ARC exhibit Belady's anomaly: miss ratio increases with the cache size for some workloads" — [SIEVE](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf), which cites its refs [50, 85]. I did not trace the original sources.
- **Losing to LRU.** ARC "occasionally fail[s] to reduce LRU's byte miss ratio" — [3L-Cache FAST'25](https://www.usenix.org/conference/fast25/presentation/zhou-wenbin). In the S3-FIFO Twitter/large case, ARC is only marginally better than LRU (0.0483 vs 0.0488) — [S3-FIFO](https://dl.acm.org/doi/10.1145/3600006.3613147)
- **Concurrency.** An LRU hit "serializes cache hits behind a single global lock". ARC's T1/T2 are LRU lists, which motivated CAR — [CAR FAST'04](https://www.usenix.org/legacy/publications/library/proceedings/fast04/tech/full_papers/bansal/bansal.pdf). A more intricate algorithm has "potentially longer critical sections, reducing both throughput and scalability" — [SIEVE](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
- **Code complexity.** ARC needs 64/108/20 LOC for hit/eviction/insertion against 5/4/3 for LRU — [SIEVE Table 2](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)
- **Stale ghosts.** OpenZFS caps each ghost list at half the size of the other states because "excessive ghost size may result in false ghost hits (too far back), that may never result in real cache hits" — [OpenZFS arc.c](https://github.com/openzfs/zfs/blob/master/module/zfs/arc.c)

### Inferences
- These failure modes are precisely where a per-object learned reuse predictor should help: distinguishing new-and-popular from one-hit objects inside T1, and capturing frequency inside T2. ARC's global split then becomes less critical. Conversely, ARC's p can act as a cheap regime detector, since a large p signals recency pressure and a small p signals frequency pressure. That could be a feature for the learner, or a gate for its fallback.
- Adversarial pattern: a phase that ends right after a long scan drives p to 0, and ARC then recovers only in steps of about 1 per B1 hit. A synthetic trace of scan, then churn, then LRU-friendly accesses is a good stress test for any ARC-flavoured component in the project.

### Gaps
- I found no formal competitive-ratio or adversarial lower-bound result for ARC in the sources I read.
- I have no quantitative multi-threaded throughput measurement of ARC itself, only CAR's qualitative argument and SIEVE/S3-FIFO numbers for LRU.

## 5. libCacheSim (C and Python) ARC support; small Python references

### Takeaway
Yes. libCacheSim implements ARC, and also CAR, 2Q, LIRS, Clock-Pro, LeCaR, Cacheus, S3-FIFO, SIEVE, W-TinyLFU, LHD, LRB, GLCache, 3LCache and Belady. It is pip-installable as `libcachesim` and supports custom Python policies via `PluginCache`. The trauzti/cache `ARC.py` is the small Python reference that libCacheSim's ARC was cross-checked against. The ~30-line implementation in Section 1 was tested here.

### Cited Findings
- libCacheSim's algorithm list includes "Adaptive algorithms: ARC, CAR, 2Q, LIRS, Clock-Pro, MQ, and LeCaR"; "Recent: S3-FIFO, SIEVE, QDLP, Cacheus, Hyperbolic, WTinyLFU"; "ML: LHD, LRB (-DENABLE_LRB=ON), GLCache (-DENABLE_GLCACHE=ON)"; plus 3LCache, S4-FIFO, LRU-K and Belady. It claims "over 20M requests/sec" trace replay — [libCacheSim GitHub](https://github.com/1a1a11a/libCacheSim)
- Python: `pip install libcachesim`. Usage example: `from libcachesim import SyntheticReader, FIFO; cache = FIFO(cache_size=...); obj_miss_ratio, byte_miss_ratio = cache.process_trace(reader)`. Custom policies go through `PluginCache` hooks (init/hit/miss/eviction) — [libCacheSim GitHub](https://github.com/1a1a11a/libCacheSim)
- The C implementation of ARC is `libCacheSim/cache/eviction/ARC.c`. It maps L1_data = T1, L1_ghost = B1, and so on, and is "cross checked with https://github.com/trauzti/cache/blob/master/ARC.py" — [libCacheSim ARC.c](https://github.com/1a1a11a/libCacheSim/blob/develop/libCacheSim/cache/eviction/ARC.c)
- SIEVE, S3-FIFO and CACHEUS-era comparisons were run in libCacheSim, so numbers from libCacheSim ARC are directly comparable with those papers — [SIEVE](https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf)

### Inferences
- Lowest-effort way to get an ARC baseline in the Python simulator: use the Section 1 class, which is dependency-free, uses uniform sizes and has invariant checks. For cross-validation, replay the same trace through `libcachesim` with unit object sizes and compare miss ratios.

### Gaps
- I did not confirm whether the Python package exposes `ARC` as a top-level class name. The README example only shows `FIFO`, and I didn't fetch the package's API list.
- I didn't fetch or check `trauzti/cache/ARC.py` myself; its existence is based on libCacheSim's comment.
