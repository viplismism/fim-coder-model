# Diagnosis: AST-trained vs random-trained (6.7B, seed 42)

Scope: **validation split only** (518 AST-hole and 518 random-hole examples; 14 packages). Test, benchmark and
the fresh held-out set were not used. Script `diagnose.py`, full numbers in `report.json`.
Char perplexity, lower is better; AST − random with paired bootstrap 95% CI (2,000 resamples of examples).
About 25 buckets are compared, so an occasional CI excluding 0 is expected by chance.

## AST-hole validation set (holes are exact syntax nodes)

| bucket | n | base | AST | random | AST − random [95% CI] |
|---|---|---|---|---|---|
| overall | 518 | 1.2113 | 1.1534 | 1.1534 | −0.0000 [−0.0004, +0.0004] |
| 1–8 tokens | 52 | 1.1750 | 1.1233 | 1.1277 | −0.0044 [−0.0183, +0.0063] |
| 9–32 | 115 | 1.2559 | 1.1856 | 1.1783 | +0.0073 [−0.0001, +0.0158] |
| 33–128 | 114 | 1.1577 | 1.1013 | 1.1029 | −0.0017 [−0.0044, +0.0006] |
| 129–512 | 119 | 1.1877 | 1.1304 | 1.1304 | +0.0000 [−0.0008, +0.0008] |
| 513+ | 118 | 1.2193 | 1.1613 | 1.1614 | −0.0000 [−0.0005, +0.0004] |
| code | 494 | 1.2108 | 1.1528 | 1.1529 | −0.0001 [−0.0005, +0.0003] |
| **comment** | 24 | 1.4020 | 1.3618 | 1.3363 | **+0.0255 [+0.0126, +0.0405]** |
| whole lines | 200 | 1.2017 | 1.1488 | 1.1490 | −0.0001 [−0.0007, +0.0003] |
| starts mid-line | 121 | 1.2281 | 1.1634 | 1.1631 | +0.0003 [−0.0004, +0.0009] |
| ends mid-line | 69 | 1.1803 | 1.1192 | 1.1192 | −0.0000 [−0.0019, +0.0034] |
| both mid-line | 128 | 1.2062 | 1.1467 | 1.1473 | −0.0006 [−0.0023, +0.0013] |

All AST holes are node-aligned. (The 19 flagged "one end mid-node" are `///` doc comments whose node includes
the trailing newline; an artifact of the whitespace trim in the boundary check, i.e. the comment bucket.)

## Random-hole validation set (arbitrary character spans)

| bucket | n | base | AST | random | AST − random [95% CI] |
|---|---|---|---|---|---|
| overall | 518 | 1.2138 | 1.1672 | 1.1651 | **+0.0021 [+0.0016, +0.0027]** |
| 1–8 tokens | 58 | 1.1348 | 1.0706 | 1.0647 | +0.0060 [−0.0034, +0.0169] |
| 9–32 | 108 | 1.1929 | 1.1340 | 1.1314 | +0.0026 [−0.0012, +0.0067] |
| 33–128 | 109 | 1.1666 | 1.1138 | 1.1109 | **+0.0028 [+0.0004, +0.0056]** |
| 129–512 | 120 | 1.2105 | 1.1640 | 1.1602 | **+0.0038 [+0.0025, +0.0053]** |
| 513+ | 123 | 1.2171 | 1.1709 | 1.1692 | **+0.0017 [+0.0011, +0.0023]** |
| code | 488 | 1.2133 | 1.1666 | 1.1645 | **+0.0021 [+0.0016, +0.0027]** |
| comment-only | 30 | 1.3481 | 1.3286 | 1.3212 | **+0.0074 [+0.0006, +0.0170]** |
| both mid-line | 420 | 1.2124 | 1.1649 | 1.1628 | **+0.0021 [+0.0015, +0.0028]** |
| ends mid-line | 80 | 1.2179 | 1.1798 | 1.1775 | **+0.0023 [+0.0011, +0.0042]** |
| starts mid-line | 16 | 1.2447 | 1.1835 | 1.1830 | +0.0006 [−0.0016, +0.0040] |
| both ends mid-node | 325 | 1.2237 | 1.1761 | 1.1739 | **+0.0022 [+0.0015, +0.0030]** |
| one end mid-node | 178 | 1.1933 | 1.1489 | 1.1470 | **+0.0019 [+0.0010, +0.0029]** |
| both ends node-aligned | 15 | 1.2539 | 1.1998 | 1.1962 | +0.0036 [+0.0010, +0.0074] |

## What each arm was trained on (7,549 training holes each)

| | AST arm | random arm |
|---|---|---|
| both ends node-aligned | 96.8% | 6.0% |
| both ends mid-node | **0.0%** | 58.7% |
| whole lines / both mid-line | 41.4% / 26.0% | 0.5% / 78.5% |
| comment-only holes | 4.2% | 6.4% (plus many partial-comment spans) |
| hole length (characters) | identical by construction | identical by construction |

## Verdict on the hypotheses

1. **Holes too clean (boundary robustness): supported.** The AST-trained model never saw a hole with both ends
   mid-node and loses on exactly those holes (random set: +0.0021 overall; also in every mid-line/mid-node bucket),
   while tying on node-aligned holes. The "too big" part is **not** supported: on node-aligned holes there is no gap at
   any size, and hole lengths are identical between arms by construction, so reweighting node size is not the lever.
   **No bucket shows an AST advantage**: even on the holes it was trained for, AST training is no better than random.
2. **Half of the paper's recipe: not testable from this data.** Note that aligned-span holes are also node-aligned at
   both ends, so aligned-span alone would not fix the boundary-robustness gap; the 10% random FIM hygiene targets that.
3. **Too little data: not testable from this data;** neither supported nor ruled out.
4. **No left-to-right examples: consistent but not decomposed here** (validation L2R: random better by +0.0048).
   Moot for the next round: L2R examples become hygiene for every arm.

Additional finding: **comments.** The AST arm is worse on comment holes in both sets (+0.0255, n=24; +0.0074, n=30).
The random arm sees more comment text as targets. Small n and multiple comparisons, so tentative.
