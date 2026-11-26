#!/usr/bin/env python3
"""Stop-effect check for telekom/controlplane: fine-tuned vs base Qwen-14B on all examples and on the subset where
NEITHER model hit the 256-token limit (both generations ended on a stop token). If the gain disappears on that subset,
it came from the model learning where to stop rather than from writing better code.
Metrics: build_test_pass (sensitive examples) and exact_match (all), split all / source / test files.
Paired bootstrap over commits, B = 2000, seed 20261118. Logic is identical to results/controlplane/stop_effect.json.
Usage: python -m eval.stop_effect_controlplane <scores_dir> <out.json>
"""
import collections, json, random, sys

D, OUT = sys.argv[1], sys.argv[2]
S = {n: {r['id']: r for r in map(json.loads, open(f'{D}/{n}.scores.jsonl'))} for n in ('qwen14b-ft', 'qwen14b-base')}
C = {n: {r['id']: r for r in map(json.loads, open(f'{D}/{n}.completions.jsonl'))} for n in S}
ids = sorted(set(S['qwen14b-ft']) & set(S['qwen14b-base']))
both_stop = [i for i in ids if C['qwen14b-ft'][i]['stopped'] and C['qwen14b-base'][i]['stopped']]
print('examples', len(ids), 'neither hit limit', len(both_stop))
B = 2000; out = {}
for metric, sel in (('build_test_pass', lambda r: r.get('sensitive')), ('exact_match', lambda r: True)):
    for part, pf in (('all', lambda r: True), ('source', lambda r: not r['is_test_file']), ('test', lambda r: r['is_test_file'])):
        for subset_name, pool in (('full', ids), ('neither_hit_limit', both_stop)):
            ex = [i for i in pool if sel(S['qwen14b-ft'][i]) and pf(S['qwen14b-ft'][i])]
            byc = collections.defaultdict(list)
            for i in ex: byc[i.split(':')[0]].append(i)
            cs = sorted(byc); v = lambda n, idx: sum(bool(S[n][i].get(metric)) for i in idx) / max(1, len(idx))
            rng = random.Random(20261118); ds = []
            for _ in range(B):
                smp = [i for c in (cs[rng.randrange(len(cs))] for _ in cs) for i in byc[c]]
                ds.append(v('qwen14b-ft', smp) - v('qwen14b-base', smp))
            ds.sort(); d = v('qwen14b-ft', ex) - v('qwen14b-base', ex); ci = [round(ds[int(.025 * B)], 4), round(ds[int(.975 * B) - 1], 4)]
            out[f'{metric}/{part}/{subset_name}'] = {'n': len(ex), 'ft': round(v('qwen14b-ft', ex), 4), 'base': round(v('qwen14b-base', ex), 4),
                                                     'diff': round(d, 4), 'ci': ci, 'significant': ci[0] > 0}
            print(f"{metric:16s} {part:6s} {subset_name:18s} n={len(ex):3d} diff={d:+.4f} CI={ci}")
json.dump(out, open(OUT, 'w'), indent=1)
