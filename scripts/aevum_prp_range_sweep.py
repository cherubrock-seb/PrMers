#!/usr/bin/env python3
"""PRP range sweep: real AUTO baseline vs bounded explicit FFT/PFA plans.

Engine-only by design: word-exact A/B, alternating order, medians, winner confirmation.
No Gerbicz/full-PRP validation is run here; this script maps plan throughput only.
"""
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import zipfile
from pathlib import Path

import aevum_bench_runner as h

ROOT = h.ROOT
OUT = h.OUT
BASE = h.BASE | {
    'AEVUM_PRP_MIDDLE1': '0',
    'AEVUM_RADIX1K': '4',
}
h.BASE = BASE
REPS = max(3, int(os.getenv('AEVUM_REPEATS', '3')))
CONFIRM_REPS = max(5, int(os.getenv('AEVUM_CONFIRM_REPEATS', '5')))
DEFAULT_EXPONENTS = [
    100000007,
    120000007,
    130000001,
    140000041,
    150000001,
    160000003,
    180000017,
    197000003,
    200000033,
    210000017,
    220000013,
]


def exponents():
    text = os.getenv('AEVUM_SWEEP_EXPONENTS', '').strip()
    if not text:
        return DEFAULT_EXPONENTS
    vals = []
    for item in text.split(','):
        item = item.strip().replace('_', '')
        if not item:
            continue
        p = int(item)
        if p < 1000000:
            raise RuntimeError('sweep exponents must be >= 1,000,000')
        vals.append(p)
    if not vals:
        raise RuntimeError('AEVUM_SWEEP_EXPONENTS is empty')
    return vals


def prepare():
    dst = OUT / 'build' / 'optimized'
    shutil.copytree(ROOT, dst, ignore=h.ignore)
    (dst / 'third_party/aevum/src/bundle.cpp').unlink(missing_ok=True)
    tracked = [
        ROOT / 'third_party/aevum/src/FFTConfig.cpp',
        ROOT / 'third_party/aevum/src/EngineApi.cpp',
        ROOT / 'src/aevum/AutoPolicy.cpp',
        ROOT / 'scripts/aevum_prp_range_sweep.py',
    ]
    (OUT / 'source-sha256.json').write_text(json.dumps({
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tracked if p.exists()
    }, indent=2))


def engine(device, p, plan, n=256, profile=False):
    r = h.engine(BASE, device, p, plan, 'prp', n, False, profile)
    text = (Path(r['path']) / 'run.log').read_text(errors='replace')
    if 'may be too small' in text:
        raise h.CandidateFailure(f'capacity warning: {plan} at p={p}')
    return r


def exact_same(a, b):
    av = (Path(a['path']) / 'residue.bin').read_bytes()
    bv = (Path(b['path']) / 'residue.bin').read_bytes()
    if a['p'] != b['p'] or a['mode'] != b['mode'] or av != bv:
        first = next((i for i in range(min(len(av), len(bv))) if av[i] != bv[i]), min(len(av), len(bv))) // 4
        raise h.CandidateFailure(
            f'WORD MISMATCH p={a["p"]} word={first}: {a["path"]} vs {b["path"]}'
        )


def paired(device, p, candidate_plan, n, reps):
    samples = [[], []]
    last = None
    for rep in range(reps + 1):
        pair = {}
        order = [0, 1] if rep % 2 == 0 else [1, 0]
        for side in order:
            plan = '' if side == 0 else candidate_plan
            pair[side] = engine(device, p, plan, n)
            if rep:
                samples[side].append(pair[side]['seconds'])
        exact_same(pair[0], pair[1])
        last = pair
    med = [statistics.median(x) for x in samples]
    return {
        'auto_median_s': med[0],
        'candidate_median_s': med[1],
        'speedup_vs_auto': med[0] / med[1],
        'auto_samples_s': samples[0],
        'candidate_samples_s': samples[1],
        'iterations': n,
        'auto_resolved_plan': last[0].get('resolved_plan', ''),
        'candidate_resolved_plan': last[1].get('resolved_plan', candidate_plan),
        'transform_auto': last[0].get('transform'),
        'transform_candidate': last[1].get('transform'),
    }


def plans_for(p):
    # Bounded, high-value plans only. The known 100M and issue-36 shapes are first.
    plans = []

    # 4M Type1: exact paired GF31/GF61. Safe capacity region ends near 165M.
    if p <= 165_000_000:
        plans += [
            '1:512:8:512:101',
            '1:512:8:512:202',
            '1:1K:8:256:101',
            '1:1K:8:256:202',
            '1:1K:4:512:101',
            '1:1K:4:512:202',
        ]

    # 8M Type1: important once 4M approaches/exceeds its capacity window.
    if p >= 140_000_000:
        plans += [
            '1:512:16:512:101',
            '1:512:16:512:202',
            '1:1K:8:512:101',
            '1:1K:8:512:202',
            '1:1K:16:256:101',
            '1:1K:16:256:202',
            '1:4K:4:256:101',
            '1:4K:4:256:202',
        ]

    # Measured/implemented Type4 power-of-two bridge region.
    if 150_000_000 <= p <= 198_000_000:
        plans += [
            '4:1K:8:256:101',
            '4:512:8:512:202',
        ]

    # PFA9: small exact/adaptive shapes around the 165-220M transition,
    # plus the 9.44M shape previously exercised around ~197M.
    if 160_000_000 <= p <= 185_000_000:
        plans += [
            'pfa9:1:512:9:512:202',
            'pfa9:1:1K:9:256:202',
        ]
    if 160_000_000 <= p <= 223_000_000:
        plans += [
            'pfa9:4:512:9:512:202',
            'pfa9:4:1K:9:256:202',
        ]
    if p >= 180_000_000:
        plans += [
            'pfa9:1:512:9:1K:202',
            'pfa9:4:512:9:1K:202',
        ]

    # Stable de-duplication preserving priority order.
    seen = set()
    return [x for x in plans if not (x in seen or seen.add(x))]


def run():
    device_list = sys.argv[4].split(',')
    summary = []
    csv_rows = []
    (OUT / 'controls.json').write_text(json.dumps({
        'devices': device_list,
        'exponents': exponents(),
        'repeats': REPS,
        'confirm_repeats': CONFIRM_REPS,
        'note': 'TRUE AUTO is the reference. No forced Type4 baseline. Engine-only word-exact sweep.',
    }, indent=2))

    for device in device_list:
        int(device)
        ds = {'device': device, 'cases': []}
        summary.append(ds)
        print(f'device {device}: PRP range sweep vs TRUE AUTO', flush=True)

        for p in exponents():
            case = {'p': p, 'candidates': [], 'winner': None}
            ds['cases'].append(case)
            try:
                pilot = engine(device, p, '', 256)
                # Aim for about 0.30 s per timed sample; keep the sweep bounded.
                n = int(os.getenv(
                    'AEVUM_BENCH_ITERS',
                    str(min(2048, max(256, 256 * math.ceil(0.30 / max(pilot['seconds'], 1e-9))))),
                ))
                auto_check = engine(device, p, '', 1)
                case['auto'] = {
                    'resolved_plan': pilot.get('resolved_plan', ''),
                    'transform': pilot.get('transform'),
                    'pilot_seconds_256': pilot.get('seconds'),
                    'iterations': n,
                }
                print(f'  p={p} AUTO={case["auto"]["resolved_plan"]} n={n}', flush=True)
            except (h.CandidateFailure, RuntimeError) as exc:
                case['baseline_failure'] = str(exc)
                h.summary = summary
                h.save()
                continue

            resolved_seen = {case['auto']['resolved_plan']}
            best = None
            for plan in plans_for(p):
                attempt = {'requested_plan': plan}
                case['candidates'].append(attempt)
                try:
                    # Cheap preflight + exact single-step check.
                    trial = engine(device, p, plan, 1)
                    exact_same(auto_check, trial)
                    resolved = trial.get('resolved_plan', plan)
                    attempt['resolved_plan'] = resolved
                    attempt['transform'] = trial.get('transform')
                    if resolved in resolved_seen:
                        attempt['skipped'] = 'same effective plan already tested (or identical to AUTO)'
                        print(f'    {plan}: SAME-AS-AUTO/DUP ({resolved})', flush=True)
                        continue
                    resolved_seen.add(resolved)

                    timing = paired(device, p, plan, n, REPS)
                    attempt['screen'] = timing
                    ratio = timing['speedup_vs_auto']
                    tag = 'STRONG' if ratio >= 1.05 else ('small+' if ratio > 1.0 else 'slower')
                    print(f'    {plan}: {ratio:.4f}x vs AUTO [{tag}]', flush=True)
                    if best is None or ratio > best['screen']['speedup_vs_auto']:
                        best = attempt
                except (h.CandidateFailure, RuntimeError) as exc:
                    attempt['failure'] = str(exc)
                    print(f'    {plan}: SKIP/REVERT; {exc}', flush=True)
                h.summary = summary
                h.save()

            if best and best['screen']['speedup_vs_auto'] > 1.0:
                try:
                    confirm = paired(device, p, best['requested_plan'], n, CONFIRM_REPS)
                    best['confirmation'] = confirm
                    case['winner'] = {
                        'requested_plan': best['requested_plan'],
                        'resolved_plan': confirm['candidate_resolved_plan'],
                        'speedup_vs_auto_screen': best['screen']['speedup_vs_auto'],
                        'speedup_vs_auto_confirmed': confirm['speedup_vs_auto'],
                        'production_candidate': confirm['speedup_vs_auto'] >= 1.05,
                    }
                    # Event profile is diagnostic only; wall-time A/B above is authoritative.
                    a = engine(device, p, '', 128, profile=True)
                    b = engine(device, p, best['requested_plan'], 128, profile=True)
                    exact_same(a, b)
                    case['profile'] = {
                        'auto': a.get('kernels', {}),
                        'candidate': b.get('kernels', {}),
                        'note': '128-square event diagnostic; A/B median wall time is authoritative.',
                    }
                    print(
                        f'    CONFIRM best={best["requested_plan"]}: '
                        f'{confirm["speedup_vs_auto"]:.4f}x vs AUTO '
                        f'{"PRODUCTION-CANDIDATE" if confirm["speedup_vs_auto"] >= 1.05 else "informational"}',
                        flush=True,
                    )
                except (h.CandidateFailure, RuntimeError) as exc:
                    case['winner_failure'] = str(exc)
                    print(f'    CONFIRM failed: {exc}', flush=True)

            winner = case.get('winner') or {}
            csv_rows.append({
                'device': device,
                'p': p,
                'auto_plan': case.get('auto', {}).get('resolved_plan', ''),
                'auto_transform': case.get('auto', {}).get('transform', ''),
                'best_plan': winner.get('resolved_plan', 'AUTO'),
                'confirmed_speedup_vs_auto': winner.get('speedup_vs_auto_confirmed', 1.0),
                'production_candidate_ge_1_05': winner.get('production_candidate', False),
            })
            h.summary = summary
            h.save()

    h.summary = summary
    h.save()
    with (OUT / 'prp-range-summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'device', 'p', 'auto_plan', 'auto_transform', 'best_plan',
            'confirmed_speedup_vs_auto', 'production_candidate_ge_1_05'
        ])
        w.writeheader()
        w.writerows(csv_rows)
    print('\nSweep complete. TRUE AUTO is the baseline for every row.', flush=True)
    print('Return tuning-output.zip plus prp-range-summary.csv.', flush=True)


def pack():
    h.pack()
    csv_path = OUT / 'prp-range-summary.csv'
    if csv_path.exists():
        with zipfile.ZipFile(OUT / 'tuning-output.zip', 'a', zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, csv_path.name)


if __name__ == '__main__':
    try:
        {'prepare': prepare, 'run': run, 'pack': pack}[sys.argv[1]]()
    except Exception as exc:
        (OUT / 'FAILED.txt').write_text(str(exc) + '\n')
        try:
            pack()
        except Exception:
            pass
        raise
