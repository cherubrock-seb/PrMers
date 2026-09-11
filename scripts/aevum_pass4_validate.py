#!/usr/bin/env python3
import argparse, hashlib, json, os, re, statistics, subprocess, time
from pathlib import Path

BENCH_RE = re.compile(r'AEVUM_BENCH\s+(\{.*\})')
DEVICE_RE = re.compile(
    r"AEVUM_DEVICE vendor_id='([^']*)' name='([^']*)' driver='([^']*)' runtime='([^']*)' device=(\d+)")
AUTOTUNE_START_RE = re.compile(
    r'Aevum autotune: cache (\S+) workload=(\S+) band=(\d+)-(\d+) native=(\S+) candidates<=(\d+) budget=(\d+)ms\.')
CANDIDATE_RE = re.compile(
    r'Aevum autotune: candidate=(\S+) median=([0-9.eE+-]+)s native=([0-9.eE+-]+)s speedup=([0-9.eE+-]+)x\.')
MISMATCH_RE = re.compile(r'Aevum autotune: reject (\S+) \(WORD MISMATCH\)\.')
REJECT_RE = re.compile(r'Aevum autotune: candidate=(\S+) rejected \((.*)\)\.')
SELECTED_RE = re.compile(
    r'Aevum autotune: selected=(\S+) advantage=([0-9.eE+-]+)x tested=(\d+) tune=(\d+)ms\.')
CACHE_HIT_RE = re.compile(
    r'Aevum autotune: cache hit workload=(\S+) plan=(\S+) prep-lead=(\d+) '
    r'cached-plan-advantage=([0-9.eE+-]+)x cached-impl-advantage=([0-9.eE+-]+)x prior-tune=(\d+)ms\.')
BYPASS_RE = re.compile(r'Aevum autotune: bypassed \((.*)\); manual/source precedence preserved\.')
BRIDGE_AB_RE = re.compile(
    r'Aevum prepared-multiply lead bridge: exact=yes canonical=([0-9.eE+-]+)s '
    r'bridge=([0-9.eE+-]+)s speedup=([0-9.eE+-]+)x selected=(\d+)\.')
BRIDGE_MANUAL_RE = re.compile(r'Aevum prepared-multiply lead bridge: manual override=(\d+)\.')
PROFILE_RE = re.compile(r'AEVUM_PROFILE name=(\S+) calls=(\d+) exec_ns=(-?\d+)')
AUTO_PLAN_RE = re.compile(r'Aevum auto FFT: (\S+) for exponent')
TUNED_PLAN_RE = re.compile(r'Aevum tuned FFT: (\S+) for exponent')


def parse_bench(text):
    m = BENCH_RE.search(text)
    if not m:
        raise RuntimeError('AEVUM_BENCH JSON missing')
    return json.loads(m.group(1))


def parse_diagnostics(text):
    d = {
        'device': None,
        'default_plan': None,
        'autotune': {
            'cache_decision': None,
            'workload_class': None,
            'native_plan': None,
            'candidate_budget': None,
            'budget_ms': None,
            'candidates': [],
            'selected_plan': None,
            'selected_speedup': None,
            'tested': None,
            'tune_ms': None,
            'correctness': 'PASS',
        },
        'prepared_width_bridge': {
            'state': None,
            'exact': None,
            'selected': None,
            'canonical_seconds': None,
            'bridge_seconds': None,
            'speedup': None,
            'reason': None,
        },
        'kernel_profile': [],
    }

    m = DEVICE_RE.search(text)
    if m:
        d['device'] = {
            'vendor_id': m.group(1), 'name': m.group(2), 'driver': m.group(3),
            'runtime': m.group(4), 'flattened_device': int(m.group(5)),
        }

    auto_plans = AUTO_PLAN_RE.findall(text)
    tuned_plans = TUNED_PLAN_RE.findall(text)
    if tuned_plans:
        d['default_plan'] = tuned_plans[-1]
    elif auto_plans:
        d['default_plan'] = auto_plans[-1]

    a = d['autotune']
    m = AUTOTUNE_START_RE.search(text)
    if m:
        a.update({
            'cache_decision': m.group(1),
            'workload_class': m.group(2),
            'band': [int(m.group(3)), int(m.group(4))],
            'native_plan': m.group(5),
            'candidate_budget': int(m.group(6)),
            'budget_ms': int(m.group(7)),
        })
        if d['default_plan'] is None:
            d['default_plan'] = m.group(5)

    for m in CANDIDATE_RE.finditer(text):
        a['candidates'].append({
            'plan': m.group(1), 'candidate_seconds': float(m.group(2)),
            'native_seconds': float(m.group(3)), 'speedup': float(m.group(4)),
            'correctness': 'PASS',
        })
    for m in MISMATCH_RE.finditer(text):
        a['candidates'].append({'plan': m.group(1), 'correctness': 'WORD_MISMATCH', 'rejected': True})
        a['correctness'] = 'PASS_WITH_REJECTED_MISMATCH_CANDIDATE'
    for m in REJECT_RE.finditer(text):
        a['candidates'].append({'plan': m.group(1), 'correctness': 'NOT_TIMED', 'rejected': True, 'reason': m.group(2)})

    m = SELECTED_RE.search(text)
    if m:
        a.update({
            'selected_plan': m.group(1), 'selected_speedup': float(m.group(2)),
            'tested': int(m.group(3)), 'tune_ms': int(m.group(4)),
        })

    m = CACHE_HIT_RE.search(text)
    if m:
        a.update({
            'cache_decision': 'hit', 'workload_class': m.group(1),
            'selected_plan': m.group(2), 'selected_speedup': float(m.group(4)),
            'implementation_speedup': float(m.group(5)), 'tune_ms': int(m.group(6)),
            'cached_prepared_width_bridge': bool(int(m.group(3))),
        })
        if d['default_plan'] is None:
            d['default_plan'] = m.group(2)

    m = BYPASS_RE.search(text)
    if m:
        a['cache_decision'] = 'bypassed'
        a['bypass_reason'] = m.group(1)
    elif 'Aevum autotune: OFF;' in text:
        a['cache_decision'] = 'off'

    b = d['prepared_width_bridge']
    m = BRIDGE_MANUAL_RE.search(text)
    if m:
        b['selected'] = bool(int(m.group(1)))
        b['reason'] = 'manual override'
    m = BRIDGE_AB_RE.search(text)
    if m:
        b.update({
            'exact': True, 'canonical_seconds': float(m.group(1)),
            'bridge_seconds': float(m.group(2)), 'speedup': float(m.group(3)),
            'selected': bool(int(m.group(4))), 'reason': 'runtime differential/A-B',
        })
    if 'prepared-multiply lead bridge: WORD MISMATCH' in text:
        b.update({'exact': False, 'selected': False, 'reason': 'WORD MISMATCH; auto-reverted'})
    elif 'prepared-multiply lead bridge: auto-reverted' in text:
        b.update({'selected': False, 'reason': 'runtime exception/regression; auto-reverted'})
    if 'prepared-multiply lead bridge: ON ' in text:
        b['state'] = 'enabled'
        if b['selected'] is None:
            b['selected'] = True
    elif 'prepared-multiply lead bridge: OFF ' in text:
        b['state'] = 'rejected_or_disabled'
        if b['selected'] is None:
            b['selected'] = False

    d['kernel_profile'] = [
        {'name': m.group(1), 'calls': int(m.group(2)), 'exec_ns': int(m.group(3))}
        for m in PROFILE_RE.finditer(text)
    ]
    return d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(args, out, label, exponent, mode, workload, iterations, *, plan='', autotune='off', bridge=None, fused_ll=None, profile=False, cache=None):
    case_dir = out / label
    case_dir.mkdir(parents=True, exist_ok=True)
    residue = case_dir / 'residue.bin'
    log_path = case_dir / 'run.log'
    env = os.environ.copy()
    env['AEVUM_AUTOTUNE'] = autotune
    if cache:
        env['AEVUM_AUTOTUNE_CACHE'] = str(cache)
    else:
        env.pop('AEVUM_AUTOTUNE_CACHE', None)
    if bridge is None:
        env.pop('AEVUM_PREPARED_MUL_LEAD', None)
    else:
        env['AEVUM_PREPARED_MUL_LEAD'] = str(int(bridge))
    if fused_ll is None:
        env.pop('AEVUM_FUSED_LL', None)
    else:
        env['AEVUM_FUSED_LL'] = str(int(fused_ll))
    env['AEVUM_PROFILE_KERNELS'] = '1' if profile else '0'
    cmd = [str(args.bench), str(args.lib), str(args.device), str(exponent), plan,
           str(args.tune_dir), mode, str(iterations), str(residue), str(workload)]
    t0 = time.monotonic()
    cp = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.monotonic() - t0
    log_path.write_text(cp.stdout)
    if cp.returncode:
        raise RuntimeError(f'{label} failed rc={cp.returncode}; see {log_path}')
    result = parse_bench(cp.stdout)
    residue_hash = sha(residue)
    # Keep the validation artifact compact: exactness is represented by SHA-256
    # and structured diagnostics, not dozens of transform-sized residue files.
    residue.unlink(missing_ok=True)
    result.update(label=label, exponent=exponent, mode=mode, workload=workload,
                  plan_request=plan or 'plugin-auto', autotune=autotune,
                  bridge=bridge, fused_ll=fused_ll, wall_seconds=wall,
                  residue_sha256=residue_hash, diagnostics=parse_diagnostics(cp.stdout))
    return result


def median_cases(args, out, prefix, reps=3, **kwargs):
    rows=[]
    for i in range(reps):
        rows.append(run_case(args, out, f'{prefix}-r{i+1}', **kwargs))
    result = dict(rows[0])
    result['seconds'] = statistics.median(r['seconds'] for r in rows)
    result['wall_seconds'] = statistics.median(r['wall_seconds'] for r in rows)
    result['repetitions'] = reps
    result['sample_seconds'] = [r['seconds'] for r in rows]
    if len({r['residue_sha256'] for r in rows}) != 1:
        raise RuntimeError(f'{prefix}: nondeterministic residue across repetitions')
    return result, rows


def seed_plan(target, exponent):
    if 80000000 <= exponent < 170000000:
        return '1:512:8:512:101'
    if 170000000 <= exponent <= 197999999:
        return '4:512:8:512:202'
    if 198000000 <= exponent <= 230000000:
        return '1:512:16:512:202' if target == 'rtx3080' else '1:1K:8:512:202'
    return ''


def iters_for(p):
    return 64 if p < 30000000 else 28 if p < 120000000 else 18 if p < 170000000 else 12


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--lib', type=Path, required=True)
    ap.add_argument('--tune-dir', type=Path, required=True)
    ap.add_argument('--device', type=int, required=True)
    ap.add_argument('--target', choices=['rtx3080','radeonvii'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows=[]
    points=[21000013,100000007,150000007,180000007,197000003,210000007]

    # Plan tuner: native TRUE AUTO, supplied measured seed where applicable,
    # then first-use force-retune and benchmark-free cache hit.  Exact residue
    # hashes must agree or validation aborts.
    for p in points:
        iters=iters_for(p)
        native=run_case(args,args.out,f'p{p}-native-auto',p,'prp',1,iters,autotune='off')
        rows.append(native)
        seed=seed_plan(args.target,p)
        if seed:
            seeded=run_case(args,args.out,f'p{p}-measured-seed',p,'prp',1,iters,plan=seed,autotune='off')
            if seeded['residue_sha256'] != native['residue_sha256']:
                raise RuntimeError(f'p={p}: measured seed WORD MISMATCH')
            rows.append(seeded)
        cache=args.out/f'cache-p{p}.tsv'
        if cache.exists(): cache.unlink()
        miss=run_case(args,args.out,f'p{p}-runtime-miss',p,'prp',1,iters,autotune='retune',cache=cache)
        hit=run_case(args,args.out,f'p{p}-runtime-hit',p,'prp',1,iters,autotune='auto',cache=cache)
        if miss['residue_sha256'] != native['residue_sha256'] or hit['residue_sha256'] != native['residue_sha256']:
            raise RuntimeError(f'p={p}: runtime autotune WORD MISMATCH')
        hit_log=(args.out/f'p{p}-runtime-hit'/'run.log').read_text()
        if 'Aevum autotune: cache hit' not in hit_log:
            raise RuntimeError(f'p={p}: second use was not a cache hit')
        rows += [miss,hit]

    # Shared-engine A/B. Explicit engine gate + autotune off isolates the
    # implementation change. Three repetitions are used for medians.
    bridge_summary=[]
    bridge_runtime_cache=[]
    for workload,name in [(3,'pm1'),(6,'ecm')]:
        p=100000007
        base,_=median_cases(args,args.out,f'bridge-{name}-off', exponent=p,mode='mixed',workload=workload,
                            iterations=16,autotune='off',bridge=False)
        opt,_=median_cases(args,args.out,f'bridge-{name}-on', exponent=p,mode='mixed',workload=workload,
                           iterations=16,autotune='off',bridge=True)
        if base['residue_sha256'] != opt['residue_sha256']:
            raise RuntimeError(f'{name}: prepared-multiply bridge WORD MISMATCH')
        bridge_summary.append({'workload':name,'baseline_seconds':base['seconds'],'bridge_seconds':opt['seconds'],
                               'speedup':base['seconds']/opt['seconds'],'correctness':'PASS'})
        rows += [base,opt]

        # Exercise production AUTO for the implementation flag itself: first-use
        # retune performs the bridge differential/A-B, second-use must reuse the
        # cached plan + bridge decision without benchmarking.
        cache=args.out/f'cache-{name}-runtime.tsv'
        if cache.exists(): cache.unlink()
        miss=run_case(args,args.out,f'bridge-{name}-runtime-miss',p,'mixed',workload,16,autotune='retune',cache=cache)
        hit=run_case(args,args.out,f'bridge-{name}-runtime-hit',p,'mixed',workload,16,autotune='auto',cache=cache)
        if miss['residue_sha256'] != base['residue_sha256'] or hit['residue_sha256'] != base['residue_sha256']:
            raise RuntimeError(f'{name}: runtime bridge/cache WORD MISMATCH')
        hit_log=(args.out/f'bridge-{name}-runtime-hit'/'run.log').read_text()
        if 'Aevum autotune: cache hit' not in hit_log:
            raise RuntimeError(f'{name}: runtime bridge second use was not a cache hit')
        bridge_runtime_cache.append({
            'workload':name,
            'miss':miss['diagnostics'],
            'hit':hit['diagnostics'],
            'correctness':'PASS',
        })
        rows += [miss,hit]

    # Profiles show the changed shared hot path. They are outside timing and
    # retained in structured summary.json plus the individual compact logs.
    profile_off=run_case(args,args.out,'profile-bridge-off',100000007,'mixed',3,12,autotune='off',bridge=False,profile=True)
    profile_on=run_case(args,args.out,'profile-bridge-on',100000007,'mixed',3,12,autotune='off',bridge=True,profile=True)

    # LL regression: exact residue with and without the previously validated
    # FUSED_LL path, plus median timing retained in the artifact.
    ll_off,_=median_cases(args,args.out,'ll-fused-off', exponent=100000007,mode='ll',workload=2,
                          iterations=20,autotune='off',fused_ll=False)
    ll_on,_=median_cases(args,args.out,'ll-fused-on', exponent=100000007,mode='ll',workload=2,
                         iterations=20,autotune='off',fused_ll=True)
    if ll_off['residue_sha256'] != ll_on['residue_sha256']:
        raise RuntimeError('FUSED_LL regression WORD MISMATCH')
    rows += [ll_off,ll_on]

    gpu_identity = next((r['diagnostics']['device'] for r in rows if r['diagnostics']['device']), None)
    summary={
        'schema':'aevum-pass4-validation-v1',
        'target':args.target,'device':args.device,'gpu_identification':gpu_identity,'points':points,
        'plan_validation':[
            {
                'label':r['label'],'exponent':r['exponent'],'workload':r['workload'],
                'default_plan':r['diagnostics']['default_plan'],
                'tuned_plan':r['diagnostics']['autotune']['selected_plan'],
                'candidate_timings':r['diagnostics']['autotune']['candidates'],
                'correctness':'PASS','cache_decision':r['diagnostics']['autotune']['cache_decision'],
                'autotune':r['diagnostics']['autotune'],
                'prepared_width_bridge':r['diagnostics']['prepared_width_bridge'],
                'seconds':r['seconds'],'wall_seconds':r['wall_seconds'],
                'residue_sha256':r['residue_sha256'],
            }
            for r in rows if r['label'].startswith('p')
        ],
        'bridge_ab':bridge_summary,
        'bridge_runtime_cache':bridge_runtime_cache,
        'bridge_profiles':{
            'off':profile_off['diagnostics']['kernel_profile'],
            'on':profile_on['diagnostics']['kernel_profile'],
        },
        'fused_ll_ab':{'off_seconds':ll_off['seconds'],'on_seconds':ll_on['seconds'],
                       'speedup':ll_off['seconds']/ll_on['seconds'],'correctness':'PASS'},
        'cases':rows,
        'validation':'PASS',
        'gpu_performance_status':'MEASURED_ON_THIS_REAL_HARDWARE_RUN',
        'notes':[
            'Runtime tuner performs its own exact differential and 3-sample alternating medians before selecting a plan.',
            'Cache-hit runs require the cache-hit log marker and do not retune.',
            'PFA9 is never generated by the runtime candidate set.',
            'Prepared-width bridge speedup is reported only from this hardware run; the source package itself makes no unmeasured speedup claim.',
        ]
    }
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'validation':'PASS','summary':str(args.out/'summary.json')},indent=2))
if __name__=='__main__': main()
