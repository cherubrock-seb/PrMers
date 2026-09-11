from pathlib import Path
import json

def oracle(record):
    p=record['p']; n=((p+31)//32)*4; raw=(Path(record['path'])/'residue.bin').read_bytes()
    values=[int.from_bytes(raw[i:i+n],'little') for i in range(0,len(raw),n)]
    if len(values)!=15: raise RuntimeError('bad snapshot count')
    m=(1<<p)-1; x=values[0]; expected=[x]
    x=(x*x-2)%m; expected.append(x)
    for _ in range(5): x=(x*x-2)%m
    expected.append(x)
    x=(x*x-4)%m; expected.append(x)
    x=(3*x*x)%m; expected.append(x)
    x=(2*x*x)%m; expected.append(x)
    x=(x*x-2)%m; expected.append(x)
    x=x*x%m; expected.extend([x,x])
    x=7*x%m; expected.append(x)
    x=2*x%m; expected.append(x)
    saved=x; x=3*x*x%m; expected.append(x)
    x=(x-saved)%m; expected.append(x)
    expected.extend([m-2,14])
    if values!=expected: raise RuntimeError(f'CPU bigint oracle mismatch: {record["path"]}')

def result_fields(text, mode):
    # Compare mathematical output only, using actual JSON records rather than timed progress.
    fields={'status','exponent','worktype','res64','res2048','residue-type','factors',
            'known-factors','factor','B1','B2','b1','b2','curves_tested','errors','mode'}
    results=[]
    for line in text.splitlines():
        try: obj=json.loads(line.strip())
        except (ValueError,TypeError): continue
        if isinstance(obj,dict) and 'status' in obj and ('exponent' in obj):
            if 'errors' in obj and any(obj['errors'].values()): raise RuntimeError('nonzero arithmetic error count')
            results.append({k:v for k,v in obj.items() if k in fields})
    if not results: raise RuntimeError(f'{mode}: missing completed result JSON; inspect run.log')
    if mode in ('prp','ll') and not any('res64' in r for r in results):
        raise RuntimeError('missing final residue')
    return results
