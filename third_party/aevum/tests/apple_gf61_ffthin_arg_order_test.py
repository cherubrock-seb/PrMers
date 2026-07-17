from pathlib import Path
root = Path(__file__).resolve().parents[1]
cl = (root / 'src/cl/ffthin.cl').read_text()
gpu = (root / 'src/Gpu.cpp').read_text()
assert 'fftHinGF61FftTwiddleApple(P(T2) data, u32 f, Trig smallTrig)' in cl
assert 'kfftHinGF61FftTwiddleApple.setFixedArgs(2, bufTrigH);' in gpu
assert 'kfftHinGF61FftTwiddleApple(*current, stage);' in gpu
assert 'fftHinGF61FftTwiddleApple(P(T2) data, Trig smallTrig, u32 f)' not in cl
print('PASS: Apple fftHinGF61 twiddle argument order matches Kernel fixed/dynamic calling convention')
