from pathlib import Path

root = Path(__file__).resolve().parents[1]
args = (root / "src/Args.cpp").read_text()
args_h = (root / "src/Args.h").read_text()
gpuid_cpp = (root / "src/gpuid.cpp").read_text()
gpuid_h = (root / "src/gpuid.h").read_text()

assert '-device <P>:<D>' in args
assert "s.find(':')" in args
assert 'parseStrictInt(s.substr(0, colon), platform)' in args
assert 'parseStrictInt(s.substr(colon + 1), platformDevice)' in args
assert 'platformDevicePlatform = platform' in args
assert 'platformDeviceIndex = platformDevice' in args
assert '-device expects <N> or <platform>:<device>' in args
assert 'device = getPosFromPlatformDevice(platformDevicePlatform, platformDeviceIndex)' in args
assert 'if (device < 0) { throw "-device: no such platform:device"; }' in args
assert 'getPlatformDeviceFromPos(i).c_str()' not in args

assert 'int platformDevicePlatform = -1;' in args_h
assert 'int platformDeviceIndex = -1;' in args_h

assert 'getGpusByPlatform()' in gpuid_cpp
assert 'CL_DEVICE_TYPE_GPU' in gpuid_cpp
assert 'int getPosFromPlatformDevice(int platform, int device)' in gpuid_cpp
assert 'std::find(all.begin(), all.end(), devices[device])' in gpuid_cpp
assert 'getPlatformDeviceFromPos' not in gpuid_cpp

assert 'int getPosFromPlatformDevice(int platform, int device);' in gpuid_h
assert 'getPlatformDeviceFromPos' not in gpuid_h

print("Aevum OpenCL platform device selector source test passed")
