# PrMers and Aevum attribution

PrMers is developed by cherubrock-seb and its PrMers-authored source files are released under the MIT License.

This source bundle also contains the optional Aevum engine in `third_party/aevum/`. Aevum is a modified GPLv3 derivative of GPUOwl/PRPLL:

- original GPUOwl author: Mihai Preda, https://github.com/preda/gpuowl
- imported upstream fork and NTT work: George Woltman, https://github.com/gwoltman/gpuowl
- imported commit: `294cc485ac8cf53c8b69144a3039832eda573849`
- register-engine design reference: Marin by Yves Gallot, https://github.com/galloty/marin
- Aevum API and PrMers integration: cherubrock-seb, https://github.com/cherubrock-seb

Aevum is not an official GPUOwl, PRPLL or Marin release. Its GPLv3 license, upstream README and modification history are preserved under `third_party/aevum/`.

For a clean public layout, publish Aevum separately at https://github.com/cherubrock-seb/aevum-engine and consume it from PrMers as an optional submodule or external shared library.
