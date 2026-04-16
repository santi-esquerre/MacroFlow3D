# sccache runbook

Build cache for MacroFlow3D using [sccache](https://github.com/mozilla/sccache).

## Status

Optional. The build works without it. When installed, CMake auto-detects it.

## How it works

`CMakeLists.txt` contains:

```cmake
find_program(SCCACHE_PROGRAM sccache)
if(SCCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
endif()
```

No action needed beyond installing sccache.

## Installation (no root required)

### Option A: cargo

```bash
cargo install sccache --locked
```

### Option B: conda

```bash
conda install -c conda-forge sccache
```

### Option C: prebuilt binary

Download from <https://github.com/mozilla/sccache/releases> and place in `$PATH`.

## Verify

```bash
# Check availability
which sccache

# Check CMake detects it
cmake --preset wsl-debug 2>&1 | grep -i sccache

# Check stats
sccache --show-stats
```

## Helper script

```bash
# Check status and set env vars
source scripts/setup-sccache.sh
```

## Local WSL

Install once. All subsequent `cmake --preset wsl-debug` invocations will use it automatically.

Cache directory defaults to `~/.cache/sccache/`. To customize:

```bash
export SCCACHE_DIR=~/.cache/sccache
export SCCACHE_CACHE_SIZE="10G"
```

## Remote V100

If sccache is installed on the remote server, it will be auto-detected there too. Install the same way. To verify:

```bash
ssh v100 'which sccache && sccache --show-stats'
```

## Disabling

To explicitly disable sccache for one build:

```bash
cmake -S . -B build/wsl-debug \
  -DCMAKE_CXX_COMPILER_LAUNCHER="" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="" \
  ...
```

## Troubleshooting

- **NVCC compatibility**: sccache supports CUDA. If you see caching failures, check `sccache --show-stats` for the "not cached" reason.
- **Stale cache**: `sccache --zero-stats` resets counters. `rm -rf ~/.cache/sccache` clears cache.
- **Build correctness**: sccache is a transparent compiler wrapper. If a build behaves differently with vs. without sccache, that's a sccache bug, not a project bug.
