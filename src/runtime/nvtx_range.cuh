#pragma once

// Optional NVTX range support
// If NVTX is not available, this becomes a no-op

#ifdef USE_NVTX
#include <nvToolsExt.h>

namespace macroflow3d {

class NvtxRange {
  public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }

    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

// Token pasting helpers (guard against redefinition)
#ifndef MACROFLOW3D_CAT
#define MACROFLOW3D_CAT_INNER(a, b) a##b
#define MACROFLOW3D_CAT(a, b) MACROFLOW3D_CAT_INNER(a, b)
#endif

#define MACROFLOW3D_NVTX_RANGE(name)                                                               \
    ::macroflow3d::NvtxRange MACROFLOW3D_CAT(__nvtx_range__, __COUNTER__)(name)

} // namespace macroflow3d

#else

namespace macroflow3d {

// No-op implementation when NVTX is not available
class NvtxRange {
  public:
    explicit NvtxRange(const char*) {}
};

#define MACROFLOW3D_NVTX_RANGE(name)                                                               \
    do {                                                                                           \
    } while (0)

} // namespace macroflow3d

#endif // USE_NVTX
