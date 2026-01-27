#pragma once

// Optional NVTX range support
// If NVTX is not available, this becomes a no-op

#ifdef USE_NVTX
#include <nvToolsExt.h>

namespace rwpt {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    
    ~NvtxRange() {
        nvtxRangePop();
    }
    
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

#define RWPT_NVTX_RANGE_CONCAT_IMPL(a, b) a##b
#define RWPT_NVTX_RANGE_CONCAT(a, b) RWPT_NVTX_RANGE_CONCAT_IMPL(a, b)
#define RWPT_NVTX_RANGE(name) ::rwpt::NvtxRange RWPT_NVTX_RANGE_CONCAT(__nvtx_range__, __COUNTER__)(name)

} // namespace rwpt

#else

namespace rwpt {

// No-op implementation when NVTX is not available
class NvtxRange {
public:
    explicit NvtxRange(const char*) {}
};

#define RWPT_NVTX_RANGE(name) do {} while(0)

} // namespace rwpt

#endif // USE_NVTX
