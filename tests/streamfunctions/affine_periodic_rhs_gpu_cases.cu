#include "reference_operators.hpp"
#include "streamfunction_operator_test_cases.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/physics/streamfunctions/affine_periodic_rhs.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {
namespace ref = macroflow3d::streamfunctions::reference;

constexpr double kCpuGpuTolerance = 1.0e-12;
constexpr double kConstantFactor = 100.0;
constexpr double kRawMeanFactor = 1000.0;
constexpr double kProjectedMeanFactor = 200.0;
constexpr double kOrderThreshold = 1.8;
const ref::Vec3 kG1{0.0, 1.0, 0.0};
const ref::Vec3 kG2{0.0, 0.0, 1.0};
const ref::Vec3 kOblique{0.7, 1.1, -0.9};

[[nodiscard]] std::string grid_description(const ref::Grid& g) {
    std::ostringstream out; out << g.nx << 'x' << g.ny << 'x' << g.nz << " h=" << g.spacing.x; return out.str();
}
[[nodiscard]] double rms(const std::vector<real>& v) {
    long double s = 0; for (real x : v) s += static_cast<long double>(x) * x;
    return std::sqrt(static_cast<double>(s / v.size()));
}
[[nodiscard]] double rms_double(const std::vector<double>& v) { return ref::rms_norm(v); }
[[nodiscard]] double rms_difference(const std::vector<real>& a, const std::vector<double>& b) {
    long double s = 0; for (std::size_t i=0;i<a.size();++i) { const long double d=static_cast<long double>(a[i])-b[i]; s+=d*d; }
    return std::sqrt(static_cast<double>(s/a.size()));
}
[[nodiscard]] double boundary_linf_difference(const ref::Grid& grid, const std::vector<real>& a,
                                              const std::vector<double>& b) {
    double maximum = 0.0;
    for (std::size_t z = 0; z < grid.nz; ++z) for (std::size_t y = 0; y < grid.ny; ++y) for (std::size_t x = 0; x < grid.nx; ++x) {
        if (x != 0 && x + 1 != grid.nx && y != 0 && y + 1 != grid.ny && z != 0 && z + 1 != grid.nz) continue;
        const auto i = grid.index(x, y, z);
        maximum = std::max(maximum, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    return maximum;
}
[[nodiscard]] double normalized(double error, double scale) { return error / std::max(scale, 1.0); }
[[nodiscard]] double eps() { return std::numeric_limits<real>::epsilon(); }

struct GpuRhsResult { std::vector<real> rhs1, rhs2; AffineRhsHostDiagnostics diagnostics; };
class Fixture {
  public:
    explicit Fixture(const ref::AffineRhsFixture& source)
        : grid_(static_cast<int>(source.grid.nx), static_cast<int>(source.grid.ny), static_cast<int>(source.grid.nz), source.grid.spacing.x, source.grid.spacing.y, source.grid.spacing.z),
          context_(0), q_(source.q.size()), rhs1_(source.q.size()), rhs2_(source.q.size()) { workspace_.prepare(source.q.size()); }
    [[nodiscard]] GpuRhsResult run(const std::vector<double>& q, const AffineGauge& gauge) {
        if (q.size()!=q_.size()) throw std::invalid_argument("fixture q size mismatch");
        std::vector<real> upload(q.begin(),q.end()); GpuRhsResult out; out.rhs1.resize(q.size()); out.rhs2.resize(q.size());
        // H2D -> assembly -> D2H, then the diagnostics helper performs the one
        // explicit synchronization that orders the whole stream.
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(q_.data(), upload.data(), q.size()*sizeof(real), cudaMemcpyHostToDevice, context_.cuda_stream()));
        const auto d = assemble_affine_periodic_rhs(context_, grid_, q_.span(), gauge, rhs1_.span(), rhs2_.span(), workspace_);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(out.rhs1.data(), rhs1_.data(), q.size()*sizeof(real), cudaMemcpyDeviceToHost, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(out.rhs2.data(), rhs2_.data(), q.size()*sizeof(real), cudaMemcpyDeviceToHost, context_.cuda_stream()));
        out.diagnostics = synchronize_affine_rhs_diagnostics(context_, d); return out;
    }
    [[nodiscard]] CudaContext& context() { return context_; }
    [[nodiscard]] Grid3D grid() const { return grid_; }
    [[nodiscard]] DeviceBuffer<real>& q() { return q_; }
    [[nodiscard]] DeviceBuffer<real>& rhs1() { return rhs1_; }
    [[nodiscard]] DeviceBuffer<real>& rhs2() { return rhs2_; }
    [[nodiscard]] AffinePeriodicRhsWorkspace& workspace() { return workspace_; }
  private:
    Grid3D grid_; CudaContext context_; DeviceBuffer<real> q_, rhs1_, rhs2_; AffinePeriodicRhsWorkspace workspace_;
};

[[nodiscard]] AffineGauge gauge(const ref::Vec3& a, const ref::Vec3& b) {
    return {{a.x,a.y,a.z},{b.x,b.y,b.z}};
}
[[nodiscard]] CaseResult case_affine_gauge_defaults() {
    const AffineGauge defaults{}; const auto b=AffineGauge::benchmark(real(2.75));
    const bool pass=defaults.psi1_gradient.x==0 && defaults.psi1_gradient.y==1 && defaults.psi1_gradient.z==0 && defaults.psi2_gradient.x==0 && defaults.psi2_gradient.y==0 && defaults.psi2_gradient.z==1 && b.psi1_gradient.y==real(2.75) && b.psi2_gradient.z==1;
    return {pass,"affine_gauge_defaults","sf06-types","no scalar affine storage",0,0,"n/a","n/a","default g1=(0,1,0), g2=(0,0,1); benchmark(vbar) separates periodic fluctuations"};
}
[[nodiscard]] CaseResult case_affine_rhs_cpu_oracle() {
    const auto c=ref::make_affine_rhs_fixture(16,true); const auto v=ref::make_affine_rhs_fixture(16);
    const auto zero=ref::affine_rhs_discrete(c.grid,c.q,kOblique); const auto coarse=ref::affine_rhs_discrete(v.grid,v.q,kOblique);
    const auto vf=ref::make_affine_rhs_fixture(32); const auto fine=ref::affine_rhs_discrete(vf.grid,vf.q,kOblique);
    const auto ec_exact=ref::affine_rhs_continuous(v.grid,kOblique); const auto ef_exact=ref::affine_rhs_continuous(vf.grid,kOblique);
    const double e0=ref::linf_norm(zero), ec=normalized(rms_difference(std::vector<real>(coarse.begin(),coarse.end()),ec_exact),rms_double(ec_exact)), ef=normalized(rms_difference(std::vector<real>(fine.begin(),fine.end()),ef_exact),rms_double(ef_exact));
    const auto o=ref::observed_order(ec,ef,v.grid.spacing.x,vf.grid.spacing.x);
    return {e0<=kConstantFactor*eps()*1.7 && o.valid()&&o.value>=kOrderThreshold,"affine_rhs_cpu_oracle","cpu-long-double",grid_description(v.grid)+"->"+grid_description(vf.grid),ec,ef,">=1.8",o.valid()?std::to_string(o.value):"n/a","constant Linf<=100 eps*1.7; independent discrete smooth order>=1.8"};
}
[[nodiscard]] CaseResult case_affine_rhs_constant() {
    const auto f=ref::make_affine_rhs_fixture(16,true); Fixture gpu(f); const auto r=gpu.run(f.q,gauge(kG1,kG2));
    const double scale=1.7, r1=rms(r.rhs1), r2=rms(r.rhs2);
    double linf = 0.0; for (real value : r.rhs1) linf = std::max(linf, std::abs(static_cast<double>(value))); for (real value : r.rhs2) linf = std::max(linf, std::abs(static_cast<double>(value)));
    const double raw=std::max(std::abs(static_cast<double>(r.diagnostics.raw_means[0])),std::abs(static_cast<double>(r.diagnostics.raw_means[1]))), proj=std::max(std::abs(static_cast<double>(r.diagnostics.projected_means[0])),std::abs(static_cast<double>(r.diagnostics.projected_means[1])));
    std::cout<<std::setprecision(16)<<"affine_rhs_metrics case=affine_rhs_constant raw_mean1="<<r.diagnostics.raw_means[0]<<" raw_mean2="<<r.diagnostics.raw_means[1]<<" projected_mean1="<<r.diagnostics.projected_means[0]<<" projected_mean2="<<r.diagnostics.projected_means[1]<<'\n';
    return {r1<=kConstantFactor*eps()*scale&&r2<=kConstantFactor*eps()*scale&&linf<=kConstantFactor*eps()*scale&&raw<=kRawMeanFactor*eps()&&proj<=kProjectedMeanFactor*eps(),"affine_rhs_constant","gpu-production",grid_description(f.grid),std::max(r1,r2),linf,"n/a","n/a","both RHS RMS/Linf<=100 eps*scale; raw<=1000 eps; projected<=200 eps"};
}
[[nodiscard]] CaseResult case_affine_rhs_gpu_oracle() {
    const auto f=ref::make_affine_rhs_fixture(16); Fixture gpu(f); const auto r=gpu.run(f.q,gauge(kG1,kOblique)); const auto a=ref::mean_zero_projected(ref::affine_rhs_discrete(f.grid,f.q,kG1)); const auto b=ref::mean_zero_projected(ref::affine_rhs_discrete(f.grid,f.q,kOblique));
    const double eg=std::max(normalized(rms_difference(r.rhs1,a),rms_double(a)),normalized(rms_difference(r.rhs2,b),rms_double(b))), eb=std::max(boundary_linf_difference(f.grid,r.rhs1,a),boundary_linf_difference(f.grid,r.rhs2,b))/std::max({rms_double(a),rms_double(b),1.0});
    return {eg<=kCpuGpuTolerance&&eb<=kCpuGpuTolerance,"affine_rhs_gpu_oracle","gpu-vs-independent-cpu",grid_description(f.grid),eg,eb,"n/a","n/a","global and boundary normalized <=1e-12"};
}
[[nodiscard]] CaseResult case_affine_rhs_smooth_order() {
    const auto c=ref::make_affine_rhs_fixture(16), f=ref::make_affine_rhs_fixture(32); Fixture cg(c), fg(f); double worst=0, worst_f=0, minorder=1e9;
    for (const ref::Vec3 g: {kG1,kG2,kOblique}) { const auto cr=cg.run(c.q,gauge(g,g)); const auto fr=fg.run(f.q,gauge(g,g)); const auto ce=normalized(rms_difference(cr.rhs1,ref::affine_rhs_continuous(c.grid,g)),rms_double(ref::affine_rhs_continuous(c.grid,g))); const auto fe=normalized(rms_difference(fr.rhs1,ref::affine_rhs_continuous(f.grid,g)),rms_double(ref::affine_rhs_continuous(f.grid,g))); const auto o=ref::observed_order(ce,fe,c.grid.spacing.x,f.grid.spacing.x); worst=std::max(worst,ce); worst_f=std::max(worst_f,fe); minorder=std::min(minorder,o.valid()?o.value:-1.0); }
    return {minorder>=kOrderThreshold,"affine_rhs_smooth_order","gpu-continuum",grid_description(c.grid)+"->"+grid_description(f.grid),worst,worst_f,">=1.8",std::to_string(minorder),"min order(g1,g2,oblique)>=1.8"};
}
[[nodiscard]] CaseResult case_affine_rhs_compatibility() {
    const auto f=ref::make_affine_rhs_fixture(16); Fixture gpu(f); const auto r=gpu.run(f.q,gauge(kG1,kG2)); const double s1=std::max(rms(r.rhs1),1.0),s2=std::max(rms(r.rhs2),1.0); const double raw=std::max(std::abs(static_cast<double>(r.diagnostics.raw_means[0]))/s1,std::abs(static_cast<double>(r.diagnostics.raw_means[1]))/s2), projected=std::max(std::abs(static_cast<double>(r.diagnostics.projected_means[0]))/s1,std::abs(static_cast<double>(r.diagnostics.projected_means[1]))/s2);
    std::cout<<std::setprecision(16)<<"affine_rhs_metrics case=affine_rhs_compatibility raw_mean1="<<r.diagnostics.raw_means[0]<<" raw_mean2="<<r.diagnostics.raw_means[1]<<" projected_mean1="<<r.diagnostics.projected_means[0]<<" projected_mean2="<<r.diagnostics.projected_means[1]<<" raw_normalized="<<raw<<" projected_normalized="<<projected<<'\n';
    return {raw<=kRawMeanFactor*eps()&&projected<=kProjectedMeanFactor*eps(),"affine_rhs_compatibility","gpu-diagnostics",grid_description(f.grid),raw,projected,"n/a","n/a","raw before P<=1000 eps; projected after P<=200 eps"};
}

template <class F> [[nodiscard]] bool rejects(F&& f) { try { f(); } catch (const std::exception&) { return true; } return false; }
[[nodiscard]] CaseResult case_affine_rhs_error_contract() {
    const auto f=ref::make_affine_rhs_fixture(16); Fixture gpu(f); const auto normal=gauge(kG1,kG2); const auto n=f.q.size();
    AffinePeriodicRhsWorkspace unprepared;
    const bool workspace=rejects([&]{ (void)assemble_affine_periodic_rhs(gpu.context(),gpu.grid(),gpu.q().span(),normal,gpu.rhs1().span(),gpu.rhs2().span(),unprepared); });
    const bool bad_grid=rejects([&]{ Grid3D x(1,16,16,1./16,1./16,1./16); (void)assemble_affine_periodic_rhs(gpu.context(),x,gpu.q().span(),normal,gpu.rhs1().span(),gpu.rhs2().span(),gpu.workspace()); }) && rejects([&]{ Grid3D x(16,16,16,1./16,1./15,1./16); (void)assemble_affine_periodic_rhs(gpu.context(),x,gpu.q().span(),normal,gpu.rhs1().span(),gpu.rhs2().span(),gpu.workspace()); });
    auto nan=normal; nan.psi1_gradient.x=std::numeric_limits<real>::quiet_NaN(); const bool nonfinite=rejects([&]{(void)assemble_affine_periodic_rhs(gpu.context(),gpu.grid(),gpu.q().span(),nan,gpu.rhs1().span(),gpu.rhs2().span(),gpu.workspace());});
    const bool sizes=rejects([&]{DeviceSpan<real> short_rhs(gpu.rhs1().data(),n-1); (void)assemble_affine_periodic_rhs(gpu.context(),gpu.grid(),gpu.q().span(),normal,short_rhs,gpu.rhs2().span(),gpu.workspace());});
    const bool aliases=rejects([&]{(void)assemble_affine_periodic_rhs(gpu.context(),gpu.grid(),gpu.q().span(),normal,gpu.q().span(),gpu.rhs2().span(),gpu.workspace());}) && rejects([&]{(void)assemble_affine_periodic_rhs(gpu.context(),gpu.grid(),gpu.q().span(),normal,gpu.rhs1().span(),gpu.rhs1().span(),gpu.workspace());});
    return {workspace&&bad_grid&&nonfinite&&sizes&&aliases,"affine_rhs_error_contract","gpu-api-contract",grid_description(f.grid),0,0,"n/a","n/a","reject unprepared, invalid/nonisotropic, nonfinite gauge, size and q/rhs/rhs alias"};
}

[[nodiscard]] std::vector<double> mutant_rhs(const ref::Grid& grid, const std::vector<double>& q, const ref::Vec3& g, int mutant) {
    std::vector<double> out(grid.cell_count());
    const bool arithmetic=mutant==1, inverse_hk=mutant==2, no_h=mutant==3, wrong_h=mutant==4, sign=mutant==0;
    for(std::size_t z=0;z<grid.nz;++z)for(std::size_t y=0;y<grid.ny;++y)for(std::size_t x=0;x<grid.nx;++x){
        const auto c=grid.index(x,y,z); const auto id=[&](std::ptrdiff_t dx,std::ptrdiff_t dy,std::ptrdiff_t dz){return grid.index(ref::wrap_index(static_cast<std::ptrdiff_t>(x)+dx,grid.nx),ref::wrap_index(static_cast<std::ptrdiff_t>(y)+dy,grid.ny),ref::wrap_index(static_cast<std::ptrdiff_t>(z)+dz,grid.nz));};
        const auto face=[&](std::size_t n){ if(arithmetic) return 0.5*(q[c]+q[n]); if(inverse_hk) return (q[c]+q[n])/2.0; return 2*q[c]*q[n]/(q[c]+q[n]); };
        double v=g.x*(face(id(1,0,0))-face(id(-1,0,0)))+g.y*(face(id(0,1,0))-face(id(0,-1,0)))+g.z*(face(id(0,0,1))-face(id(0,0,-1)));
        const double divisor=no_h?1.0:(wrong_h?2.0*grid.spacing.x:grid.spacing.x); out[c]=(sign?-1:1)*v/divisor;
    } return out;
}
[[nodiscard]] double sawtooth_boundary_error(const ref::Grid& grid,const std::vector<double>& q,const ref::Vec3& g) {
    const auto correct=ref::affine_rhs_discrete(grid,q,g); long double s=0; std::size_t count=0;
    for(std::size_t z=0;z<grid.nz;++z)for(std::size_t y=0;y<grid.ny;++y)for(std::size_t x=0;x<grid.nx;++x){ if(y!=0&&y+1!=grid.ny)continue; const auto c=grid.index(x,y,z); const auto yp=grid.index(x,ref::wrap_index(static_cast<std::ptrdiff_t>(y)+1,grid.ny),z),ym=grid.index(x,ref::wrap_index(static_cast<std::ptrdiff_t>(y)-1,grid.ny),z); const double u= g.y*(static_cast<double>(y)+.5)*grid.spacing.y, up=g.y*(static_cast<double>((y+1)%grid.ny)+.5)*grid.spacing.y, um=g.y*(static_cast<double>((y+grid.ny-1)%grid.ny)+.5)*grid.spacing.y; const double hp=2*q[c]*q[yp]/(q[c]+q[yp]),hm=2*q[c]*q[ym]/(q[c]+q[ym]); const double bad=-(hp*(up-u)-hm*(u-um))/(grid.spacing.y*grid.spacing.y); const long double d=bad+correct[c]; s+=d*d;++count; } return std::sqrt(static_cast<double>(s/count))/std::max(rms_double(correct),1.0);
}
[[nodiscard]] CaseResult case_affine_rhs_mutation_sensitivity() {
    const auto f=ref::make_affine_rhs_fixture(16); const auto exact=ref::affine_rhs_discrete(f.grid,f.q,kOblique); const double scale=std::max(rms_double(exact),1.0); const double correct=normalized(rms_double(exact),scale);
    std::array<double,5> errors{}, mismatches{}; bool rejected=true; for(int i=0;i<5;++i){errors[i]=normalized(rms_double(mutant_rhs(f.grid,f.q,kOblique,i)),scale); mismatches[i]=normalized(rms_difference(std::vector<real>(exact.begin(),exact.end()),mutant_rhs(f.grid,f.q,kOblique,i)),scale); rejected=rejected&&mismatches[i]>kCpuGpuTolerance;}
    const double saw=sawtooth_boundary_error(f.grid,f.q,kG1); auto offset=exact; for(double& v:offset)v+=1e-2*scale; const double raw=std::abs(static_cast<double>(ref::long_double_mean(offset)))/scale; const auto hidden=ref::mean_zero_projected(offset); const auto projected_exact=ref::mean_zero_projected(exact); const double post=normalized(rms_difference(std::vector<real>(projected_exact.begin(),projected_exact.end()),hidden),scale);
    std::cout<<std::setprecision(12)<<"affine_rhs_mutants correct="<<correct<<" sign="<<errors[0]<<" arithmetic="<<errors[1]<<" inverse_hk="<<errors[2]<<" no_h="<<errors[3]<<" wrong_h="<<errors[4]<<" mismatch_sign="<<mismatches[0]<<" mismatch_arithmetic="<<mismatches[1]<<" mismatch_inverse_hk="<<mismatches[2]<<" mismatch_no_h="<<mismatches[3]<<" mismatch_wrong_h="<<mismatches[4]<<" sawtooth_boundary="<<saw<<" offset_raw="<<raw<<" offset_projected="<<post<<'\n';
    return {rejected&&saw>1e-2&&raw>kRawMeanFactor*eps()&&post<=kCpuGpuTolerance,"affine_rhs_mutation_sensitivity","negative-oracle",grid_description(f.grid),saw,raw,"n/a","n/a","all RHS mutants rejected; sawtooth boundary>1e-2; offset raw fails while P hides it"};
}
} // namespace

CaseRegistry affine_periodic_rhs_case_registry() {
    return {{"affine_gauge_defaults",case_affine_gauge_defaults},{"affine_rhs_cpu_oracle",case_affine_rhs_cpu_oracle},{"affine_rhs_constant",case_affine_rhs_constant},{"affine_rhs_gpu_oracle",case_affine_rhs_gpu_oracle},{"affine_rhs_smooth_order",case_affine_rhs_smooth_order},{"affine_rhs_compatibility",case_affine_rhs_compatibility},{"affine_rhs_error_contract",case_affine_rhs_error_contract},{"affine_rhs_mutation_sensitivity",case_affine_rhs_mutation_sensitivity}};
}
} // namespace macroflow3d::streamfunctions::test
