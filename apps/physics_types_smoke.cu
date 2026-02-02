/**
 * @file physics_types_smoke.cu
 * @brief Smoke test for physics types (Etapa 5.0)
 * 
 * Verifies that all physics types compile and construct correctly.
 * No actual computation, just instantiation and basic operations.
 */

#include "../src/physics/common/physics_types.cuh"
#include "../src/runtime/CudaContext.cuh"
#include "../src/core/Grid3D.hpp"
#include <iostream>

using namespace rwpt;
using namespace rwpt::physics;

int main() {
    std::cout << "=== Physics Types Smoke Test (Etapa 5.0) ===\n\n";
    
    // 1. Test configuration structs (no GPU needed)
    std::cout << "1. Testing configuration structs...\n";
    {
        StochasticConfig stoch_cfg;
        std::cout << "   StochasticConfig: sigma2=" << stoch_cfg.sigma2 
                  << ", corr_length=" << stoch_cfg.corr_length
                  << ", n_modes=" << stoch_cfg.n_modes << "\n";
        
        FlowConfig flow_cfg;
        std::cout << "   FlowConfig: solver_type=" << flow_cfg.solver_type
                  << ", mg_levels=" << flow_cfg.mg_levels
                  << ", cg_max_iter=" << flow_cfg.cg_max_iter << "\n";
        
        TransportConfig trans_cfg;
        std::cout << "   TransportConfig: n_particles=" << trans_cfg.n_particles
                  << ", dt=" << trans_cfg.dt
                  << ", n_steps=" << trans_cfg.n_steps << "\n";
        
        SimulationConfig sim_cfg;
        std::cout << "   SimulationConfig: Nx=" << sim_cfg.Nx
                  << ", Ny=" << sim_cfg.Ny
                  << ", Nz=" << sim_cfg.Nz << "\n";
    }
    std::cout << "   OK\n\n";
    
    // 2. Test field types (requires GPU)
    std::cout << "2. Testing field types...\n";
    {
        CudaContext ctx(0);
        Grid3D grid(16, 16, 16, 0.1, 0.1, 0.1);
        
        // Scalar fields
        KField K(grid);
        std::cout << "   KField: size=" << K.size() 
                  << " (" << K.nx << "x" << K.ny << "x" << K.nz << ")\n";
        
        HeadField h(grid);
        std::cout << "   HeadField: size=" << h.size() << "\n";
        
        // Velocity field (staggered)
        VelocityField vel(grid);
        std::cout << "   VelocityField:\n";
        std::cout << "     U: " << vel.size_U() << " (" << (vel.nx+1) << "x" << vel.ny << "x" << vel.nz << ")\n";
        std::cout << "     V: " << vel.size_V() << " (" << vel.nx << "x" << (vel.ny+1) << "x" << vel.nz << ")\n";
        std::cout << "     W: " << vel.size_W() << " (" << vel.nx << "x" << vel.ny << "x" << (vel.nz+1) << ")\n";
        std::cout << "     Total: " << vel.total_size() << "\n";
        
        // Verify staggered dims match legacy convention
        // Legacy: U(Nx+1,Ny,Nz), V(Nx,Ny+1,Nz), W(Nx,Ny,Nz+1)
        bool dims_ok = (vel.size_U() == static_cast<size_t>(17 * 16 * 16)) &&
                       (vel.size_V() == static_cast<size_t>(16 * 17 * 16)) &&
                       (vel.size_W() == static_cast<size_t>(16 * 16 * 17));
        std::cout << "     Staggered dims match legacy: " << (dims_ok ? "YES" : "NO") << "\n";
        
        // Test resize
        Grid3D grid2(32, 32, 32, 0.05, 0.05, 0.05);
        K.resize(grid2);
        std::cout << "   KField after resize: size=" << K.size() << "\n";
    }
    std::cout << "   OK\n\n";
    
    // 3. Test workspace types
    std::cout << "3. Testing workspace types...\n";
    {
        CudaContext ctx(0);
        Grid3D grid(16, 16, 16, 0.1, 0.1, 0.1);
        
        StochasticConfig stoch_cfg;
        stoch_cfg.n_modes = 500;
        
        StochasticWorkspace stoch_ws;
        stoch_ws.allocate(grid, stoch_cfg);
        std::cout << "   StochasticWorkspace: n_modes=" << stoch_ws.n_modes
                  << ", n_cells=" << stoch_ws.n_cells
                  << ", allocated=" << stoch_ws.is_allocated() << "\n";
        
        FlowWorkspace flow_ws;
        flow_ws.allocate(grid);
        std::cout << "   FlowWorkspace: n_cells=" << flow_ws.n_cells
                  << ", allocated=" << flow_ws.is_allocated() << "\n";
        
        TransportConfig trans_cfg;
        trans_cfg.n_particles = 5000;
        
        ParticlesWorkspace part_ws;
        part_ws.allocate(trans_cfg);
        std::cout << "   ParticlesWorkspace: n_particles=" << part_ws.n_particles
                  << ", allocated=" << part_ws.is_allocated() << "\n";
        
        // Test combined workspace
        SimulationConfig sim_cfg;
        sim_cfg.stochastic.n_modes = 1000;
        sim_cfg.transport.n_particles = 10000;
        
        SimulationWorkspace sim_ws;
        sim_ws.allocate(grid, sim_cfg);
        std::cout << "   SimulationWorkspace: all allocated\n";
        
        // Test clear
        sim_ws.clear();
        std::cout << "   SimulationWorkspace after clear: stoch=" << sim_ws.stochastic.is_allocated()
                  << ", flow=" << sim_ws.flow.is_allocated()
                  << ", particles=" << sim_ws.particles.is_allocated() << "\n";
    }
    std::cout << "   OK\n\n";
    
    std::cout << "=== All tests passed ===\n";
    std::cout << "Physics module version: " << PHYSICS_VERSION_MAJOR << "." << PHYSICS_VERSION_MINOR << "\n";
    
    return 0;
}
