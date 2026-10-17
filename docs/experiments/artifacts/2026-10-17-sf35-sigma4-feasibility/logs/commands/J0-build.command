cmake --preset v100-release && cmake --build build/v100-release -j && ctest --test-dir build/v100-release -R inlet_slab --output-on-failure
