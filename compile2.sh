 #! /bin/bash
 nvcc -std=c++11 main_transport_JSON_input.cu compute_velocity_from_head_for_par2.cu -Iinclude -o run_flow2.out -lcublas -L. ./lib_flow.a 