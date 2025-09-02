#!/bin/bash
rm -rf ./output/log
mkdir -p ./output/log
nohup ./run_flow2.out 1 4 > ./output/log/var_4.log 2>&1 &
PID0=$!
nohup ./run_flow2.out 0 0.25 > ./output/log/var_0.25.log 2>&1 &
PID1=$!
wait $PID0
nohup ./run_flow2.out 1 1 > ./output/log/var_1.log 2>&1 &
wait $PID1
nohup ./run_flow2.out 0 2.25 > ./output/log/var_2.25.log 2>&1 &