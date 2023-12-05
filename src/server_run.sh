#!/bin/bash
for i in {0..10}
do
    for j in {0..3}
    do
        python3 run_exp.py $i $j &
    done
done