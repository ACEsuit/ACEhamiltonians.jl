#!/bin/bash

for f in $@; do 
    echo julia --project=. ./test/extract_errors.jl $f
    julia --project=. ./test/extract_errors.jl $f &
done

wait
