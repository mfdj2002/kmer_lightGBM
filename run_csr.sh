#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 start_value end_value"
    exit 1
fi

start=$1
end=$2

for i in $(seq $start $end)
do
    python3 create_csr.py $i > csr${i}.log 2>&1
done