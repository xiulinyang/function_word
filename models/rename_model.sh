#!/bin/bash
model_id="$1"
cd "$model_id"

i=1
for ckpt in $(find . -maxdepth 1 -type d -name "checkpoint-*" \
                | sed 's|./checkpoint-||' | sort -n); do
    old="checkpoint-$ckpt"
    new="epoch-$i"
    mv "$old" "$new"
    echo "$old → $new"
    i=$((i+1))
done

cd ..
