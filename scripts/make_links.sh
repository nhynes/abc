#!/bin/bash

for f in $(git ls-tree --full-tree -r --name-only HEAD | grep "\.py$")
do
    if [ "$1" == "--cp" ]; then
        rm "$f"
        cp "../../$f" "$f"
    else
        ln -Lrsf "../../$f" "$f"
    fi
done
