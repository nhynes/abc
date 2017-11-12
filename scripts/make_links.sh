#!/bin/bash

for f in $(git ls-tree --full-tree -r --name-only HEAD | grep "\.py$")
do
    ln -Lrsf "../../$f" "$f"
done
