#!/bin/bash
dir=$(find . -maxdepth 1 -type d -printf '%f\n')

for i in $dir; do
 echo $i
 zip -r ${i}.zip  ${i}
 rm -rf ${i}
done
