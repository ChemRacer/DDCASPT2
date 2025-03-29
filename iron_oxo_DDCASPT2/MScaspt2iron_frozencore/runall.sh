#!/bin/bash
files=$(find . -name "GMJ_e2_A*" | sort -h)
for i in *; do
   if [[ ! -e "$i/$i.csv" && -d "$i" ]]; then
       echo "$i"
       echo "File does not exist."
       cd $i
       pymolcas --new --clean $i.input -oe $i.output
       cd ../
   fi
done

