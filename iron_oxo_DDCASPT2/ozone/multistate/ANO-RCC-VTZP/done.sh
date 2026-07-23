#!/bin/bash
donedirs=$(find . -name "DDCASPT2_e2_H_P_3_.csv")

for i in $donedirs; do
 dirnames=$(dirname "${i}")
 if [ -f "finished/${dirnames#./}.zip" ]; then
  echo "${dirnames#./} done"
 else
  echo "${dirnames#./} NOT done"
  zip -r "${dirnames#./}.zip" "${dirnames}/"
  mv "${dirnames#./}.zip" finished/
 fi

done
