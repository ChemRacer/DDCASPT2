#!/bin/bash
dirs=$(find . -mindepth 1 -maxdepth 1 -type d | sort)

for i in $dirs; do
	echo $i
	cd $i
	cd ../
done
