#!/bin/bash

runfiles=$(find $(pwd) -name "newddcaspt2.ipynb")
topdir=$(pwd)
for i in $runfiles; do
	dirname=$(dirname $i)
	cd $dirname
	echo "Running $dirname"
  papermill $i $i
	cd ../
done
