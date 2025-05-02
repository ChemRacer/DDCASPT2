current_dir=$(pwd | sed 's/\//\\\//g')
sed -i "s/\${pwd}/$current_dir/g" */*input

