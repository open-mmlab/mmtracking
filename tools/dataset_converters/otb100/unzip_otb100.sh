#!/bin/bash

data_dir=$1
if [ ! -d $data_dir/data ]; then
    mkdir $data_dir/data
fi
for zips in $(ls "${data_dir}/zips"); do
    if [ ${zips##*.} == "zip" ]; then
        vid_name=${zips%.zip}
        # echo $vid_name
        unzip -q $data_dir/zips/$zips -d $data_dir/data/

        # clean up unnecessary files
        for x in $(ls "$data_dir/data/$vid_name/img" -a); do
            if [ ! -d $x ] && [[ ${x#*.} != "jpg" ]]; then
                echo "delete $data_dir/data/$vid_name/img/$x"
                rm -f $data_dir/data/$vid_name/img/$x
            fi
        done
    fi
done
# clean up unnecessary folds
if [ -d "${data_dir}/data/__MACOSX" ]; then
    echo delete "${data_dir}/data/__MACOSX"
    rm -rf "${data_dir}/data/__MACOSX"
fi
