#!/bin/bash

data_dir=$1
if [ ! -d $data_dir/frames ]; then
    mkdir $data_dir/frames
fi
for zips in $(ls "${data_dir}/zips"); do
    if [ ${zips##*.} == "zip" ]; then
       vid_name=${zips%.zip}
       echo $vid_name
       if [ ! -d $data_dir/frames/$vid_name ]; then
           mkdir $data_dir/frames/$vid_name
       fi
       unzip $data_dir/zips/$zips -d $data_dir/frames/$vid_name
    fi
done
