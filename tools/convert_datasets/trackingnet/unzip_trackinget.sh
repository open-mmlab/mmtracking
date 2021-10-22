#!/bin/bash

data_dir=$1
for chunk in $(ls "${data_dir}"); do
    # unzip chunk zip
    if [ ${chunk##*.} == "zip" ]; then
        chunk_name=${chunk%.zip}
        unzip_dir=$data_dir/$chunk_name
        if [ ! -d $unzip_dir ]; then
           mkdir $unzip_dir
        fi
        unzip -n $data_dir/$chunk -d  $unzip_dir

        # unzip zips in every chunk
        for zips in $(ls "${unzip_dir}/zips"); do
            if [ ${zips##*.} == "zip" ]; then
                vid_name=${zips%.zip}
                if [ ! -d $unzip_dir/frames/$vid_name ]; then
                    mkdir -p $unzip_dir/frames/$vid_name
                fi
                unzip -n $unzip_dir/zips/$zips -d $unzip_dir/frames/$vid_name
            fi
        done
    fi
done
