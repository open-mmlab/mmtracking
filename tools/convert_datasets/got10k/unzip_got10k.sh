#!bin/bash

train_dir=$1/full_data/train_data
new_train_dir=$1/train
if [ ! -d $new_train_dir ]; then
    mkdir $new_train_dir
fi
cp $train_dir/list.txt $new_train_dir
for x in $(ls $train_dir); do
    if [ ${x##*.} == zip ]; then
        unzip $train_dir/$x -d $new_train_dir
    fi
done
test_zip=$1/full_data/test_data.zip
val_zip=$1/full_data/val_data.zip
unzip $test_zip -d $1
unzip $val_zip -d $1
