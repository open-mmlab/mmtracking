PARTITION=$1
ROOT_DIR=$2

# MOT
CONFIG=configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py
WORK_DIR=faster-rcnn_r50_fpn_4e_mot17-half
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/reid/resnet50_b32x8_MOT17.py
WORK_DIR=resnet50_b32x8_MOT17
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &

# VID
CONFIG=configs/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid.py
WORK_DIR=dff_faster_rcnn_r50_dc5_1x_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/vid/fgfa/fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.py
WORK_DIR=fgfa_faster_rcnn_r50_dc5_1x_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py
WORK_DIR=selsa_faster_rcnn_r50_dc5_1x_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py
WORK_DIR=selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &

# SOT
CONFIG=configs/sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py
WORK_DIR=siamese_rpn_r50_1x_lasot
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options checkpoint_config.max_keep_ckpts=1 >/dev/null &
