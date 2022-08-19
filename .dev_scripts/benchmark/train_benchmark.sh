PARTITION=$1
ROOT_DIR=$2

# VID
CONFIG=configs/vid/dff/dff_faster_rcnn_r50_dc5_1x_imagenetvid.py
WORK_DIR=dff_faster_rcnn_r50_dc5_7e_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/vid/fgfa/fgfa_faster_rcnn_r50_dc5_7e_imagenetvid.py
WORK_DIR=fgfa_faster_rcnn_r50_dc5_1x_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/vid/selsa/selsa_faster_rcnn_r50_dc5_7e_imagenetvid.py
WORK_DIR=selsa_faster_rcnn_r50_dc5_1x_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py
WORK_DIR=selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

# MOT
CONFIG=configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py
WORK_DIR=bytetrack_yolox_x_crowdhuman_mot17-private-half
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py
WORK_DIR=qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

# VIS
CONFIG=configs/vis/masktrack_rcnn/masktrack-rcnn_mask-rcnn-resnet50-fpn_8x1bs-12e_youtubevis2019.py
WORK_DIR=masktrack-rcnn_mask-rcnn-resnet50-fpn_8x1bs-12e_youtubevis2019
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

# SOT
CONFIG=configs/sot/siamese_rpn/siamese-rpn_resnet50_8x28bs-20e_imagenetvid-imagenetdet-coco_test-lasot.py
WORK_DIR=siamese-rpn_resnet50_8x28bs-20e_imagenetvid-imagenetdet-coco_test-lasot
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${WORK_DIR} ${CONFIG} ${ROOT_DIR}/${WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null &

CONFIG=configs/sot/stark/stark-st1_resnet50_8x16bs-500e_got10k-lasot-trackingnet-coco_test-lasot.py
ST1_WORK_DIR=stark-st1_resnet50_8x16bs-500e_got10k-lasot-trackingnet-coco_test-lasot
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${ST1_WORK_DIR} ${CONFIG} ${ROOT_DIR}/${ST1_WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 >/dev/null

CONFIG=configs/sot/stark/stark-st2_resnet50_8x16bs-50e_got10k-lasot-trackingnet-coco_test-lasot.py
ST2_WORK_DIR=stark-st2_resnet50_8x16bs-50e_got10k-lasot-trackingnet-coco_test-lasot
echo ${CONFIG} &
./tools/slurm_train.sh ${PARTITION} ${ST2_WORK_DIR} ${CONFIG} ${ROOT_DIR}/${ST2_WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 load_from=${ROOT_DIR}/${ST1_WORK_DIR}/epoch_500.pth >/dev/null

# MOT
REID_CONFIG=configs/reid/resnet50_b32x8_MOT17.py
REID_WORK_DIR=resnet50_b32x8_MOT17
echo ${REID_CONFIG}
./tools/slurm_train.sh ${PARTITION} ${REID_WORK_DIR} ${REID_CONFIG} ${ROOT_DIR}/${REID_WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1

DET_CONFIG=configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py
DET_WORK_DIR=faster-rcnn_r50_fpn_4e_mot17-half
echo ${DET_CONFIG}
./tools/slurm_train.sh ${PARTITION} ${DET_WORK_DIR} ${DET_CONFIG} ${ROOT_DIR}/${DET_WORK_DIR} 8 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1

CONFIG=configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py
WORK_DIR=deepsort_faster-rcnn_fpn_4e_mot17-private-half
echo ${CONFIG}
./tools/slurm_test.sh ${PARTITION} ${WORK_DIR} ${CONFIG} 8 --work-dir ${ROOT_DIR}/${WORK_DIR} --cfg-options model.detector.init_cfg.checkpoint=${ROOT_DIR}/${DET_WORK_DIR}/epoch_4.pth

CONFIG=configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py
WORK_DIR=tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half
echo ${CONFIG}
./tools/slurm_test.sh ${PARTITION} ${WORK_DIR} ${CONFIG} 8 --work-dir ${ROOT_DIR}/${WORK_DIR} --cfg-options model.detector.init_cfg.checkpoint=${ROOT_DIR}/${DET_WORK_DIR}/epoch_4.pth model.reid.init_cfg.checkpoint=${ROOT_DIR}/${REID_WORK_DIR}/epoch_6.pth

# VIS
CONFIG=configs/vis/masktrack_rcnn/masktrack-rcnn_mask-rcnn-resnet50-fpn_8x1bs-12e_youtubevis2019.py
WORK_DIR=masktrack-rcnn_mask-rcnn-resnet50-fpn_8x1bs-12e_youtubevis2019
echo ${CONFIG}
./tools/slurm_test.sh ${PARTITION} ${WORK_DIR} ${CONFIG} 8 --cfg-options test_evaluator.outfile_prefix=${ROOT_DIR}/${WORK_DIR} --checkpoint ${ROOT_DIR}/${WORK_DIR}/epoch_12.pth
