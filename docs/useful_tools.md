## Useful tools

We provide lots of useful tools under `tools/` directory.

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/dff_faster_rcnn_r101_dc5_1x_imagenetvid/latest.pth dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201230.pth
```

The final output filename will be `dff_faster_rcnn_r101_dc5_1x_imagenetvid_20201230-{hash id}.pth`.
