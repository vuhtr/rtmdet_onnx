# Convert RTMDet-Ins to ONNX

## Installation

We need an conda environment with Python verison 3.8.

```bash
conda create -n rtmdet_onnx python=3.8 -y
conda activate rtmdet_onnx
bash scripts/install.sh
```

## Download weights

```bash
gdown 1aipuUOF5CwKuRhwdmwvjcAdza5heQBYj&confirm=t
unzip rtmdetins_weights.zip
rm rtmdetins_weights.zip
```

## Convert to ONNX

Change the inference input size at the first line of config files (default to 640).

```bash
CONFIG=weights/rtmdet-ins-s/rtmdet-ins_s_8xb32-300e_coco.py
WEIGHTS=weights/rtmdet-ins-s/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth
OUTDIR=./work_dirs/rtmdet-ins-s

python ./mmdeploy/tools/deploy.py \
    ./mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py \
    $CONFIG \
    $WEIGHTS \
    ./test_imgs/1.jpeg \
    --work-dir $OUTDIR \
    --device cpu
```

## Float32 to float16

```bash
OUTDIR=./work_dirs/rtmdet-ins-s

python convert_f16.py \
	--input_model $OUTDIR/end2end.onnx \
    --output_model $OUTDIR/end2end_f16.onnx \
    --infer_size 640
```
