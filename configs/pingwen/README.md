# Pingwen Experiments

Run each experiment from the repository root. The runner trains from the
COCO-pretrained checkpoint specified by the config, finds the best checkpoint
selected by `coco/bbox_mAP`, and then evaluates it on the test split.

Training outputs, test metrics, and test artifacts are kept together in the
same `work_dirs/<run-name>/` directory. The test console output is saved as
`test_metrics.log`, and the evaluated checkpoint path is saved as
`test_checkpoint.txt`.

```bash
conda activate mmdet
cd /home/tkz/code/github/mmdetection

python tools/run_experiment.py \
  configs/pingwen/detr_r50_300e.py \
  --work-dir work_dirs/detr_r50_pretrained \
  --device 1

python tools/run_experiment.py \
  configs/pingwen/deformable_detr_r50_300e.py \
  --work-dir work_dirs/deformable_detr_r50_pretrained \
  --device 1

python tools/run_experiment.py \
  configs/pingwen/faster_rcnn_r50_300e.py \
  --work-dir work_dirs/faster_rcnn_r50_pretrained \
  --device 1
```

`--device` sets `CUDA_VISIBLE_DEVICES` for both training and test evaluation.
This runner is intended for a single-process, single-GPU experiment. Use the
native distributed MMDetection launch workflow for multi-GPU training.
