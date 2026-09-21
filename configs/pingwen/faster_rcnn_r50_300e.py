_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'


# =========================================================
# Dataset
# =========================================================

data_root = '/home/tkz/datasets/pingwen_coco/'

metainfo = {
    'classes': ('row', 'col', 'hole')
}


# =========================================================
# Model
# =========================================================

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=3
        )
    )
)


# =========================================================
# Dataloader
# 注意：
# COCO json 中的 file_name 已经包含 train2017/ 等目录
# 所以这里 img 必须为空字符串
# =========================================================

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img=''),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img=''),
        metainfo=metainfo
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img=''),
        metainfo=metainfo
    )
)


# =========================================================
# Evaluator
# =========================================================

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json'
)

test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test2017.json'
)


# =========================================================
# Optimizer
#
# Faster R-CNN 官方：
# batch size = 16
# lr = 0.02
#
# 当前：
# batch size = 2
#
# 线性缩放：
# 0.02 * 2 / 16 = 0.0025
# =========================================================

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.0025,
        momentum=0.9,
        weight_decay=0.0001
    )
)


# =========================================================
# Training schedule
# =========================================================

max_epochs = 300

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

param_scheduler = [
    # warmup
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),

    # 300 epoch schedule
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[200, 275],
        gamma=0.1
    )
]


# =========================================================
# Checkpoint
# =========================================================

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=3
    )
)


# =========================================================
# COCO pretrained Faster R-CNN
# =========================================================

load_from = (
    'https://download.openmmlab.com/mmdetection/v2.0/'
    'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/'
    'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
)


# =========================================================
# Output
# =========================================================

work_dir = './work_dirs/faster_rcnn_r50'
