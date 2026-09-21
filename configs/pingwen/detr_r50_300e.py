_base_ = '../detr/detr_r50_8xb2-150e_coco.py'


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
    bbox_head=dict(
        num_classes=3
    )
)


# =========================================================
# Dataloader
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
# DETR 官方配置：
# Transformer lr = 1e-4
# Backbone lr = 1e-5
# AdamW
# clip_grad = 0.1
#
# 这里因为是完整 COCO pretrained fine-tune，
# 保留 DETR 官方优化器设置。
# =========================================================

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4
    ),
    clip_grad=dict(
        max_norm=0.1,
        norm_type=2
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(
                lr_mult=0.1,
                decay_mult=1.0
            )
        }
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

# 原始官方 DETR:
# 150 epoch
# epoch 100 decay
#
# 等比例：
# 300 epoch
# epoch 200 decay
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[200],
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
# COCO pretrained DETR-R50
# =========================================================

load_from = (
    'https://download.openmmlab.com/mmdetection/v3.0/'
    'detr/detr_r50_8xb2-150e_coco/'
    'detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
)


# =========================================================
# Output
# =========================================================

work_dir = './work_dirs/detr_r50'
