_base_ = '../deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'


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
# 官方 Deformable DETR：
# AdamW
# lr = 2e-4
# batch size = 32
#
# 但是现在是完整 COCO pretrained + 极小数据集，
# 不建议用官方从头训练时那么激进的 2e-4。
#
# 这里用 5e-5 做 fine-tune。
# backbone / sampling offsets / reference points
# 继续保持 0.1 倍学习率。
# =========================================================

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=1e-4
    ),
    clip_grad=dict(
        max_norm=0.1,
        norm_type=2
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(
                lr_mult=0.1
            ),
            'sampling_offsets': dict(
                lr_mult=0.1
            ),
            'reference_points': dict(
                lr_mult=0.1
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

# 官方：
# 50 epochs
# decay @ 40
#
# 等比例扩展：
# 300 epochs
# decay @ 240
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[240],
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
# COCO pretrained Deformable DETR-R50
# =========================================================

load_from = (
    'https://download.openmmlab.com/mmdetection/v3.0/'
    'deformable_detr/deformable-detr_r50_16xb2-50e_coco/'
    'deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth'
)


# =========================================================
# Output
# =========================================================

work_dir = './work_dirs/deformable_detr_r50'
