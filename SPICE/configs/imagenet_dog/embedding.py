model_name = "embedding"
weight = "./results/imagenet_dog/moco/checkpoint_2999.pth.tar"
model_type = "resnet34"
device = 0
num_cluster = 10
batch_size = 1000
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True

data_test = dict(
    type="imagenet_lmdb",
    lmdb_file='./datasets/imagenet_dog_lmdb',
    meta_info_file='./datasets/imagenet_dog_lmdb_meta_info.pkl',
    embedding=None,
    split='train',
    resize=(224, 224),
    shuffle=False,
    ims_per_batch=1,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
    trans1=dict(
        aug_type="test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2=dict(
        aug_type="test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),

)


model_sim = dict(
    type=model_type,
    num_classes=128,
    in_channels=3,
    in_size=224,
    batchnorm_track=True,
    test=False,
    feature_only=True,
    pretrained=weight,
    model_type="moco_sim_feature",
)


results = dict(
    output_dir="./results/imagenet_dog/{}".format(model_name),
)