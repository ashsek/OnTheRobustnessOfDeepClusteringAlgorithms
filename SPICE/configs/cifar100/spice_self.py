model_name = "spice_self"
pre_model = "./results/cifar100/moco_cls/checkpoint_0999.pth.tar"
embedding = "./results/cifar100/embedding_cls/feas_moco_512_l2.npy"
resume = "./results/cifar100/{}/checkpoint_last.pth.tar".format(model_name)
model_type = "clusterresnet"
# model_type = "resnet18_cifar"
num_head = 10
num_workers = 4
device_id = 0
num_train = 5
num_cluster = 20
batch_size = 10000
target_sub_batch_size = 1000
train_sub_batch_size = 128
batch_size_test = 500
num_trans_aug = 1
num_repeat = 8
fea_dim = 512
att_conv_dim = num_cluster
att_size = 7
center_ratio = 0.5
sim_center_ratio = 0.9
epochs = 100
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True

start_epoch = 0
print_freq = 1
test_freq = 1
eval_ent = False
eval_ent_weight = 0

data_train = dict(
    type="cifar100",
    root_folder="./datasets/cifar100",
    embedding=embedding,
    train=True,
    all=True,
    ims_per_batch=batch_size,
    shuffle=True,
    aspect_ratio_grouping=False,
    show=False,
    trans1=dict(
        aug_type="weak",
        crop_size=32,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),

    trans2=dict(
        aug_type="scan",
        crop_size=32,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=16,
                           random=True)
    ),
)

data_test = dict(
    type="cifar100",
    root_folder="./datasets/cifar100",
    embedding=embedding,
    train=True,
    all=True,
    shuffle=False,
    ims_per_batch=50,
    aspect_ratio_grouping=False,
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

model = dict(
    feature=dict(
        type=model_type,
        num_classes=num_cluster,
        in_channels=3,
        in_size=32,
        batchnorm_track=True,
        test=False,
        feature_only=True
    ),

    head=dict(type="sem_multi",
              multi_heads=[dict(classifier=dict(type="mlp", num_neurons=[fea_dim, fea_dim, num_cluster], last_activation="softmax"),
                                feature_conv=None,
                                num_cluster=num_cluster,
                                loss_weight=dict(loss_cls=1),
                                iter_start=epochs,
                                iter_up=epochs,
                                iter_down=epochs,
                                iter_end=epochs,
                                ratio_start=1.0,
                                ratio_end=1.0,
                                center_ratio=center_ratio,
                                )]*num_head,
              ),
    model_type="moco",
    pretrained=pre_model,
    freeze_conv=True,
)


solver = dict(
    type="adam",
    base_lr=0.005,
    bias_lr_factor=1,
    weight_decay=0,
    weight_decay_bias=0,
    target_sub_batch_size=target_sub_batch_size,
    batch_size=batch_size,
    train_sub_batch_size=train_sub_batch_size,
    num_repeat=num_repeat,
)

results = dict(
    output_dir="./results/cifar100/{}".format(model_name),
)
