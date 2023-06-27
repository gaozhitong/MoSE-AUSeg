from data.cityscapes_dataset import cityscapes_Dataloader
from utils.OTloss import OT_loss
import data.transformations as transformations
from data.cityscapes_labels import label_switches, switched_name2Id, name2trainId
import torchvision.transforms as transforms
from models.MoSE import MoSE
import numpy as np

experiment_name = 'run'

# Setup directories
log_root = './log'
run_root = './runs'
log_dir_name = 'cityscapes'
data_folder = '../data/cityscape_npy_5/'

# Setup data
data_loader = cityscapes_Dataloader
n_classes = 25
input_channels = 4
image_size =(128,128)
num_w = 12
label_range = 'all'

use_bbpred = True
eval_class_names = list(label_switches.keys()) + list(switched_name2Id.keys())
eval_class_ids = [name2trainId[n] for n in eval_class_names]

transform = {
    'train': transforms.Compose([
              transformations.RandomCrop(scales=np.linspace(.6, .99, 4)),
              transformations.Resize(imsize=image_size),
              transformations.RandomHorizontalFlip()
              ]),
    'val': transforms.Compose([transformations.Crop(imsize=image_size)]),
    'test': None

}

# loss function
loss_fn = OT_loss( cost_fn='ce', beta = 1, gamma0=1/32, G0=1.0, num_chunks=16)

# Setup Model
net = MoSE(input_channels=input_channels,
                num_classes=n_classes,
                num_filters=[32, 64, 128, 192, 192, 192, 192],
                latent_dim = 5,
                num_expert = 35,
                sample_per_mode=1,
                loss_fn = loss_fn,
                masked_pred = True,
                eval_class_ids = eval_class_ids,
                softmax = False,
                eval_sample_num = None,
                )

# Training settings
pretrained_model_full_path = ''

train_batch_size = 12
val_batch_size = 12
test_batch_size = 2
epochs = 1000

lr = 1e-3
scheduler_options = {'min_lr': 1e-5,
                     'patience':10,
                     'factor': 0.5}

gamma_scheduler_options = {'updated_values': [1],
                            'patience': 30}
S_scheduler_options = {'updated_values': [2],
                            'patience': 30}
G_scheduler_options = {'updated_values': [0.95, 0.925, 0.9],
                            'patience': 30}

logging_frequency = 2
validation_frequency = 2
