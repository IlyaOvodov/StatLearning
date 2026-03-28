import torch
import torchvision

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

# Note that statistics are wrt to uin8 range, [0,255].
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

CIFAR_MEAN = tuple(x*255 for x in (0.4914, 0.4822, 0.4465))
CIFAR_STD = tuple(x*255 for x in (0.2023, 0.1994, 0.2010))

def create_cifar_loaders(batch_size, num_workers=6, device='cuda', use_amp=True):
    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device(device)), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=4),
                # Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(torch.device(device), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16 if use_amp else torch.float32),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        loaders[name] = Loader(f'./data/cifar_{name}.beton',
                                batch_size=batch_size,
                                num_workers=num_workers,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                        'label': label_pipeline})
    return loaders['train'], loaders['test']        

