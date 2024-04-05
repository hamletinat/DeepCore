from torchvision import transforms
from torchvision.datasets import ImageFolder
import os


def CUSTOM(data_path):
    channel = 3
    im_size = (256, 256)
    num_classes = 4
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    transform = transforms.Compose([transforms.Resize(im_size), transforms.ToTensor()])
    dst_train = ImageFolder(root=os.path.join(data_path, 'Training'), transform=transform)
    dst_test = ImageFolder(root=os.path.join(data_path, 'Testing'), transform=transform)
    
    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test