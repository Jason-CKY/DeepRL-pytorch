import glob
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset
from torch.nn import functional as F

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=transforms.ToTensor()):
        self.paths = glob.glob(data_dir+"/*.png")
        self.transforms = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transforms(image)

        return image