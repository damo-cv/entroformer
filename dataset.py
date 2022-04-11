import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
import pickle


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, cache=True ):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames.sort()
        self.input_transform = input_transform
        self.cache = cache
        if cache :
            self.image_list = []
            for image_file in self.image_filenames:
                self.image_list.append(load_img(image_file))
            print('load image finished')

    def __getitem__(self, index):
        if not self.cache:
            input = load_img(self.image_filenames[index])
        else:
            input = self.image_list[index]
        if self.input_transform:
            input = self.input_transform(input)
        return input, self.image_filenames[index]

    def __len__(self):
        return len(self.image_list) if self.cache else len(self.image_filenames)


class DatasetFromList(data.Dataset):
    def __init__(self, list_path, input_transform=None, cache=True):
        super(DatasetFromList, self).__init__()
        
        with open(list_path) as f:
            self.image_filenames = []
            for line in f.readlines():
                self.image_filenames.append(line.strip('\n'))

        self.image_filenames = [x for x in self.image_filenames if is_image_file(x)]
        self.image_filenames.sort()

        self.input_transform = input_transform
        self.cache = cache
        if cache :
            self.image_list = []
            for image_file in self.image_filenames:
                self.image_list.append(load_img(image_file))
            print('load image finished')

    def __getitem__(self, index):
        if not self.cache:
            input = load_img(self.image_filenames[index])
        else:
            input = self.image_list[index]
        if self.input_transform:
            input = self.input_transform(input)
        return input, self.image_filenames[index]

    def __len__(self):
        return len(self.image_list) if self.cache else len(self.image_filenames)
