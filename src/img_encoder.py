# -*- coding: UTF-8 -*-
'''
Image encoder.  Get visual embeddings of images.
From RSME (https://github.com/wangmengsd/RSME)
'''

import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import pickle
from pytorch_pretrained_vit import ViT
import os
import imagehash
from tqdm import tqdm
import numpy as np

class ImageEncoder():
    TARGET_IMG_SIZE = 224
    img_to_tensor = transforms.ToTensor()
    Normalizer = transforms.Normalize((0.5,), (0.5,))

    @staticmethod
    def get_embedding(self, filter_gate=True):
        pass

    # 特征提取
    def extract_feature(self, base_path, filter_gate=True):
        print("start extract")
        self.model.eval()
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        dict = {}


        ents = os.listdir(base_path)

        pbar = tqdm(total=len(ents))

        while len(ents) > 0:
            # print(len(ents))
            ents_50 = []
            ents_50_ok = []

            for i in range(5):
                if len(ents) > 0:
                    ent = ents.pop()
                    try:
                        ents_50.append(base_path + '/' + ent)
                        # ents_50.append(base_path + ent + '/' + os.listdir(base_path + ent + '/')[0])
                    except Exception as e:
                        print(e)
                        continue

            tensors = []
            for imgpath in ents_50:
                try:
                    img = Image.open(imgpath).convert('RGB').resize((384, 384))
                except Exception as e:
                    print(e)
                    continue
                img_tensor = self.img_to_tensor(img)
                img_tensor = self.Normalizer(img_tensor)
                if img_tensor.size()[0] == 3:
                    tensors.append(img_tensor)
                    ents_50_ok.append(imgpath)
                else:
                    print(imgpath)
                    print(img_tensor.shape)

            if len(tensors) > 0:
                tensor = torch.stack(tensors, 0)
                tensor = tensor.cuda()

            result = self.model(Variable(tensor))
            result_npy = result.data.cpu().numpy()
            for i in range(len(result_npy)):
                dict[ents_50_ok[i]] = result_npy[i]
            pbar.update(5)
        pbar.close()
        return dict


class VisionTransformer(ImageEncoder):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.model = ViT('B_16_imagenet1k', pretrained=True)

    def get_embedding(self, base_path, filter_gate=True):
        self.model.eval()
        self.model.cuda()
        self.d = self.extract_feature(base_path, filter_gate=filter_gate)
        return self.d

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


class Resnet50(ImageEncoder):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=True)

    def get_embedding(self, base_path, filter_gate=True):
        self.model.eval()
        self.model.cuda()
        self.d = self.extract_feature(base_path, filter_gate)
        return self.d

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


class VGG16(ImageEncoder):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)

    def get_embedding(self, base_path, filter_gate=True):
        self.model.eval()
        self.model.cuda()
        self.d = self.extract_feature(base_path, filter_gate)
        return self.d

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


class PHash(ImageEncoder):
    def __init__(self):
        super(PHash, self).__init__()
        self.model = imagehash

    def get_embedding(self, base_path, hash_size):
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        ents = os.listdir(base_path)
        dict = {}
        while len(ents) > 0:
            print(len(ents))
            ents_50 = []
            ents_50_ok = []
            for i in range(5):
                if len(ents) > 0:
                    ent = ents.pop()
                    try:
                        ents_50.append(base_path + ent + '/' + os.listdir(base_path + ent + '/')[0])
                    except Exception as e:
                        print(e)
                        continue

            result_npy = []
            for imgpath in ents_50:
                try:
                    image_hash = imagehash.phash(Image.open(imgpath), hash_size=hash_size)
                except Exception as e:
                    print(e)
                    continue

                result_npy.append(image_hash)
                ents_50_ok.append(imgpath)

            for i in range(len(result_npy)):
                dict[ents_50_ok[i]] = result_npy[i]

        self.d = dict
        return dict

    def save_embedding(self, output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


if __name__ == "__main__":

    model = VisionTransformer()
    base_path = '../img_data/WN9'
    img_vec = model.get_embedding(base_path, filter_gate=True)
    f = open('../data/WN9/ent_id', 'r')
    Lines = f.readlines()

    id2ent = {}
    img_array = []
    dim = 1000
    for l in Lines:
        ent, id = l.strip().split()
        id2ent[id] = ent
        if ent in img_vec.keys():
            print(id, ent)
            img_array.append(img_vec[ent])
        else:
            img_array.append(np.random.normal(size=(1000,)))

    output_file = '../data/WN9/img_feature.pickle'
    img_array = np.array(img_array)
    with open(output_file, 'wb') as out:
        pickle.dump(img_array, out)
