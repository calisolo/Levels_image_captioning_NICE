import json

from torch.utils.data import Dataset
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
from loguru import logger


class CaptionDataset(Dataset):

    def __init__(self, caption_file, image_file):
        logger.info('loading data from:{} and {}'.format(caption_file, image_file))
        # 读取每个图片的内容
        image_id2content = {}
        with open(image_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                image_id, image_content = line.split('\t')
                image_id2content[image_id] = image_content

         #각 사진의 모든 캡션을 읽고 모든 훈련 데이터를 얻습니다.
        data_list = []
        with open(caption_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = json.loads(line)
                image_id = line['image_id']
                captions = line['text']
                encoder_prefix = line['encoder_prefix'] #added 
                for caption in captions:
                    data = {'caption': caption, 'image_base64': image_id2content[image_id], 'image_id': image_id, 'encoder_prefix': encoder_prefix}
                    data_list.append(data)

        logger.info('len of data:{}'.format(len(data_list)))
        self.data_list = data_list

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.patch_resize_transform = patch_resize_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        row = self.data_list[index]
        caption = row['caption'].strip()
        image_base64 = row['image_base64']
        image_id = row['image_id']
        encoder_prefix = row['encoder_prefix']#added
        # 이미지를 로드하고 전처리
        try:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
            patch_image = self.patch_resize_transform(image).unsqueeze(0)
        except Exception as e:
            # 图片加载失败
            logger.info('open image error, image_id: {}'.format(image_id))
            logger.info(e)
            patch_image = None

        data = {'patch_image': patch_image, 'caption': caption, 'encoder_prefix': encoder_prefix}
        return data
