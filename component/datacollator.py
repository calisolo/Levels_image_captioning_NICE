from typing import Any, Dict, List
import torch


class CaptionCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        captions, patch_images, encoded_prefix,gt_index = [], [], [], []
        for data in features:
            # 如果图片预处理失败，则跳过该图片
            if data['patch_image'] is None:
                continue
            captions.append(data['caption'])
            gt_index.append(data['image_id'])
            patch_images.append(data['patch_image'])

            encoded_prefix.append(' what does the image describe?'+data['encoder_prefix'])
            
            #[' what does the image describe?']*len(captions)
        # 获得encoder的输入
        input_ids = self.tokenizer(
            encoded_prefix, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True
        ).input_ids
        patch_images = torch.concat(patch_images, dim=0)

        # 获得decoder的输入
        inputs = self.tokenizer(
            captions, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True
        )
        decoder_input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        inputs = {
            'input_ids': input_ids,
            'patch_images': patch_images,
            'decoder_input_ids': decoder_input_ids,
            'attention_mask': attention_mask,
            'return_loss': True,
            'gt':captions,
            'gt_index': gt_index
        }
        return inputs
