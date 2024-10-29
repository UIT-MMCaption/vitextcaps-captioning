import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from io import BytesIO
import re


def image_parser(image_file,
                 sep):
    out = image_file.split(sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(model,
               tokenizer,
               image_processor,
               context_len,
               query,
               image_file,
               conv_mode="mistral_instruct",
               sep=",",
               temperature=0.01,
               top_p=0.9,
               num_beams=1,
               max_new_tokens=2048
               ):

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(image_file, sep)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs




def inference(args):
    captions = {
        'image_id': [],
        'caption': []
    }

    # Model
    disable_torch_init()

    tokenizer, model, image_processor, context_len = load_pretrained_model(load_8bit=True)

    for img_name in tqdm(os.listdir(args.image_path)):
        img_file = os.path.join(args.image_path, img_name)
        ocr = os.path.join(args.ocr_path, img_name.split('.')[0] + '.npy')

        if ocr is True:
            texts = np.load(ocr, allow_pickle=True)[()]['texts']
            texts = ', '.join(texts)
            query = f"Bạn là chuyên gia mô tả hình ảnh và nhận dạng văn bản. Nhiệm vụ là tạo mô tả chi tiết. Hãy xác định đối tượng, màu sắc, bố cục và quét toàn bộ văn bản để hiểu vai trò và mối liên hệ với hình ảnh. Kết hợp các yếu tố này và sử dụng các văn bản có trong hình ảnh {texts} để viết mô tả ngắn, độ dài 1 câu, rõ ràng và chính xác, nhấn mạnh ý nghĩa của văn bản. Nếu văn bản bị che khuất hoặc khó đọc, hãy ghi chú và phỏng đoán. Đảm bảo mô tả phản ánh đúng bối cảnh hình ảnh khi cần. Lưu ý chỉ cần sử dụng một số văn bản có trong hình ảnh để tạo một câu mô tả hàm súc."
        else:
            query = "Bạn là chuyên gia phân tích hình ảnh và nhận dạng văn bản. Nhiệm vụ là tạo chú thích chi tiết, ưu tiên diễn giải văn bản trong ảnh. Hãy xác định đối tượng, màu sắc, bố cục và quét toàn bộ văn bản để hiểu vai trò và mối liên hệ với hình ảnh. Kết hợp các yếu tố này để viết chú thích ngắn, độ dài 1 câu, rõ ràng và chính xác, nhấn mạnh ý nghĩa của văn bản. Nếu văn bản bị che khuất hoặc khó đọc, hãy ghi chú và phỏng đoán. Đảm bảo chú thích phản ánh đúng bối cảnh hình ảnh khi cần."
        output = eval_model(
            model,
            tokenizer,
            image_processor,
            context_len,
            conv_mode="mistral_instruct",
            image_file=img_file,
            query=query
        )

        captions['image_id'].append(img_name)
        captions['caption'].append(output)

    inference_result = pd.DataFrame(captions)
    inference_result.to_csv(args.save_name, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--ocr_path', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--ocr', type=bool, required=True)
    args = parser.parse_args()

    eval_model(args)