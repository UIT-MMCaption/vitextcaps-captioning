from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import argparse
import os
import pandas as pd 

def main():
    parser = argparse.ArgumentParser(description='Process image captioning.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file to save results.')
    args = parser.parse_args()

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        load_in_4bit=True)  
    model.to("cuda:0")

    image_files = [f for f in os.listdir(args.folder_path) if f.endswith(('.jpg', '.png'))]
    results = []

    for image_file in image_files:
        image_path = os.path.join(args.folder_path, image_file)
        image = Image.open(image_path)

        conversation = [
            {
              "role": "system",  
              "content": "Bạn là chuyên gia phân tích hình ảnh và nhận dạng văn bản. Nhiệm vụ là tạo chú thích chi tiết, ưu tiên diễn giải văn bản trong ảnh. Hãy xác định đối tượng, màu sắc, bố cục và quét toàn bộ văn bản để hiểu vai trò và mối liên hệ với hình ảnh. Kết hợp các yếu tố này để viết chú thích ngắn, độ dài 1 câu, rõ ràng và chính xác, nhấn mạnh ý nghĩa của văn bản. Nếu văn bản bị che khuất hoặc khó đọc, hãy ghi chú và phỏng đoán. Đảm bảo chú thích phản ánh đúng bối cảnh hình ảnh khi cần."
            },
            {
              "role": "user",  
              "content": "Vui lòng mô tả hình ảnh dưới đây."
            },
            {
              "role": "assistant",  
              "content": "Chú thích hình ảnh sẽ được tạo ra ở đây."
            }
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        result = processor.decode(output[0], skip_special_tokens=True)
        
        results.append({'image': image_file, 'caption': result})

    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
