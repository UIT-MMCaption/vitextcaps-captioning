# LaVy

## LaVy 

Clone LaVy repository
```
git clone https://github.com/baochi0212/LaVy
```
After cloning, copy code from the existed builder.py to LaVy/model/builder.py and the existed run_llava.py to LaVy/eval/run_llava.py

## Requirements
Make sure to install all the requirements

```
pip install flash-attn --no-build-isolation
pip isntall -r requirements.txt
```

## Usage

1. **Prepare your images**: Place all images you want to caption in a single folder. Supported formats are `.jpg`, and `.png`.

2. **Run the script**: Use the following command to run the script, replacing `<image_path>`, `<ocr_path>`, `<ocr>` and `<save_name>` with your desired values.


- `<image_path>`: Path to the folder containing images.
- `<ocr_path>`: Path to the folder containing ocr.
- `<save_name>`: Save name for the inference file.
- `<ocr>`: Include ocr tokens in your promp or not.

## Example

```
run_llava.py --image_path path_to_image_folder
             --ocr_path  path_to_ocr_folder
             --save_name output 
             --ocr True
```