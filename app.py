# OCR Translate v0.2
# 创建人：曾逸夫
# 创建时间：2022-07-19

import os

os.system("sudo apt-get install xclip")

import gradio as gr
import nltk
import pyclip
import pytesseract
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer

nltk.download('punkt')

OCR_TR_DESCRIPTION = '''# OCR Translate v0.2
<div id="content_align">OCR translation system based on Tesseract</div>'''

# 图片路径
img_dir = "./data"

# 获取tesseract语言列表
choices = os.popen('tesseract --list-langs').read().split('\n')[1:-1]


# 翻译模型选择
def model_choice(src="en", trg="zh"):
    # https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
    # https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"  # 模型名称

    tokenizer = MarianTokenizer.from_pretrained(model_name)  # 分词器
    model = MarianMTModel.from_pretrained(model_name)  # 模型

    return tokenizer, model


# tesseract语言列表转pytesseract语言
def ocr_lang(lang_list):
    lang_str = ""
    lang_len = len(lang_list)
    if lang_len == 1:
        return lang_list[0]
    else:
        for i in range(lang_len):
            lang_list.insert(lang_len - i, "+")

        lang_str = "".join(lang_list[:-1])
        return lang_str


# ocr tesseract
def ocr_tesseract(img, languages):
    ocr_str = pytesseract.image_to_string(img, lang=ocr_lang(languages))
    return ocr_str


# 清除
def clear_content():
    return None


# 复制到剪贴板
def cp_text(input_text):
    # sudo apt-get install xclip
    try:
        pyclip.copy(input_text)
    except Exception as e:
        print("sudo apt-get install xclip")
        print(e)


# 清除剪贴板
def cp_clear():
    pyclip.clear()


# 翻译
def translate(input_text, inputs_transStyle):
    # 参考：https://huggingface.co/docs/transformers/model_doc/marian
    if input_text is None or input_text == "":
        return "System prompt: There is no content to translate!"

    # 选择翻译模型
    trans_src, trans_trg = inputs_transStyle.split("-")[0], inputs_transStyle.split("-")[1]
    tokenizer, model = model_choice(trans_src, trans_trg)

    translate_text = ""
    input_text_list = input_text.split("\n\n")

    translate_text_list_tmp = []
    for i in range(len(input_text_list)):
        if input_text_list[i] != "":
            translate_text_list_tmp.append(input_text_list[i])

    for i in range(len(translate_text_list_tmp)):
        translated_sub = model.generate(
            **tokenizer(sent_tokenize(translate_text_list_tmp[i]), return_tensors="pt", truncation=True, padding=True))
        tgt_text_sub = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_sub]
        translate_text_sub = "".join(tgt_text_sub)
        translate_text = translate_text + "\n\n" + translate_text_sub

    return translate_text[2:]


def main():

    with gr.Blocks(css='style.css') as ocr_tr:
        gr.Markdown(OCR_TR_DESCRIPTION)

        # -------------- OCR 文字提取 --------------
        with gr.Box():

            with gr.Row():
                gr.Markdown("### Step 01: Text Extraction")

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        inputs_img = gr.Image(image_mode="RGB", source="upload", type="pil", label="image")
                    with gr.Row():
                        inputs_lang = gr.CheckboxGroup(choices=["chi_sim", "eng"],
                                                       type="value",
                                                       value=['eng'],
                                                       label='language')

                    with gr.Row():
                        clear_img_btn = gr.Button('Clear')
                        ocr_btn = gr.Button(value='OCR Extraction', variant="primary")

                with gr.Column():
                    with gr.Row():
                        outputs_text = gr.Textbox(label="Extract content", lines=20)
                    with gr.Row():
                        inputs_transStyle = gr.Radio(choices=["zh-en", "en-zh"],
                                                     type="value",
                                                     value="zh-en",
                                                     label='translation mode')
                    with gr.Row():
                        clear_text_btn = gr.Button('Clear')
                        translate_btn = gr.Button(value='Translate', variant="primary")

            with gr.Row():
                example_list = [["./data/test.png", ["eng"]], ["./data/test02.png", ["eng"]],
                                ["./data/test03.png", ["chi_sim"]]]
                gr.Examples(example_list, [inputs_img, inputs_lang], outputs_text, ocr_tesseract, cache_examples=False)

        # -------------- 翻译 --------------
        with gr.Box():

            with gr.Row():
                gr.Markdown("### Step 02: Translation")

            with gr.Row():
                outputs_tr_text = gr.Textbox(label="Translate Content", lines=20)

            with gr.Row():
                cp_clear_btn = gr.Button(value='Clear Clipboard')
                cp_btn = gr.Button(value='Copy to clipboard', variant="primary")

        # ---------------------- OCR Tesseract ----------------------
        ocr_btn.click(fn=ocr_tesseract, inputs=[inputs_img, inputs_lang], outputs=[
            outputs_text,])
        clear_img_btn.click(fn=clear_content, inputs=[], outputs=[inputs_img])

        # ---------------------- 翻译 ----------------------
        translate_btn.click(fn=translate, inputs=[outputs_text, inputs_transStyle], outputs=[outputs_tr_text])
        clear_text_btn.click(fn=clear_content, inputs=[], outputs=[outputs_text])

        # ---------------------- 复制到剪贴板 ----------------------
        cp_btn.click(fn=cp_text, inputs=[outputs_tr_text], outputs=[])
        cp_clear_btn.click(fn=cp_clear, inputs=[], outputs=[])

    ocr_tr.launch(inbrowser=True)


if __name__ == '__main__':
    main()
