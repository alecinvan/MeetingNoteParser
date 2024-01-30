# AI Meeting note parser
# Author：Alec Li
# Date：2024-01-26
# Location: Richmond Hospital Canada

import os

os.system("sudo apt-get install xclip")


import gradio as gr
import nltk
import pyclip
import pytesseract
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
import openai


nltk.download('punkt')


OCR_TR_DESCRIPTION = '''
<div id="content_align">
  <span style="color:darkred;font-size:32px;font-weight:bold">  
    EPF壹平台-模多多会议记录总结神器 
  </span>
</div>

<div id="content_align">
  <span style="color:blue;font-size:16px;font-weight:bold">  
  会议记录拍照 -> 转文字 -> 翻译 -> 提炼会议纪要 -> 识别待办事项 -> 分配任务
</div>

<div id="content_align" style="margin-top: 10px;">
  作者: Dr.  Alec Li
</div>
'''


# Image path
img_dir = "./data"

# Get tesseract language list
choices = os.popen('tesseract --list-langs').read().split('\n')[1:-1]



# Translation model selection
def model_choice(src="en", trg="zh"):
    # https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
    # https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"  # Model name

    tokenizer = MarianTokenizer.from_pretrained(model_name)  # Tokenizer
    model = MarianMTModel.from_pretrained(model_name)  # model

    return tokenizer, model


# Convert tesseract language list to pytesseract language
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


#import pytesseract


# Set Tesseract executable path in Colab virtal environment
#pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Set up the Tesseract data directory
#os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"

#def ocr_tesseract(img, languages):
#    custom_config = f'--oem 3 --psm 6 -l {ocr_lang(languages)}'
#    ocr_str = pytesseract.image_to_string(img, config=custom_config)
#    return ocr_str

# ocr tesseract
def ocr_tesseract(img, languages):
    ocr_str = pytesseract.image_to_string(img, lang=ocr_lang(languages))
    return ocr_str


# Clear content
def clear_content():
    return None


# copy to clipboard
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


# 在 https://platform.openai.com/signup 注册并获取 API 密钥

openai.api_key = os.getenv('OPENAI_API_KEY')



def generate_summary(text_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
             {"role": "system", "content": "你是一个非常能干的办公助手. 请把会议记录再此总结成会议纪要，根据时间线详细识别出不同的会议主题，并进行总结。 请识别出待办事项，并根据时间点进行任务分配，并在最后总结出决策点给领导决策。"},
             {"role": "user", "content": text_input}
         ]
     )
    summary = response["choices"][0]["message"]["content"].strip()
    return summary


def main():

    with gr.Blocks(css='style.css') as ocr_tr:
        gr.Markdown(OCR_TR_DESCRIPTION)

        # -------------- OCR 文字提取 --------------
        with gr.Column():

            with gr.Row():
                gr.Markdown("### Step 01: 文本抽取")

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        inputs_img = gr.Image(image_mode="RGB", type="pil", label="image")
                    with gr.Row():
                        inputs_lang = gr.CheckboxGroup(choices=["chi_sim", "eng"],
                                                       type="value",
                                                       value=['eng'],
                                                       label='language')

                    with gr.Row():
                        clear_img_btn = gr.Button('清除')
                        ocr_btn = gr.Button(value='图片文本抽取', variant="primary")

                with gr.Column():
                    with gr.Row():
                        outputs_text = gr.Textbox(label="抽取的文本", lines=20)
                    with gr.Row():
                        inputs_transStyle = gr.Radio(choices=["zh-en", "en-zh"],
                                                     type="value",
                                                     value="zh-en",
                                                     label='翻译模式')
                    with gr.Row():
                        clear_text_btn = gr.Button('清除')
                        translate_btn = gr.Button(value='翻译', variant="primary")

            # Add a text box to display the generated summary
            with gr.Row():
                  outputs_summary_text = gr.Textbox(label="生成的摘要", lines=20)
            
           
            with gr.Row():
                  with gr.Row():
                       generate_summary_btn = gr.Button('生成摘要', variant="primary")
                  with gr.Row():
                       clear_summary_btn = gr.Button('清除摘要')

            with gr.Row():
                example_list = [["./data/test.png", ["eng"]], ["./data/test02.png", ["eng"]],
                                ["./data/test03.png", ["chi_sim"]]]
                gr.Examples(example_list, [inputs_img, inputs_lang], outputs_text, ocr_tesseract, cache_examples=False)

        # -------------- 翻译 --------------
        with gr.Column():

            with gr.Row():
                gr.Markdown("### Step 02: 翻译")

            with gr.Row():
                outputs_tr_text = gr.Textbox(label="Translate Content", lines=20)

            with gr.Row():
                cp_clear_btn = gr.Button(value='清除剪贴板')
                cp_btn = gr.Button(value='复制到剪贴板', variant="primary")

        # ---------------------- OCR Tesseract ----------------------
        ocr_btn.click(fn=ocr_tesseract, inputs=[inputs_img, inputs_lang], outputs=[
            outputs_text,])
        clear_img_btn.click(fn=clear_content, inputs=[], outputs=[inputs_img])


        # ---------------------- Summarization ----------------------
        # To update the click event of the button, use generate_summary directly
        generate_summary_btn.click(fn=generate_summary, inputs=[outputs_text],   
                outputs=[outputs_summary_text])
        clear_summary_btn.click(fn=clear_content, inputs=[], outputs=[outputs_summary_text])


        # ---------------------- Translate ----------------------
        translate_btn.click(fn=translate, inputs=[outputs_text, inputs_transStyle], outputs=[outputs_tr_text])
        clear_text_btn.click(fn=clear_content, inputs=[], outputs=[outputs_text])

        # ---------------------- Copy to clipboard ----------------------
        cp_btn.click(fn=cp_text, inputs=[outputs_tr_text], outputs=[])
        cp_clear_btn.click(fn=cp_clear, inputs=[], outputs=[])

    ocr_tr.launch(inbrowser=True)


if __name__ == '__main__':
    main()



