import gradio as gr
from prediction_on_text import Text_Classification_USE
obj = Text_Classification_USE()



def text_predict(text:str):
    o = obj.predict(text)  
    if o == 0:
        return "Positive"
    elif o == 2:
        return "Negative"
    else:
        return "Neutral"
inter = gr.Interface(fn=text_predict,inputs="text",outputs="text")
inter.launch(share=True)
