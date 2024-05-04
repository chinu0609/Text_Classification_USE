#Ganesha
import numpy as np
import tensorflow_hub as hub
from keras.models import load_model
#model = load_model("./text_classification_NN.keras")
class Text_Classification_USE():
    def __init__(self):
        use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.use = hub.load(use_url)
        self.text_classification_model = load_model("./model/text_classification_NN.keras")

    def predict(self,text:str):
        embeddings = self.use([text])
        prediction = self.text_classification_model(embeddings.numpy())
        prediction = np.array(prediction)
        output = np.argmax(prediction)
        return output 


if __name__ == "__main__":
    Ganu = Text_Classification_USE()
    o = Ganu.predict("This is positive")
    print(o)
    
