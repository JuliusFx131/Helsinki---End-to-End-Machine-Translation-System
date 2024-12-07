"""Kserve inference script."""

import argparse
import re
#import torch
from kserve import (InferOutput,InferRequest,InferResponse,Model,ModelServer,model_server,)
from kserve.utils.utils import generate_uuid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "/app/saved_model"
CHARS_TO_REMOVE_REGEX = r'[!"&\(\),-./:;=?+.\n\[\]«»]'
MODEL_KWARGS = {"max_new_tokens": 30,"temperature":0.01} #"do_sample":True,"top_k":20,"top_p":0.95,
def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting
    to lower case.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters based on regex
    text = re.sub(CHARS_TO_REMOVE_REGEX, ' ', text)
    # Replace multiple consecutive dots with a single dot
    text = re.sub(r'\.{2,}', '', text)
    # Replace typographic apostrophe with straight apostrophe
    text = text.replace("’", "'")
    # Remove em dashes or other dashes if needed
    text = text.replace('—', '')
    # Replace ellipses with a single dot or handle as needed
    text = text.replace('…', '')
    # Remove extra white spaces (convert multiple spaces to a single space)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class MyModel(Model):
    """Kserve inference implementation of model."""
    def __init__(self, name: str):
        """Initialise model."""
        super().__init__(name)
        self.name = name
        self.model = None
        self.tokenizer = None
        self.ready = False
        self.load()
    def load(self):
        """Reconstitute model from disk."""
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        self.ready = True
    async def preprocess(self, payload: InferRequest, *_args, **_kwargs) -> str:
        """Preprocess inference request."""
        # Get sentence from payload
        raw_data = payload.inputs[0].data[0]
        prepared_data = f"{clean_text(raw_data)}"
        return prepared_data
    async def predict(self, data: str, *_predict_args, **_kwargs) -> InferResponse:
        """Pass inference request to model to make prediction."""
        # Model prediction preprocessed sentence
        inference_input = self.tokenizer(data, return_tensors="pt").input_ids
        #with torch.no_grad():
            #output = self.model.generate(inference_input, **MODEL_KWARGS)
        output = self.model.generate(inference_input, **MODEL_KWARGS)
        #output = self.model.generate(inference_input) #NO KWARGS
        translation = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response_id = generate_uuid()
        infer_output = InferOutput(name="output-0", shape=[1], datatype="STR", data=[translation])
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response
parser = argparse.ArgumentParser(parents=[model_server.parser],conflict_handler='resolve')
parser.add_argument("--model_name", default="model", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(args.model_name)
    ModelServer().start([model])
##python:3.11.8-slim-buster
# python test_no_kserve.py
