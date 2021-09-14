from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from transformers import BartModel, BartForConditionalGeneration
import os

def create_bart():
    return BartForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), 'bart-base'))

def get_embeddings(model):
    if isinstance(model, BartEncoder):
        return model.embed_tokens
    elif isinstance(model, BartDecoder) or isinstance(model, BartModel):
        return model.get_input_embeddings()
    else:
        return model.model.get_input_embeddings()

def get_embedding_dims(model):
    return model.config.d_model
