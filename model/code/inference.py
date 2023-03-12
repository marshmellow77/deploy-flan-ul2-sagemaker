from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import subprocess


def model_fn(model_dir):
    #load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2",
                                                       load_in_8bit=True, device_map="auto", cache_dir="/tmp/model_cache/")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
    
    # print 'nvidia-smi' output to see how much VRAM is being used by the model
    sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8")
    print(out_list)
    
    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    # load model and tokenizer and retrieve prompt
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)

    # tokenize prompt and use it (together with other generation parameters) to create the model response
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, **data)
    
    # return model output and skip special tokens (such as "<s>")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)