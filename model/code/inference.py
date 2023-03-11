from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import os
import subprocess



def model_fn(model_dir):
    model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2",
                                                       load_in_8bit=True, device_map="auto", cache_dir="/tmp/model_cache/")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
    
    print("Device map after loading")
    print(model.hf_device_map)
    
    sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8")#.split('\n')
    print(out_list)
    
    return model, tokenizer



def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)

    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, **data)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)