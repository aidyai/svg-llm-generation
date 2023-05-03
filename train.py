import pickle

import huggingface_hub
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from torch import nn
import gc
import wandb
import pydiffvg
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import os
import torch
from torch.utils.data import IterableDataset
from datasets import Dataset
from sklearn.model_selection import train_test_split
import argparse

# def img_gen(svg_str):
#     canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_str)
#     _render = pydiffvg.RenderFunction.apply
#     scene_args = pydiffvg.RenderFunction.serialize_scene(
#         canvas_width, canvas_height, shapes, shape_groups)
#     img = _render(128,  # width
#                   128,  # height
#                   5,  # num_samples_x
#                   5,  # num_samples_y
#                   0,  # seed
#                   None,
#                   *scene_args)
#     return img
#
#
# first = img_gen("D:\\dloads\\svgicons\\svgicons\\processed\\_0_deleted_303484_apple1-logo.svg")
# pydiffvg.imwrite(first.cpu(), "./1.png", gamma=2.2)

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dataset_path", type=str)
parser.add_argument("-m", "--model_path", type=str)
parser.add_argument("-t", "--tokenizer_path", type=str)
parser.add_argument("-wkey", "--wandb_key", type=str)
parser.add_argument("-hkey", "--hugging_key", type=str)
parser.add_argument("-sr", "--save_repo", type=str)
args = parser.parse_args()

wandb.login(key=args.wandb_key)
wandb.init()

set_seed(17)
model_name = args.model_path
dataset_dir_name = args.dataset_path

model = AutoModelForCausalLM.from_pretrained(
    model_name
)
# config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=128, lora_dropout=0.1)
# model = get_peft_model(model, config)
model.to("cuda")

ds = []
pathlist = Path(dataset_dir_name).rglob('*.pkl')
for path in pathlist:
    svg = pickle.load(open(path, "rb"))
    ds.append(svg)
print(ds[0])

train, test = train_test_split(ds, test_size=0.1)
len(train)


def to_list(color):
    res = ["F"]
    if color == "None":
        return ["F", "_", "_", "_"]
    print(color)
    res.append(str(int(color[1:3], 16)))
    res.append(str(int(color[3:5], 16)))
    res.append(str(int(color[5:7], 16)))
    return res


class DatasetImpl(IterableDataset):
    def __init__(self, tokenizer, dataset, context_size=1024):
        self.cuted = []
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.context_size = context_size
        self.cut()

    def __iter__(self):
        for s in self.cuted:
            yield s

    def __len__(self):
        return len(self.cuted)

    def to_map(self, tokens):
        pad_size = (self.context_size - len(tokens))
        tokens = [self.tokenizer.pad_token_id] * pad_size + tokens
        return {
            "input_ids": tokens,
            "attention_mask": [0] * pad_size + [1] * (self.context_size - pad_size),
            "labels": tokens,
        }

    def cut(self):
        for s in self.dataset:
            paths = s["paths"]
            prompt = s["prompt"]
            if prompt is not None:
                prompt = prompt.replace("<!--", "")
                prompt = prompt.replace("-->", "").strip()
            else:
                continue
            tokens = []
            for k, path in enumerate(paths):
                tokens += tokenizer.encode(prompt + " fill" + str(k) + ":")
                tokens += [tokenizer.encode(i)[0] for i in to_list(path["fill"])]
                tokens += tokenizer.encode(" path" + str(k) + ":")
                tokens += [tokenizer.encode(i)[0] for i in path["path"]]
                tokens += tokenizer.encode(';')
            if len(tokens) > self.context_size:
                continue
            self.cuted.append(self.to_map(tokens))


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

train_ds = DatasetImpl(
    tokenizer,
    train)
test_ds = DatasetImpl(
    tokenizer,
    test)


class CustomTrainer(Trainer):
    header = '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ' \
             'xmlns:ev="http://www.w3.org/2001/xml-events" width="512" height="512">'
    footer = '</svg>'

    @staticmethod
    def to_fill(r, g, b):
        if "_" in [r, g, b]:
            return "none"

        def to_hex(n):
            a = str(hex(int(n)).split('x')[-1])
            return "0" * (2 - len(a)) + a

        return "#" + to_hex(r) + to_hex(g) + to_hex(b)

    @staticmethod
    def img_gen(svg_str):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_str_to_scene(svg_str)
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = _render(128,  # width
                      128,  # height
                      5,  # num_samples_x
                      5,  # num_samples_y
                      0,  # seed
                      None,
                      *scene_args)
        return img

    def detokenize(self, tokens, size=None):
        print("detokenize")
        untokened = [self.tokenizer.decode(i, skip_special_tokens=True) for i in tokens]
        print(untokened)
        if size is None:
            size = untokened.count(";")
        print(size)
        i = 0
        result = []
        prompt = None
        for k in range(size):
            while "fill" not in untokened[i]:
                if prompt is None:
                    prompt = "".join(untokened[:i])
                i += 1
            i += 3
            r, g, b = untokened[i], untokened[i + 1], untokened[i + 2]
            print(r, g, b)
            while "path" not in untokened[i]:
                i += 1
            i += 2
            path = []
            while ";" not in untokened[i]:
                path.append(untokened[i])
                i += 1
            print(path)
            d = " ".join(path)
            fill = self.to_fill(r, g, b)
            result.append('<path d="' + d + '" fill="' + fill + '"/>')
        return "\n".join([self.header] + result + [self.footer]), prompt

    def compute_loss(self, model, inputs, return_outputs=False):
        print("loss")
        filtered = [token.item() for token in inputs["labels"][0] if token != tokenizer.pad_token_id]
        print(filtered)
        input, prompt = self.detokenize(filtered)
        print(input, prompt)
        size = input.count("path")
        enc = tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(input_ids=enc["input_ids"],
                                      attention_mask=enc["attention_mask"],
                                      max_new_tokens=1024)
        print(outputs)
        try:
            expected = self.img_gen(input)
            print(expected)
            generated = self.img_gen(self.detokenize(outputs.detach().cpu().numpy(), size))
            print(generated)
            loss_fct = nn.MSELoss()
            print("pictured")
            return loss_fct(expected, generated)
        except:
            print("default")
            return model(**inputs)["loss"]


trainer = CustomTrainer(
    model=model, args=TrainingArguments(
        "bloom-1b7_lora",
        num_train_epochs=1,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        logging_steps=50,
        save_strategy="no"
    ),
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds
)
trainer.train()

huggingface_hub.login(token=args.hugging_key)
model.push_to_hub(args.save_repo)
