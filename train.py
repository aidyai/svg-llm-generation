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
#         canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_str)
#         _render = pydiffvg.RenderFunction.apply
#         scene_args = pydiffvg.RenderFunction.serialize_scene(
#             canvas_width, canvas_height, shapes, shape_groups)
#         img = _render(canvas_width,  # width
#                       canvas_height,  # height
#                       2,  # num_samples_x
#                       2,  # num_samples_y
#                       0,  # seed
#                       None,
#                       *scene_args)
#         return img
#
# first = img_gen("C:\\Users\\glebm\\Desktop\\fruits\\processed\\0orange-heart.svg")
# second = img_gen("C:\\Users\\glebm\\Desktop\\fruits\\processed\\0fruit-food-lemon.svg")
# pydiffvg.imwrite(first.cpu(), "./1.png", gamma=2.2)
# pydiffvg.imwrite(second.cpu(), "./2.png", gamma=2.2)
#
#
#
# print(nn.MSELoss()(first, second))

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dataset_path", type=str)
parser.add_argument("-key", "--wandb_key", type=str)
parser.add_argument("-sp", "--save_path", type=str)
args = parser.parse_args()

Path(args.save_path).mkdir(parents=True, exist_ok=True)

wandb.login(key=args.wandb_key)
wandb.init()

set_seed(17)
model_name = "bigscience/bloom-560m"
dataset_dir_name = args.dataset_path

model = AutoModelForCausalLM.from_pretrained(
    model_name
)
config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=128, lora_dropout=0.1)
model = get_peft_model(model, config)
# model.to("cuda")
model.print_trainable_parameters()

ds = []
pathlist = Path(dataset_dir_name).rglob('*.svg')
for path in pathlist:
    path_in_str = str(path)
    svg_string = open(path_in_str).read()
    ds.append(svg_string)
print(ds[0])

train, test = train_test_split(ds, test_size=0.1)
len(train)


class DatasetImpl(IterableDataset):
    def __init__(self, tokenizer, dataset, context_size=1024):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.context_size = context_size
        self.cut()

    def __iter__(self):
        for s in self.cuted:
            yield s

    def __len__(self):
        return len(self.cuted)

    def cut(self):
        self.cuted = []
        cut = []
        for s in self.dataset:
            if "path" not in s:
                continue
            prompt, code = s.split('\n')
            prompt_tok = self.tokenizer(prompt)
            code_tok = self.tokenizer(code)
            whole_tok = prompt_tok["input_ids"] + code_tok["input_ids"] + [self.tokenizer.eos_token_id]
            if len(whole_tok) > self.context_size:
                continue
            self.cuted.append({
                "input_ids": whole_tok + [self.tokenizer.pad_token_id] * (self.context_size - len(whole_tok)),
                "attention_mask": [1] * len(whole_tok) + [0] * (self.context_size - len(whole_tok)),
                "labels": whole_tok + [self.tokenizer.pad_token_id] * (self.context_size - len(whole_tok))
            })


tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer

train_ds = DatasetImpl(
    tokenizer,
    train)
test_ds = DatasetImpl(
    tokenizer,
    test)

class CustomTrainer(Trainer):

    def img_gen(self, svg_str):
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_str_to_scene(svg_str)
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = _render(canvas_width,  # width
                      canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      0,  # seed
                      None,
                      *scene_args)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model.generate(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 max_new_tokens=1024)
        gen_decoded = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        label_decoded = tokenizer.batch_decode(inputs["labels"], skip_special_tokens=True)[0]
        try:
            generated = self.img_gen(gen_decoded)
            expected = self.img_gen(label_decoded)
            loss_fct = nn.MSELoss()
            return loss_fct(expected, generated)
        except Exception:
            return model(**inputs)["loss"]



trainer = CustomTrainer (
    model=model, args=TrainingArguments(
        "bloom-1b7_lora",
        num_train_epochs=1,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        logging_steps=100,
        save_strategy="no"
    ),
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds
)
trainer.train()

model.save_pretrained(args.save_path)
