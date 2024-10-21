import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import json 
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from themis_model import get_Themis
import re

import warnings
warnings.filterwarnings(action="ignore")

from datasets import get_dataset, Fakeddit_Dataset, Recovery_Dataset, XRay_Dataset, xray_load_annotations_file, recovery_load_annotations_file

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--name_llm", type=str)
    parser.add_argument("--name_img_embed", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--n_tokens", type=int, default=128)
    parser.add_argument("--set_params", type=bool, default=False)
    parser.add_argument("--save_preds", type=bool, default=False)

    
    name_llm = parser.parse_args().name_llm
    name_img_embed = parser.parse_args().name_img_embed
    batch_size = parser.parse_args().batch_size
    merge_tokens = parser.parse_args().merge_tokens
    if merge_tokens == 0:
        merge_tokens = None
    lora_alpha = parser.parse_args().lora_alpha
    lora_r = parser.parse_args().lora_r
    lora_dropout = parser.parse_args().lora_dropout
    use_lora = parser.parse_args().use_lora
    model_path = parser.parse_args().model_path
    n_tokens = parser.parse_args().n_tokens
    set_params = parser.parse_args().set_params
    save_preds = parser.parse_args().save_preds

    if set_params:
        p = model_path.split('\\')[-1].split('_')
        lora_alpha = int(p[2])
        lora_r = int(p[3])
        lora_dropout = float(p[4])
        use_lora = True if 'True' in p[5] else False 
    
    model_dir = ''
    for i in model_path.split('\\')[:-1]:
        model_dir += i + '\\'
   
    themis, tokenizer, processor = get_Themis(
        name_llm = name_llm,
        name_img_embed = name_img_embed,
        use_lora = use_lora,
        is_pythia = True if "pythia" in name_llm else False,
        lora_alpha = lora_alpha,
        lora_r = lora_r,
        lora_dropout = lora_dropout,
        merge_tokens = merge_tokens
    )
    themis.to("cuda")

    base_dir = "recovery"
    dataset_test = get_dataset(Recovery_Dataset, recovery_load_annotations_file, n_tokens, processor, tokenizer, 
                f"{base_dir}/synonim_aug\\test_aug_sy.csv",
                f"{base_dir}/synonim_aug\images", 500)
    
    
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))

    #extract the predictions for the test set
    themis.load_state_dict(torch.load(model_path,map_location='cpu'))
    preds = []
    accumulated_labels = []
    running_loss = 0
    paths = []
    loss = nn.BCELoss()
    with torch.no_grad():
        for images, labels, text, img_path in tqdm(dataloader_test):
            images = images.to("cuda")
            text = text.to("cuda")
            labels = labels.to("cuda")

            paths.append(img_path)
            
            outputs = themis(images, text)
            
            loss_test = loss(outputs.float(), labels.float().unsqueeze(1))
            running_loss += loss_test.item()
            preds.extend(outputs.cpu().detach().numpy())
            accumulated_labels.extend(labels.cpu().numpy())
            
        paths = [p for path in paths for p in path]
        total_loss = running_loss / len(dataloader_test)
        preds = [1 if i > 0.5 else 0 for i in preds]

        if save_preds:
            datas = []
            for label, pred, path in zip(accumulated_labels,preds, paths):
                data = {
                    'label':label,
                    'pred':pred,
                    'path':path
                }
                datas.append(data)

            name_out = model_dir + 'pred.txt'
            with open(name_out, "w") as output:
                output.write(str(datas))
            
        acc = accuracy_score(accumulated_labels, preds)
        prec0 = precision_score(accumulated_labels, preds, pos_label=0)
        rec0 = recall_score(accumulated_labels, preds,pos_label=0)
        f10 = f1_score(accumulated_labels, preds, pos_label=0)
        prec1 = precision_score(accumulated_labels, preds, pos_label=1)
        rec1 = recall_score(accumulated_labels, preds,pos_label=1)
        f11 = f1_score(accumulated_labels, preds, pos_label=1)
        macro_f1 = f1_score(accumulated_labels, preds, average='macro')
        conf_matr = confusion_matrix(accumulated_labels, preds)
        print(f"Test loss: {total_loss}")
        print(f"Accuracy: {acc}")
        print(f"Label: 0 || Precision: {prec0} || Recall: {rec0} || F1: {f10}")
        print(f"Label: 1 || Precision: {prec1} || Recall: {rec1} || F1: {f11}")
        print(f"Macro-F1: {macro_f1}")
        print(conf_matr)
   
        
        metrics= {
            "loss": total_loss,
            "accuracy": acc,
            "precision_0": prec0,
            "precision_1": prec1,
            "recall_0": rec0,
            "recall_1": rec1,
            "F1_0": f10,
            "F1_1": f11,
            "macro_F1": macro_f1
        }
        
        # save confusion matrix
        ax= plt.subplot()
        sns.heatmap(conf_matr, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['unreliable', 'reliable']); ax.yaxis.set_ticklabels(['unreliable', 'reliable'])
        plt.savefig(model_dir + 'conf_matr.png')
       
        # save json metrics
        name_out = model_dir + 'metrics.json'
        with open(name_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print("Done!")


