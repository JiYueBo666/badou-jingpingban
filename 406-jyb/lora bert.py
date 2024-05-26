import pandas as pd
import torch
from peft import LoraConfig,TaskType,get_peft_model
from main import load_data
import torch.nn as nn
from torch.nn import LSTM
from transformers import BertModel,BertTokenizer,BertForSequenceClassification
from torch.utils.data import Dataset,DataLoader
from collections import defaultdict
from collections import Counter
from utils import FocalLoss
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report
from sklearn.metrics import confusion_matrix as  cmx


class Bert_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=BertForSequenceClassification.from_pretrained(r"E:\python\pre_train_model\bert_file",num_labels=len(targets))
        self.softmax=nn.Softmax(dim=-1)
        self.focalloss=FocalLoss()
    def forward(self,inputs,label=None):
        output=self.bert(**inputs).logits
       # output=self.fc(self.dropout(output))
        if label is not None:
            return self.focalloss(output.view(-1,len(targets)),label)
        else:
            return torch.argmax(self.softmax(output),dim=1)


def train_bert():
    epoch = 3
    #使用这个封装模型无法微调
    model=Bert_model()

    #直接使用这个模型的话可以正常微调。
   # model=BertForSequenceClassification.from_pretrained(r"E:\python\pre_train_model\bert_file",num_labels=len(targets))


    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    focal_loss=FocalLoss()

    for e in range(epoch):
        for idx, batch in enumerate(dataloader):
            # 送入device
            inputs, label, binary_label = batch# inputs 是[batch,length], label 是[batch,num_labels]
            label = label.cuda()
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            model = model.cuda()
            optimizer.zero_grad()
            model.train()

            # logits=model(**inputs).logits
            # loss = focal_loss(logits,label)

            loss=model(inputs,label)#这里会报错，提示model没有config属性。

            print('epoch:', e, 'batch:', idx, 'loss:', loss.item())
            loss.backward()
            optimizer.step()
        #只保存新添加的参数
        save_params={
            k:v.to('cpu')
            for k,v in model.named_parameters() if v.requires_grad
        }
        torch.save(save_params, 'peft.pth')