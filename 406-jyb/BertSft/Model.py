import torch.nn as nn
from transformers import BertModel
import torch
class languageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.bert=BertModel.from_pretrained(r"E:\python\pre_train_model\bert_file",return_dict=False)

        self.classify=nn.Linear(768,vocab_size)
        self.loss=nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,x,mask=None,y=None):

        if y is not None:
            x,_=self.bert(x,attention_mask=mask)
            y_pred=self.classify(x)

            return self.loss(y_pred.view(-1,y_pred.shape[-1]),y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred/145, dim=-1)