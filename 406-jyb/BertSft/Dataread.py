import torch
from config import config
from torch.utils.data import DataLoader
config=config()
import json
def load_corpus(path):
    corpus=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=json.loads(line)
            corpus.append([line['title'],line['content']])

    test_size=config.test_size
    length=len(corpus)

    train_corpus=corpus[:int(length*(1-test_size))]
    test_corpus=corpus[int(length*(1-test_size)):]

    return train_corpus,test_corpus

def build_dataset(tokenizer,corpus,max_length,batch_size):
    dataset=[]
    for i,(prompt,answer) in enumerate(corpus):
        prompt_encode=tokenizer.encode(prompt,add_special_tokens=False)
        answer_encode=tokenizer.encode(answer,add_special_tokens=False)

        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]

        #构建sft的mask矩阵
        mask=creat_mask(len(prompt_encode),len(answer_encode))

        #padding，截断或者补全
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)



def creat_mask(prompt_length,answer_length):

    matrix_size=prompt_length+1+1+answer_length+1#两个sep token，一个cls token
    mask=torch.ones((matrix_size,matrix_size))

    #前prompt+cls+sep行，prompt部分为1，后面部分为0
    for i in range(prompt_length+2):
        mask[i,prompt_length+2:]=0
    for i in range(answer_length):
        mask[prompt_length+2+i,prompt_length+2+i+1:]=0
    return mask
def pad_mask(tensor,target_shape):
    height,width=tensor.shape
    target_height,target_width=target_shape

    #创建全0张量，形状为输出形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)

    #计算需要截断或者补充的区域

    h_start=0
    w_start=0
    h_end = min(height, target_height)
    w_end = min(width, target_width)

    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]

    return result
