from Model import languageModel
from Dataread import load_corpus
from Dataread import build_dataset
from config import config
import torch
import numpy as np
def train(model,train_data):

    optimizer=torch.optim.Adam(model.parameters(),lr=config.lr)

    train_data = build_dataset(tokenizer, train_data, config.max_len, config.batch_size)

    if config.use_gpu:
        model=model.cuda()

    print("开始训练")

    for e in range(config.epoch):
        model.train()

        watch_loss=[]

        for x,mask,y in train_data:
            if config.use_gpu:
                x,mask,y=x.cuda(),mask.cuda(),y.cuda()
            optimizer.zero_grad()

            loss=model(x,mask,y)

            loss.backward()
            optimizer.step()

            watch_loss.append(loss.item())

            print("=========\n第%d轮平均loss:%f" % (e + 1, np.mean(watch_loss)))
            print(generator("北京明年拟推工作日半价观看电影", model, tokenizer))
            print(generator("南京一合金厂锅炉发生爆炸", model, tokenizer))
        torch.save(model,"bert_model.pth")

def generator(sentence,model,tokenizer):
    model.eval()

    sentence_encode=tokenizer.encode(sentence)

    with torch.no_grad():
        while len(sentence_encode)<=128:
            x=torch.LongTensor([sentence_encode])
            if config.use_gpu:
                x=x.cuda()

            y=model(x)[0][-1]
            index=sampling_strategy(y)

            sentence_encode.append(index)
    return tokenizer.decode(sentence_encode)
import random
def sampling_strategy(prob_distribution):
    if random.random() > 0.7:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)




if __name__ == '__main__':
    config=config()
    train_data,test_data=load_corpus(config.corpus_path)

    model=languageModel(vocab_size=21128)
    model=torch.load("bert_model.pth")
    tokenizer=config.tokenizer

    train(model,train_data)
