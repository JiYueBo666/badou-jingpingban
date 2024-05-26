from transformers import BertTokenizer
class config:
    def __init__(self):
        self.corpus_path=r"E:\badouFile\第十周\week10 文本生成问题\week10 文本生成问题\transformers-生成文章标题\sample_data.json"
        self.epoch=50
        self.lr=0.001
        self.batch_size=16
        self.max_len=70
        self.use_gpu=False
        self.test_size=0.1
        self.tokenizer=BertTokenizer.from_pretrained(r"E:\python\pre_train_model\bert_file\vocab.txt")