import os
from transformers import BertModel, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data1_train = pd.read_excel(os.path.join(BASE_DIR, 'dataset', '감성대화말뭉치Training.xlsx'))
data1_validation = pd.read_excel(os.path.join(BASE_DIR, 'dataset','감성대화말뭉치Validation.xlsx'))
data1 = pd.concat([data1_train, data1_validation])
data2 = pd.read_csv(os.path.join(BASE_DIR, 'dataset','대화음성데이터셋.csv'), encoding='cp949')

data1['사람문장1'] = data1['사람문장1'].fillna("")
data1['사람문장2'] = data1['사람문장2'].fillna("")
data1['사람문장3'] = data1['사람문장3'].fillna("")

# 감성 레이블을 정수로 변환 (0, 2, 3, 5)
data1.loc[(data1['감정_대분류'] == "불안"), '감정_대분류'] = '0' #불안 => 0
data1.loc[(data1['감정_대분류'] == "분노"), '감정_대분류'] = '2' #분노 => 2
data1.loc[(data1['감정_대분류'] == "슬픔"), '감정_대분류'] = '3' #슬픔 => 3
data1.loc[(data1['감정_대분류'] == "상처"), '감정_대분류'] = '3' #상처 => 3
data1.loc[(data1['감정_대분류'] == "기쁨"), '감정_대분류'] = '5' #기쁨 => 5

# 데이터셋에서 나누어져있는 '사람문장1, 사람문장2, 사람문장3'을 병합, 당황 감정 제거
data1 = data1.drop(data1[data1['감정_대분류'] == '당황'].index)
data1['사람문장'] = data1['사람문장1'] + data1['사람문장2'] + data1['사람문장3']

data1_list = []
for q, label in zip(data1['사람문장'], data1['감정_대분류']):
    data = []
    data.append(str(q))
    data.append(str(label))

    data1_list.append(data)

# 감성 레이블을 정수로 변환 (0~6)
data2.loc[(data2['상황'] == "fear"), '상황'] = '0'      #공포 => 0
data2.loc[(data2['상황'] == "surprise"), '상황'] = '1'   #놀람 => 1
data2.loc[(data2['상황'] == "angry"), '상황'] = '2'      #분노 => 2
data2.loc[(data2['상황'] == "sadness"), '상황'] = '3'    #슬픔 => 3
data2.loc[(data2['상황'] == "neutral"), '상황'] = '4'    #중립 => 4
data2.loc[(data2['상황'] == "happiness"), '상황'] = '5'  #행복 => 5
data2.loc[(data2['상황'] == "disgust"), '상황'] = '6'    #혐오 => 6

data2_list = []
for q, label in zip(data2['발화문'], data2['상황']) :
    data = []
    data.append(str(q))
    data.append(str(label))

    data2_list.append(data)

merge_data = data1_list + data2_list

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        self.dataset = dataset
        self.sent_idx = sent_idx
        self.label_idx = label_idx
        self.tokenizer = bert_tokenizer
        self.max_len = max_len

    def __getitem__(self, i):
        # 데이터에서 텍스트와 라벨 추출
        text = str(self.dataset[i][self.sent_idx])
        label = np.int32(self.dataset[i][self.label_idx])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        token_ids = encoding['input_ids'].squeeze(0)

        if 'token_type_ids' in encoding:
            segment_ids = encoding['token_type_ids'].squeeze(0)
        else:
            segment_ids = torch.zeros(self.max_len, dtype=torch.long)

        valid_length = torch.sum(encoding['attention_mask']).item()

        return token_ids, valid_length, segment_ids, label

    def __len__(self):
        return len(self.dataset)

# class BERTClassifier
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

train_set, test_set = train_test_split(merge_data, test_size=0.2, shuffle=True, random_state=20)

data_train = BERTDataset(train_set, 0, 1, tokenizer, max_len, True, False)
data_test = BERTDataset(test_set, 0, 1, tokenizer, max_len, True, False)
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

# 정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

