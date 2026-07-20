import os
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, AutoTokenizer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setting parameters
max_len = 64
batch_size = 64

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

# 소프트 맥스 추가
def new_softmax(a) :
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 1)

def predict(sentence, chunk_size=77):
    # 텍스트를 평균 길이로 나눔
    chunks = [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]
    all_emotions = []

    for chunk in chunks:
        dataset = [[chunk, '0']]
        test = BERTDataset(dataset, 0, 1, tokenizer, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=0)
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)

            emotions = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]
            for i in out:
                logits = i.detach().cpu().numpy()
                probabilities = new_softmax(logits).tolist()
                emotion_probs = {
                    emotion: round(prob, 1)
                    for emotion, prob in zip(emotions, probabilities)
                }
                all_emotions.append((emotion_probs, len(chunk)))

    # 전체 감정을 종합
    final_emotion = combine_emotions(all_emotions)
    print(final_emotion)

def combine_emotions(emotions_list):
    combined = {}
    total_length = sum(length for _, length in emotions_list)

    for emotions, length in emotions_list:
        for emotion, prob in emotions.items():
            if emotion not in combined:
                combined[emotion] = 0
            combined[emotion] += prob * (length / total_length)  # 길이로 가중 평균

    return {emotion: round(prob, 2) for emotion, prob in combined.items()}

device = torch.device("cpu")
weights_path = os.path.join(BASE_DIR, 'weights','SentimentAnalysisKOBert_StateDict.pt')
weights = torch.load(weights_path, weights_only=True, map_location = 'cpu')
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(weights)

# 질문 반복하기. 0 입력시 종료
end = 1
while end == 1 :
    sentence = input("일기 쓰기 : ")
    if sentence == 0 :
        break
    predict(sentence)