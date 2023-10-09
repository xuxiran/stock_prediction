import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tqdm
from sklearn.metrics import f1_score
import os
import json
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

# parameters
batch_size = 256
num_epochs = 50
lr = 2e-5
device_ids = 0
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2024)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(2024)

# Read all data

# The train_data is from the financial reports obtained through various channels
# Train_labels are from the stock market situation the next day
# The valid_data is from the financial reports obtained from competition organizer
# Valid_labels are from competition organizer

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    train_labels = pickle.load(f)

with open('val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)
    val_labels = pickle.load(f)

# Balance the positive and negative labels
num_0 = len([i for i in train_labels if i == 0])
num_1 = len([i for i in train_labels if i == 1])
weight_0 = 1.0
weight_1 = num_0 / num_1
weights = [weight_1 if label == 1 else weight_0 for label in train_labels]
trainsampler = WeightedRandomSampler(weights, len(weights), replacement=True)


# model: Bert-chinese
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./bert-base-chinese')
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False
model.to(device)
optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()


# To get Dataset and DataLoader
def preprocess_text(text, label=None, max_seq_length=64):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    data = {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
    }
    if label is not None:
        data['label'] = torch.tensor([label])

    return data


class TextDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][0:64]
        label = self.label[idx]
        return preprocess_text(text, label)


train_dataset = TextDataset(train_data,train_labels)
val_dataset = TextDataset(val_data,val_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=trainsampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Trainning and validating model
val_f1_max = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    cnt = 0
    for batch in tqdm.tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epochs}", leave=False):
        cnt = cnt + 1
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels[:,0]).sum().item()

    train_acc = correct / total
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_labels = []
    val_preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc=f"Validing epoch {epoch + 1}/{num_epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits


            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels[:,0]).sum().item()
            val_labels.extend(labels[:, 0].tolist())
            val_preds.extend(predicted.tolist())

    val_acc = correct / total
    val_loss /= len(val_loader)

    val_f1 = f1_score(val_labels, val_preds)
    print(
        f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}')


    if val_f1>val_f1_max:
        val_f1_max = val_f1
        output_dir = './model'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)

        tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        tokenizer.save_pretrained(output_dir)

        best_val_model = model.state_dict()
        torch.save(best_val_model, os.path.join(output_dir, 'pytorch_model.bin'))
        result = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
        }
        with open(os.path.join(output_dir, 'result.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


