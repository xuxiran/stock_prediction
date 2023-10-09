import PyPDF2
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import csv
# # s3://ai-competition/6yh55f7q/torch_experiment-pkg3.tar.gz
# # tar -cvzf torch_experiment-pkg4.tar.gz test_three

device = torch.device(f"cpu")
batch_size = 1
datadir = '../xfdata/'
modeldir = '../user_data/model'
result_dir = '../prediction_result/result.csv'

def readtext(pdfname):
    mystr = ''
    try:
        pdf_file = open(pdfname, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        for i in range(min(total_pages,3)):
            page_obj = pdf_reader.pages[i]
            page_text = page_obj.extract_text()
            page_text = page_text.replace('\n', '')
            page_text = page_text.replace(' ', '')
            mystr = mystr + page_text
        pdf_file.close()
    except:
        mystr = 'error'
    return mystr

all_data = [0] * 495
for i in range(1,496):
    filename = datadir + '{:04d}.pdf'.format(i)
    tmp = readtext(filename)
    all_data[i-1] = tmp
    print(i)
test_data = all_data


# Bert-chinese
tokenizer = BertTokenizer.from_pretrained(modeldir)
model = BertForSequenceClassification.from_pretrained(modeldir)
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



# Get data
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][0:64]
        return preprocess_text(text)

test_dataset = TextDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# model
model.to(device)
model.eval()
allres = torch.zeros((495,1))
cnt = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        allres[cnt] = predicted
        cnt = cnt + 1


filenames = [f'{i:04d}.pdf' for i in range(1, 496)]
allres_np = allres.numpy()
rows = [[filename, int(prediction)] for filename, prediction in zip(filenames, allres_np)]


with open(result_dir, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])
    writer.writerows(rows)






