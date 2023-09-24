import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from transformers import GPT2Tokenizer
from params import configs


def clean_outliers(df):
    # some data points dont have label, or some labels are empty
    df = df.dropna(subset=configs.emotion_columns)
    df = df[(df[configs.emotion_columns].T != 0).any()]
    return df


class goEmotion_individuals(Dataset):
    def __init__(self, csv_file):
        self.data = clean_outliers(pd.read_csv(csv_file))
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=configs.max_len_of_sequence)

        input_ids = tokenized_text['input_ids'].squeeze()
        attention_mask = tokenized_text['attention_mask'].squeeze()

        rater_id = self.data.iloc[idx]['rater_id']

        emotions = [col for col in configs.emotion_columns if self.data.iloc[idx][col] == 1]

        label = torch.zeros(configs.num_classes)
        label[configs.emotion_mapping[emotions[0]]] = 1

        return input_ids, attention_mask, label

class goEmotion_overall(Dataset):
    def __init__(self, tsv_file):
        self.data = pd.read_csv(tsv_file, delimiter='\t')
        self.data = self.data[self.data.iloc[:, 1] != "27"]
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx][0]
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=configs.max_len_of_sequence)

        input_ids = tokenized_text['input_ids'].squeeze()
        attention_mask = tokenized_text['attention_mask'].squeeze()

        emotion_label = int(self.data.iloc[idx][1].split(",")[0])
        
        emotion_label = configs.emotion_mapping[configs.emotion_labels_27[emotion_label]]

        label = torch.zeros(configs.num_classes)
        label[emotion_label] = 1
        
        return input_ids, attention_mask, label




def get_dataloader():
    torch.manual_seed(configs.random_seed)
    
    dataset1 = goEmotion_individuals('./data/goemotions_1.csv')
    dataset2 = goEmotion_individuals('./data/goemotions_2.csv')
    dataset3 = goEmotion_individuals('./data/goemotions_3.csv')

    merged_dataset = ConcatDataset([dataset1, dataset2, dataset3])

    total_samples = len(merged_dataset)
    train_size = int(configs.train_ratio * total_samples)
    val_size = int(configs.val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, remaining_dataset = random_split(merged_dataset, [train_size, total_samples-train_size])
    val_dataset, test_dataset = random_split(remaining_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, num_workers=configs.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, num_workers=configs.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, num_workers=configs.num_workers, drop_last=True)
    
    print("train_size:", len(train_loader.dataset))
    print("val_size:", len(val_loader.dataset))
    print("test_size:", len(test_loader.dataset))

    return train_loader, val_loader, test_loader
    
def get_dataloader_majority_vote():
    torch.manual_seed(configs.random_seed)
    
    train_set = goEmotion_overall('./data/train.tsv')
    val_set = goEmotion_overall('./data/dev.tsv')
    test_set = goEmotion_overall('./data/test.tsv')


    train_loader = DataLoader(train_set, batch_size=configs.batch_size, num_workers=configs.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=configs.batch_size, num_workers=configs.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=configs.batch_size, num_workers=configs.num_workers, drop_last=True)
    
    print("train_size:", len(train_loader.dataset))
    print("val_size:", len(val_loader.dataset))
    print("test_size:", len(test_loader.dataset))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    
    train_loader = DataLoader(goEmotion_overall("/projects/nie/goEmotion/data/train.tsv"), batch_size=configs.batch_size, shuffle=True, drop_last=True)
    for i in train_loader:
        input_ids, attention_mask, emotion_label = i
        print(emotion_label.shape)
        break