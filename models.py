from transformers import GPT2Config
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2ForSequenceClassification, AdamW
from params import configs
from dataset import get_dataloader
import pytorch_lightning as pl


class GPT2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=configs.num_classes, pad_token_id=self.tokenizer.eos_token_id, return_dict=True)
        self.classifier = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, label=None):
        logits = self.model(input_ids, attention_mask=attention_mask)["logits"]
        out = self.classifier(logits)
        
        if label is not None:
            loss = self.loss_fn(out, label)
            return out, loss
        else:
            return out
        

class GPT2Classifier_reduced(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model(
                GPT2Config(
                    vocab_size=50257,
                    n_positions=30, # given that our max_seq_len is 30
                    n_embd=768,
                    n_layer=6,
                    n_head=6,
                    n_inner=1024,
                    activation_function="gelu_new",
                    resid_pdrop=0.1,
                    embd_pdrop=0.1,
                    attn_pdrop=0.1,
                    layer_norm_epsilon=1e-5,
                    initializer_range=0.02,
                    summary_type="cls_index",
                    summary_use_proj=True,
                    summary_activation=None,
                    summary_proj_to_labels=True,
                    summary_first_dropout=0.1,
                    scale_attn_weights=True,
                    use_cache=True,
                    bos_token_id=50256,
                    eos_token_id=50256,
                    scale_attn_by_inverse_layer_idx=False,
                    reorder_and_upcast_attn=False,
                    return_dict=True
                    )
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 128), 
            nn.ReLU(),
            nn.Linear(128, configs.num_classes),
            nn.Softmax(dim=-1)
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, label=None):
        logits = self.model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        cls_token_indices = torch.argmin(attention_mask, dim=1)-1
        logits = logits[range(configs.batch_size), cls_token_indices.tolist()]
        
        out = self.classifier(logits)
        
        if label is not None:
            loss = self.loss_fn(out, label)
            return out, loss
        else:
            return out
        
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader()
    test_model = GPT2Classifier_reduced()
    for batch in val_loader:
        input_ids, attention_mask, rater_id, label = batch
        out, loss = test_model(input_ids, attention_mask, label)
        print(loss)
        break