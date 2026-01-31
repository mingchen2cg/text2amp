import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from Bio import SeqIO
from tqdm import tqdm
import wandb  # 1. 引入 wandb

# 引入模型组件
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from models.Text2ProteinGenModel import Text2ProteinGenModel

# ================= 配置与初始化 =================
# 请确保这些路径是正确的
CHECKPOINT_PATH = "./weights/text2protein_model/text2protein_complete.pt"
SWISS_PROT_PATH = "./data/uniprot_sprot.fasta"
T5_PATH = "./weights/pinal-official-t5-large"
PROGEN_TOK_PATH = "./models/progen3_module/tokenizer.json"
SAVE_DIR = "./checkpoints"  # Checkpoint 保存目录

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 动态 Batch Size 设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 # 默认

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")
    # 简单的关键词匹配，涵盖 A100 和 H20
    if "A100" in gpu_name or "H20" in gpu_name:
        BATCH_SIZE = 48
        print(f"High-end GPU detected. Setting Batch Size to {BATCH_SIZE}")
    else:
        print(f"Standard GPU detected. Setting Batch Size to {BATCH_SIZE}")

# ================= WandB 初始化 =================
# 1. 初始化 wandb 项目
wandb.init(
    project="text2protein-training",
    config={
        "learning_rate": 5e-4,
        "batch_size": BATCH_SIZE,
        "epochs": 20,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "max_len": 1024
    }
)

# ================= Tokenizers & Model =================
print("Loading Tokenizers...")
text_tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
progen_tokenizer = Tokenizer.from_file(PROGEN_TOK_PATH)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

print(f"Loading Model from {CHECKPOINT_PATH}...")
model = Text2ProteinGenModel(CHECKPOINT_PATH)
model.to(device)

# 1. 监控模型梯度
wandb.watch(model, log="gradients", log_freq=100)

# ================= 冻结参数逻辑 =================
print("\n=== Freezing Parameters ===")

# 1. 冻结 T5 Encoder
for param in model.lm.parameters():
    param.requires_grad = False

# 2. 冻结 ProGen3 的原始参数，只训练 cross_attn
for name, param in model.plm.named_parameters():
    if "cross_attn" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 检查可训练参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")

# ================= Dataset 定义 =================
class SwissProtDataset(Dataset):
    def __init__(self, fasta_file, limit=None):
        self.data = []
        print(f"Loading {fasta_file}...")
        # 这里的 limit 可以用于调试，正式训练可设为 None
        for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            if limit and i >= limit: break
            self.data.append({"text": record.description, "protein": str(record.seq)})
            
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Text
        text_enc = text_tokenizer(item['text'], max_length=512, truncation=True, return_tensors='pt')
        # Protein (这里硬编码了截断长度，回答了您关于长度处理的问题)
        prot_ids = [BOS_ID] + progen_tokenizer.encode(item['protein']).ids[:1022] + [EOS_ID]
        
        return {
            "text_ids": text_enc.input_ids.squeeze(0),
            "text_mask": text_enc.attention_mask.squeeze(0),
            "protein_ids": torch.tensor(prot_ids, dtype=torch.long)
        }

def collate_fn(batch):
    text_ids = pad_sequence([b['text_ids'] for b in batch], batch_first=True, padding_value=text_tokenizer.pad_token_id)
    text_masks = pad_sequence([b['text_mask'] for b in batch], batch_first=True, padding_value=0)
    protein_ids = pad_sequence([b['protein_ids'] for b in batch], batch_first=True, padding_value=PAD_ID)
    return {"text_ids": text_ids, "text_masks": text_masks, "protein_ids": protein_ids, "labels": protein_ids}

# 实例化 DataLoader
dataset = SwissProtDataset(SWISS_PROT_PATH) 
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

# ================= 训练循环 =================
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
epochs = 20

model.train()
print("\n=== Starting Training ===")

for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    total_loss = 0
    
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch) # Forward pass
        loss = outputs['loss']
        
        loss.backward()
        
        # 获取梯度范数用于监控
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # WandB Log
        wandb.log({
            "epoch": epoch + 1,
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "avg_loss": total_loss / (step + 1)
        })
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss/len(dataloader)
    print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
    
    # 3. 保存 Checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

wandb.finish()