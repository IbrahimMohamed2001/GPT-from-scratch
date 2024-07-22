# some important imports
import torch
from model import GPT
import os

def save_checkpoint(model, optimizer, step, filename='checkpoint_step_{step}.pth'):
    os.makedirs('weights', exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join('weights', filename.format(step=step)))

def load_checkpoint(model, optimizer, folder='weights'):
    if not os.path.exists(folder):
        print("No checkpoints found.")
        return 0  # No checkpoint to load

    checkpoints = [f for f in os.listdir(folder) if f.startswith('checkpoint_step_') and f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found.")
        return 0  # No checkpoint to load

    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[2].split('.')[0]))
    checkpoint_path = os.path.join(folder, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path} at step {checkpoint['step']}")
    return checkpoint['step']

def tokenize(string):
    return [chars.index(c) for c in string]
def decode(idx):
    return "".join([chars[i] for i in idx])

def get_batch(batch_size, seq_len, split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(batch_size, seq_len, split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

with open("dataset.txt", "r", encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
data = torch.tensor(tokenize(text), dtype=torch.long)

# model, training, and validating configurations
vocab_size = len(chars)
batch_size = 64
seq_len = 400
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 512
n_heads = 8
n_layers = 6
dropout = 0.2
max_steps = 10000
learning_rate = 6e-5
eval_iters = 200
eval_interval = 1000

train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]

model = GPT(vocab_size, seq_len, n_embed, n_heads, n_layers, dropout).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"Trainable parameters: {trainable_params}")
print(f"Non-trainable parameters: {nontrainable_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_step = load_checkpoint(model, optimizer)
for step in range(start_step, max_steps):
    
    if (step + 1) % eval_interval == 0:
        save_checkpoint(model, optimizer, step + 1)
        losses = estimate_loss(model)
        print(f"step {step + 1} ,, train loss: {losses['train']:.4f} ,, val loss: {losses['val']:.4f}")
    
    xb, yb = get_batch(batch_size, seq_len, 'train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(context, 500)[0].tolist()))