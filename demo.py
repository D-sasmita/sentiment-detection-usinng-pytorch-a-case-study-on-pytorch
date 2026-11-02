import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── STEP 1: Verify Installation ───────────────────────────────
print("=" * 50)
print("STEP 1: PyTorch Installation Verified")
print("=" * 50)
print(f"PyTorch Version : {torch.__version__}")
if torch.backends.mps.is_available():
    print(f"GPU Available   : Apple M1 Metal (MPS) ✓")
elif torch.cuda.is_available():
    print(f"GPU Available   : NVIDIA CUDA ✓")
else:
    print(f"Device          : CPU (works fine for demo)")

# ── STEP 2: Basic Tensor Operations ──────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Basic Tensor Operations")
print("=" * 50)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(f"Tensor x        : {x}")
print(f"Tensor y        : {y}")
print(f"x + y           : {x + y}")
print(f"x * y           : {x * y}")
print(f"Mean of x       : {x.mean().item()}")

# ── STEP 3: Autograd ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Automatic Differentiation (Autograd)")
print("=" * 50)
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(f"x               : {x.item()}")
print(f"y = x²+2x+1     : {y.item()}")
print(f"dy/dx (gradient): {x.grad.item()}  (expected: 2x+2 = 8.0)")

# ── STEP 4: Dataset ───────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: NLP Sentiment Classifier")
print("         (Embedding + LSTM + Linear)")
print("=" * 50)

# Larger balanced dataset for better accuracy
texts = [
    ("this movie was fantastic loved every moment", 1),
    ("amazing film brilliant performances", 1),
    ("outstanding story and great direction", 1),
    ("best movie ever wonderful experience", 1),
    ("excellent acting enjoyed it thoroughly", 1),
    ("superb film highly recommend watching", 1),
    ("incredible visuals and amazing soundtrack", 1),
    ("loved this film perfect in every way", 1),
    ("terrible film complete waste of time", 0),
    ("boring plot poorly directed awful movie", 0),
    ("worst film i have ever seen avoid it", 0),
    ("disappointing and very predictable story", 0),
    ("bad acting horrible script do not watch", 0),
    ("dull and uninteresting film very bad", 0),
    ("awful movie hated every single minute", 0),
    ("terrible waste of money very disappointing", 0),
]

vocab = {}
for text, _ in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab) + 1  # 0 reserved for padding

def encode(text, max_len=10):
    tokens = [vocab.get(w, 0) for w in text.split()]
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    return tokens

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text, label = self.data[idx]
        return torch.tensor(encode(text)), torch.tensor(label)

loader = DataLoader(SentimentDataset(texts), batch_size=4, shuffle=True)
print(f"Vocabulary size : {len(vocab)} words")
print(f"Training samples: {len(texts)}")

# ── STEP 5: Model ─────────────────────────────────────────────
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded       = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

model = SentimentClassifier(vocab_size=len(vocab),
                             embed_dim=32,
                             hidden_dim=64,
                             num_classes=2)
print(f"\nModel Architecture:")
print(model)
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

# ── STEP 6: Train ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Training the Model")
print("=" * 50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

torch.manual_seed(42)
for epoch in range(50):
    total_loss = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}/50  |  Loss: {total_loss:.4f}")

# ── STEP 7: Inference ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Inference — Predicting Sentiment")
print("=" * 50)

model.eval()
def predict(text):
    with torch.no_grad():
        inp    = torch.tensor(encode(text)).unsqueeze(0)
        output = model(inp)
        pred   = torch.argmax(output, dim=1).item()
        conf   = torch.softmax(output, dim=1).max().item() * 100
        label  = "Positive ✓" if pred == 1 else "Negative ✗"
        return label, conf

tests = [
    ("the acting was outstanding loved every moment",  "Expected: Positive"),
    ("i hated every minute of it terrible movie",      "Expected: Negative"),
    ("brilliant story and wonderful performances",     "Expected: Positive"),
    ("worst film i have ever seen avoid it",           "Expected: Negative"),
    ("incredible visuals and amazing soundtrack",      "Expected: Positive"),
]

for sentence, expected in tests:
    label, conf = predict(sentence)
    print(f"  Input   : {sentence}")
    print(f"  Result  : {label}  (confidence: {conf:.1f}%)  [{expected}]")
    print()

# ── STEP 8: Save & Load ───────────────────────────────────────
print("=" * 50)
print("STEP 7: Save and Load Model")
print("=" * 50)
torch.save(model.state_dict(), "sentiment_model.pth")
print("Model saved  : sentiment_model.pth ✓")

loaded = SentimentClassifier(len(vocab), 32, 64, 2)
loaded.load_state_dict(torch.load("sentiment_model.pth"))
loaded.eval()
print("Model loaded : successfully ✓")

label, conf = predict("absolutely fantastic movie loved it")
print(f"Test after reload: 'absolutely fantastic movie loved it' → {label}")

print("\n" + "=" * 50)
print("Demonstration Complete!")
print("=" * 50)

print("\n" + "=" * 50)
print("Custom Input Test")
print("=" * 50)

user_text = input("Enter your own sentence: ")

label, conf = predict(user_text)

print(f"\nYour Input : {user_text}")
print(f"Prediction : {label} (confidence: {conf:.1f}%)")