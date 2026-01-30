import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# =====================================================================
# 1. KONFIGURACJA I STAŁE
# =====================================================================
BATCH_SIZE = 64  # Większy batch niż w LSTM, bo model jest lżejszy
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping
RANDOM_SEED = 42

# Ustawienie ziarna losowości dla powtarzalności wyników
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Wykrywanie urządzenia (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")


# =====================================================================
# 2. MODEL BAZOWY (DENSE / MLP)
# =====================================================================

class EEG_Dense_Classifier(nn.Module):
    """
    Prosty model bazowy typu MLP (Multi-Layer Perceptron).

    Architektura:
    Input (868) -> Dense(128) -> ReLU -> Dropout -> Dense(64) -> ReLU -> Dropout -> Dense(2)
    """

    def __init__(self, input_size=868, hidden_size_1=128, hidden_size_2=64, num_classes=2, dropout=0.5):
        super(EEG_Dense_Classifier, self).__init__()

        # Flatten nie jest konieczny, jeśli dane wchodzą jako (batch, 868),
        # ale dodajemy dla pewności, gdyby wchodziły jako (batch, 1, 868)
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            # Warstwa 1
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Warstwa 2
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Warstwa wyjściowa
            nn.Linear(hidden_size_2, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.layers(x)
        return out


# =====================================================================
# 3. NARZĘDZIA POMOCNICZE (DATASET, WCZYTYWANIE)
# =====================================================================

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(data_dir='processed_data'):
    print(f"\n[INFO] Wczytywanie danych z {data_dir}...")
    try:
        X = np.load(f"{data_dir}/X_processed.npy")
        y = np.load(f"{data_dir}/y_labels.npy")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        return X, y
    except FileNotFoundError:
        print(f"[BLĄD] Nie znaleziono plików w katalogu {data_dir}!")
        print("Upewnij się, że uruchomiłeś najpierw 'eeg_preprocessing_pipeline.py'.")
        exit(1)


def split_data(X, y):
    """
    Stratyfikowany podział:
    - 70% Train
    - 15% Validation
    - 15% Test
    """
    print("[INFO] Dzielenie danych na zbiory...")

    # 1. Wydziel zbiór testowy (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )

    # 2. Resztę (85%) podziel na trening (70% całości) i walidację (15% całości)
    # 0.15 / 0.85 ≈ 0.1765
    val_split_ratio = 0.15 / 0.85

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split_ratio, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# 4. PĘTLA TRENINGOWA
# =====================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("\n[START] Rozpoczęcie treningu modelu Dense...")

    for epoch in range(NUM_EPOCHS):
        # --- TRENING ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # --- WALIDACJA ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        # Zapisz historię
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoka {epoch + 1}/{NUM_EPOCHS} | "
              f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}")

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [STOP] Early stopping w epoce {epoch + 1} (brak poprawy od {PATIENCE} epok)")
                break

    # Przywróć najlepszy model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n[INFO] Przywrócono najlepszy model (Val Loss: {best_val_loss:.4f})")

    return history


# =====================================================================
# 5. EWALUACJA
# =====================================================================

def evaluate_model(model, test_loader):
    print("\n[EWALUACJA] Testowanie na zbiorze testowym...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            # ZMIANA: używamy .tolist() zamiast .numpy() żeby ominąć błąd wersji
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    # Metryki
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }

    print("\n" + "=" * 50)
    print("WYNIKI NA ZBIORZE TESTOWYM (Model Bazowy - Dense)")
    print("=" * 50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("-" * 30)
    # Tu też mała zmiana kosmetyczna przy wyświetlaniu
    print(f"Macierz pomyłek:\n{results['confusion_matrix']}")
    print("=" * 50)

    return results

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Bazowy (Dense): Loss')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Bazowy (Dense): Accuracy')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dense_training_history.png')
    print("\n[INFO] Zapisano wykres historii treningu: dense_training_history.png")


# =====================================================================
# 6. MAIN
# =====================================================================

if __name__ == "__main__":
    print("--- URUCHAMIANIE MODELU BAZOWEGO (DENSE) ---\n")

    # 1. Dane
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # DataLoadery
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # 2. Model
    # Input size = 868 (długość jednego segmentu)
    model = EEG_Dense_Classifier(input_size=X.shape[1]).to(device)

    # Wyświetl architekturę
    print(f"\nArchitektura modelu:\n{model}")

    # 3. Trening
    # Class weights: [1.0, 4.0] bo klasa 1 jest 4x mniejsza niż klasa 0
    class_weights = torch.FloatTensor([1.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = train_model(model, train_loader, val_loader, criterion, optimizer)

    # 4. Wyniki i Zapis
    results = evaluate_model(model, test_loader)
    plot_history(history)

    # Zapis wyników do JSON
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Dense (MLP)',
        'metrics': results,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'dropout': 0.5
        }
    }

    with open('dense_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    print("[INFO] Zapisano wyniki do dense_results.json")
    print("\n--- ZAKOŃCZONO ---")