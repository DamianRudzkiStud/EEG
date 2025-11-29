#!/usr/bin/env python3

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
# 1. WCZYTYWANIE DANYCH
# =====================================================================

class EEG_LSTM_Classifier(nn.Module):
    def __init__(self, input_size=868, hidden_size_1=64,
                 hidden_size_2=32, num_classes=2, dropout=0.3):
        super(EEG_LSTM_Classifier, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size_1,
                             num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2,
                             num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class EEGDataset(Dataset):
    """Custom Dataset dla sekwencji EEG"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_preprocessed_data(data_dir='processed_data'):
    """Wczytuje przetworzone dane z Raportu 1"""

    data_path = Path(data_dir)

    print("=" * 80)
    print("WCZYTYWANIE PRZETWORZONYCH DANYCH")
    print("=" * 80)

    X = np.load(data_path / "X_processed.npy")
    y = np.load(data_path / "y_labels.npy")

    with open(data_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    print(f"\nWymiary danych:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"\nRozkład klas:")
    print(f"  Klasa 0 (brak napadu): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Klasa 1 (napad):       {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"\nMetadane:")
    print(f"  Długość okna: {metadata['window_size']} próbek ({metadata['window_duration_seconds']}s)")
    print(f"  Częstotliwość próbkowania: {metadata['sampling_frequency']} Hz")
    print(f"  Metoda normalizacji: {metadata['normalization_method']}")
    print("=" * 80)
    print()

    return X, y, metadata


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Podział danych na train/val/test ze stratyfikacją

    70% train, 15% validation, 15% test
    """

    print("PODZIAŁ DANYCH NA TRAIN/VAL/TEST")
    print("=" * 80)

    # Najpierw oddziel test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Potem podziel pozostałe na train i val
    val_size_adjusted = val_size / (1 - test_size)  # Dostosuj proporcję
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    print(f"\nPodział danych:")
    print(f"  Train: {len(X_train)} próbek ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Klasa 0: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"    - Klasa 1: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} próbek ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Klasa 0: {np.sum(y_val == 0)} ({np.sum(y_val == 0)/len(y_val)*100:.1f}%)")
    print(f"    - Klasa 1: {np.sum(y_val == 1)} ({np.sum(y_val == 1)/len(y_val)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} próbek ({len(X_test)/len(X)*100:.1f}%)")
    print(f"    - Klasa 0: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"    - Klasa 1: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
    print("=" * 80)
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# 2. ARCHITEKTURA MODELU LSTM
# =====================================================================

class EEG_LSTM_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size_1=64, hidden_size_2=32, num_classes=2, dropout=0.3):
        super(EEG_LSTM_Classifier, self).__init__()

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        # Pierwsza warstwa LSTM
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        # Druga warstwa LSTM
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Warstwa Dense (fully connected)
        self.fc = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, sequence_length)

        # LSTM1
        out, _ = self.lstm1(x)
        out = self.dropout(out)

        # LSTM2
        out, _ = self.lstm2(out)
        out = self.dropout(out)

        # Weź tylko ostatni output z sekwencji
        out = out[:, -1, :]

        # Dense layer
        out = self.fc(out)

        return out


def get_model_summary(model, input_size):
    """Wyświetla podsumowanie architektury modelu"""

    print("=" * 80)
    print("ARCHITEKTURA MODELU")
    print("=" * 80)
    print(f"\n{model}\n")

    # Policz parametry
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parametry modelu:")
    print(f"  Łącznie:  {total_params:,}")
    print(f"  Trenowalne: {trainable_params:,}")
    print(f"  Rozmiar wejścia: {input_size}")
    print("=" * 80)
    print()


# =====================================================================
# TRENING MODELU
# =====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trening przez jedną epokę"""

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.unsqueeze(1).to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Walidacja przez jedną epokę"""

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.unsqueeze(1).to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=10):
    """
    Główna pętla treningu z early stopping
    """

    print("=" * 80)
    print("ROZPOCZĘCIE TRENINGU")
    print("=" * 80)
    print()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Trening
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Walidacja
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Zapisz historię
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Wyświetl postęp
        print(f"Epoka {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ Nowy najlepszy model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping po epoce {epoch+1}")
                break

        print()

    # Przywróć najlepszy model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Przywrócono najlepszy model (val_loss: {best_val_loss:.4f})")

    print("=" * 80)
    print()

    return history


# =====================================================================
# EWALUACJA I WIZUALIZACJA
# =====================================================================

def evaluate_model(model, test_loader, device):

    print("=" * 80)
    print("EWALUACJA NA ZBIORZE TESTOWYM")
    print("=" * 80)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.unsqueeze(1).to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Oblicz metryki
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"\nMetryki klasyfikacji:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nMacierz pomyłek:")
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    print("=" * 80)
    print()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_training_history(history, output_file='training_history.png'):

    print(f"Generowanie wykresów treningu...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Zapisano: {output_file}")
    print()


# =====================================================================
# PIPELINE TRENINGU
# =====================================================================

def main():

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  RAPORT 2: BUDOWA I TRENING MODELU LSTM".center(78) + "║")
    print("║" + "  Projekt: Wykrywanie napadów padaczkowych (Bonn EEG)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    # Konfiguracja
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10
    RANDOM_SEED = 42

    # Ustaw seed dla reprodukowalności
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}\n")

    # 1. Wczytaj dane
    X, y, metadata = load_preprocessed_data()

    # 2. Podziel dane
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Utwórz DataLoadery
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Zbuduj model
    input_size = X.shape[1]  # 868 próbek
    model = EEG_LSTM_Classifier(
        input_size=input_size,
        hidden_size_1=64,
        hidden_size_2=32,
        num_classes=2,
        dropout=0.3
    ).to(device)

    get_model_summary(model, input_size)

    # 5. Konfiguracja treningu
    # Class weights dla niezbalansowanych danych
    class_weights = torch.FloatTensor([1.0, 4.0]).to(device)  # 80:20 -> waga 4:1
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Parametry treningu:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Optimizer: Adam")
    print(f"  Loss function: CrossEntropyLoss (class weights: {class_weights.tolist()})")
    print(f"  Max epochs: {NUM_EPOCHS}")
    print(f"  Early stopping patience: {PATIENCE}")
    print()

    # 6. Trening
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )

    # 7. Ewaluacja
    results = evaluate_model(model, test_loader, device)

    # 8. Wizualizacja
    plot_training_history(history, output_file='lstm_training_history.png')

    # 9. Zapisz model i wyniki
    print("Zapisywanie modelu i wyników...")

    # Zapisz model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'test_results': results,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'input_size': input_size,
            'hidden_size_1': 64,
            'hidden_size_2': 32,
            'dropout': 0.3
        }
    }, 'lstm_model.pth')

    # Zapisz wyniki do JSON
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'LSTM (2 warstwy)',
        'dataset': 'Bonn EEG',
        'test_metrics': {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score'])
        },
        'confusion_matrix': results['confusion_matrix'],
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs_trained': len(history['train_loss']),
            'architecture': '2xLSTM(64,32) + Dense(2)'
        }
    }

    with open('lstm_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)

    print("  Zapisano: lstm_model.pth")
    print("  Zapisano: lstm_results.json")
    print()

    print("=" * 80)
    print("TRENING ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 80)
    print()
    print("Wygenerowane pliki:")
    print("  1. lstm_model.pth - wytrenowany model")
    print("  2. lstm_results.json - wyniki i metryki")
    print("  3. lstm_training_history.png - wykresy loss i accuracy")
    print()


if __name__ == "__main__":
    main()



