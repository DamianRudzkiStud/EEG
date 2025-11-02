import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict
import json

SAMPLING_FREQUENCY = 173.61
SAMPLES_PER_FILE = 4097
DURATION_PER_FILE = SAMPLES_PER_FILE / SAMPLING_FREQUENCY
WINDOW_DURATION = 5.0
WINDOW_SIZE = int(WINDOW_DURATION * SAMPLING_FREQUENCY)
OVERLAP_RATIO = 0.5


class BonnEEGPreprocessor:
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.categories = {
            'S': 'Set A - Zdrowi ochotnicy, oczy otwarte (EEG powierzchniowe)',
            'F': 'Set B - Zdrowi ochotnicy, oczy zamknięte (EEG powierzchniowe)',
            'N': 'Set C - Pacjenci z epilepsją, rejon hipokampa (międzynapadowo)',
            'O': 'Set D - Pacjenci z epilepsją, strefa epileptogenna (międzynapadowo)',
            'Z': 'Set E - Pacjenci z epilepsją, PODCZAS NAPADU PADACZKOWEGO'
        }
        self.dataset_stats = {}

    def load_single_file(self, filepath: Path) -> np.ndarray:
        try:
            data = np.loadtxt(filepath)
            return data
        except Exception as e:
            print(f"Błąd wczytywania pliku {filepath}: {e}")
            return None

    def analyze_dataset_structure(self) -> Dict:
        print("=" * 80)
        print("ANALIZA STRUKTURY DATASETU BONN EEG")
        print("=" * 80)
        print()

        stats = {
            'categories': {},
            'total_files': 0,
            'total_duration_seconds': 0,
            'sampling_frequency': SAMPLING_FREQUENCY,
            'samples_per_file': SAMPLES_PER_FILE
        }

        for category, description in self.categories.items():
            category_path = self.dataset_path / category

            files_txt = list(category_path.glob("*.txt"))
            files_TXT = list(category_path.glob("*.TXT"))
            all_files = files_txt + files_TXT

            num_files = len(all_files)
            duration = num_files * DURATION_PER_FILE

            stats['categories'][category] = {
                'description': description,
                'num_files': num_files,
                'duration_seconds': duration,
                'duration_minutes': duration / 60
            }

            stats['total_files'] += num_files
            stats['total_duration_seconds'] += duration

            print(f"Kategoria {category}: {description}")
            print(f"  Liczba plików: {num_files}")
            print(f"  Czas nagrań: {duration:.1f} sek ({duration/60:.2f} min)")
            print()

        stats['total_duration_minutes'] = stats['total_duration_seconds'] / 60

        print("-" * 80)
        print(f"ŁĄCZNIE:")
        print(f"  Liczba plików: {stats['total_files']}")
        print(f"  Całkowity czas: {stats['total_duration_seconds']:.1f} sek ({stats['total_duration_minutes']:.2f} min)")
        print(f"  Częstotliwość próbkowania: {SAMPLING_FREQUENCY} Hz")
        print(f"  Próbek na plik: {SAMPLES_PER_FILE}")
        print("=" * 80)
        print()

        self.dataset_stats = stats
        return stats

    def segment_signal(self, signal: np.ndarray,
                       window_size: int = WINDOW_SIZE,
                       overlap: float = OVERLAP_RATIO) -> np.ndarray:

        step_size = int(window_size * (1 - overlap))
        num_windows = (len(signal) - window_size) // step_size + 1

        segments = []
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            if end_idx <= len(signal):
                segment = signal[start_idx:end_idx]
                segments.append(segment)

        return np.array(segments)

    def normalize_zscore(self, data: np.ndarray) -> np.ndarray:
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return data - mean
        else:
            return (data - mean) / std

    def process_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print("=" * 80)
        print("PRZETWARZANIE DANYCH")
        print("=" * 80)
        print()

        all_segments = []
        all_labels = []

        for category in self.categories.keys():
            print(f"Przetwarzanie kategorii {category}...")

            category_path = self.dataset_path / category
            files = list(category_path.glob("*.txt")) + list(category_path.glob("*.TXT"))

            # Etykieta: 1 dla napadu (Z), 0 dla pozostałych
            label = 1 if category == 'Z' else 0

            category_segments = []

            for file_path in files:

                signal = self.load_single_file(file_path)
                if signal is None:
                    continue

                segments = self.segment_signal(signal)

                normalized_segments = []
                for segment in segments:
                    normalized_segment = self.normalize_zscore(segment)
                    normalized_segments.append(normalized_segment)

                category_segments.extend(normalized_segments)

            num_segments = len(category_segments)
            all_segments.extend(category_segments)
            all_labels.extend([label] * num_segments)

            print(f"  Wygenerowano {num_segments} segmentów 5-sekundowych")

        X = np.array(all_segments)
        y = np.array(all_labels)

        print()
        print("-" * 80)
        print(f"WYNIK PRZETWARZANIA:")
        print(f"  Wymiary macierzy cech X: {X.shape}")
        print(f"  Wymiary wektora etykiet y: {y.shape}")
        print(f"  Liczba segmentów klasy 0 (zdrowy): {np.sum(y == 0)}")
        print(f"  Liczba segmentów klasy 1 (napad): {np.sum(y == 1)}")
        print(f"  Rozkład klas: {np.sum(y == 0)/len(y)*100:.1f}% vs {np.sum(y == 1)/len(y)*100:.1f}%")
        print("=" * 80)
        print()

        return X, y


    def visualize_raw_signals(self, output_file: str = "eeg_raw_signals.png"):

        print(f"Generowanie wizualizacji surowych sygnałów...")

        fig, axes = plt.subplots(5, 1, figsize=(15, 12))
        fig.suptitle('Surowe sygnały EEG z datasetu Bonn - pierwsze 5 sekund',
                     fontsize=16, fontweight='bold')

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

        for idx, (category, description) in enumerate(self.categories.items()):
            category_path = self.dataset_path / category
            files = list(category_path.glob("*.txt")) + list(category_path.glob("*.TXT"))

            if files:
                signal = self.load_single_file(files[0])

                samples_to_show = int(5 * SAMPLING_FREQUENCY)
                time_axis = np.arange(samples_to_show) / SAMPLING_FREQUENCY

                axes[idx].plot(time_axis, signal[:samples_to_show],
                             color=colors[idx], linewidth=0.8, alpha=0.9)
                axes[idx].set_title(f'{category}: {description}',
                                   fontsize=11, fontweight='bold')
                axes[idx].set_ylabel('Amplituda (μV)', fontsize=10)
                axes[idx].grid(True, alpha=0.3, linestyle='--')
                axes[idx].set_xlim(0, 5)

                if idx == 4:
                    axes[idx].set_xlabel('Czas (sekundy)', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Zapisano: {output_file}")
        print()

    def visualize_processed_signals(self, X: np.ndarray, y: np.ndarray,
                                   output_file: str = "eeg_processed_signals.png"):

        print(f"Generowanie wizualizacji przetworzonych sygnałów...")

        fig, axes = plt.subplots(5, 1, figsize=(15, 12))
        fig.suptitle('Przetworzone segmenty EEG (5 sek, znormalizowane Z-score)',
                     fontsize=16, fontweight='bold')

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        categories_list = list(self.categories.keys())

        segments_per_category = len(X) // 5

        for idx, category in enumerate(categories_list):
            segment_idx = idx * segments_per_category
            segment = X[segment_idx]

            time_axis = np.arange(len(segment)) / SAMPLING_FREQUENCY

            axes[idx].plot(time_axis, segment,
                         color=colors[idx], linewidth=0.8, alpha=0.9)
            axes[idx].set_title(f'{category}: {self.categories[category]}',
                               fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Amplituda\n(Z-score)', fontsize=10)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            axes[idx].set_xlim(0, 5)

            if idx == 4:
                axes[idx].set_xlabel('Czas (sekundy)', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Zapisano: {output_file}")
        print()

    def visualize_class_distribution(self, y: np.ndarray,
                                    output_file: str = "class_distribution.png"):

        print(f"Generowanie wizualizacji rozkładu klas...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Rozkład klas w datasecie', fontsize=16, fontweight='bold')

        # Wykres słupkowy
        class_counts = [np.sum(y == 0), np.sum(y == 1)]
        class_labels = ['Brak napadu\n(S, F, N, O)', 'Napad padaczkowy\n(Z)']
        colors_bar = ['#2E86AB', '#C73E1D']

        bars = ax1.bar(class_labels, class_counts, color=colors_bar, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Liczba segmentów 5-sekundowych', fontsize=11)
        ax1.set_title('Liczność klas', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax2.pie(class_counts, labels=class_labels, colors=colors_bar, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Proporcje klas', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Zapisano: {output_file}")
        print()

    def save_processed_data(self, X: np.ndarray, y: np.ndarray,
                          output_dir: str = "processed_data"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("Zapisywanie przetworzonych danych...")

        np.save(output_path / "X_processed.npy", X)
        np.save(output_path / "y_labels.npy", y)

        metadata = {
            'num_samples': int(len(X)),
            'window_size': int(X.shape[1]),
            'window_duration_seconds': WINDOW_DURATION,
            'sampling_frequency': SAMPLING_FREQUENCY,
            'overlap_ratio': OVERLAP_RATIO,
            'num_class_0': int(np.sum(y == 0)),
            'num_class_1': int(np.sum(y == 1)),
            'normalization_method': 'z-score'
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"  Zapisano przetworzone dane w katalogu: {output_dir}")
        print(f"    - X_processed.npy: {X.shape}")
        print(f"    - y_labels.npy: {y.shape}")
        print(f"    - metadata.json")
        print()


def main():

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  PIPELINE PRZETWARZANIA DANYCH EEG - DATASET BONN".center(78) + "║")
    print("║" + "  Projekt: Analiza sygnałów biologicznych RNN/LSTM".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    preprocessor = BonnEEGPreprocessor(dataset_path="dataset")

    dataset_stats = preprocessor.analyze_dataset_structure()

    preprocessor.visualize_raw_signals(output_file="eeg_raw_signals.png")

    X, y = preprocessor.process_all_data()

    preprocessor.visualize_processed_signals(X, y, output_file="eeg_processed_signals.png")

    preprocessor.visualize_class_distribution(y, output_file="class_distribution.png")

    preprocessor.save_processed_data(X, y, output_dir="processed_data")

    print("=" * 80)
    print("PIPELINE ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 80)
    print()
    print("Wygenerowane pliki:")
    print("  1. eeg_raw_signals.png - wizualizacja surowych sygnałów")
    print("  2. eeg_processed_signals.png - wizualizacja przetworzonych segmentów")
    print("  3. class_distribution.png - rozkład klas")
    print("  4. processed_data/ - katalog z przetworzonymi danymi")
    print()
    print("Dane gotowe do dalszej analizy i budowy modeli LSTM/GRU!")
    print()


if __name__ == "__main__":
    main()
