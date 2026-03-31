# Hierarchical Offensive Language Detection: BERT vs DeBERTa

---

## Deskripsi

Penelitian ini membandingkan performa **BERT** dan **DeBERTa** untuk deteksi bahasa ofensif secara hierarkis menggunakan dataset **OLID** (SemEval 2019 Task 6 — OffensEval).

| Task   | Input          | Label           |
| ------ | -------------- | --------------- |
| Task A | Semua tweet    | OFF / NOT       |
| Task B | Tweet OFF saja | TIN / UNT       |
| Task C | Tweet TIN saja | IND / GRP / OTH |

---

## Struktur Folder

```
├── dataset/                  # File dataset OLID (tidak diupload, lihat bagian Dataset)
├── notebook/
│   ├── 01_eksplorasi_olid.ipynb   # Eksplorasi dan visualisasi data
│   └── 02_finetune_task_a.ipynb   # Fine-tuning Task A (BERT & DeBERTa)
├── output/                   # Hasil training (checkpoint, grafik, JSON)
├── finetune_task_a.py        # Script versi .py (alternatif notebook)
├── eksplorasi_olid.py        # Script eksplorasi versi .py
└── README.md
```

---

## Dataset

Dataset OLID tersedia di [SemEval 2019 Task 6](https://competitions.codalab.org/competitions/20011).

Letakkan file berikut di folder `dataset/`:

```
dataset/
├── olid-training-v1.0.tsv
├── testset-levela.tsv
├── testset-levelb.tsv
├── testset-levelc.tsv
├── labels-levela.csv
├── labels-levelb.csv
└── labels-levelc.csv
```

---

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn statsmodels
```

### 2. Jalankan notebook secara berurutan

```
notebook/01_eksplorasi_olid.ipynb   ← jalankan dulu
notebook/02_finetune_task_a.ipynb   ← ubah MODEL_CHOICE & SEED di STEP 2
```

### 3. Konfigurasi di Notebook 02 (STEP 2)

```python
MODEL_CHOICE = 'bert'     # 'bert' atau 'deberta'
SEED         = 42         # 42, 123, atau 999
USE_WEIGHTED = True       # True (utama) atau False (ablasi)
```

Jalankan 12x total sesuai tabel di akhir notebook untuk mendapatkan semua hasil.

---

## Model

| Model   | HuggingFace ID              |
| ------- | --------------------------- |
| BERT    | `bert-base-uncased`         |
| DeBERTa | `microsoft/deberta-v3-base` |

---
