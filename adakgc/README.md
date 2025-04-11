# AdaKGC: Adaptive Knowledge Graph Construction

This project implements a pipeline for adaptive knowledge graph construction using the WebNLG dataset. It follows the Extract, Define, Canonicalize (EDC) framework approach for processing text data and constructing a knowledge graph with well-defined relations.

## Overview

The pipeline processes text data from the WebNLG dataset containing 159 distinct schema relations, performing the following steps:

1. For each batch of relations:
   - Process text/triplets into a batch structure
   - Apply EDC (Extract, Define, Canonicalize) to improve schema representation
   - Choose nearest similarity match with existing relations
   - Add new relations when similarity is below threshold

2. Generate embeddings for training to make relation labels and predictions closer

## Architecture

```
adakgc/
├── src/                               # Source code
│   ├── data/                          # Data processing modules
│   ├── models/                        # Model implementations
│   │   ├── edc/                       # Extract, Define, Canonicalize
│   │   ├── embedding/                 # Embedding models
│   │   └── clustering/                # Relation clustering
│   ├── pipeline/                      # Pipeline implementation
│   └── utils/                         # Utilities
├── data/                              # Data files
├── models/                            # Model checkpoints
├── configs/                           # Configuration files
├── notebooks/                         # Jupyter notebooks
├── scripts/                           # Utility scripts
└── tests/                             # Unit tests
```

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/adakgc.git
cd adakgc
```

2. Set up the environment
```bash
# Using conda
conda env create -f environment.yml
conda activate adakgc

# Or using pip
pip install -r requirements.txt
```

3. Download and prepare data
```bash
bash scripts/download_data.sh
python scripts/preprocess.py
```

4. Run the pipeline
```bash
python src/main.py --config configs/default.yaml
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers
- Sentence-Transformers
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
