# Dynamic Knowledge Graph with EDC

## Prerequisite

To run the simulation of the test, prepare your `GEMINI_API_KEY` in `.env` and the embedding model.

For just 1 GEMINI API KEY, it's very likely to be rated limit, so here's how you can set up multiple keys:
```
./set_api_keys.sh "AIzaSyA1B2C3D4..." "AIzaSyE5F6G7H8..." "AIzaSyI9J0K1L2..."
```
We should rotate using them when one become restricted.

Download embedding model :
```bash
pip install -U ctranslate2
ct2-transformers-converter --model BAAI/bge-m3 --output_dir bge_model_ctranslate2 --force
```

## Usage

### Step 1: Build dataset for evaluating EDC's performances on triplets extraction + construct NOTA sets.

In this step we aiming at building dataset for:
- Evaluation on triplets extracting technique (EDC in this case)
- Training embedding models to make schema extracted and label schema closer 

For no pre-defined schema usage:

```bash
OIE_LLM=gemini-2.0-flash
SD_LLM=gemini-2.0-flash
SC_LLM=gemini-2.0-flash
SC_EMBEDDER=BAAI/bge-m3
EE_LLM=gemini-2.0-flash
DATASET=webnlg

python run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --sc_embedder $SC_EMBEDDER \
    --ee_llm $EE_LLM \
    --input_text_file_path "./datasets/${DATASET}_simple.txt" \
    --output_dir "./output/${DATASET}_free_alignment" \
    --enrich_schema \
    --suffle true \
    --extraction_length $EXTRACTION_LENGTH \
    --schema_length $SCHEMA_LENGTH \
    --embedding_threshold $EMBEDDING_THRESHOLD \
    --logging_verbose
    --logging_verbose
```

For pre-defined schema usage:

```bash
OIE_LLM=gemini-2.0-flash
SD_LLM=gemini-2.0-flash
SC_LLM=gemini-2.0-flash
SC_EMBEDDER=BAAI/bge-m3
EE_LLM=gemini-2.0-flash
DATASET=webnlg
EXTRACTION_LENGTH=200
SCHEMA_LENGTH=50
EMBEDDING_THRESHOLD=0.7
SUFFLE=true

python run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --sc_embedder $SC_EMBEDDER \
    --ee_llm $EE_LLM \
    --input_text_file_path "./dataset_1_test.txt" \
    --target_schema_path "./schemas/webnlg_first_50_schema.csv" \
    --output_dir "./output/dataset_1_target_alignment" \
    --suffle true \
    --extraction_length $EXTRACTION_LENGTH \
    --schema_length $SCHEMA_LENGTH \
    --embedding_threshold $EMBEDDING_THRESHOLD \
    --logging_verbose
```

After running one of these, you will have your own dataset in folder output, each row includes:
```json
{
  'text': str // the text (one single line) from your dataset
  'index': int // index in dataset of the text 
  'labeled_schema' : list // text's equivalent schema (indexed from your dataset)
  'labeled_schema_definition' : list // definitions of schemas in equivalent labeled_schema (indexed from your dataset)
  'open_extraction_information': list // triplets extracted from edc
  'schema_definition': list // schema definition from edc
  'schema_canonicalization': list // schema canonicalization from edc for alignment with labeled_schema_definitions 
  'nota': boolean // this will be insert into nota or not
}
```

Evaluation of triplet-extracting methods should locate in `/evaluation/` folder

### Step 2: Clustering on NOTA sets, then evaluate clustering performances

Run this script `clustering.sh` for clustering:

```bash
ALGORITHM=kmeans
DATASET=webnlg
# ALGORITHM=h-clustering
# ALGORITHM=dbscan
# ALGORITHM=fuzzy_cmeans
# ALGORITHM=spectral_clustering

python clutering.py \
    --algorithm $ALGORITHM
    --dataset_path "./output/{$DATASET}_target_alignment" 
```

Evaluation:
```bash
python eval_clustering.py \
    --dataset_path "./output/clustering/{$DATASET}"
```

Evaluation of clustering model should locate in `/evaluation/` folder

### Bonus: GraphRAG

Run this script to build Graph Database:

```bash
NAME_OF_GRAPH = webnlg

python kg_construct.py \
    --name $NAME_OF_GRAPH
```



