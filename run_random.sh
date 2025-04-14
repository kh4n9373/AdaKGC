# An example to run EDC (without refinement) on the example dataset
# This script randomly selects N lines from the source dataset

OIE_LLM=gemini-2.0-flash
SD_LLM=gemini-2.0-flash
SC_LLM=gemini-2.0-flash
SC_EMBEDDER=intfloat/e5-mistral-7b-instruct
EE_LLM=gemini-2.0-flash
DATASET=webnlg
RANDOMIZE=true
NO_LINES=50
EMBEDDING_THRESHOLD=0.7

# Prepare the randomize parameter - set for random selection
RANDOMIZE_ARG="--randomize"

# Example for running with line selection from source dataset (random lines)
python run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --sc_embedder $SC_EMBEDDER \
    --ee_llm $EE_LLM \
    --source_dataset_path "./datasets/${DATASET}.txt" \
    $RANDOMIZE_ARG \
    --no_lines $NO_LINES \
    --target_schema_path "./schemas/webnlg_first_50_schema.csv" \
    --embedding_threshold $EMBEDDING_THRESHOLD \
    --logging_verbose 