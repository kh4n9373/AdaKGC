# An example to run EDC (without refinement) on the example dataset
# This script takes the first N lines instead of random sampling

OIE_LLM=gemini-2.0-flash
SD_LLM=gemini-2.0-flash
SC_LLM=gemini-2.0-flash
SC_EMBEDDER=intfloat/e5-mistral-7b-instruct
EE_LLM=gemini-2.0-flash
DATASET=webnlg
RANDOMIZE=false
NO_LINES=200
EMBEDDING_THRESHOLD=0.7

# Prepare the randomize parameter - in this case we want sequential lines
RANDOMIZE_ARG=""
# if [ "$RANDOMIZE" = true ]; then
#     RANDOMIZE_ARG="--randomize"
# fi

# Example for running with line selection from source dataset (sequential lines)
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
    --output_dir "./output/dataset_sequential_alignment" \
    --embedding_threshold $EMBEDDING_THRESHOLD \
    --logging_verbose 