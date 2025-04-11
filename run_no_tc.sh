# An example to run EDC (without refinement) on the example dataset

OIE_LLM=gemini-2.0-flash
SD_LLM=gemini-2.0-flash
SC_LLM=gemini-2.0-flash
SC_EMBEDDER=intfloat/e5-mistral-7b-instruct
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
    --logging_verbose