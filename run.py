from argparse import ArgumentParser
from edc.edc_framework import EDC
import os
import logging
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def select_lines_from_file(file_path, randomize=False, no_lines=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            all_lines = [line.strip() for line in file if line.strip()]
            
        if no_lines is None or no_lines >= len(all_lines):
            return all_lines
        
        if randomize:
            selected_lines = random.sample(all_lines, no_lines)
        else:
            selected_lines = all_lines[:no_lines]
            
        return selected_lines
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def generate_input_file_path(source_path, randomize, no_lines):
    source_basename = os.path.splitext(os.path.basename(source_path))[0]
    
    if randomize:
        return f"./dataset_{source_basename}_random_{no_lines}.txt"
    else:
        return f"./dataset_{source_basename}_sequential_{no_lines}.txt"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--oie_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for open information extraction."
    )
    parser.add_argument(
        "--oie_prompt_template_file_path",
        default="./prompt_templates/oie_template.txt",
        help="Promp template used for open information extraction.",
    )
    parser.add_argument(
        "--oie_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_examples.txt",
        help="Few shot examples used for open information extraction.",
    )

    parser.add_argument(
        "--sd_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for schema definition."
    )
    parser.add_argument(
        "--sd_prompt_template_file_path",
        default="./prompt_templates/sd_template.txt",
        help="Prompt template used for schema definition.",
    )
    parser.add_argument(
        "--sd_few_shot_example_file_path",
        default="./few_shot_examples/example/sd_few_shot_examples.txt",
        help="Few shot examples used for schema definition.",
    )

    parser.add_argument(
        "--sc_llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="LLM used for schema canonicaliztion verification.",
    )
    parser.add_argument(
        "--sc_embedder", default="intfloat/e5-mistral-7b-instruct", help="Embedder used for schema canonicalization. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--sc_prompt_template_file_path",
        default="./prompt_templates/sc_template.txt",
        help="Prompt template used for schema canonicalization verification.",
    )

    parser.add_argument("--sr_adapter_path", default=None, help="Path to adapter of schema retriever.")
    parser.add_argument(
        "--sr_embedder", default="intfloat/e5-mistral-7b-instruct", help="Embedding model used for schema retriever. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--oie_refine_prompt_template_file_path",
        default="./prompt_templates/oie_r_template.txt",
        help="Prompt template used for refined open information extraction.",
    )
    parser.add_argument(
        "--oie_refine_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_refine_examples.txt",
        help="Few shot examples used for refined open information extraction.",
    )
    parser.add_argument(
        "--ee_llm", default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM used for entity extraction."
    )
    parser.add_argument(
        "--ee_prompt_template_file_path",
        default="./prompt_templates/ee_template.txt",
        help="Prompt templated used for entity extraction.",
    )
    parser.add_argument(
        "--ee_few_shot_example_file_path",
        default="./few_shot_examples/example/ee_few_shot_examples.txt",
        help="Few shot examples used for entity extraction.",
    )
    parser.add_argument(
        "--em_prompt_template_file_path",
        default="./prompt_templates/em_template.txt",
        help="Prompt template used for entity merging.",
    )

    parser.add_argument(
        "--input_text_file_path",
        default=None,
        help="Optional: File containing input texts to extract KG from. If not provided, will be generated from source_dataset_path.",
    )
    parser.add_argument(
        "--source_dataset_path",
        default=None,
        help="Path to the source dataset file (e.g., webnlg.txt) for line selection.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Whether to randomly select lines from source dataset.",
    )
    parser.add_argument(
        "--no_lines",
        type=int,
        default=None,
        help="Number of lines to select from source dataset.",
    )
    parser.add_argument(
        "--target_schema_path",
        default="./schemas/example_schema.csv",
        help="File containing the target schema to align to.",
    )
    parser.add_argument("--refinement_iterations", default=0, type=int, help="Number of iteration to run.")
    parser.add_argument(
        "--enrich_schema",
        action="store_true",
        help="Whether un-canonicalizable relations should be added to the schema.",
    )
    parser.add_argument(
        "--suffle",
        action="store_true",
        help="Whether to shuffle the input text lines before processing.",
    )
    parser.add_argument(
        "--extraction_length", 
        type=int, 
        default=None, 
        help="Number of texts to extract triplets from."
    )
    parser.add_argument(
        "--schema_length", 
        type=int, 
        default=None, 
        help="Number of schemas to extract."
    )
    parser.add_argument(
        "--embedding_threshold", 
        type=float, 
        default=0.7, 
        help="Threshold for embedding similarity in schema canonicalization."
    )

    parser.add_argument("--output_dir", default="./output/tmp", help="Directory to output to.")
    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()
    args = vars(args)
    
    if args["source_dataset_path"] and args["no_lines"]:
        print(f"Selecting {args['no_lines']} lines from {args['source_dataset_path']}")
        print(f"Randomize: {args['randomize']}")
        
        selected_lines = select_lines_from_file(
            args["source_dataset_path"], 
            randomize=args["randomize"], 
            no_lines=args["no_lines"]
        )
        
        if not args["input_text_file_path"]:
            args["input_text_file_path"] = generate_input_file_path(
                args["source_dataset_path"],
                args["randomize"],
                args["no_lines"]
            )
        
        temp_file_path = args["input_text_file_path"]
        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            for line in selected_lines:
                temp_file.write(line + '\n')
        
        print(f"Created input file with {len(selected_lines)} lines at {temp_file_path}")
        
        if args["output_dir"] == "./output/tmp" and not args.get("input_text_file_path_provided", False):
            output_dir_name = os.path.splitext(os.path.basename(temp_file_path))[0]
            args["output_dir"] = f"./output/{output_dir_name}_alignment"
            print(f"Using auto-generated output directory: {args['output_dir']}")
    
    edc = EDC(**args)
    
    input_text_list = open(args["input_text_file_path"], "r").readlines()
    output_kg = edc.extract_kg(
        input_text_list,
        args["output_dir"],
        refinement_iterations=args["refinement_iterations"],
    )
