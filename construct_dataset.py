import json
import csv
import os
import ast
import difflib

def load_original_texts(file_path):
    """Load original texts from the input file."""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    texts.append(line.strip())
        return texts
    except FileNotFoundError:
        print(f"Error: Original text file {file_path} not found.")
        return []

def load_canon_kg(file_path):
    """Load canonical knowledge graph from the output file. This uses Python's ast.literal_eval to parse the list format."""
    kg_by_line = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    # Use ast.literal_eval instead of json.loads to parse Python list syntax
                    triplets = ast.literal_eval(line.strip())
                    kg_by_line.append(triplets)
        return kg_by_line
    except FileNotFoundError:
        print(f"Error: Canon KG file {file_path} not found.")
        return []
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing Canon KG file {file_path}: {e}")
        return []

def load_schema_definitions(file_path):
    """Load schema definitions from the CSV file."""
    schema_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    relation_name = row[0]
                    relation_definition = row[1]
                    schema_dict[relation_name] = relation_definition
        return schema_dict
    except FileNotFoundError:
        print(f"Error: Schema file {file_path} not found.")
        return {}

def load_reference_data(file_path):
    """Load reference data if available."""
    reference_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    reference_data.append(line.strip())
        return reference_data
    except FileNotFoundError:
        print(f"Warning: Reference file {file_path} not found. Proceeding without reference data.")
        return []

def load_extraction_data(file_path):
    """Load OIE, schema definition, and schema canonicalization from result_at_each_stage.json."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: Extraction data file {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Extraction data file {file_path} contains invalid JSON.")
        return []

def find_line_numbers_in_original_file(texts_to_find, original_file_path, similarity_threshold=0.8):
    """Find the line numbers of given texts in the original file using fuzzy matching."""
    # Load all lines from the original file
    all_original_lines = []
    try:
        with open(original_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    all_original_lines.append(line.strip())
    except FileNotFoundError:
        print(f"Error: Original file {original_file_path} not found for line number mapping.")
        return {i: i+1 for i in range(len(texts_to_find))}  # Return 1-indexed sequential numbers
    
    # Map each text to its best matching line number in the original file
    result = {}
    matched_lines = set()  # Keep track of already matched lines to avoid duplicates
    
    for i, text in enumerate(texts_to_find):
        # Find the closest matching line in the original file
        best_match = None
        best_ratio = 0
        best_index = -1
        
        for j, orig_line in enumerate(all_original_lines, 1):
            # Skip already matched lines to ensure one-to-one mapping
            if j in matched_lines:
                continue
                
            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, text, orig_line).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = orig_line
                best_index = j
        
        # If we found a good match, use that line number
        if best_ratio >= similarity_threshold:
            result[i] = best_index
            matched_lines.add(best_index)
            print(f"Mapped text at index {i} to line {best_index} with similarity {best_ratio:.2f}")
        else:
            # If no good match found, use sequential numbering
            result[i] = i + 1
            print(f"Warning: No good match found for text at index {i}. Using index {i+1} instead.")
    
    return result

def load_reference_triplets(file_path):
    """Load the reference triplets from the webnlg reference file."""
    reference_triplets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    # Use ast.literal_eval to safely parse the triplets
                    triplets = ast.literal_eval(line.strip())
                    reference_triplets.append(triplets)
        return reference_triplets
    except FileNotFoundError:
        print(f"Error: Reference triplets file {file_path} not found.")
        return []
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing reference triplets file {file_path}: {e}")
        return []

def construct_dataset(original_file, canon_kg_file, schema_file, reference_file, extraction_file, output_file, original_source_file):
    """Construct the complete dataset by combining all sources."""
    # Load all data sources
    original_texts = load_original_texts(original_file)
    canon_kg = load_canon_kg(canon_kg_file)
    schema_definitions = load_schema_definitions(schema_file)
    reference_data = load_reference_data(reference_file)
    extraction_data = load_extraction_data(extraction_file)
    
    # Load reference triplets from webnlg.txt
    reference_triplets_file = "evaluate/references/webnlg.txt"
    reference_triplets = load_reference_triplets(reference_triplets_file)
    if not reference_triplets:
        print(f"Warning: No reference triplets loaded from {reference_triplets_file}. The labeled_triplets column will be empty.")
    
    # Check if we have enough data to proceed
    if not original_texts or not canon_kg or not extraction_data:
        print("Error: Cannot construct dataset due to missing critical data.")
        return []
    
    # Find the original line numbers in the source dataset using fuzzy matching
    print("Mapping texts to original line numbers...")
    line_number_map = find_line_numbers_in_original_file(original_texts, original_source_file)
    
    # Ensure we have the same number of entries in main data sources
    min_length = min(len(original_texts), len(canon_kg), len(extraction_data))
    if len(original_texts) != len(canon_kg) or len(original_texts) != len(extraction_data):
        print(f"Warning: Data sources have different lengths. Using the minimum length: {min_length}")
    
    # Extend reference_data if needed
    if len(reference_data) < min_length:
        reference_data.extend([""] * (min_length - len(reference_data)))
    
    # Create the dataset
    dataset = []
    for i in range(min_length):
        # Extract extraction data (OIE, schema definition, schema canonicalization)
        extraction_entry = extraction_data[i] if i < len(extraction_data) else {}
        oie = extraction_entry.get("oie", [])
        schema_definition = extraction_entry.get("schema_definition", {})
        schema_canonicalization = extraction_entry.get("schema_canonicalizaiton", [])
        
        # Use the original line number from the source dataset
        original_line_number = line_number_map.get(i, i+1)
        
        # Get labeled triplets from reference file if available
        labeled_triplets = []
        if reference_triplets and original_line_number <= len(reference_triplets):
            labeled_triplets = reference_triplets[original_line_number-1]
        
        entry = {
            "index": original_line_number,
            "text": original_texts[i],
            "canon_kg": canon_kg[i],
            "reference": reference_data[i] if i < len(reference_data) else "",
            "oie": oie,
            "schema_definition": schema_definition,
            "schema_canonicalization": schema_canonicalization,
            "labeled_triplets": labeled_triplets
        }
        
        # Get labeled schema and definitions based on relation names in canon_kg
        labeled_schema = []
        labeled_schema_definition = []
        for triplet in canon_kg[i]:
            if len(triplet) >= 2:  # Ensure the triplet has a subject and relation
                relation = triplet[1]
                if relation in schema_definitions:
                    labeled_schema.append(relation)
                    labeled_schema_definition.append(schema_definitions[relation])
        
        entry["labeled_schema"] = labeled_schema
        entry["labeled_schema_definition"] = labeled_schema_definition
        
        # Determine if this should be in NOTA set (this is a placeholder logic - adjust as needed)
        # For example, if no labeled schema matches any canonicalized schema
        entry["nota"] = not any(
            triplet[1] in labeled_schema if triplet and len(triplet) > 1 else False 
            for triplet in schema_canonicalization if triplet
        )
        
        dataset.append(entry)
    
    # Sort the dataset by index (original line number)
    dataset.sort(key=lambda x: x["index"])
    
    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["index", "text", "canon_kg", "labeled_schema", "labeled_schema_definition",
                         "labeled_triplets", "reference", "oie", "schema_definition", "schema_canonicalization", "nota"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in dataset:
                # Convert complex data structures to JSON strings for CSV
                for field in ["canon_kg", "labeled_schema", "labeled_schema_definition", 
                              "labeled_triplets", "oie", "schema_definition", "schema_canonicalization"]:
                    entry[field] = json.dumps(entry[field])
                writer.writerow(entry)
        
        print(f"Dataset successfully created: {output_file}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
    
    return dataset

if __name__ == "__main__":
    # Define paths
    original_file = "./dataset_webnlg_sequential_50.txt"
    canon_kg_file = "./output/test_50_no_suffle/iter0/canon_kg.txt"
    schema_file = "./schemas/webnlg_first_50_schema.csv"
    original_source_file = "./datasets/webnlg.txt"  # The original source file for line number mapping
    
    # Check if reference directory exists, use a placeholder if not
    reference_dir = "./evaluate/reference"
    if os.path.exists(reference_dir):
        reference_file = os.path.join(reference_dir, "webnlg.txt")
    else:
        print(f"Warning: Reference directory {reference_dir} not found. Reference data will be empty.")
        reference_file = ""
    
    extraction_file = "./output/test_50_no_suffle/iter0/result_at_each_stage.json"
    output_file = "./complete_dataset.csv"
    
    # Construct the dataset
    construct_dataset(original_file, canon_kg_file, schema_file, reference_file, extraction_file, output_file, original_source_file) 