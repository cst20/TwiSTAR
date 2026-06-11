import json
import gzip
import os

def load_amazon_meta(meta_file_path):
    """Load Amazon metadata and build a dictionary keyed by title."""
    print(f"Loading Amazon metadata from {meta_file_path}...")
    amazon_data = {}
    
    open_func = gzip.open if meta_file_path.endswith('.gz') else open
    mode = 'rt' if meta_file_path.endswith('.gz') else 'r'
    
    with open_func(meta_file_path, mode, encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                # Fallback for old amazon dataset using eval
                item = eval(line.strip())
                
            title = item.get('title', '')
            if title:
                # Extract rich features
                enriched_info = {}
                if 'brand' in item: enriched_info['brand'] = item['brand']
                if 'price' in item: enriched_info['price'] = item['price']
                if 'feature' in item: enriched_info['feature'] = item['feature']
                
                # Keep a robust description if available
                if 'description' in item and isinstance(item['description'], list):
                    enriched_info['description'] = " ".join(item['description'])
                elif 'description' in item:
                    enriched_info['description'] = item['description']
                
                amazon_data[title] = enriched_info
                
    print(f"Loaded {len(amazon_data)} items from Amazon metadata.")
    return amazon_data

def enrich_pretrain_data(pretrain_file, amazon_meta_file, output_file):
    amazon_data = load_amazon_meta(amazon_meta_file)
    
    print(f"Loading current pretrain data from {pretrain_file}...")
    with open(pretrain_file, 'r', encoding='utf-8') as f:
        pretrain_data = json.load(f)
        
    matched_count = 0
    for item_id, item_info in pretrain_data.items():
        title = item_info.get('title', '')
        if title in amazon_data:
            matched_count += 1
            rich_info = amazon_data[title]
            
            # Merge rich information
            if 'brand' in rich_info: item_info['brand'] = rich_info['brand']
            if 'price' in rich_info: item_info['price'] = rich_info['price']
            if 'feature' in rich_info: item_info['feature'] = rich_info['feature']
            if 'description' in rich_info and not item_info.get('description'):
                item_info['description'] = rich_info['description']
                
        pretrain_data[item_id] = item_info

    print(f"Matched {matched_count} out of {len(pretrain_data)} items.")
    
    print(f"Saving enriched data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pretrain_data, f, indent=4, ensure_ascii=False)
    print("Done!")

if __name__ == "__main__":
    META_FILE = "meta_Beauty.json.gz" # Update this path to where you downloaded the file
    PRETRAIN_FILE = "Beauty.pretrain.json"
    OUTPUT_FILE = "Beauty.pretrain.ranking.json"
    
    if not os.path.exists(META_FILE):
        print(f"Error: {META_FILE} not found. Please download it from jmcauley.ucsd.edu/data/amazon/ and place it here.")
    else:
        enrich_pretrain_data(PRETRAIN_FILE, META_FILE, OUTPUT_FILE)
