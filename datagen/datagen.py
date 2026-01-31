#!/usr/bin/env python3
"""AST-based FIM dataset generator for Rust code."""

import json, os, random, sys, time
from pathlib import Path

TRAIN_SPLIT = 0.98
CHUNK_SIZES = [8000, 16000]
MIN_SPAN, MAX_SPAN_RATIO = 10, 10

def parse_span(span_str):
    try:
        parts = span_str.split(':')
        return (int(parts[0]), int(parts[1])) if len(parts) == 2 else (None, None)
    except (ValueError, AttributeError):
        return None, None

def line_to_char(content, line_num, _cache={}):
    cache_key = id(content)
    if cache_key not in _cache:
        positions = [0]
        for line in content.split('\n'):
            positions.append(positions[-1] + len(line) + 1)
        _cache[cache_key] = positions
    positions = _cache[cache_key]
    return 0 if line_num <= 0 else positions[min(line_num - 1, len(positions) - 1)]

def extract_blocks(content, ast_nodes):
    blocks = []
    max_span = max(len(content) // MAX_SPAN_RATIO, 30)
    
    for node in ast_nodes:
        node_type = next(iter(node.keys()))
        data = node[node_type]
        
        if 'span' in data:
            s, e = parse_span(data['span'])
            if s and e:
                start, end = line_to_char(content, s), line_to_char(content, e + 1)
                if MIN_SPAN <= (end - start) <= max_span:
                    blocks.append({'type': node_type, 'start': start, 'end': end})
        
        # Extract function body - find the body by locating opening brace in function span
        if node_type == 'Function' and 'body' in data and 'span' in data:
            s, e = parse_span(data['span'])
            if s and e:
                func_start = line_to_char(content, s)
                func_end = min(line_to_char(content, e + 1), len(content))
                func_content = content[func_start:func_end]
                
                # Find the function body by locating the opening brace
                brace_pos = func_content.find('{')
                if brace_pos != -1:
                    body_start = func_start + brace_pos
                    # Find matching closing brace (the function ends with it)
                    body_end = func_end
                    # Trim trailing whitespace/newlines after closing brace
                    while body_end > body_start + 1 and body_end <= len(content) and content[body_end-1] in ' \t\n':
                        body_end -= 1
                    
                    body_len = body_end - body_start
                    if MIN_SPAN <= body_len <= max_span:
                        blocks.append({'type': 'FunctionBody', 'start': body_start, 'end': body_end})
    
    return blocks

def create_fim_sample(content, node, file_path):
    max_len = random.choice(CHUNK_SIZES)
    mid_len = node['end'] - node['start']
    if mid_len <= 0 or mid_len > max_len:
        return None
    
    remaining = max_len - mid_len
    prefix_len = remaining // 2
    prefix_start = max(0, node['start'] - prefix_len)
    suffix_end = min(len(content), node['end'] + (remaining - prefix_len))
    
    return {
        'filePath': file_path,
        'prefix': content[prefix_start:node['start']],
        'middle': content[node['start']:node['end']],
        'suffix': content[node['end']:suffix_end],
        'nodeType': node['type']
    }

def generate_dataset(ast_data, output_dir, prefix="ast"):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = output_dir / f'{prefix}_train.jsonl', output_dir / f'{prefix}_test.jsonl'
    
    stats = {'files': 0, 'success': 0, 'train': 0, 'test': 0}
    start, total = time.time(), len(ast_data)
    
    print(f"Processing {total:,} files → {output_dir}")
    
    with open(train_path, 'w') as train_f, open(test_path, 'w') as test_f:
        for i, (path, data) in enumerate(ast_data.items()):
            stats['files'] += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start
                speed = stats['files'] / elapsed if elapsed > 0 else 0
                print(f"  [{stats['files']:,}/{total:,}] {speed:.0f} files/s | train={stats['train']:,} test={stats['test']:,}")
            
            if 'code' not in data:
                continue
            
            content = data['code']
            for block in extract_blocks(content, data.get('ast', [])):
                sample = create_fim_sample(content, block, path)
                if not sample:
                    continue
                
                line = json.dumps(sample) + '\n'
                if random.random() < TRAIN_SPLIT:
                    train_f.write(line)
                    stats['train'] += 1
                else:
                    test_f.write(line)
                    stats['test'] += 1
            
            stats['success'] += 1
    
    elapsed = time.time() - start
    total_samples = stats['train'] + stats['test']
    print(f"\nDone in {elapsed:.1f}s | {stats['success']:,} files | {total_samples:,} samples (train={stats['train']:,}, test={stats['test']:,})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate FIM training data from AST")
    parser.add_argument("--ast", type=str, default=None, help="Path to AST JSON file (default: data/ast.json)")
    parser.add_argument("--output_prefix", type=str, default="ast", help="Output file prefix (default: ast)")
    args = parser.parse_args()
    
    random.seed(42)
    project_root = Path(__file__).parent.parent
    
    ast_path = Path(args.ast) if args.ast else project_root / 'data' / 'ast.json'
    
    if not ast_path.exists():
        sys.exit(f"ERROR: {ast_path} not found")
    
    print(f"Loading {ast_path}...")
    with open(ast_path) as f:
        ast_data = json.load(f)
    print(f"Loaded {len(ast_data):,} files\n")
    
    generate_dataset(ast_data, project_root / 'data', args.output_prefix)