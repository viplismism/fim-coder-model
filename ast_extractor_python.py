#!/usr/bin/env python3
"""Python AST extractor for FIM dataset generation."""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any


NODE_TYPES = [
    'FunctionDef',
    'AsyncFunctionDef',
    'ClassDef',
    'Import',
    'ImportFrom',
    'Assign',
    'If',
    'For',
    'AsyncFor',
    'While',
    'With',
    'AsyncWith',
    'Try',
    'Match',
]


class ASTExtractor(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.nodes = []

    def generic_visit(self, node: ast.AST):
        node_type = type(node).__name__
        if node_type in NODE_TYPES:
            self.nodes.append({
                'type': node_type,
                'lineno': getattr(node, 'lineno', None),
                'end_lineno': getattr(node, 'end_lineno', None),
            })
        super().generic_visit(node)


def extract_file(file_path: Path) -> dict[str, Any] | None:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
    except (UnicodeDecodeError, PermissionError, OSError):
        return None

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return None

    extractor = ASTExtractor(source)
    extractor.visit(tree)

    nodes = []
    for node in extractor.nodes:
        lineno = node['lineno']
        end_lineno = node['end_lineno']
        if lineno and end_lineno:
            nodes.append({node['type']: {'span': f'{lineno}:{end_lineno}'}})

    return {
        'code': source,
        'ast': nodes,
    }


def scan_directory(root: Path, extensions: list[str]) -> list[Path]:
    files = []
    for ext in extensions:
        files.extend(root.rglob(f'*{ext}'))
    return files


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <output_json>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not source_dir.exists():
        sys.exit(f"Error: {source_dir} does not exist")

    print(f"Scanning for Python files in {source_dir}...")
    py_files = scan_directory(source_dir, ['.py'])
    print(f"Found {len(py_files)} Python files")

    ast_data = {}
    node_counts = {}

    for i, file_path in enumerate(py_files):
        if i > 0 and i % 500 == 0:
            print(f"  Processed {i}/{len(py_files)} files...")

        rel_path = str(file_path.relative_to(source_dir))

        result = extract_file(file_path)
        if result and result['ast']:
            ast_data[rel_path] = result
            for node in result['ast']:
                node_type = next(iter(node.keys()))
                node_counts[node_type] = node_counts.get(node_type, 0) + 1

    print(f"\nProcessed: {len(ast_data)} files successfully")
    print("\nNode counts:")
    for node_type, count in sorted(node_counts.items(), key=lambda x: -x[1]):
        print(f"  {node_type}: {count}")

    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(ast_data, f)
    print(f"Done! Output: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
