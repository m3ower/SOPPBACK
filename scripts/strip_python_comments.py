#!/usr/bin/env python3
"""
Strip comments from Python files recursively.
Uses tokenize to safely remove comments while preserving strings and code.
Docstrings are left intact (to avoid changing runtime semantics when used).
"""
from __future__ import annotations
import io
import os
import sys
import tokenize

SKIP_DIRS = {'.git', '__pycache__', 'env', '.venv', 'venv', 'dist', 'build'}

def strip_comments_from_code(code: str) -> str:
	out = []
	last_lineno = -1
	last_col = 0
	tokgen = tokenize.generate_tokens(io.StringIO(code).readline)
	for tok_type, tok_str, (srow, scol), (erow, ecol), _ in tokgen:
		if tok_type == tokenize.COMMENT:
			# drop comments entirely
			continue
		if tok_type == tokenize.NL:
			# standalone newline (blank or end-of-line from comment) keep it
			out.append(tok_str)
			continue
		if srow > last_lineno:
			last_col = 0
		if scol > last_col:
			out.append(" " * (scol - last_col))
		out.append(tok_str)
		last_lineno = erow
		last_col = ecol
	return "".join(out)

def process_file(path: str) -> None:
	try:
		with open(path, 'r', encoding='utf-8') as f:
			src = f.read()
		out = strip_comments_from_code(src)
		if out != src:
			with open(path, 'w', encoding='utf-8') as f:
				f.write(out)
	except Exception as e:
		print(f"[strip] Failed {path}: {e}")

def walk(root: str) -> None:
	for dirpath, dirnames, filenames in os.walk(root):
		dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
		for fn in filenames:
			if fn.endswith('.py'):
				process_file(os.path.join(dirpath, fn))

if __name__ == '__main__':
	root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
	print(f"[strip] Python: walking {root}")
	walk(root)
	print("[strip] Python: done")

