"""Execute and validate the notebook with the project's GPU virtual environment."""
import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


parser = argparse.ArgumentParser()
parser.add_argument('--validate-recompute', action='store_true', help='Exercise full feature extraction without replacing the delivered notebook')
args = parser.parse_args()
project = Path(__file__).resolve().parent.parent
notebook_path = project / 'notebooks' / '手工与CNN特征融合_SVM全过程.ipynb'
kernel_directory = project / '.venv_gpu' / 'share' / 'jupyter' / 'kernels'
runtime = project / 'notebooks' / '.runtime'
runtime.mkdir(exist_ok=True)
spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_directory)], ensure_native_kernel=False)
manager = KernelManager(kernel_name='cpchild-gpu', kernel_spec_manager=spec_manager,
                        connection_file=str(runtime / 'fusion-kernel.json'))
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
if args.validate_recompute:
    modified = 0
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            if 'USE_FEATURE_CACHE = True' in cell.source:
                cell.source = cell.source.replace('USE_FEATURE_CACHE = True', 'USE_FEATURE_CACHE = False')
                modified += 1
    if modified != 1:
        raise RuntimeError('Expected exactly one feature-cache switch')
client = NotebookClient(notebook, km=manager, kernel_name='cpchild-gpu', timeout=600,
                        resources={'metadata': {'path': str(project)}})
try:
    client.execute()
except Exception:
    # Save the executed portion on error as well, so diagnostics remain inspectable.
    diagnostic_path = runtime / 'recompute_diagnostics.ipynb' if args.validate_recompute else notebook_path
    nbformat.write(notebook, diagnostic_path)
    raise
nbformat.validate(notebook)
executed = sum(cell.cell_type == 'code' and cell.execution_count is not None for cell in notebook.cells)
errors = [output for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs
          if output.get('output_type') == 'error']
if errors:
    raise RuntimeError(f'{len(errors)} notebook cells produced errors')
if args.validate_recompute:
    outputs = project / 'notebook_outputs'
    run = max((path for path in outputs.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    metadata = {'executed_code_cells': executed, 'errors': len(errors), 'feature_cache_used': False, 'output_directory': str(run)}
    (run / 'full_recompute_validation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Full recomputation validation:', json.dumps(metadata, ensure_ascii=False))
else:
    nbformat.write(notebook, notebook_path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    html_path = notebook_path.with_suffix('.html')
    html_path.write_text(html, encoding='utf-8')
    print('Notebook:', notebook_path)
    print('HTML:', html_path)
print(f'Completed: {executed} executed code cells, {len(errors)} errors')
