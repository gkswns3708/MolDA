"""DEPRECATED: This file has been replaced by the dataset_generation/ package.

Use instead:
    python -m dataset_generation.run --config smiles
    python -m dataset_generation.run --config smiles selfies --toy 100

See src/dataset_generation/ for the refactored code with cross-source decontamination.
"""

import sys

print(
    "WARNING: dataset_generator.py is deprecated.\n"
    "Use: python -m dataset_generation.run --config <config_name>\n"
    "See src/dataset_generation/ for details.",
    file=sys.stderr,
)

# Re-export for any potential backward compatibility
from dataset_generation.run import main  # noqa: F401

if __name__ == "__main__":
    main()
