#!/bin/bash

# Clone the repository
git clone https://github.com/dna-llm/BEND.git

# Install the requirements
pip install -r /BEND/requirements.txt

# Install the BEND package
pip install -e /BEND

# Install pysam
pip install pysam

# Download BEND
python /BEND/scripts/download_bend.py

# Precompute embeddings
python /BEND/scripts/precompute_embeddings.py model=virus_pythia_1.2M_1024_compliment task=gene_finding

# Train on task
python /BEND/scripts/train_on_task.py --config-name gene_finding embedder=virus_pythia_1.2M_1024_compliment
