#!/bin/bash

source ../.bashrc1
echo Running OmegaFold with ../out.fasta results will be saved at /agh/projects/noelia/sebastian
python main.py ../out.fasta  .. --subbatch_size 27 --weights_file /agh/projects/noelia/sebastian/OmegaFold/.cache
