#!/usr/bin/env bash
source ~soft_bio_267/initializes/init_python
#mkdir -p py_venv
#python -m venv py_venv --system-site-packages
#source py_venv/bin/activate
#pip install -e ~/dev_py/py_semtools
./launch_plots.py -p ./data/profiles.txt -O HPO -t ./template.txt -o ./profiles_freqs
