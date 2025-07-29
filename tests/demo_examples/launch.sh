#!/usr/bin/env bash
source ~soft_bio_267/initializes/init_python
#mkdir -p py_venv
#python -m venv venv --system-site-packages
#source venv/bin/activate
#pip install -e ~/dev_py/py_exp_calc
#pip install -e ~/dev_py/py_report_html
#pip install -e ~/dev_py/py_semtools
#pip install --upgrade kaleido
#pip install --upgrade plotly

./launch_plots.py -p ./data/profiles.txt -O HPO -t ./template.txt -o ./profiles_freqs
