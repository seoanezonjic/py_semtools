#! /usr/bin/env bash

. ~soft_bio_267/initializes/init_python
mkdir -p envf
#python -m venv envf --system-site-packages
source envf/bin/activate
#pip install -e ~/dev_py/py_semtools/
#pip install -e ~/dev_py/py_report_html

report_html -t template.txt -d ./example


