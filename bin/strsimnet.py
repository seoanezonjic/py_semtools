#! /usr/bin/env python

import argparse
import sys
import os

ROOT_PATH=os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_PATH, '..'))

from py_semtools.sim_handler import *
#########################################################
# Define useful functions
#########################################################

def load_table_file(input_file, splitChar = "\t", targetCol = 1, filterCol = -1, filterValue = None):
  texts = []
  with open(input_file) as f:
    for line in f:
      row = line.rstrip().split(splitChar)
      if filterCol >= 0 and row[filterCol] != filterValue: continue 
      texts.append(row[targetCol]) 
    # Remove uniques
    texts = list(set(texts))
    return texts

#########################################################
# OPT parser
#########################################################
parser = argparse.ArgumentParser(description='Perform text similarity analysis')
parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
          help="Input OMIM diseases file.")
parser.add_argument("-s", "--split_char", dest="split_char", default="\t", 
          help="Character for splitting input file. Default: tab.")
parser.add_argument("-c", "--column", dest="cindex", default=0, type=int, 
          help="Column index wich contains texts to be compared. Default: 0.")
parser.add_argument("-C", "--filter_column", dest="findex", default=-1, type=int, 
          help="[OPTIONAL] Column index wich contains to be used as filters. Default: -1.")
parser.add_argument("-f", "--filter_value", dest="filter_value", default=None, 
          help="[OPTIONAL] Value to be used as filter.")
parser.add_argument("-r", "--remove_chars", dest="rm_char", default="", 
          help="Chars to be excluded from comparissons.")
parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
          help="Output similitudes file.")
options = parser.parse_args()

#########################################################
# MAIN
#########################################################
texts2compare = load_table_file(input_file = options.input_file,
                                 splitChar = options.split_char,
                                 targetCol = options.cindex,
                                 filterCol = options.findex,
                                 filterValue = options.filter_value)
# Obtain all Vs all
similitudes_AllVsAll = similitude_network(texts2compare, charsToRemove = options.rm_char)

# Iter and store
with open(options.output_file, "w") as f:
  for item, item_similitudes in similitudes_AllVsAll.items():
    for item2, sim in item_similitudes.items():
      f.write("\t".join([item, item2 , str(sim)]) + "\n" )