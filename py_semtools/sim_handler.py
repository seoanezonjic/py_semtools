import re
import sys

def get_white_similarity(textA, textB):
  string_A_pairs = get_string_pairs(textA)
  string_B_pairs = get_string_pairs(textB) 
  sim = len(set(string_A_pairs) & set(string_B_pairs)) * 2 / (len(string_A_pairs) + len(string_B_pairs))
  return sim

def get_string_pairs(string, k =2):
  num_kmers = len(string) - k + 1
  kmers = [ string[i:i+k] for i in range(num_kmers) ]
  return kmers

# Applies the WhiteSimilarity from 'text' package over two given texts
# Param:
# +textA+:: text to be compared with textB
# +textB+:: text to be compared with textA
# Returns the similarity percentage between [0,1]
def text_similitude(textA, textB):
  return get_white_similarity(textA.strip(), textB.strip())

# Applies the WhiteSimilarity from 'text' package over two given text sets and returns the similitudes
# of the each element of the first set over the second set 
# Param:
# +textsA+:: text set to be compared with textsB
# +textsB+:: text set to be compared with textsA
# Returns the maximum similarity percentage between [0,1] for each element of textsA against all elements of textsB
def ctext_AtoB(textsA, textsB):
  # Calculate similitude
  similitudesA = []
  for fragA in textsA:
    similitudesA.append(max([ text_similitude(fragA, fragB) for fragB in textsB]))
  return similitudesA

# Applies the WhiteSimilarity from 'text' package over two given complex texts.
# Complex texts will be splitted and compared one by one from A to B and B to A
# Param:
# +textA+:: text to be compared with textB
# +textB+:: text to be compared with textA
# +splitChar+:: char to split text* complex names
# +charsToRemove+:: char (or chars set) to be removed from text to be compared
# Returns the similarity percentage between [0,1] obtained by bidirectional all Vs all similarity
def complex_text_similitude(textA, textB, splitChar = ";", charsToRemove = ""):
  # Split&Clean both sets
  textA_splitted = textA.split(splitChar)
  textB_splitted = textB.split(splitChar)
  if len(charsToRemove) > 0:
    textA_splitted = clean_fragments(textA_splitted, charsToRemove)
    textB_splitted = clean_fragments(textB_splitted, charsToRemove)
  # Per each X elemnt, compare against all Y elements
  similitudesA = ctext_AtoB(textA_splitted, textB_splitted)
  similitudesB = ctext_AtoB(textB_splitted, textA_splitted)
  # Obtain bidirectional similitude
  similitudesA = sum(similitudesA) / len(similitudesA)
  similitudesB = sum(similitudesB) / len(similitudesB)
  bidirectional_sim = (similitudesA + similitudesB) / 2
  return bidirectional_sim

def clean_fragments(fragments, charsToRemove):
  cleaned_fragments = []
  for frag in fragments:
    f = re.sub(r"["+charsToRemove+"]", '', frag)
    if len(f) > 0: cleaned_fragments.append(f)
  return cleaned_fragments
  

# Applies the WhiteSimilarity from 'text' package over all complex text stored into an array.
# Complex texts will be splitted and compared one by one from A to B and B to A
# Param:
# +items_array+:: text elements to be compared all against others
# +splitChar+:: char to split text* complex names
# +charsToRemove+:: char (or chars set) to be removed from texts to be compared
# +unique+:: boolean flag which indicates if repeated elements must be removed
# Returns the similarity percentage for all elements into array
def similitude_network(items_array, splitChar = ";", charsToRemove = "", unique = False):
  # Remove repeated elements
  if unique: 
    items_array_unique = []
    for i in items_array:
      if i not in items_array_unique: items_array_unique.append(i)
    items_array = items_array_unique
  sims = {}
  # Per each item into array => Calculate similitude
  while(len(items_array) > 1):
    current = items_array.pop(0)
    sims[current] = {}
    for item in items_array:
      sims[current][item] = complex_text_similitude(current,item,splitChar,charsToRemove)
  return sims
