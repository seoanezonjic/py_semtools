import re
import sys

def get_white_similarity(textA, textB):
  string_A_pairs = get_string_pairs(textA)
  string_B_pairs = get_string_pairs(textB) 
  sim = len(set(string_A_pairs) & set(string_B_pairs)) * 2 / (len(string_A_pairs) + len(string_B_pairs))
  return sim

def longest_common_subsequence(xs, ys):
  # Uses the Ukkonen-Myers algorithm
  t = len(xs) + len(ys)
  front = [0] * (2 * t + 1)
  candidates = [None] * (2 * t + 1)
  for d in range(t + 1):
      for k in range(-d, d+1, 2):
          if k == -d or (k != d and front[t + k - 1] < front[t + k + 1]):
              index = t + k + 1
              x = front[index]
          else:
              index = t + k - 1
              x = front[index] + 1
          y = x - k
          chain = candidates[index]
          while x < len(xs) and y < len(ys) and xs[x] == ys[y]:
              chain = ((x, y), chain)
              x += 1
              y += 1
          if x >= len(xs) and y >= len(ys):
              result = []
              while chain:
                  result.append(chain[0])
                  chain = chain[1]
              result.reverse()
              return result
          front[t + k] = x
          candidates[t + k] = chain

def get_ukkonen_myers_similarity(textA, textB):
  # Original code: https://gist.github.com/sbp/c2b94bc3826f38fed037
  # Adapted to similarity index based in perl implementation: https://metacpan.org/release/MLEHMANN/String-Similarity-1.04/source/fstrcmp.c ( see return in c module)
  i = -1
  j = -1
  matches = longest_common_subsequence(textA, textB)
  #result = []
  char_changes = 0
  for (mi, mj) in matches:
      if mi - i > 1 or mj - j > 1:
          char_changes += (( mi - i - 1) + (mj - j - 1))
          #result.append((i + 1, mi - i - 1, j + 1, mj - j - 1))
          #result.append((i + 1, textA[i + 1:mi], j + 1, textB[j + 1:mj]))
      i = mi
      j = mj
  #print(result)
  sum_len = len(textA) + len(textB)
  result = (sum_len - char_changes)/sum_len
  return result


def get_string_pairs(string, k =2):
  num_kmers = len(string) - k + 1
  kmers = [ string[i:i+k] for i in range(num_kmers) ]
  return kmers

# Applies the WhiteSimilarity from 'text' package over two given texts
# Param:
# +textA+:: text to be compared with textB
# +textB+:: text to be compared with textA
# Returns the similarity percentage between [0,1]
def text_similitude(textA, textB, algorithm = 'white'):
  if algorithm == 'white':
    result = get_white_similarity(textA.strip(), textB.strip())
  elif algorithm == 'ukkonen_myers':
    result = get_ukkonen_myers_similarity(textA.strip(), textB.strip())
  return result

# Applies the WhiteSimilarity from 'text' package over two given text sets and returns the similitudes
# of the each element of the first set over the second set 
# Param:
# +textsA+:: text set to be compared with textsB
# +textsB+:: text set to be compared with textsA
# Returns the maximum similarity percentage between [0,1] for each element of textsA against all elements of textsB
def ctext_AtoB(textsA, textsB, algorithm = 'white'):
  # Calculate similitude
  similitudesA = []
  for fragA in textsA:
    similitudesA.append(max([ text_similitude(fragA, fragB, algorithm = algorithm) for fragB in textsB]))
  return similitudesA

# Applies the WhiteSimilarity from 'text' package over two given complex texts.
# Complex texts will be splitted and compared one by one from A to B and B to A
# Param:
# +textA+:: text to be compared with textB
# +textB+:: text to be compared with textA
# +splitChar+:: char to split text* complex names
# +charsToRemove+:: char (or chars set) to be removed from text to be compared
# Returns the similarity percentage between [0,1] obtained by bidirectional all Vs all similarity
def complex_text_similitude(textA, textB, splitChar = ";", charsToRemove = "", algorithm = 'white'):
  # Split&Clean both sets
  textA_splitted = textA.split(splitChar)
  textB_splitted = textB.split(splitChar)
  if len(charsToRemove) > 0:
    textA_splitted = clean_fragments(textA_splitted, charsToRemove)
    textB_splitted = clean_fragments(textB_splitted, charsToRemove)
  # Per each X elemnt, compare against all Y elements
  similitudesA = ctext_AtoB(textA_splitted, textB_splitted, algorithm = algorithm)
  similitudesB = ctext_AtoB(textB_splitted, textA_splitted, algorithm = algorithm)
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
def similitude_network(items_array, splitChar = ";", charsToRemove = "", unique = False, algorithm = 'white'):
  if len(items_array[0]) == 1: #One column case: [["Blaba"], ["Blala"], ["dadad"]] all_vs_all
    items_array = [item[0] for item in items_array] #Flattening the nested list
    sims_dict = calc_all_vs_all_sims(items_array, splitChar, charsToRemove, unique, algorithm = algorithm)
    sims = _convert_to_nested_lists(sims_dict)
  elif len(items_array[0]) == 2: #Two columns case: [["Blaba", "heyy"], ["heho", "hello"], ["dadad", "dad"]] only first_vs_second
    sims = calc_a_vs_b_sims(items_array, splitChar, charsToRemove, algorithm = algorithm)
  else:
    raise Exception(f"Wrong number of expected items. Expected 1 column or two columns, but got {len(items_array[0])}")
  return sims

def calc_all_vs_all_sims(items_array, splitChar = ";", charsToRemove = "", unique = False, algorithm = 'white'):
  # Remove repeated elements
  if unique: 
    items_array_unique = []
    for i in items_array:
      if i not in items_array_unique: items_array_unique.append(i)
    items_array = items_array_unique
  sims_dict = {}
  # Per each item into array => Calculate similitude
  while(len(items_array) > 1):
    current = items_array.pop(0)
    sims_dict[current] = {}
    for item in items_array:
      sims_dict[current][item] = complex_text_similitude(current,item,splitChar,charsToRemove, algorithm = algorithm)
  return sims_dict

def calc_a_vs_b_sims(items_array, splitChar = ";", charsToRemove = "", algorithm = 'white'):
  sims = []
  for item1, item2 in items_array:
    sim = complex_text_similitude(item1, item2, splitChar, charsToRemove, algorithm = algorithm)
    sims.append([item1, item2, sim])  
  return sims


########### UTILS FUNCTIONS

def _convert_to_nested_lists(dictionary):
  nested_list = []
  for key1, inner_dict in dictionary.items():
      for key2, value in inner_dict.items():
        nested_list.append([key1, key2 , value])
  return nested_list