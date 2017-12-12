# import os
# import json

# postreqdata = json.loads(open(os.environ['req']).read())
# response = open(os.environ['res'], 'w')
# response.write("hello world from "+postreqdata['name'])
# response.close()

import os
import json
import pandas as pd

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import ngrams

from functools import partial


response = open(os.environ['res'], 'w')
response.write("hello world from ")
response.close()

def tokenize(name, x):
    output = list()
    my_list = map(lambda x:x.lower(), word_tokenize(str(x[name])))
    output.append(my_list)
    return output

def build(name, inv, df):
    df_de_tokens = df[[name]].apply(partial(tokenize, name), axis = 1)    

    for i, map_list in df_de_tokens[name].iteritems():
        tokenized_string = [tmp_token for tmp_token in map_list]

        for token in tokenized_string:
            c = tokenized_string.count(token)
            tokenized_string.count(token)
            if not token in inv:
                # Initializing the index and the counter in the corpus
                inv[token] = [[i], 0, [c/len(tokenized_string)]]
                continue
            # ID of the document in which the token appears
            inv[token][0].append(i)
            # Count of appearances in all the corpus
            inv[token][1] += 1
            # Frequence of appearance within the document
            inv[token][2].append(c/len(tokenized_string))

    return inv

def build_index(df_raw):
    inv_index = dict()

    inv_index = build("Description", inv_index, df_raw)
    inv_index = build("Solution Name", inv_index, df_raw)
    inv_index = build("Account", inv_index, df_raw)
    
    return inv_index

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def search2(search_term, inv, df_raw, verbose = True):
    search_tokens = [token.lower() for token in word_tokenize(search_term)]
    index_unions = set()
    
    def mean(x):
        return sum(x)/len(x)
    
    def intern_search(x):
        output = None
        if not x in inv:
            for k in inv.keys():
                if levenshtein(k, x) <= 5:
                    x = k
                    
        output = inv[x]
        
        return output
    
    def ranking(x):
        return mean(x)
    
    try:
        output = None
        
        search_results = [result for result in map(intern_search, search_tokens)]
        docs, count, freqs = zip(*search_results)

        result_docs = list()
        result_freqs = list()

        [result_docs.extend(doc) for doc in docs]
        [result_freqs.extend(freq) for freq in freqs ]

        aggregated_results = [x for x in zip(result_docs, result_freqs)]

        # Summarizing the results that share the same document
        results = dict()
        for k, v in aggregated_results:
            if not k in results:
                results[k] = list()
            results[k].append(v)


        # A simple formula to rank the results (using the average over all results)
        results = [x for x in zip(results.keys(), [x for x in map(ranking, results.values())])]
        best = sorted(results, key = lambda x: x[1], reverse = True)


        docs, _ = zip(*best)
        output = df_raw.loc[list(docs)][["Solution Name","Description"]] 
    except:
        output = pd.DataFrame()
        
    return output
    
#======================
# df_result = search2("blue zone", inv = inv_index, df_raw=df_de, verbose = True)
# df_result
            
# inv_index = build_index(df_de)