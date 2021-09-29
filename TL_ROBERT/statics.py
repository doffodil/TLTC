import numpy as np
import pandas as pd

def load_data(root_path, domain):
    positive_data_file = root_path + domain + "_positive_1000.txt"
    negative_data_file = root_path + domain + "_negative_1000.txt"
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8', errors='ignore').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8', errors='ignore').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    text = positive_examples + negative_examples
    label = [0] * len(positive_examples) + [1] * len(negative_examples)
    data_df  = pd.DataFrame({'text':text, 'label':label})
    return data_df

data_path = "../data/semantic_data/"
source_domain = 'book'
source_data_df = load_data(data_path, source_domain)
text = source_data_df.text.tolist()
source_data_df['text_len'] = [len(s) for s in text]
print()