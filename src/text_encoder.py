import os
import json
import pickle
from tqdm import tqdm
from utils import read_tab_file
import numpy as np

import torch
from transformers import BertTokenizer, BertModel
import logging

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_text_vec(text):
    masked_text = "[CLS] " + text + " [SEP]"

    tokenized_text = tokenizer.tokenize(masked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    indexed_tokens = indexed_tokens[:510]  # bert word count 512 = 510 + [CLS] + [SEP]
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')

    model.to('cuda')

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_vecs = hidden_states[-2][0]
    text_embedding = torch.mean(token_vecs, dim=0)
    return text_embedding.cpu()



def get_entity_description(imputation='name'):
    ent_dscp_path = '../src_data/WN9/ent_dscp.txt'
    ents, descriptions = read_tab_file(ent_dscp_path)
    ent_dscp = {}
    ent_id_path = '../data/WN9/ent_id'
    id_ents, ids = read_tab_file(ent_id_path)
    ent_name_path = '../src_data/WN9/ent_name.txt'
    name_ents, names = read_tab_file(ent_name_path)
    for ent in tqdm(id_ents):
        n_ent = ent
        # n_ent = 'n' + ent  # some entity starts with an "n"
        try:
            if ent in ents and n_ent in name_ents:
                name = names[name_ents.index(n_ent)]
                ent_dscp[ent] = name + ': ' + descriptions[ents.index(ent)]
            elif n_ent in name_ents:
                name = names[name_ents.index(ent)]
                ent_dscp[ent] = name
            elif ent in ents:
                ent_dscp[ent] = descriptions[ents.index(ent)]
        except Exception as e:
            print(e)
    return ent_dscp


if __name__ == "__main__":

    imputation = 'name'
    ent_dscp = get_entity_description(imputation)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
    model.eval()
    model.cuda()

    ents = sorted(list(ent_dscp.keys()))

    array_filename = '../data/WN9/text_feature.pickle'

    dscp_vec = {}
    for ent in tqdm(ents):
        dscp = ent_dscp[ent]
        if dscp == '':
            dscp_vec[ent] = np.random.normal(size=(1, 768))
        else:
            vec = get_text_vec(dscp)
            dscp_vec[ent] = vec.cpu().numpy()

    dscp_array = np.array(list(dscp_vec.values()))
    with open(array_filename, 'wb') as out:
        pickle.dump(dscp_array, out)
