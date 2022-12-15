from transformers import AutoTokenizer
from config import hparams
from datautils import read_patent_jsonl
hp = hparams()
from tqdm import tqdm
import os
import glob
import json
import torch

def generate_keyphrases_paths_dict(keyphrases_folder=hp.output_dir):
    patent_id_keyphrases_path_dict = {}
    for path in tqdm(glob.glob(os.path.join(keyphrases_folder, "*.jsonl"))):
        patent_id = os.path.basename(path)[:-6].split('_')[1]
        patent_id_keyphrases_path_dict[patent_id] = path
    return patent_id_keyphrases_path_dict

def find_sublist_index(main_list, sub_list):
    results=[]
    sub_list_len=len(sub_list)
    for ind in (i for i,e in enumerate(main_list) if e==sub_list[0]):
        if main_list[ind:ind+sub_list_len]==sub_list:
            results.append((ind,ind+sub_list_len-1))

    return results

def filter_out_special_tokens(id):
    return id not in hp.special_token_ids

def main(data_path = hp.train_claims_json):
    tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
    patent_id_keyphrases_path_dict = generate_keyphrases_paths_dict()
    # breakpoint()
    # data_list = read_patent_jsonl(hp.patent_jsonl_path)
    data = json.load(open(data_path))
    data_ids = list(data.keys())
    final_data = {}
    breakpoint()
    for i in tqdm(range(10)):
        # for each patent application
        breakpoint()
        id = data_ids[i]
        claims_text_list = data[id]
        if id not in patent_id_keyphrases_path_dict:
            continue
        claims_kp = json.load(open(patent_id_keyphrases_path_dict[id]))['claims'][0]
        claims_kp_tokenized = tokenizer(claims_kp)
        attention_masks_list = []
        claims_tokenized_list = []
        claims_len_list = []
        for claims_text in claims_text_list:
            claims_tokenized = tokenizer(claims_text, return_tensors=True)
            attention_mask = [0] * len(claims_tokenized['input_ids'])
            for i, kp_tokenized in enumerate(claims_kp_tokenized['input_ids']):
                filtered_kp_ids = list(filter(filter_out_special_tokens, kp_tokenized))
                mask_indices = find_sublist_index(claims_tokenized['input_ids'], filtered_kp_ids)
                for start, end in mask_indices:
                    attention_mask[start: end+1] = [1]*(end+1 - start)
            attention_masks_list.append(attention_mask)
            claims_tokenized_list.append(claims_tokenized)
            claims_len_list.append(len(attention_mask))
        torch.save({'id': id, 'claims_list': claims_text_list, 'claims_tokenized_list': claims_tokenized_list, 'attention_masks_list': attention_masks_list, 'claims_len_list': claims_len_list}, f'./tokenized_attention_masks/{id}.pt')
        # final_data[id] = {'id': id, 'claims_list': claims_text_list, 'claims_tokenized_list': [], 'attention_masks_list': [], 'claims_len_list': []}
    breakpoint()
    # torch.save(final_data, 'tokenized_claims_with_attention_mask.pt')

if __name__ == '__main__':
    main()