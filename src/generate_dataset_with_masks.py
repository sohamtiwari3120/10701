from transformers import AutoTokenizer
from config import hparams
import argparse
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
    for ind in (j for j,e in enumerate(main_list) if e==sub_list[0]):
        if main_list[ind:ind+sub_list_len]==sub_list:
            results.append((ind,ind+sub_list_len-1))

    return results

def filter_out_special_tokens(id):
    return id not in hp.special_token_ids

def main(args, data_path = hp.train_claims_json):
    tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
    patent_id_keyphrases_path_dict = generate_keyphrases_paths_dict()
    data = json.load(open(data_path))
    data_ids = list(data.keys())
    ctr = args.start_index
    if args.num_lines == -1:
        data_ids = data_ids[args.start_index:]
    else:
        data_ids = data_ids[args.start_index:args.start_index + args.num_lines]
    print(len(data_ids))
    os.makedirs(args.output_dir, exist_ok=True)
    for _, id  in enumerate(tqdm(data_ids)):
        # for each patent application
        claims_text_list = data[id]
        ctr += 1
        if id not in patent_id_keyphrases_path_dict:
            continue
        claims_kp = json.load(open(patent_id_keyphrases_path_dict[id]))['claims'][0]
        if not claims_kp:
            continue
        claims_kp_tokenized = tokenizer(claims_kp)
        attention_masks_list = []
        claims_tokenized_list = []
        claims_len_list = []
        for claims_text in claims_text_list:
            claims_tokenized = tokenizer(claims_text)
            attention_mask = [0] * len(claims_tokenized['input_ids'])
            for _, kp_tokenized in enumerate(claims_kp_tokenized['input_ids']):
                filtered_kp_ids = list(filter(filter_out_special_tokens, kp_tokenized))
                mask_indices = find_sublist_index(claims_tokenized['input_ids'], filtered_kp_ids)
                for start, end in mask_indices:
                    attention_mask[start: end+1] = [1]*(end+1 - start)
            attention_masks_list.append(attention_mask)
            claims_tokenized_list.append(claims_tokenized)
            claims_len_list.append(len(attention_mask))
        torch.save({'id': id, 'claims_list': claims_text_list, 'claims_tokenized_list': claims_tokenized_list, 'attention_masks_list': attention_masks_list, 'claims_len_list': claims_len_list}, os.path.join(args.output_dir, f'{ctr-1}_{id}.pt'))

        # final_data[id] = {'id': id, 'claims_list': claims_text_list, 'claims_tokenized_list': [], 'attention_masks_list': [], 'claims_len_list': []}
    # torch.save(final_data, 'tokenized_claims_with_attention_mask.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir',
                        type=str, default=hp.tokenized_attn_mask_output_dir)
    parser.add_argument('-si', '--start_index', type=int, default=0)
    parser.add_argument('-nl', '--num_lines', type=int, default=-1)
    args = parser.parse_args()
    main(args)