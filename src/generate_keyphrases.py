import argparse
import sys
from tqdm import tqdm
import json
import time
from datautils import read_patent_jsonl
from config import hparams
hp = hparams()
sys.path.append(hp.launch_path)  # noqa
import launch  # noqa

embedding_distributor = launch.load_local_embedding_distributor(
    hp.config_ini_path)
pos_tagger = launch.load_local_corenlp_pos_tagger(hp.config_ini_path)

def extract_keyphrase(raw_text: str, num_keyphrase: int = hp.num_keyphrases_to_extract, fail_tries: int = hp.fail_tries):
    while True:
        try:
            kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, num_keyphrase,
                                            'en', beta=hp.mmr_beta, alias_threshold=hp.alias_threshold_to_group_candidate_kps)
            return kp1
        except Exception as e:
            fail_tries -= 1
            if fail_tries == 0:
                raise e
            time.sleep(0.5)


def generate_and_save_keyphrases(data, output_dir, start_index, fail_limit = 3):
    i = start_index
    for line in tqdm(data):
        output_data = {}
        id = line['id']
        output_data['id'] = id
        skip_to_next = False
        for key in hp.jsonl_keys_for_generating_kp:
            fail_count = 0
            loop = True
            while loop:
                try:
                    output_data[key] = extract_keyphrase(line[key])
                    loop = False
                except:
                    fail_count += 1
                    if fail_count == fail_limit:
                        loop = False
            if fail_count == fail_limit:
                skip_to_next = True
                print(f'[ERROR] skipping {i}')
                break
        if not skip_to_next:
            with open(output_dir+f"/{i}_{id}.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps(output_data)+"\n")
        i += 1


def read_keyphrases_file(file_path: str):
    data = read_patent_jsonl(file_path)
    return data


def generate_keyphrases(output_dir, start_index: int = 0, num_lines: int = -1):
    data = read_patent_jsonl(hp.patent_jsonl_path)
    if num_lines == -1:
        generate_and_save_keyphrases(data[start_index:], output_dir, start_index)
    else:
        end_index = min(start_index+num_lines, len(data))
        generate_and_save_keyphrases(
            data[start_index:end_index], output_dir, start_index)


def main(args):
    generate_keyphrases(
        args.output_dir, start_index=args.start_index, num_lines=args.num_lines)
    # read_keyphrases_file(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir',
                        type=str, default=hp.output_dir)
    parser.add_argument('-si', '--start_index', type=int, default=0)
    parser.add_argument('-nl', '--num_lines', type=int, default=-1)
    args = parser.parse_args()
    main(args)
