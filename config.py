class hparams:
    def __init__(self) -> None:
        self.patent_jsonl_path = "/home/sohamdit/10701-files/10701/data/patent_acceptance_pred_subs=0.2_no_desc.jsonl"
        # self.patent_jsonl_path = "/home/arnaik/ml_project/data/patent_acceptance_pred_subs=0.2_no_desc.jsonl"
        self.output_dir = "/home/sohamdit/10701-files/10701/data/keyphrases/"
        self.launch_path = "/home/sohamdit/10701-files/ai-research-keyphrase-extraction"
        self.config_ini_path = "/home/sohamdit/10701-files/ai-research-keyphrase-extraction/config.ini"
        # keyphrase extraction
        self.num_keyphrases_to_extract = 10000
        self.mmr_beta=0.55
        self.alias_threshold_to_group_candidate_kps=0.7
        self.jsonl_keys_for_generating_kp = [ "title", "abstract", "summary", "claims", "background"]