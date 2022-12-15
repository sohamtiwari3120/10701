class hparams:
    def __init__(self) -> None:
        self.patent_jsonl_path = "/home/sohamdit/10701-files/10701/data/patent_acceptance_pred_subs=0.2.jsonl"
        # self.patent_jsonl_path = "/home/arnaik/ml_project/data/patent_acceptance_pred_subs=0.2_no_desc.jsonl"
        self.output_dir = "/home/sohamdit/10701-files/10701/data/keyphrases2/"
        self.tokenized_attn_mask_output_dir = "/home/sohamdit/10701-files/10701/src/tokenized_attention_masks/"
        self.launch_path = "/home/sohamdit/10701-files/ai-research-keyphrase-extraction"
        self.config_ini_path = "/home/sohamdit/10701-files/ai-research-keyphrase-extraction/config.ini"
        # keyphrase extraction
        self.num_keyphrases_to_extract = 10000
        self.mmr_beta = 0.55
        self.alias_threshold_to_group_candidate_kps = 0.7
        self.jsonl_keys_for_generating_kp = [
            "title", "abstract", "summary", "claims", "background"]
        self.fail_tries = 5
        # sentence transformer
        self.model_name = 'anferico/bert-for-patents'
        # self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        # data paths
        self.train_pkl = '/home/sohamdit/10701-files/10701/src/train_data_topic_groups.pkl'
        self.val_pkl = '/home/sohamdit/10701-files/10701/src/val_data_topic_groups.pkl'
        self.test_pkl = '/home/sohamdit/10701-files/10701/src/test_data_topic_groups.pkl'
        
        self.train_claims_json = "/home/arnaik/ml_project/data/train_claims.json"
        self.test_claims_json = "/home/arnaik/ml_project/data/test_claims.json"
        self.val_claims_json = "/home/arnaik/ml_project/data/val_claims.json"
        # train args
        # finetune
        self.evaluation_strategy = "epoch"
        self.save_strategy = "epoch"
        self.learning_rate = 2e-5
        self.num_train_epochs = 3  # early stopping
        self.weight_decay = 0.01
        self.push_to_hub = True
        self.report_to = 'wandb'
        self.load_best_model_at_end = True
        self.metric_for_best_model = "f1"
        self.greater_is_better = True
        self.early_stopping_patience = 10
        self.early_stopping_threshold = 0.005
        self.per_device_train_batch_size = 8
        # self.output_dir = "/home/sohamdit/10701-files/10701/src/results"
        # paths to parsed claims jsons
        self.validation_claims_path = "/home/arnaik/ml_project/data/val_claims.json"
        self.train_claims_path = "/home/arnaik/ml_project/data/train_claims.json"
        self.test_claims_path = "/home/arnaik/ml_project/data/test_claims.json"
        self.special_token_ids = [0, 100, 101, 102, 103]
