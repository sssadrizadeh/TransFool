import argparse

def TransFool_parser():
    parser = argparse.ArgumentParser(description="TransFool Attack")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result", type=str,
        help="folder for loading trained models")

    # Data
    parser.add_argument("--dataset_name", default="wmt14", type=str,
        choices=["wmt14", "opus100"],
        help="translation dataset to use")
    parser.add_argument("--dataset_config_name", default="fr-en", type=str,
        choices=["fr-en", "de-en", "en-zh", "cs-en", "ru-en"],
        help="config of the translation dataset to use")
    parser.add_argument("--num_samples", default=100, type=int,
        help="number of samples to attack")
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--part", default="", type=str,
        help="dataset part to attack")
        
    # Model  
    parser.add_argument("--model_name", default="marian", type=str,
        choices=["marian", "mbart", "marian_adv_adv", "marian_adv_all"],
        help="model which we have its gradient: target NMT model in white-box attack or reference NMT model in black-box attack")
    parser.add_argument("--source_lang", default="en", type=str,
        choices=["en"],
        help="source language")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr", "de", "zh", "ru", "cs"],
        help="target language") 
    parser.add_argument("--black_model_name", default="mbart", type=str,
        choices=["marian", "mbart"],
        help="target NMT model in black_box attack")
    parser.add_argument("--black_target_lang", default="fr", type=str,
        choices=["fr", "de"],
        help="target language for two language attack") 


    # Attack setting
    parser.add_argument("--LM", default="gpt2", type=str,
        choices=["gpt2", "fine_tune", "no_LM"],
        help="LM model to use for similarity")
    parser.add_argument("--w_sim", default=20.0, type=float,
        help="similarity loss coefficient")
    parser.add_argument("--w_perp", default=1.8, type=float,
        help="LM loss coefficient")
    parser.add_argument("--lr", default=0.016, type=float,
        help="learning rate")
    parser.add_argument("--mode", default="avg", type=str,
        choices=["avg", "idf"],
        help="score mode")
    parser.add_argument("--bleu", default=0.4, type=float,
        help="bleu score ratio for stopping criteria")
    parser.add_argument("--SAR", default=0.5, type=float,
        help="bleu score ratio for success attack rate")
    
    return parser