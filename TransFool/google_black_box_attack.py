import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


import torch
import pickle
from parser import TransFool_parser
import time
from utils.load import load_tokenized_dataset, load_model_tokenizer, load_LM_FC
from utils.attack import google_attack, save_outputs
from datasets import load_dataset
from attacker import Attacker


def main(args):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the NMT model and its tokenizer
    model, tokenizer = load_model_tokenizer(args.model_name, args.source_lang, args.target_lang, device)

    # load tokenized dataset 
    _, tokenized_dataset = load_tokenized_dataset(tokenizer,args.source_lang,args.target_lang,args.dataset_name,args.dataset_config_name)

    # create Attacker
    attacker = Attacker(args, model, tokenizer, tokenized_dataset, device, attack_type='google')    

    # load LM model and FC layer
    LM_model, fc = load_LM_FC('marian', args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)
   
    
    attack_dict = {}

    time_begin = time.time()
    for idx in range(args.start_index, args.start_index+args.num_samples):
        best_output = attacker.google_attack(idx, LM_model, fc)
        attack_dict[idx]= best_output 
        save_outputs(args, best_output, 'google', attack_alg="TransFool")

        if idx%100==0 or idx==3002:    
            os.makedirs(f'TransFool/{args.result_folder}/google', exist_ok=True)
            with open(f'TransFool/{args.result_folder}/google/google_{args.source_lang}_{args.target_lang}_{idx}.pkl', 'wb') as f:
                pickle.dump(attack_dict, f)
    
    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    print(time.time()-time_begin)


if __name__ == '__main__':
    parser = TransFool_parser()
    args = parser.parse_args()

    main(args)


