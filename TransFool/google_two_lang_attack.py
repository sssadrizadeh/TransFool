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
from datasets import load_dataset, DatasetDict
from attacker import Attacker


def main(args):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target language of the target black-box model (the source language is the same)
    args.target_lang = "de" if args.black_target_lang=="fr" else "fr"

    # load the NMT model and its tokenizer
    model, tokenizer = load_model_tokenizer('marian', args.source_lang, args.target_lang, device)

    # dataset tokenized by reference tokenizer
    dataset_name = "wmt14"
    dataset_config_name = f'{args.target_lang}-en'
    black_dataset_config_name = f'{args.black_target_lang}-en'
    _, tokenized_dataset = load_tokenized_dataset(tokenizer,args.source_lang,args.target_lang,dataset_name,dataset_config_name)

    # black dataset 
    black_dataset = load_dataset(dataset_name, black_dataset_config_name,split="validation")
    black_dataset = DatasetDict({"validation": black_dataset})


    # create Attacker
    attacker = Attacker(args, model, tokenizer, tokenized_dataset, device, black_tokenized_dataset=black_dataset, attack_type='google_two_lang')    

    # load LM model and FC layer
    LM_model, fc = load_LM_FC('marian', args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)
   
    
    attack_dict = {}

    time_begin = time.time()
    flag = 0
    for idx in range(args.start_index, args.start_index+args.num_samples):
        best_output = attacker.google_attack(idx, LM_model, fc, flag)
        attack_dict[idx]= best_output 
        flag = 0
        if (idx)%10==0 or idx==2999:    
            os.makedirs(f'TransFool/{args.result_folder}/google_two_lang', exist_ok=True)
            with open(f'TransFool/{args.result_folder}/google_two_lang/google__{args.target_lang}_{args.black_target_lang}_{idx}.pkl', 'wb') as f:
                pickle.dump(attack_dict, f)
            flag = 1
            time.sleep(60)
    
        # save_outputs(args, best_output, 'google', attack_alg="TransFool")

        
    
    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    print(time.time()-time_begin)


if __name__ == '__main__':
    parser = TransFool_parser()
    args = parser.parse_args()

    main(args)


