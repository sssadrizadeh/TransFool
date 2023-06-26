import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


import torch
import pickle
from utils.load import load_tokenized_dataset, load_model_tokenizer, load_LM_FC 
from utils.attack import save_outputs
from attacker import Attacker
from parser import TransFool_parser
import time


def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.model_name = "mbart" if args.black_model_name=="marian" else "marian"

    # load reference NMT model and its tokenizer
    model, tokenizer = load_model_tokenizer(args.model_name, args.source_lang, args.target_lang, device)

    # load target black-box NMT model and its tokenizer
    black_model, black_tokenizer = load_model_tokenizer(args.black_model_name, args.source_lang, args.target_lang, device)


    # dataset tokenized by reference tokenizer
    dataset, tokenized_dataset = load_tokenized_dataset(tokenizer,args.source_lang,args.target_lang,args.dataset_name,args.dataset_config_name)

    # dataset tokenized by target tokenizer
    _, black_tokenized_dataset = load_tokenized_dataset(black_tokenizer,args.source_lang,args.target_lang,dataset = dataset)

    # create attacker
    attacker = Attacker(args, model, tokenizer, tokenized_dataset, device, black_model, black_tokenizer, black_tokenized_dataset, attack_type='black')
    
    # load LM model and FC layer
    LM_model, fc = load_LM_FC(args.model_name, args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)

    

    
    attack_dict = {}

    time_begin = time.time()
    for idx in range(args.start_index, args.start_index+args.num_samples):
        best_output = attacker.gen_adv(idx, LM_model, fc)
        attack_dict[idx]= best_output 
        # save_outputs(args, best_output, 'black', attack_alg=TransFool)

    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    print(time.time()-time_begin)
        
    os.makedirs(f'TransFool/{args.result_folder}/black_box', exist_ok=True)
    with open(f'TransFool/{args.result_folder}/black_box/{args.black_model_name}_{args.source_lang}_{args.target_lang}_{args.mode}_{args.start_index}_{args.start_index+args.num_samples}_sim_{args.w_sim*10}_perp_{args.w_perp*10}_lr_{args.lr*1000}_ratio_{args.bleu*10}.pkl', 'wb') as f:
        pickle.dump(attack_dict, f)

if __name__ == '__main__':

    parser = TransFool_parser()
    args = parser.parse_args()

    main(args)
