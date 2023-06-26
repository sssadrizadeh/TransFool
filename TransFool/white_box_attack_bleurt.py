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
from utils.load import load_tokenized_dataset, load_model_tokenizer, load_LM_FC, get_ids_adv_text
from utils.attack import save_outputs
import time
from attacker_bleurt import Attacker


def main(args):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the NMT model and its tokenizer
    model, tokenizer = load_model_tokenizer(args.model_name, args.source_lang, args.target_lang, device)

    # load tokenized dataset 
    _, tokenized_dataset = load_tokenized_dataset(tokenizer,args.source_lang,args.target_lang,args.dataset_name,args.dataset_config_name)

    if "marian" in args.model_name:
        args.model_name = "marian"
        
    # create Attacker
    if args.LM=='no_LM':
        # args.lr = 0.040
        attacker = Attacker(args, model, tokenizer, tokenized_dataset, device, attack_type='white_noLM')    
    else:
        attacker = Attacker(args, model, tokenizer, tokenized_dataset, device)

    # load LM model and FC layer
    if args.LM=='fine_tune':
        LM_model, fc = load_LM_FC('fine_tune_LM_marian', args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)
    else:
        LM_model, fc = load_LM_FC(args.model_name, args.source_lang, args.target_lang, tokenizer.vocab_size, attacker.embeddings.size()[-1], device)

   
    attack_dict = {}
    attack_type = "" if args.LM=='gpt2' else f'_{args.LM}'

    if args.part=="train":
        ids_to_attack = get_ids_adv_text(len(tokenized_dataset['train']),args.num_samples,args.start_index)
    else:
        ids_to_attack = list(range(args.start_index, args.start_index+args.num_samples))

    time_begin = time.time()
    for idx in ids_to_attack:
        idx = int(idx)
        
        if args.part=="train":
            if len(tokenized_dataset['train'][idx]['input_ids'])>127:
                continue
        
        best_output = attacker.gen_adv(idx, LM_model, fc)
        attack_dict[idx]= best_output 
        # save_outputs(args, best_output, 'white'+attack_type, attack_alg="TransFool")


    print(f'finished attack for {args.num_samples} samples in {time.time()-time_begin} seconds!')
    print(time.time()-time_begin)
        
    
    folder_name_dict = { '':'white_box',\
                         '_fine_tune':'fine_tune',\
                         '_no_LM':'noLM'}

    os.makedirs(f'TransFool/{args.result_folder}/{folder_name_dict[attack_type]}', exist_ok=True)
    with open(f'TransFool/{args.result_folder}/{folder_name_dict[attack_type]}/{args.model_name}_{args.source_lang}_{args.target_lang}_{args.mode}_{args.start_index}_{args.start_index+args.num_samples}_sim_{args.w_sim*10}_perp_{args.w_perp*10}_lr_{args.lr*1000}_ratio_{args.bleu*10}.pkl', 'wb') as f:
        pickle.dump(attack_dict, f)

if __name__ == '__main__':
    
    parser = TransFool_parser()
    args = parser.parse_args()

    main(args)
