import pickle
import torch
import argparse    
from utils.eval import eval, Eval 
from utils.attack_output import attack_output
from tabulate import tabulate
# from TransFool import attacker
# from kNN import attacker

import sys
# sys.modules['attacker'] = attacker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    
   
    if "TransFool" in args.attack_alg:
            
        if args.target_model_name!="google" and args.attack_type!="black_box2" and args.attack_type!="two_lang":
            if args.num_beam!=0:
                PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.mode}_{args.start_index}_{args.start_index+args.num_samples}_sim_{args.w_sim*10}_perp_{args.w_perp*10}_lr_{args.lr*1000}_ratio_{args.bleu*10}_beam_{args.num_beam}.pkl'
            elif args.epoch!=34:
                PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.mode}_{args.start_index}_{args.start_index+args.num_samples}_sim_{args.w_sim*10}_perp_{args.w_perp*10}_lr_{args.lr*1000}_ratio_{args.bleu*10}_epoch_{args.epoch}.pkl'
            else:
                PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.mode}_{args.start_index}_{args.start_index+args.num_samples}_sim_{args.w_sim*10}_perp_{args.w_perp*10}_lr_{args.lr*1000}_ratio_{args.bleu*10}.pkl'
        elif args.attack_type=="black_box2":
            PATH = f'{args.attack_alg}/{args.result_folder}/black_box/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}.pkl'
        elif args.attack_type=="two_lang":
            ref_lang = "de" if args.target_lang=="fr" else "fr"
            PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{ref_lang}_{args.target_lang}_avg_{args.start_index}_{args.start_index+args.num_samples}_sim_{args.w_sim*10}_perp_{args.w_perp*10}_lr_{args.lr*1000}_ratio_{args.bleu*10}.pkl'
        else:
            PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}.pkl'
    
    elif "kNN" in args.attack_alg:
        if args.attack_type=="black_box" or args.attack_type=="google":
            PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}.pkl'
        else:
            # PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}_grad_sign_iter_1.pkl'
            PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}_grad_sign_swap_{args.max_swap}.pkl'
    
    elif args.attack_alg=="WSLS":
        PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.source_lang}_{args.target_lang}.pkl'#_job_0.pkl'
    
    elif "Seq2Sick" in args.attack_alg:
        if args.attack_type=="black_box" or args.attack_type=="google":
            PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}.pkl'
        else:
            PATH = f'{args.attack_alg}/{args.result_folder}/{args.attack_type}/{args.target_model_name}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.start_index+args.num_samples}_const_{args.const}_lr_{args.lr}_itr_200.pkl'#_job_0.pkl'
    
    
    
    with (open(PATH, "rb")) as f:
        d = (pickle.load(f))


    E = Eval(d, device, args, part = 'both')
    results,results_all = E.results_success, E.results_all


    print("\n\n")
    print("*********** RESULTS ***********")
    print("\n")               

    print("\n")

    print("In All Attacks:")
    print(tabulate(results_all, tablefmt='psql', showindex=False, numalign="left", floatfmt=".8f"))

    print("\n")

    print("In Successful Attacks:")
    print(tabulate(results, tablefmt='psql', showindex=False, numalign="left", floatfmt=".8f"))


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation.")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result", type=str,
        help="folder for loading trained models")

    # Data
    parser.add_argument("--num_samples", default=100, type=int,
        help="number of samples to attack")
    parser.add_argument("--attack_alg", default="TransFool", type=str,
        choices=["TransFool", "kNN", "WSLS", "Seq2Sick", "TransFool_scale", "Seq2Sick_scale", "kNN_scale"],
        help="attack method to load reasults from corresponding folder")
    parser.add_argument("--attack_type", default="white_box", type=str,
        choices=["white_box", "black_box", "two_lang", "google", "fine_tune", "noLM", "LM", "black_box2"],
        help="attack type to load reasults from corresponding folder")

    # Model
    parser.add_argument("--target_model_name", default="marian", type=str,
        choices=["marian", "mbart", "google"],
        help="target NMT model")
    parser.add_argument("--source_lang", default="en", type=str,
        choices=["en", "fr"],
        help="source language")
    parser.add_argument("--target_lang", default="fr", type=str,
        choices=["fr", "de", "zh", "en", "de2", "ru", "cs"],
        help="target language")
    # Eval setup
    parser.add_argument("--bad_perp", default=100, type=float,
        help="threshold for the bad perplexity")
    parser.add_argument("--bad_sim", default=0, type=float,
        help="threshold for the bad similaroty")

    # Attack setting
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")

    parser.add_argument("--w_sim", default=15.0, type=float,
        help="similarity loss coefficient")
    
    parser.add_argument("--w_perp", default=1.8, type=float,
        help="LM loss coefficient")
    
    parser.add_argument("--lr", default=0.016, type=float,
        help="learning rate")

    parser.add_argument("--weights", default="", type=str,
        help="weighted attack")

    parser.add_argument("--mode", default="avg", type=str,
        choices=["avg", "idf"],
        help="score mode")

    parser.add_argument("--bleu", default=0.5, type=float,
        help="bleu score ratio for success")

    parser.add_argument("--num_beam", default=0, type=int,
        help="beam size of target model")

    parser.add_argument("--const", default=1, type=float,
        help="const in seq2sick")

    parser.add_argument("--max-swap", default=1, type=int,
        help="max-swap in knn")
    
    parser.add_argument("--epoch", default=34, type=int,
        help="number of epoch for LM checkpoint")

    args = parser.parse_args()

    main(args)