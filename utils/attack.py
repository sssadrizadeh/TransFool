
import os
from googletrans import Translator
from datasets import load_metric
import torch 
from utils.eval import wer
from utils.attack_output import attack_output


# tokeniser decode only for zh is different
def decode(ids,tokenizer, model_name, target_lang):
        if model_name=="mbart" and target_lang=="zh":
            return tokenizer.decode(ids, skip_special_tokens=True).replace(" ","")
        else:
            return tokenizer.decode(ids, skip_special_tokens=True)

def save_outputs(args, best, attack_type, attack_alg="TransFool"):

    folder_name_dict = { 'white':f'white_box/{args.model_name}',\
                         'white_fine_tune':f'fine_tune/{args.model_name}',\
                         'white_no_LM':f'noLM/{args.model_name}',\
                         'black':f'black_box/{args.black_model_name}',\
                         'black_transfer':f'black_transfer/{args.black_model_name}',\
                         'two_lang':f'two_lang/{args.black_model_name}',\
                         'google':'google/google',
                         'google_transfer':'google_transfer/google',
                         'white_LM':f'LM/{args.model_name}' }

    model_name = folder_name_dict[attack_type]
    if attack_type!="two_lang":
        PATH = f'{attack_alg}/{args.result_folder}/{model_name}_{args.source_lang}_{args.target_lang}'
    else:
        PATH = f'{attack_alg}/{args.result_folder}/{model_name}_{args.target_lang}_{args.black_target_lang}'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # write original sentence
    with open(f'{PATH}/original_sentences', 'a') as f:
        f.write("%s\n" % best.org_sent)
    
    # write adversarial sentence
    with open(f'{PATH}/adversarial_sentences', 'a') as f:
        f.write("%s\n" % best.adv_sent)
    
    # write original translation
    with open(f'{PATH}/original_translations', 'a') as f:
        f.write("%s\n" % best.org_tr)
    
    # write adversarial translation
    with open(f'{PATH}/adversarial_translations', 'a') as f:
        f.write("%s\n" % best.adv_tr)
    
    # write true translation
    with open(f'{PATH}/true_translations', 'a') as f:
        f.write("%s\n" % best.ref_tr)

    # write original bleu socre
    with open(f'{PATH}/org_bleu_socre', 'a') as f:
        f.write("%s\n" % best.org_bleu)

    # write adversarial bleu socre
    with open(f'{PATH}/adv_bleu_socre', 'a') as f:
        f.write("%s\n" % best.adv_bleu)

    # write original cHRF
    with open(f'{PATH}/org_chrf', 'a') as f:
        f.write("%s\n" % best.org_chrf)
        
    # write adversarial cHRF
    with open(f'{PATH}/adv_chrf', 'a') as f:
        f.write("%s\n" % best.adv_chrf)



def save_outputs_from_pickle(d, attack_type, attack_alg="TransFool", model_name=None, black_model_name=None, result_folder=None, source_lang=None, target_lang=None, black_target_lang=None):

    folder_name_dict = { 'white':f'white_box/{model_name}',\
                         'white_fine_tune':f'fine_tune/{model_name}',\
                         'white_no_LM':f'noLM/{model_name}',\
                         'black':f'black_box/{black_model_name}',\
                         'two_lang':f'two_lang/{black_model_name}',\
                         'google':'google/google',
                         'white_LM':f'LM/{model_name}' }

    model_name = folder_name_dict[attack_type]
    if attack_type!="two_lang":
        PATH = f'{attack_alg}/{result_folder}/{model_name}_{source_lang}_{target_lang}'
    else:
        PATH = f'{attack_alg}/{result_folder}/{model_name}_{target_lang}_{black_target_lang}'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # write original sentence
    with open(f'{PATH}/original_sentences', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_sent)
    
    # write adversarial sentence
    with open(f'{PATH}/adversarial_sentences', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_sent)
    
    # write original translation
    with open(f'{PATH}/original_translations', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_tr)
    
    # write adversarial translation
    with open(f'{PATH}/adversarial_translations', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_tr)
    
    # write true translation
    with open(f'{PATH}/true_translations', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].ref_tr)

    # write original bleu socre
    with open(f'{PATH}/org_bleu_socre', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_bleu)

    # write adversarial bleu socre
    with open(f'{PATH}/adv_bleu_socre', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_bleu)

    # write original cHRF
    with open(f'{PATH}/org_chrf', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].org_chrf)
        
    # write adversarial cHRF
    with open(f'{PATH}/adv_chrf', 'a') as f:
        for i in d.keys():
            f.write("%s\n" % d[i].adv_chrf)









