import jiwer
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score.utils import get_idf_dict
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import math
from tqdm import tqdm
from datasets import load_metric
from utils.load import load_model_tokenizer
import pickle
import logging
import os


# Word Error Rate
def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)



# Cosine similarity constraint
class cosine_distance():
    
    def __init__(self, dataset, tokenizer, device, mode="avg"):
        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        
        if self.mode=="idf":
            self.idf_dict = self.calc_idf_dict()
        
    def calc_idf_dict(self):
        print("*********** calculating idf weights ***********")
        return get_idf_dict([ex["en"] for ex in self.dataset["train"]["translation"]], self.tokenizer, nthreads=20)
        
    
    def calc_dist(self,input_ids,x1,x2):
    
        x1 = x1 / torch.unsqueeze(x1.norm(2,1), 1)
        x2 = x2 / torch.unsqueeze(x2.norm(2,1), 1)

        dists = 1-(x1*x2).sum(1)

        if self.mode=="avg":
            dists = dists/dists.size(0)
        else:   
            weights =  torch.FloatTensor([self.idf_dict[idx] for idx in input_ids]).to(self.device)
            weights = weights/weights.sum()
            dists = dists*weights

        return dists.sum()


# Universal Sentence Encoder
class USE:
    def __init__(self):
        # self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    def compute_sim(self, clean_texts, adv_texts):
        clean_embeddings = self.encoder(clean_texts)
        adv_embeddings = self.encoder(adv_texts)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))

        return float(cosine_sim.numpy())



class gpt_perp:
    def __init__(self, device):
        self.device = device
        model_id = "gpt2-large"
        self.gpt = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.gpt_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    def compute_loss(self,sent):
        self.gpt.eval()
        tokens = self.gpt_tokenizer.encode(sent+'\n')
        loss = self.gpt(torch.LongTensor(tokens).unsqueeze(0).to(self.device),labels = torch.LongTensor(tokens).unsqueeze(0).to(self.device)).loss

        return loss.item()


class eval_TER:
    def __init__(self, device,args):
        logging.basicConfig(level=logging.ERROR)
        self.device = device
        _, self.tokenizer = load_model_tokenizer(args.target_model_name, args.source_lang, args.target_lang, device)
        
    def compute_TER(self,org_sent,adv_sent):
        input_ids = self.tokenizer(org_sent, truncation=True)["input_ids"]
        adv_ids = self.tokenizer(adv_sent, truncation=True)["input_ids"]

        return wer(adv_ids,input_ids)

def get_ids_adv_text(data_len,num_ids,start_idx,seed=42):
  #Generate the ids of the sentences to attack in the dataset
  np.random.seed(seed)
  ids=np.arange(data_len)
  np.random.shuffle(ids)
  return ids[start_idx:num_ids]

class Eval:
    def __init__(self, attack_output, device, args, part = 'all'):

        self.d = attack_output
        self.device = device
        self.args = args
        self.part = part

        self.use = USE()
        self.gpt = gpt_perp(device)
        if args.target_model_name!="google":
            self.eval_TER = eval_TER(device,args)
        self.bertscore = load_metric("bertscore")

        # self.indices = list(self.d.keys())#[:1000]
        self.indices = get_ids_adv_text(len(self.d.keys()),1000,0)

        if self.part =='all' or self.part=='both':
            self.condition = {idx:self.d[idx].org_bleu!=0 for idx in self.indices}
        else:
            self.condition = {idx:self.d[idx].org_bleu!=0 and self.d[idx].attack_result=='success' for idx in self.indices}

        self.n_sent = len([idx for idx in (self.indices) if self.d[idx].org_bleu!=0])
        self.fail = len([idx for idx in (self.indices) if self.d[idx].org_bleu!=0 and self.d[idx].attack_result!='success'])

        self.sim_perp_calc()

        if self.part=='all':
            self.results_all = self.eval_calc()
            self.save_sim()
        elif self.part=='success':
            self.results_success = self.eval_calc()
        else:
            self.results_all = self.eval_calc()
            self.success_calc()
            self.results_success = self.eval_calc()

    def save_sim(self):
        folder_name_dict = { 'white_box':f'white_box/',\
                         'white_fine_tune':f'fine_tune/',\
                         'white_no_LM':f'noLM/',\
                         'black_box':f'black_box/',\
                         'two_lang':f'two_lang/',\
                         'google':'google/google',
                         'white_LM':f'LM/' }


        PATH = f'{self.args.attack_alg}/{self.args.result_folder}/{folder_name_dict[self.args.attack_type]}{self.args.target_model_name}_{self.args.source_lang}_{self.args.target_lang}'
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # write original bleu socre
        with open(f'{PATH}/sim_tr', 'w') as f:
            for item in self.sim_tr.values():
                f.write("%s\n" % item)

        # write original bleu socre
        with open(f'{PATH}/adv_sim', 'w') as f:
            for item in self.sim.values():
                f.write("%s\n" % item)

        # write original bleu socre
        with open(f'{PATH}/org_perp', 'w') as f:
            for item in self.org_lm_loss.values():
                f.write("%s\n" % item)

        # write original bleu socre
        with open(f'{PATH}/adv_perp', 'w') as f:
            for item in self.adv_lm_loss.values():
                f.write("%s\n" % item)
            
        
    def success_calc(self):
        self.condition = {idx:self.condition[idx] and self.d[idx].attack_result=='success' and self.sim[idx]>=self.args.bad_sim and self.adv_lm_loss[idx]<=self.args.bad_perp for idx in (self.indices)}
        
        self.sim = {idx:self.sim[idx] for idx in (self.indices) if self.condition[idx]}
        self.sim_tr = {idx:self.sim_tr[idx] for idx in (self.indices) if self.condition[idx]}

        self.adv_lm_loss = {idx:self.adv_lm_loss[idx] for idx in (self.indices) if self.condition[idx]}
        self.org_lm_loss = {idx:self.org_lm_loss[idx] for idx in (self.indices) if self.condition[idx]}

        self.bertscore_adv_tr = {idx:self.bertscore_adv_tr[idx] for idx in (self.indices) if self.condition[idx]}
        self.bertscore_org_tr = {idx:self.bertscore_org_tr[idx] for idx in (self.indices) if self.condition[idx]}
        self.bertscore_src = {idx:self.bertscore_src[idx] for idx in (self.indices) if self.condition[idx]}
        self.bertscore_src_tr = {idx:self.bertscore_src_tr[idx] for idx in (self.indices) if self.condition[idx]}

        self.TER = {idx:self.TER[idx] for idx in (self.indices) if self.condition[idx]}
           
    def sim_perp_calc(self):
        print("computing similarities ...")
        self.sim = {idx:self.use.compute_sim([self.d[idx].org_sent], [self.d[idx].adv_sent]) for idx in tqdm(self.indices) if self.condition[idx]}
        self.sim_tr = {idx:self.use.compute_sim([self.d[idx].org_tr], [self.d[idx].adv_tr]) for idx in tqdm(self.indices) if self.condition[idx]}
        
        print("computing CLM loss ...")
        self.adv_lm_loss = {idx:self.gpt.compute_loss(self.d[idx].adv_sent) for idx in tqdm(self.indices) if self.condition[idx]}
        self.org_lm_loss = {idx:self.gpt.compute_loss(self.d[idx].org_sent) for idx in tqdm(self.indices) if self.condition[idx]}

        self.bertscore_adv_tr = {idx:self.bertscore.compute(predictions=[self.d[idx].adv_tr], references=[[self.d[idx].ref_tr]], lang=self.args.target_lang)["f1"][0] for idx in (self.indices) if self.condition[idx]}
        self.bertscore_org_tr = {idx:self.bertscore.compute(predictions=[self.d[idx].org_tr], references=[[self.d[idx].ref_tr]], lang=self.args.target_lang)["f1"][0] for idx in (self.indices) if self.condition[idx]}
        self.bertscore_src = {idx:self.bertscore.compute(predictions=[self.d[idx].adv_sent], references=[[self.d[idx].org_sent]], lang=self.args.source_lang)["f1"][0] for idx in (self.indices) if self.condition[idx]}
        self.bertscore_src_tr = {idx:self.bertscore.compute(predictions=[self.d[idx].adv_tr], references=[[self.d[idx].org_tr]], lang=self.args.target_lang)["f1"][0] for idx in (self.indices) if self.condition[idx]}

        if self.args.target_model_name!="google":
            self.TER = {idx:self.eval_TER.compute_TER(self.d[idx].org_sent,self.d[idx].adv_sent) for idx in (self.indices) if self.condition[idx]}
        else:
            self.TER = {idx:jiwer.wer(self.d[idx].adv_sent,self.d[idx].org_sent) for idx in (self.indices) if self.condition[idx]}
            

        self.bad = len([idx for idx in (self.indices) if self.condition[idx] and (self.sim[idx]<self.args.bad_sim or self.adv_lm_loss[idx]>self.args.bad_perp) and self.d[idx].attack_result=='success'])
        if self.part=='success':
            self.success_calc()
            # self.condition = {idx:self.condition[idx] and self.sim[idx]>=self.args.bad_sim and self.adv_lm_loss[idx]<=self.args.bad_perp for idx in (self.indices)}

   
    def eval_tr(self, adv_tr, ref_tr, org_tr, adv, org):
        bleu = load_metric("sacrebleu")
        chrf = load_metric("chrf")

        bleu_corp_adv = bleu.compute(predictions=adv_tr, references=ref_tr)['score']
        bleu_corp_org = bleu.compute(predictions=org_tr, references=ref_tr)['score']

        chrf_corp_adv = chrf.compute(predictions=adv_tr, references=ref_tr)['score']
        chrf_corp_org = chrf.compute(predictions=org_tr, references=ref_tr)['score']

        chrf_src = chrf.compute(predictions=adv, references=org)['score']

        return bleu_corp_adv, bleu_corp_org, chrf_corp_adv, chrf_corp_org, chrf_src 


    def eval_calc(self):

        adv = [self.d[idx].adv_sent for idx in (self.indices) if self.condition[idx]]
        org = [[self.d[idx].org_sent] for idx in (self.indices) if self.condition[idx]]
        
        adv_tr = [self.d[idx].adv_tr for idx in (self.indices) if self.condition[idx]]
        ref_tr = [[self.d[idx].ref_tr] for idx in (self.indices) if self.condition[idx]]
        org_tr = [self.d[idx].org_tr for idx in (self.indices) if self.condition[idx]]

        org_bleus = [self.d[idx].org_bleu for idx in (self.indices) if self.condition[idx]]
        adv_bleus = [self.d[idx].adv_bleu for idx in (self.indices) if self.condition[idx]]

        org_chrfs = [self.d[idx].org_chrf for idx in (self.indices) if self.condition[idx]]
        adv_chrfs = [self.d[idx].adv_chrf for idx in (self.indices) if self.condition[idx]]

        org_bleus = np.array(org_bleus)  
        adv_bleus = np.array(adv_bleus)
        org_chrfs = np.array(org_chrfs)
        adv_chrfs = np.array(adv_chrfs)


        bleu_corp_adv, bleu_corp_org, chrf_corp_adv, chrf_corp_org, chrf_src = self.eval_tr(adv_tr, ref_tr, org_tr, adv, org) 

        # bleu_corp_adv, bleu_corp_org = adv_bleus.mean(), org_bleus.mean()

        results = []

        results.append(["Bad attacks due to low similarity/perplexity:", self.bad])
        results.append(["Failed attacks:", self.fail])
        results.append(["Attack success rate (%):", (1-(self.fail+self.bad)/self.n_sent)*100])
        results.append(["Average semantic similarity:", sum(self.sim.values()) / len(self.sim)])
        results.append(["Average semantic similarity of translations:", sum(self.sim_tr.values()) / len(self.sim)])
        results.append(["Average source chrf:", chrf_src])
        results.append(["Average source BERTScore:", sum(self.bertscore_src.values())/len(self.sim)])
        results.append(["Average source BERTScore translation:", sum(self.bertscore_src_tr.values())/len(self.sim)])
        results.append(["Token error rate (%):", 100*sum(self.TER.values())/len(self.sim)])
        results.append(["Average adv CLM loss:", sum(self.adv_lm_loss.values()) / len(self.sim)])
        results.append(["Average org CLM loss:", sum(self.org_lm_loss.values()) / len(self.sim)])
        results.append(["Average adv perp score:", math.exp(sum(self.adv_lm_loss.values()) / len(self.sim))])
        results.append(["Average org perp score:", math.exp(sum(self.org_lm_loss.values()) / len(self.sim))])
        results.append(["Corpus adv bleu score:", bleu_corp_adv])
        results.append(["Corpus org bleu score:", bleu_corp_org])
        results.append(["Relative decrease of corpus bleu score:", (bleu_corp_org-bleu_corp_adv)/bleu_corp_org])
        results.append(["Average relative decrease of sentence bleu score:", np.mean((org_bleus-adv_bleus)/org_bleus)])
        results.append(["Corpus adv chrf score:", chrf_corp_adv])
        results.append(["Corpus org chrf score:", chrf_corp_org])
        results.append(["Relative decrease of corpus chrf score:", (chrf_corp_org-chrf_corp_adv)/chrf_corp_org])
        results.append(["Average relative decrease of sentence chrf score:", np.mean((org_chrfs-adv_chrfs)/org_chrfs)])
        results.append(["Average adversarial translation BERTScore:", sum(self.bertscore_adv_tr.values())/len(self.sim)])
        results.append(["Average original translation BERTScore:", sum(self.bertscore_org_tr.values())/len(self.sim)])

        return results