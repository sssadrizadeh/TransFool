import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


from datasets import load_metric
import torch
from utils.eval import wer, cosine_distance
from utils.load import isolate_source_lang 
from utils.attack import decode
from utils.attack_output import attack_output
import numpy as np
from googletrans import Translator



class Attacker():
    def __init__(self, args, model=None, tokenizer=None, tokenized_dataset=None, device=None, black_model=None, black_tokenizer=None, black_tokenized_dataset=None, attack_type='white'): 
        self.model = model              # target NMT model
        self.model.eval()
        self.tokenizer = tokenizer      
        self.tokenized_dataset = tokenized_dataset          
        self.args = args
        self.device = device
        self.attack_type = attack_type
        self.scale = self.model.get_encoder().embed_scale

        # for black-box
        self.black_model = black_model
        if self.attack_type=='black' or self.attack_type=='two_lang':
            self.black_model.eval()
            self.args.model_name = "marian"

        if self.attack_type=='black':
            self.args.black_target_lang = self.args.target_lang
        elif "google" in self.attack_type:
            self.dest_dict = {'fr':'fr', 'de':'de', 'zh':'zh-cn'}
            self.translator = Translator()
            if self.attack_type=="google":
                self.args.black_target_lang = self.args.target_lang

        self.black_tokenizer = black_tokenizer
        self.black_tokenized_dataset = black_tokenized_dataset


        # isolating source language token ids
        self.all_source_index = isolate_source_lang(self.tokenized_dataset)

        # Find the embeddings of all tokens
        with torch.no_grad():
            self.embeddings = self.model.get_input_embeddings()(torch.Tensor(self.all_source_index).long().to(self.device))

        self.eos_id = self.tokenizer.eos_token_id
        self.EOS_embdedding = self.model.get_input_embeddings()(torch.Tensor([self.eos_id]).long().to(self.device))[0]

        # token distance calculator
        self.cos_dist = cosine_distance(self.tokenized_dataset, self.tokenizer, self.device, mode=self.args.mode)

    def tr_gen(self, ids, model):
        if self.args.num_beam==0:
            tr = model.generate(ids.unsqueeze(0).to(self.device))
        else:
            tr = model.generate(ids.unsqueeze(0).to(self.device),num_beams=self.args.num_beam)
        return tr
            
    # TransFool attack
    def gen_adv(self, sentence_number, LM_model, fc):
        
        lr, w_perp_coef, list_w_sim, ratio = self.args.lr, self.args.w_perp, [self.args.w_sim], self.args.bleu

        min_bleu = 0
        if self.args.part=="":
            if self.attack_type=="two_lang":
                part = "validation"
            else:
                part = "test"
        else:
            part = self.args.part
        metric_bleu = load_metric("sacrebleu")
        metric_chrf = load_metric("chrf")

        # LM embeddings
        fc.eval()    
        LM_model.eval()
        # Find the embeddings of all tokens
        with torch.no_grad():
            # embeddings of the LM
            LM_embeddings = fc(self.embeddings)


        for w_sim_coef in list_w_sim:
            
            ref_tr = self.tokenized_dataset[part][sentence_number]['labels']         
                        
            input_ids = self.tokenized_dataset[part][sentence_number]['input_ids']
            input_sent = self.tokenizer.decode(input_ids, skip_special_tokens=True).replace("▁"," ")
            
            

            if 'white' in self.attack_type:
                ref_tr_decode = [[decode(ref_tr,self.tokenizer,self.args.model_name,self.args.target_lang)]]

                first_tr_ids = self.tr_gen(torch.LongTensor(input_ids), self.model)
                first_tr = [decode(first_tr_ids[0],self.tokenizer,self.args.model_name,self.args.target_lang)]

            elif self.attack_type=='black' or self.attack_type=='two_lang':
                black_ref_tr = self.black_tokenized_dataset[part][sentence_number]['labels']
                ref_tr_decode = [[decode(black_ref_tr,self.black_tokenizer,self.args.black_model_name,self.args.black_target_lang)]]
                
                black_input_ids = self.black_tokenized_dataset[part][sentence_number]['input_ids']
                first_tr_ids = self.tr_gen(torch.LongTensor(black_input_ids), self.black_model)
                first_tr = [decode(first_tr_ids[0],self.black_tokenizer,self.args.black_model_name,self.args.black_target_lang)]

            org_bleu = metric_bleu.compute(predictions=first_tr, references=ref_tr_decode)['score']
            org_chrf = metric_chrf.compute(predictions=first_tr, references=ref_tr_decode)['score']
            
            
            w = [self.EOS_embdedding if i==self.eos_id else self.embeddings[self.all_source_index.index(i)].cpu().numpy() for i in input_ids]
            w = torch.tensor(w,requires_grad=False).to(self.device)
            
            LM_w = torch.tensor(fc(w[:-1].detach()),requires_grad=False)
            
            all_w = [ torch.clone(w.detach()) ]

            w_a = torch.tensor(w,requires_grad=True).to(self.device)
            optimizer = torch.optim.Adam([w_a],lr=lr)
            
            output_ids = first_tr_ids.detach()#self.model.generate(torch.LongTensor(input_ids).unsqueeze(0).to(self.device))
            pred_project_decode = first_tr#[decode(output_ids[0],self.tokenizer)]
            
            itr = 0
            adv_bleu = org_bleu

            best  = attack_output('failed',input_sent, \
                    input_sent, \
                    first_tr[0], \
                    ref_tr_decode[0][0], \
                    first_tr[0], \
                    0, org_bleu, org_bleu, org_chrf, org_chrf,0)

            while adv_bleu>max(min_bleu,org_bleu*ratio):
                
                itr+=1  

                output = self.model(inputs_embeds=self.scale*w_a.unsqueeze(0), labels = torch.tensor(ref_tr).unsqueeze(0).to(self.device))
                adv_loss = -1*output.loss

                LM_input = fc(w_a[:-1]).to(self.device)
                
                
                if itr==1:
                    perp_loss = LM_model(inputs_embeds = LM_input.unsqueeze(0),labels=torch.LongTensor(input_ids[:-1]).unsqueeze(0).to(self.device)).loss
                else:
                    perp_loss = LM_model(inputs_embeds = LM_input.unsqueeze(0),labels=torch.LongTensor(index_prime[:-1]).unsqueeze(0).to(self.device)).loss

                if self.attack_type=='white_noLM':
                    sim_loss = self.cos_dist.calc_dist(input_ids[:-1],w[:-1].detach(),w_a[:-1])
                else:
                    sim_loss = self.cos_dist.calc_dist(input_ids[:-1],LM_w,LM_input)

                total_loss = adv_loss + w_sim_coef * sim_loss + w_perp_coef * perp_loss

                optimizer.zero_grad()
                total_loss.backward()
                
                if self.args.model_name=="marian":
                    fix_idx = torch.from_numpy(np.array([len(input_ids)-1])).to(self.device)
                elif self.args.model_name=="mbart":
                    fix_idx = torch.from_numpy(np.array([0,len(input_ids)-1])).to(self.device)
                w_a.grad.index_fill_(0, fix_idx, 0)
                
                
                optimizer.step()
                
                print(f'itr: {itr} \t w_sim_coef: {w_sim_coef} \t loss: {adv_loss.data.item(), sim_loss.data.item(), perp_loss.data.item(), total_loss.data.item()}')
            
                if self.attack_type=='white_noLM':
                    cosine = -1 * torch.matmul(w_a[:-1],self.embeddings.transpose(1, 0))/torch.unsqueeze(w_a[:-1].norm(2,1), 1)/torch.unsqueeze(self.embeddings.norm(2,1), 0)
                else:
                    cosine = -1 * torch.matmul(LM_input,LM_embeddings.transpose(1, 0))/torch.unsqueeze(LM_input.norm(2,1), 1)/torch.unsqueeze(LM_embeddings.norm(2,1), 0)
                
                index_prime = torch.argmin(cosine,dim=1)
                w_prime = torch.cat((self.embeddings[index_prime],self.EOS_embdedding.unsqueeze(0)),dim=0)
                index_prime = torch.tensor([self.all_source_index[index_prime[i]] if i<len(w_prime)-1 else self.eos_id for i in range(len(w_prime))],requires_grad=False)
                if 'white' in self.attack_type:
                    ref_index_prime = torch.tensor(self.tokenizer.encode(self.tokenizer.decode(index_prime,  skip_special_tokens=True)),requires_grad=False) # to account for tokenization artefacts
                elif self.attack_type=='black' or self.attack_type=='two_lang':
                    ref_index_prime = torch.tensor(self.black_tokenizer.encode(self.tokenizer.decode(index_prime,  skip_special_tokens=True)),requires_grad=False) # to account for tokenization artefacts



                adv_sent = self.tokenizer.decode(index_prime, skip_special_tokens=True).replace("▁"," ")
                print(adv_sent)

                
                if torch.equal(w_prime,w)==False:
                    skip=False
                    for w_ in all_w:
                        if torch.equal(w_prime,w_):
                            skip=True
                    if skip==False:
                        print('*****************')
                        w_a.data=w_prime
                        all_w.append(torch.clone(w_prime.detach()))


                        if 'white' in self.attack_type:
                            output_ids = self.tr_gen(ref_index_prime, self.model)
                            pred_project_decode = [decode(output_ids[0],self.tokenizer,self.args.model_name,self.args.target_lang)]
                        elif self.attack_type=='black' or self.attack_type=='two_lang':
                            output_ids = self.tr_gen(ref_index_prime, self.black_model)
                            pred_project_decode = [decode(output_ids[0],self.black_tokenizer,self.args.black_model_name,self.args.black_target_lang)]

                        best.query=best.query+1
                        print(best.query)
                        
                        adv_bleu = metric_bleu.compute(predictions=pred_project_decode, references=ref_tr_decode)['score']
                        adv_chrf = metric_chrf.compute(predictions=pred_project_decode, references=ref_tr_decode)['score']

                        if adv_bleu<best.adv_bleu:
                            best.adv_sent = adv_sent
                            best.adv_tr = pred_project_decode[0]
                            if 'white' in self.attack_type:
                                best.error_rate =  wer(ref_index_prime,input_ids) 
                            elif self.attack_type=='black' or self.attack_type=='two_lang':
                                best.error_rate =  wer(ref_index_prime,black_input_ids)
                            best.adv_bleu = adv_bleu
                            best.adv_chrf = adv_chrf

                        print('bleu score: ', adv_bleu)

                best.itr = itr
                if itr> 500:
                    break
            if adv_bleu<=max(min_bleu,org_bleu*ratio):
                break
        
        if best.adv_bleu > self.args.SAR * best.org_bleu and best.adv_bleu>min_bleu:
            print(f'Failed to generate Adv attack on {sentence_number}!')
            best.attack_result = 'failed'
        else:
            print(f'Successed to generate Adv attack on {sentence_number}!')
            best.attack_result = 'success'


        return best


    def google_attack(self, sentence_number, LM_model, fc):
        
        lr, w_perp_coef, list_w_sim, ratio = self.args.lr, self.args.w_perp, [self.args.w_sim], self.args.bleu

        min_bleu = 0
        if self.args.part=="":
            if self.attack_type=="google_two_lang":
                part = "validation"
            else:
                part = "test"
        else:
            part = "train"
        metric_bleu = load_metric("sacrebleu")
        metric_chrf = load_metric("chrf")

        # LM embeddings
        fc.eval()    
        LM_model.eval()
        # Find the embeddings of all tokens
        with torch.no_grad():
            # embeddings of the LM
            LM_embeddings = fc(self.embeddings)


        for w_sim_coef in list_w_sim:

            ref_tr = self.tokenized_dataset[part][sentence_number]['labels']         
                        
            input_ids = self.tokenized_dataset[part][sentence_number]['input_ids']
            input_sent = self.tokenizer.decode(input_ids, skip_special_tokens=True).replace("▁"," ")

            first_tr = [self.translator.translate(input_sent, src=self.args.source_lang, dest=self.dest_dict[self.args.black_target_lang]).text]
            ref_tr_decode = [[self.black_tokenized_dataset[part][sentence_number]["translation"][self.args.black_target_lang]]]

            org_bleu = metric_bleu.compute(predictions=first_tr, references=ref_tr_decode)['score']
            org_chrf = metric_chrf.compute(predictions=first_tr, references=ref_tr_decode)['score']
            
            
            w = [self.EOS_embdedding if i==self.eos_id else self.embeddings[self.all_source_index.index(i)].cpu().numpy() for i in input_ids]
            w = torch.tensor(w,requires_grad=False).to(self.device)
            
            LM_w = torch.tensor(fc(w[:-1].detach()),requires_grad=False)
            
            all_w = [ torch.clone(w.detach()) ]

            w_a = torch.tensor(w,requires_grad=True).to(self.device)
            optimizer = torch.optim.Adam([w_a],lr=lr)
            
            # output_ids = first_tr_ids.detach()#self.model.generate(torch.LongTensor(input_ids).unsqueeze(0).to(self.device))
            pred_project_decode = first_tr#[decode(output_ids[0],self.tokenizer)]

            itr = 0
            adv_bleu = org_bleu

            best  = attack_output('failed',input_sent, \
                    input_sent, \
                    first_tr[0], \
                    ref_tr_decode[0][0], \
                    first_tr[0], \
                    0, org_bleu, org_bleu, org_chrf, org_chrf,0)

            
            while adv_bleu>max(min_bleu,org_bleu*ratio):
                itr+=1  

                output = self.model(inputs_embeds=self.scale*w_a.unsqueeze(0), labels = torch.tensor(ref_tr).unsqueeze(0).to(self.device))
                adv_loss = -1*output.loss

                LM_input = fc(w_a[:-1]).to(self.device)
                
                
                if itr==1:
                    perp_loss = LM_model(inputs_embeds = LM_input.unsqueeze(0),labels=torch.LongTensor(input_ids[:-1]).unsqueeze(0).to(self.device)).loss
                else:
                    perp_loss = LM_model(inputs_embeds = LM_input.unsqueeze(0),labels=torch.LongTensor(index_prime[:-1]).unsqueeze(0).to(self.device)).loss

                if self.attack_type=='white_noLM':
                    sim_loss = self.cos_dist.calc_dist(input_ids[:-1],w[:-1].detach(),w_a[:-1])
                else:
                    sim_loss = self.cos_dist.calc_dist(input_ids[:-1],LM_w,LM_input)

                total_loss = adv_loss + w_sim_coef * sim_loss + w_perp_coef * perp_loss

                optimizer.zero_grad()
                total_loss.backward()
                
                if self.args.model_name=="marian":
                    fix_idx = torch.from_numpy(np.array([len(input_ids)-1])).to(self.device)
                elif self.args.model_name=="mbart":
                    fix_idx = torch.from_numpy(np.array([0,len(input_ids)-1])).to(self.device)
                w_a.grad.index_fill_(0, fix_idx, 0)
                
                
                optimizer.step()
                
                print(f'itr: {itr} \t w_sim_coef: {w_sim_coef} \t loss: {adv_loss.data.item(), sim_loss.data.item(), perp_loss.data.item(), total_loss.data.item()}')

                cosine = -1 * torch.matmul(LM_input,LM_embeddings.transpose(1, 0))/torch.unsqueeze(LM_input.norm(2,1), 1)/torch.unsqueeze(LM_embeddings.norm(2,1), 0)
                
                index_prime = torch.argmin(cosine,dim=1)
                w_prime = torch.cat((self.embeddings[index_prime],self.EOS_embdedding.unsqueeze(0)),dim=0)
                index_prime = torch.tensor([self.all_source_index[index_prime[i]] if i<len(w_prime)-1 else self.eos_id for i in range(len(w_prime))],requires_grad=False)

                adv_sent = self.tokenizer.decode(index_prime, skip_special_tokens=True).replace("▁"," ")

                print(adv_sent)

                
                if torch.equal(w_prime,w)==False:
                    skip=False
                    for w_ in all_w:
                        if torch.equal(w_prime,w_):
                            skip=True
                    if skip==False:
                        print('*****************')
                        w_a.data=w_prime
                        all_w.append(torch.clone(w_prime.detach()))

                        pred_project_decode = [self.translator.translate(adv_sent, src=self.args.source_lang, dest=self.dest_dict[self.args.black_target_lang]).text]

                        best.query=best.query+1
                        print(best.query)
                        
                        adv_bleu = metric_bleu.compute(predictions=pred_project_decode, references=ref_tr_decode)['score']
                        adv_chrf = metric_chrf.compute(predictions=pred_project_decode, references=ref_tr_decode)['score']

                        if adv_bleu<best.adv_bleu:
                            best.adv_sent = adv_sent
                            best.adv_tr = pred_project_decode[0]
                            best.error_rate =  wer(index_prime,input_ids)
                            best.adv_bleu = adv_bleu
                            best.adv_chrf = adv_chrf

                        print('bleu score: ', adv_bleu)
                
                best.itr = itr
                if itr> 500:
                    break
            if adv_bleu<=max(min_bleu,org_bleu*ratio):
                break
        
        if best.adv_bleu > self.args.SAR * best.org_bleu and best.adv_bleu>min_bleu:
            print(f'Failed to generate Adv attack on {sentence_number}!')
            best.attack_result = 'failed'
        else:
            print(f'Successed to generate Adv attack on {sentence_number}!')
            best.attack_result = 'success'


        return best

            

