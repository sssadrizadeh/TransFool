class attack_output():
    def __init__(self,attack_result,adv_sent,org_sent,adv_tr,ref_tr,org_tr,error_rate,org_bleu,adv_bleu,org_chrf,adv_chrf,query,itr=0):
        self.attack_result = attack_result
        self.adv_sent=adv_sent
        self.org_sent=org_sent
        self.adv_tr=adv_tr
        self.ref_tr=ref_tr
        self.org_tr=org_tr
        self.error_rate=error_rate
        self.org_bleu=org_bleu
        self.adv_bleu=adv_bleu
        self.org_chrf = org_chrf
        self.adv_chrf = adv_chrf
        self.query = query
        self.itr = itr