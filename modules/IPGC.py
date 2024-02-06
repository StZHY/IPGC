import torch
import torch.nn as nn


class Critique(nn.Module):
    def __init__(self, args_config, user_embedding, entity_embedding):
        super(Critique, self).__init__()

        device = torch.device("cuda:"+str(args_config.gpu_id)) if args_config.cuda else torch.device("cpu")

        self.emb_size = args_config.dim

        self.user_emb = nn.Embedding.from_pretrained(user_embedding, freeze=False)
        self.entity_emb = nn.Embedding.from_pretrained(entity_embedding, freeze=True)

        self.reg_params = {}


    def forward(self, batch):
        user_id = batch['users']
        item_id = batch['pos']
        keyphrase_id = batch['neg']

        user_emb_input = self.user_emb(user_id)
        item_emb_input = self.entity_emb(item_id)
        keyphrase_emb_input = self.entity_emb(keyphrase_id)

        bpr_loss = self.create_bpr_loss(user_emb_input, item_emb_input, keyphrase_emb_input)

        return bpr_loss
    
    def create_bpr_loss(self, user_emb, item_emb, key_emb):
        pos_scores = user_emb * item_emb
        neg_scores = user_emb * key_emb

        bpr_loss = -1 * torch.mean(nn.LogSigmoid()(0 - neg_scores))

        return bpr_loss
    
    def create_output(self, batch):
        user_id = batch['users']
        item_id = batch['items']

        user_emb_input = self.user_emb(user_id)
        entity_emb_input = self.entity_emb(item_id)

        score = torch.mul(user_emb_input, entity_emb_input).mean(dim=1)

        return score
    
    def generate(self):
        return self.user_emb, self.entity_emb
    
    def rating(self, u_embeddings, i_embeddings):
        return torch.matmul(u_embeddings, i_embeddings.t())