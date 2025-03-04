import torch
from torch import nn

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
from ldm.models.vit import VisionTransformer as VIT
from ldm.models.psp_encoder.encoders import psp_encoders
import os
from torchvision.utils import save_image
DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(
        tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(
        token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            mvtec_path=None,
            return_position=False,
            **kwargs
    ):
        super().__init__()
        self.return_position=return_position
        self.spatial_encoder = False
        self.string_to_token_dict = {}

        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict()  # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'):  # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else:  # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280
        print(placeholder_strings)  # 是"*"
        print(initializer_words)  # 是一个单词
        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)
        name_anomaly_file = '/research/cvl-guoxia11/anomaly_detection_v2/anomalydiffusion/name-anomaly.txt'
        with open(name_anomaly_file,'r') as f:
            sample_anomaly_pairs=f.read().split('\n')
        for name in sample_anomaly_pairs:
            for idx, placeholder_string in enumerate(placeholder_strings):

                token = get_token_for_string(placeholder_string)

                if initializer_words and idx < len(initializer_words):
                    init_word_token = get_token_for_string(initializer_words[idx])

                    with torch.no_grad():
                        init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                    token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1),
                                                      requires_grad=True)
                    self.initial_embeddings[placeholder_string] = torch.nn.Parameter(
                        init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
                else:
                    token_params = torch.nn.Parameter(
                        torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
                self.string_to_token_dict[placeholder_string] = token
                self.string_to_param_dict[name] = token_params
        
        self.string_to_param_dict[placeholder_string] = token_params    ## GX: additional line for the mask generation.

        self.string_to_param_dict=self.string_to_param_dict.cuda()

    def forward(
            self,
            tokenized_text,
            embedded_text,
            cond_img=None,
            name=None,
            **kwargs
    ):
        img=cond_img
        b, n, device = *tokenized_text.shape, tokenized_text.device
        # if img is not None:
        #     from torchvision.utils import save_image
        #     print(img.shape,img.min(),img.max())
        #     save_image(img,'tmp.jpg',nrow=4)
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():  # GX: 只有一次，self.string_to_token_dict.items()只有*
            placeholder_embedding=[]
            # print("$$$$$$$$$$$$$")
            # print(name)
            # print("$$$$$$$$$$$$$")
            # print(self.string_to_param_dict)
            # import sys;sys.exit(0)
            if name is not None:
                for i in name:
                    placeholder_embedding.append(self.string_to_param_dict[i])  ## GX: self.string_to_param_dict has all [sample-name]-[anomaly-type], each is 4*1280.
                placeholder_embedding = torch.stack(placeholder_embedding,dim=0)  ## GX: placeholder_embedding: torch.Size([4, 4, 1280])
            else:
                placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)

            if self.max_vectors_per_token == 1:  # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                if self.spatial_encoder and img is not None:
                    embedded_text[placeholder_idx] = self.spatial_encoder_model(img)[:, 0]
                else:
                    embedded_text[placeholder_idx] = placeholder_embedding
            else:  # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:  # 执行这个
                    max_step_tokens = self.max_vectors_per_token    ## GX: 4; define when initializing the class.
                ##TODO-GX:not sure about which cases correspond to these three condition blocks.
                if self.spatial_encoder and (img is not None) and (name is not None):

                    # print(f"the first option.")
                    # import sys;sys.exit(0)
                    placeholder_embedding2 = self.spatial_encoder_model(img)    ## GX: torch.Size([4, 1, 256, 256]) ==> torch.Size([4, 4, 1280])
                    
                    ## GX: guess using spatial encoder than it will generate different things. 
                    placeholder_embedding= torch.cat([placeholder_embedding,placeholder_embedding2],dim=1)  ## placeholder_embedding: torch.Size([4, 8, 1280])
                    # placeholder_embedding= torch.cat([placeholder_embedding,placeholder_embedding],dim=1)  ## placeholder_embedding: torch.Size([4, 8, 1280])
                    # print(placeholder_embedding.size())

                    num_vectors_for_token = placeholder_embedding.shape[1]
                    # num_vectors_for_token = min(placeholder_embedding.shape[1], max_step_tokens)
                    #print(num_vectors_for_token)
                    placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                    # rows对应batchsize：0~batchsize-1;col:对应*在哪个位置
                    if placeholder_rows.nelement() == 0:
                        continue

                    sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                    sorted_rows = placeholder_rows[sort_idx]
                    position=torch.zeros(tokenized_text.size(0),2)
                    for idx in range(len(sorted_rows)):
                        row = sorted_rows[idx]
                        col = sorted_cols[idx]
                        # print(embedded_text[row][:col].shape,num_vectors_for_token, placeholder_embedding[row,:num_vectors_for_token].shape)
                        new_token_row = torch.cat(
                            [tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device),
                             tokenized_text[row][col + 1:]], dim=0)[:n]  # 把*插到77维的text中间
                        new_embed_row = torch.cat(
                            [embedded_text[row][:col], placeholder_embedding[row, :num_vectors_for_token],
                             embedded_text[row][col + 1:]], dim=0)[:n]
                        embedded_text[row] = new_embed_row
                        tokenized_text[row] = new_token_row
                        position[row][0]=col
                        position[row][1]=col+num_vectors_for_token

                else:   ## GX: mask generation entry. 
                    position = None

                    if placeholder_embedding.dim() == 2:    ## when name list is None.
                        num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)
                        # print(name) 
                        # print("placeholder_embedding: ", placeholder_embedding.size())  ## torch.Size([16, 1280])
                        # print()
                        # print("tokenized text: ", tokenized_text)
                        # print("placeholder text: ", placeholder_token)
                        placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                        # print(placeholder_rows, placeholder_cols)
                        #rows对应batchsize：0~batchsize-1;col:对应*在哪个位置
                        if placeholder_rows.nelement() == 0:
                            continue

                        sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                        sorted_rows = placeholder_rows[sort_idx]

                        for idx in range(len(sorted_rows)):
                            row = sorted_rows[idx]
                            col = sorted_cols[idx]

                            # print()
                            # print(embedded_text[row][:col].size())  # torch.Size([6, 1280])
                            # print(placeholder_embedding[:num_vectors_for_token].size())     # torch.Size([16, 1280]) ==> just select the entire placeholder_embedding
                            # print(placeholder_embedding[row,:num_vectors_for_token].size()) # torch.Size([16]) ==> just select row-th row's first num_vectors_for_token elements. 
                            # print()

                            new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n] #把*插到77维的text中间
                            new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]

                            embedded_text[row]  = new_embed_row
                            tokenized_text[row] = new_token_row
                        # print("==================")
                        # print(embedded_text.size())     ## torch.Size([4, 77, 1280])
                        # print(tokenized_text.size())    ## torch.Size([4, 77])
                        # print(f"...over...")
                        # import sys;sys.exit(0)

                    elif placeholder_embedding.dim() == 3:    ## when name list is not None
                        num_vectors_for_token = min(placeholder_embedding.shape[1], max_step_tokens)
                        # print(name) 
                        # print("placeholder_embedding: ", placeholder_embedding.size())  ## torch.Size([4, 16, 1280])

                        placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                        #rows对应batchsize：0~batchsize-1;col:对应*在哪个位置
                        if placeholder_rows.nelement() == 0:
                            continue

                        sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                        sorted_rows = placeholder_rows[sort_idx]

                        for idx in range(len(sorted_rows)):
                            row = sorted_rows[idx]
                            col = sorted_cols[idx]

                            # print()
                            # print(embedded_text[row][:col].size())  # torch.Size([6, 1280])
                            # print(placeholder_embedding[:num_vectors_for_token].size())     # torch.Size([4, 16, 1280])
                            # print(placeholder_embedding[row,:num_vectors_for_token].size()) # torch.Size([16, 1280])
                            # print(placeholder_embedding[row][:num_vectors_for_token].size()) # torch.Size([16, 1280])
                            # print()

                            new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n] #把*插到77维的text中间
                            
                            ## GX: This indexing operation directly slices the second dimension. more efficiently. 
                            new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[row, :num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]
                            
                            ## GX: updates: two steps; first 3D to 2D; then slices the second dimension 
                            # new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[row][:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]
                            embedded_text[row]  = new_embed_row
                            tokenized_text[row] = new_token_row

                        # print("==================")
                        # print(embedded_text.size())     ## torch.Size([4, 77, 1280])
                        # print(tokenized_text.size())    ## torch.Size([4, 77])
                        # print(f"...over...")
                        # import sys;sys.exit(0)

                    # print("...over...")
                    # import sys;sys.exit(0)

                # #     continue
                # # else:

                #     # print("the third option: ", name)
                #     # print(self.spatial_encoder is None)
                #     # print(img is None)
                #     # import sys;sys.exit(0)
                #     # num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)  
                #     num_vectors_for_token = placeholder_embedding.shape[1]  ## GX: not sure why swtich from the last one to this one.

                #     placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
                #     # rows对应batchsize：0~batchsize-1;col:对应*在哪个位置
                #     if placeholder_rows.nelement() == 0:
                #         continue

                #     sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                #     sorted_rows = placeholder_rows[sort_idx]

                #     for idx in range(len(sorted_rows)):
                #         row = sorted_rows[idx]
                #         col = sorted_cols[idx]
                #         new_token_row = torch.cat(
                #             [tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device),
                #              tokenized_text[row][col + 1:]], axis=0)[:n]  # 把*插到77维的text中间
                        
                #         # print()
                #         # print(embedded_text[row][:col].size())
                #         # print(placeholder_embedding[row,:num_vectors_for_token].size())
                #         # print()
                #         # # import sys;sys.exit(0)
                #         # new_embed_row = torch.cat(
                #         #     [embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token],
                #         #      embedded_text[row][col + 1:]], axis=0)[:n]
                #         new_embed_row = torch.cat(  ##TODO-GX: this is directly copied from the embedding_manager.py; not sure.
                #             [embedded_text[row][:col], placeholder_embedding[row,:num_vectors_for_token],
                #              embedded_text[row][col + 1:]], axis=0)[:n]

                #         embedded_text[row] = new_embed_row
                #         tokenized_text[row] = new_token_row
                #     position = None

        return embedded_text,position

    def prepare_spatial_encoder(self,text_num=4):
        self.spatial_encoder = True
        self.spatial_encoder_model = psp_encoders.GradualStyleEncoder(text_num=text_num).cuda()
        ## GX: what is this function for?

    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict}, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        tmp = ckpt['string_to_param']
        # for i in tmp.keys():
        #     tmp[i] = torch.cat([tmp[i], torch.zeros(4, 1280)], dim=0)
        self.string_to_param_dict = tmp.cuda()
        # self.string_to_param_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0)  # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)  # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):

        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss