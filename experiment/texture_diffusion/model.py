import torch.nn as nn 
from PIL import Image
import torch 
import copy
from diffusers import  T2IAdapter
 
class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model=None, adapter_modules=None, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    '''def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred'''
    
    def forward(self,img):
        return 

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class img_embedding(nn.Module):
    def __init__(self, ) -> None: #image_proj_model, adapter_modules
        super().__init__()
        '''  self.feature_model=feature_model
        self.type_emb=nn.Embedding(3,1024)
        for n , i in self.feature_model.named_modules():
            if n in ["layers.3","layers.2","layers.1","layers.0"]:
                i.register_forward_hook(self.forward_hook)'''
        #self.decoder=Decoder()
       
        self.features_level=[]

        self.t2i_adapters= T2IAdapter(
            in_channels=3,
            channels=(320, 640, 1280, 1280),
            num_res_blocks=2,
            downscale_factor=16,
            adapter_type="full_adapter_xl",
        )

        self.type_dict={"dyngradient:input1":0,"mirror:input0":1}
        
        #self.ip_adapter=IPAdapter(unet, image_proj_model, adapter_modules)

        self.img2token=Image2Token()

    def forward_hook(self,model,input,output):
        
        self.features_level.append(output.detach())

     

    def forward(self,unet,img, noisy_latents, timesteps, image_embeds,added_cond_kwargs=None):
        down_block_additional_residuals=self.t2i_adapters(img)
        
        encoder_hidden_states=self.img2token(image_embeds)
        
        '''with torch.no_grad():
            logits=self.feature_model(img)#.last_hidden_state
            #cls_token=logits[:,0,:]
        if self.features_level==[]:
            raise ValueError("not get features")  
        
        hs = {'res3' : self.features_level[-3], 
              'res4' : self.features_level[-2],     
              'res5' : self.features_level[-1], }
        self.features_level=[]
        #type_index=  [  torch.tensor(self.type_dict[i],dtype=torch.long) for i in type_cond]
        #type_index=torch.stack(type_index)[:,None].to(img.device)
        encoder_hidden_states=self.decoder(hs)#+self.type_emb(type_index)'''
        #noise_pred=self.ip_adapter( noisy_latents, timesteps, encoder_hidden_states, image_embeds)
        noise_pred=unet(noisy_latents, timesteps, encoder_hidden_states,down_block_additional_residuals=[
                        sample.to(dtype=img.dtype) for sample in down_block_additional_residuals
                    ],added_cond_kwargs=added_cond_kwargs).sample
        return noise_pred #cls_token.unsqueeze(1)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Image2Token(nn.Module):

    def __init__(self, visual_hidden_size=1024, text_hidden_size=2048, max_length=77, num_layers=3):
        super(Image2Token, self).__init__()
        
        self.visual_proj = nn.Linear(visual_hidden_size, text_hidden_size)
        self.text_hidden_size = text_hidden_size
        
        if num_layers>0:
            self.query = nn.Parameter(torch.randn((1, max_length, text_hidden_size)))
            decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.i2t = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            self.i2t = None

    def forward(self, x):
        b=x.shape[0]
        out = self.visual_proj(x).view(b,-1,self.text_hidden_size)
        if self.i2t is not None:
            out = self.i2t(self.query.repeat(b,1,1), out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self,
                 dim=256, 
                 feedforward_dim=1024,
                 dropout=0.1, 
                 activation="relu",
                 n_heads=8,):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = x
        h1 = self.self_attn(x, x, x, attn_mask=None)[0]
        h = h + self.dropout1(h1)
        h = self.norm1(h)

        h2 = self.linear2(self.dropout2(self.activation(self.linear1(h))))
        h = h + self.dropout3(h2)
        h = self.norm2(h)
        return h

class DecoderLayerStacked(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x):
        h = x
        for _, layer in enumerate(self.layers):
            h = layer(h)
        if self.norm is not None:
            h = self.norm(h)
        return h


class Decoder(nn.Module):
    def __init__(
            self,
            inchannels={"res3":256,"res4":512,"res5":1024},#
            trans_input_tags=["res3","res4","res5"], #["res3","res4","res5"],
            trans_num_layers=6,
            trans_dim=1024,
            trans_nheads=8,
            trans_dropout=0.1,
            trans_feedforward_dim=1024):

        super().__init__()
        trans_inchannels = {
            k: v for k, v in inchannels.items() if k in trans_input_tags}
        fpn_inchannels = {
            k: v for k, v in inchannels.items() if k not in trans_input_tags}

        self.trans_tags = sorted(list(trans_inchannels.keys()))
        self.fpn_tags   = sorted(list(fpn_inchannels.keys()))
        self.all_tags   = sorted(list(inchannels.keys()))

        if len(self.trans_tags)==0: 
            assert False # Not allowed

        self.num_trans_lvls = len(self.trans_tags)

        self.inproj_layers = nn.ModuleDict()
        for tagi in self.trans_tags:
            layeri = nn.Sequential(
                nn.Conv2d(trans_inchannels[tagi], trans_dim, kernel_size=1),
                nn.GroupNorm(32, trans_dim),)
            nn.init.xavier_uniform_(layeri[0].weight, gain=1)
            nn.init.constant_(layeri[0].bias, 0)
            self.inproj_layers[tagi] = layeri

        '''tlayer = DecoderLayer(
            dim     = trans_dim,
            n_heads = trans_nheads,
            dropout = trans_dropout,
            feedforward_dim = trans_feedforward_dim,
            activation = 'relu',)

        self.transformer = DecoderLayerStacked(tlayer, trans_num_layers)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.level_embed = nn.Parameter(torch.Tensor(len(self.trans_tags), trans_dim))
        nn.init.normal_(self.level_embed)'''

        self.GAP=nn.AdaptiveAvgPool2d((1,1))
        self.proj=nn.Sequential(nn.Linear(trans_dim,trans_dim),
                                nn.GELU(),
                                #nn.Linear(trans_dim,trans_dim),
                                nn.LayerNorm(trans_dim))

        

    def forward(self, features):
        x = []
         
        for idx, tagi in enumerate(self.trans_tags[::-1]):
            xi = features[tagi].permute(0,3,1,2)
            xi = self.inproj_layers[tagi](xi)
            bs, _, h, w = xi.shape
            xi=self.GAP(xi)
            xi=self.proj(xi.flatten(1))
            xi =xi.unsqueeze(1)   # xi.flatten(2).transpose(1, 2) 
            
            #+ self.level_embed[idx].view(1, 1, -1)
            x.append(xi)

         
        #x_concat = torch.cat(x, 1)
        #y_concat = self.transformer(x_concat)
         

        return  torch.cat(x,dim=1)