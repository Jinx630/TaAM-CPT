import torch
from mmpt.utils import load_config, set_seed
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter
from viclip import ViCLIP
from laion_clap import CLAP_Module
from clip import clip

def text_global_pool(x, text = None, pool_type = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

class FeatureLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, use_local_prompt, prompt_dim, modal_type):
        super().__init__()

        self.prompt_dim = prompt_dim

        self.obj_num = classnames
        self.use_local_prompt = use_local_prompt
        
        if hasattr(clip_model,'dtype'):
            dtype = clip_model.dtype
        else:
            dtype = torch.float32
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)
        
        text_features = torch.ones([self.obj_num, self.prompt_dim], dtype=dtype)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = nn.Parameter(text_features)

        self.text_features_neg = None
        if self.use_local_prompt:
            text_features_neg = torch.ones([self.obj_num,self.prompt_dim], dtype=dtype)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
            self.text_features_neg = nn.Parameter(text_features_neg)
 
    def forward(self):

        if self.use_local_prompt:
            return self.text_features, self.text_features_neg, self.temperature, self.spatial_T, self.ranking_scale
        else:
            return self.text_features, None, self.temperature, self.spatial_T, self.ranking_scale

class CLIP_audio_encoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        self.cfg = cfg
        self.model = clip_model.model

    def forward(self, x):

        device = next(self.model.text_projection.parameters()).device

        audio_feat = self.model.audio_branch(x, mixup_lambda=None, device=device)
        audio_feat_ = self.model.audio_projection(audio_feat['embedding'])
        audio_feats = self.model.audio_projection(audio_feat['fine_grained_embedding'])

        return audio_feat_, audio_feats

class CLIP_audio_text_encoder(nn.Module):
    def __init__(self, cfg, clap_model):
        super().__init__()

        self.cfg = cfg
        self.model = clap_model.model
    
    def forward(self, x):

        device = next(self.model.text_projection.parameters()).device
        return self.model.encode_text(x, device = device)

class CLIP_video_encoder(ViCLIP):
    def __init__(self, cfg, clip_model):
        super().__init__()

        self.cfg = cfg
        self.model = clip_model

    def forward(self, input_frames):

        if input_frames.ndim == 5:
            input_frames = input_frames.permute(0, 2, 1, 3, 4).contiguous()
        else:
            input_frames = input_frames.unsqueeze(2)

        clip_feat, clip_feats= self.model.vision_encoder.forward_video(input_frames)

        clip_feat = clip_feat.float()
        clip_feats = clip_feats.float()

        return clip_feat, clip_feats

class CLIP_video_text_encoder(ViCLIP):
    def __init__(self, cfg, clip_model):
        super().__init__()

        self.cfg = cfg
        self.model = clip_model

    def forward(self, input_text):

        text_features = self.model.text_encoder.forward_text(input_text)

        return text_features

class CLIP_image_encoder(nn.Module):
    def __init__(self, cfg, clip_model, return_interm_layers=False):
        super().__init__()

        self.cfg = cfg
        self.model = clip_model.encode_image

    def forward(self,x):
        
        image_feature_, image_features = self.model(x)

        return image_feature_, image_features

class CLIP_image_text_encoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        self.cfg = cfg
        self.model = clip_model.encode_text
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_pool_type = clip_model.text_pool_type
        self.register_buffer('attn_mask', clip_model.text_mask, persistent=False)

    def forward(self, text, normalize: bool = False):

        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, y = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
                y = self.text_projection(y)
            else:
                x = x @ self.text_projection
                y = y @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x, F.normalize(y, dim=-1) if normalize else y

class DenseCLIP_multimodal_clip(nn.Module):
    def __init__(self, cfg, classnames, image_clip_model, video_clip_model, audio_clip_model, return_interm_layers=False):
        super().__init__()

        class_names = sum(classnames)

        self.cfg = cfg
        self.use_local_prompt = self.cfg.use_local_prompt
        self.si_directional = cfg.si_directional

        self.use_image = self.cfg.use_image
        self.use_video = self.cfg.use_video
        self.use_audio = self.cfg.use_audio

        self.use_image_video_con = self.cfg.use_image_video_con
        self.use_image_audio_con = self.cfg.use_image_audio_con
        self.use_video_audio_con = self.cfg.use_video_audio_con

        self.image_clip_model = image_clip_model
        self.video_clip_model = video_clip_model
        self.audio_clip_model = audio_clip_model

        self.video_feature_learner = FeatureLearner(cfg, class_names, self.video_clip_model, self.use_local_prompt, prompt_dim = 512, modal_type='video')
        self.image_feature_learner = FeatureLearner(cfg, class_names, self.image_clip_model, self.use_local_prompt, prompt_dim = 512, modal_type='image')
        self.audio_feature_learner = FeatureLearner(cfg, class_names, self.audio_clip_model, self.use_local_prompt, prompt_dim = 512, modal_type='audio')

        # self.modal_proj = nn.Linear(512,256)

        if self.use_image:
            self.image_encoder = CLIP_image_encoder(self.cfg, self.image_clip_model)
            self.image_text_encoder = CLIP_image_text_encoder(self.cfg, self.image_clip_model)
            self.logit_scale = self.image_clip_model.logit_scale

        if self.use_video:
            self.video_encoder = CLIP_video_encoder(self.cfg, self.video_clip_model)
            self.video_text_encoder = CLIP_video_text_encoder(self.cfg, self.video_clip_model)

        if self.use_audio:
            self.audio_encoder = CLIP_audio_encoder(self.cfg, self.audio_clip_model)
            self.audio_text_encoder = CLIP_audio_text_encoder(self.cfg, self.audio_clip_model)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'sum')

        only_k400_label = torch.arange(class_names)
        only_coco_label = torch.arange(class_names)
        only_esc50_label = torch.arange(class_names)

        only_k400_label[:classnames[0]] = -100
        only_coco_label[classnames[0]:classnames[0]+classnames[1]] = -100
        only_esc50_label[classnames[0]+classnames[1]:] = -100

        self.register_buffer('only_k400_label', only_k400_label, persistent=False)
        self.register_buffer('only_coco_label', only_coco_label, persistent=False)
        self.register_buffer('only_esc50_label', only_esc50_label, persistent=False)

    def norm_feature(self, a,b,c,d):

        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        c = c / c.norm(dim=-1, keepdim=True)
        if self.use_local_prompt:
            d = d / d.norm(dim=-1, keepdim=True)

        return a,b,c,d

    def norm_feature_a_b(self, a,b):

        a = a / a.norm(dim=-1, keepdim=True)
        if b is not None:
            b = b / b.norm(dim=-1, keepdim=True)

        return a,b

    def get_logits_global(self, temperature, feature_, global_prompt):

        logit_scale = temperature.exp()  # rk_scale
        logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50 # temperature.exp()  # self.logit_scale.exp()
        logits_global = logit_scale * feature_ @ global_prompt.t()   # B * C,  cls * C, = B * cls

        return logits_global

    def get_logits_local(self, features, local_prompt, temperature, spatial_T, mask=None):

        logits_neg = features @ local_prompt.t()    #  L * B * C,  cls * C,  L * B * cls

        if mask:
            logits_neg = logits_neg.permute(2, 1, 0) + mask[None, :, :]
            logits_neg = logits_neg.permute(2, 1, 0)

        tmp_scale = spatial_T.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text
        prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale, dim=0)
        logits_local = torch.sum(logit_scale * logits_neg * prob_spatial, dim=0)

        return logits_global

    def contrastive_loss(self, pred_1, pred_2, logit_scale, label):

        pred_1_to_pred_2 = logit_scale * pred_1 @ [pred_2].t()
        loss_ce = self.ce_loss(pred_1_to_pred_2, label)

        if not self.si_directional:
            pred_2_to_pred_1 = logit_scale * pred_2 @ pred_1.t()
            return (loss_ce + self.ce_loss(pred_2_to_pred_1, label)) / 2.0

        return loss_ce
        
    def ranking_loss(self, y_pred, y_true, scale_ = 2.0, margin_ = 1):

        y_pred *= scale_
        y_true_ = y_true.float()
        tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
        partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
        loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
        loss = torch.sum(loss, dim=-1)
        loss = torch.sum(loss, dim=-1)

        return torch.mean(loss)
    
    def forward(self, test_input=None, video_input_ids=None, image_input_ids=None, audio_input_ids=None, audio_attention_mask=None, \
                video_labels=None, image_labels=None, audio_labels=None, \
                is_image=False, is_audio=False, is_video=False, device='cuda', if_test=False):
        if if_test:

            image_text_global_prompt, image_text_local_prompt, image_text_temperature, image_text_spatial_T, image_text_rk_scale = self.image_feature_learner()
            video_text_global_prompt, video_text_local_prompt, video_text_temperature, video_text_spatial_T, video_text_rk_scale = self.video_feature_learner()
            audio_text_global_prompt, audio_text_local_prompt, audio_text_temperature, audio_text_spatial_T, audio_text_rk_scale = self.audio_feature_learner()

            image_text_global_prompt, image_text_local_prompt = self.norm_feature_a_b(image_text_global_prompt, image_text_local_prompt)
            video_text_global_prompt, video_text_local_prompt = self.norm_feature_a_b(video_text_global_prompt, video_text_local_prompt)
            audio_text_global_prompt, audio_text_local_prompt = self.norm_feature_a_b(audio_text_global_prompt, audio_text_local_prompt)

            if is_image:
                
                image_feature_, image_features= self.image_encoder(test_input)
                image_features = image_features.permute(1, 0, 2)

                image_feature_, image_features = self.norm_feature_a_b(image_feature_, image_features)

                image_logits_global = self.get_logits_global(image_text_temperature, image_feature_, image_text_global_prompt)

                if self.use_local_prompt:
                    image_logits_local = self.get_logits_local(image_features, image_text_local_prompt, image_text_temperature, image_text_spatial_T)
                    return image_logits_global, image_logits_local
                
                return image_logits_global, image_feature_

            if is_video:

                video_feature_, video_features = self.video_encoder(test_input)
                video_features = video_features.permute(1, 0, 2)

                video_feature_, video_features = self.norm_feature_a_b(video_feature_, video_features)

                video_logits_global = self.get_logits_global(video_text_temperature, video_feature_, video_text_global_prompt)

                if self.use_local_prompt:
                    video_logits_local = self.get_logits_local(video_features, video_text_local_prompt, video_text_temperature, video_text_spatial_T, video_text_mask)
                    return video_logits_global, video_logits_local
                
                return video_logits_global, video_feature_

            if is_audio:

                audio_feature_, audio_features = self.audio_encoder(test_input)
                audio_features = audio_features.permute(1, 0, 2)

                audio_feature_, audio_features = self.norm_feature_a_b(audio_feature_, audio_features)

                audio_logits_global = self.get_logits_global(audio_text_temperature, audio_feature_, audio_text_global_prompt)

                if self.use_local_prompt:
                    audio_logits_local = self.get_logits_local(audio_features, audio_text_local_prompt, audio_text_temperature, audio_text_spatial_T, audio_text_mask)
                    return audio_logits_global, audio_logits_local
                
                return audio_logits_global, audio_feature_

        else:

            image_text_global_prompt, image_text_local_prompt, image_text_temperature, image_text_spatial_T, image_text_rk_scale = self.image_feature_learner()
            video_text_global_prompt, video_text_local_prompt, video_text_temperature, video_text_spatial_T, video_text_rk_scale = self.video_feature_learner()
            audio_text_global_prompt, audio_text_local_prompt, audio_text_temperature, audio_text_spatial_T, audio_text_rk_scale = self.audio_feature_learner()

            image_text_global_prompt, image_text_local_prompt = self.norm_feature_a_b(image_text_global_prompt, image_text_local_prompt)
            video_text_global_prompt, video_text_local_prompt = self.norm_feature_a_b(video_text_global_prompt, video_text_local_prompt)
            audio_text_global_prompt, audio_text_local_prompt = self.norm_feature_a_b(audio_text_global_prompt, audio_text_local_prompt)

            if self.use_image:

                image_text_feature_, image_text_features = self.image_text_encoder(image_input_ids)
                image_text_features = image_text_features.permute(1, 0, 2)  # LBD

                image_text_feature_, image_text_features = self.norm_feature_a_b(image_text_feature_, image_text_features)
                
                image_text_mask = (image_input_ids == 0).long() * (-10000)  # BL

                image_text_logits_global = self.get_logits_global(image_text_temperature, image_text_feature_, image_text_global_prompt)   # B * C,  cls * C, = B * cls

                if self.use_local_prompt:
                    image_text_logits_local = self.get_logits_local(image_text_features, image_text_local_prompt, image_text_temperature, image_text_spatial_T, image_text_mask)

            if self.use_video:

                video_text_feat = self.video_text_encoder(video_input_ids)

                video_text_feature_ = video_text_feat[torch.arange(video_text_feat.shape[0]), video_input_ids.argmax(dim=-1)]  # BD
                video_text_features = video_text_feat.permute(1, 0, 2)  # LBD

                video_text_feature_, video_text_features = self.norm_feature_a_b(video_text_feature_, video_text_features)
                
                video_text_mask = (video_input_ids == 0).long() * (-10000)  # BL

                video_text_logits_global = self.get_logits_global(video_text_temperature, video_text_feature_, video_text_global_prompt)   # B * C,  cls * C, = B * cls

                if self.use_local_prompt:
                    video_text_logits_local = self.get_logits_local(video_text_features, video_text_local_prompt, video_text_temperature, video_text_spatial_T, video_text_mask)

            if self.use_audio:

                audio_input = {
                    'input_ids': audio_input_ids,
                    'attention_mask': audio_attention_mask
                }

                audio_text_feature_, audio_text_features = self.audio_text_encoder(audio_input)
                audio_text_features = audio_text_features.permute(1, 0, 2)  # LBD

                audio_text_feature_, audio_text_features = self.norm_feature_a_b(audio_text_feature_, audio_text_features)

                audio_text_mask = (audio_input_ids == 0).long() * (-10000)  # BL
                
                audio_text_logits_global = self.get_logits_global(audio_text_temperature, audio_text_feature_, audio_text_global_prompt)   # B * C,  cls * C, = B * cls

                if self.use_local_prompt:
                    audio_text_logits_local = self.get_logits_local(audio_text_features, audio_text_local_prompt, audio_text_temperature, audio_text_spatial_T, audio_text_mask)

            loss_ranking_v = torch.tensor(0.0, device=image_input_ids.device)
            loss_ranking_i = torch.tensor(0.0, device=image_input_ids.device)
            loss_ranking_a = torch.tensor(0.0, device=image_input_ids.device)

            if self.use_image:
                loss_ranking_i = self.ranking_loss(image_text_logits_global, image_labels, scale_ = 1.0, margin_ = 1)
            if self.use_video:
                loss_ranking_v = self.ranking_loss(video_text_logits_global, video_labels, scale_ = 1.0, margin_ = 1)
            if self.use_audio:
                loss_ranking_a = self.ranking_loss(audio_text_logits_global, audio_labels, scale_ = 1.0, margin_ = 1)

            # video_text_global_prompt_proj = self.modal_proj(video_text_global_prompt)
            # audio_text_global_prompt_proj = self.modal_proj(audio_text_global_prompt)
            # image_text_global_prompt_proj = self.modal_proj(image_text_global_prompt)

            # loss_cont = torch.tensor(0.0, device=image_input_ids.device)

            # if self.use_image_video_con:
            #     loss_cont += self.contrastive_loss(video_text_global_prompt_proj, audio_text_global_prompt_proj, self.logit_scale, self.only_k400_label)
            #     loss_cont += self.contrastive_loss(video_text_global_prompt_proj, image_text_global_prompt_proj, self.logit_scale, self.only_k400_label)
            # if self.use_image_audio_con:
            #     loss_cont += self.contrastive_loss(image_text_global_prompt_proj, video_text_global_prompt_proj, self.logit_scale, self.only_coco_label)
            #     loss_cont += self.contrastive_loss(image_text_global_prompt_proj, audio_text_global_prompt_proj, self.logit_scale, self.only_coco_label)
            # if self.use_video_audio_con:
            #     loss_cont += self.contrastive_loss(audio_text_global_prompt_proj, video_text_global_prompt_proj, self.logit_scale, self.only_esc50_label)
            #     loss_cont += self.contrastive_loss(audio_text_global_prompt_proj, image_text_global_prompt_proj, self.logit_scale, self.only_esc50_label)

            loss_cont = torch.tensor(0.0, device=image_input_ids.device)

            if self.use_image_video_con:
                loss_cont += self.contrastive_loss(video_text_global_prompt, audio_text_global_prompt, self.logit_scale, self.only_k400_label)
                loss_cont += self.contrastive_loss(video_text_global_prompt, image_text_global_prompt, self.logit_scale, self.only_k400_label)
            if self.use_image_audio_con:
                loss_cont += self.contrastive_loss(image_text_global_prompt, video_text_global_prompt, self.logit_scale, self.only_coco_label)
                loss_cont += self.contrastive_loss(image_text_global_prompt, audio_text_global_prompt, self.logit_scale, self.only_coco_label)
            if self.use_video_audio_con:
                loss_cont += self.contrastive_loss(audio_text_global_prompt, video_text_global_prompt, self.logit_scale, self.only_esc50_label)
                loss_cont += self.contrastive_loss(audio_text_global_prompt, image_text_global_prompt, self.logit_scale, self.only_esc50_label)

            # loss_ranking_cont = loss_ranking + loss_cont/2

            return loss_ranking_i, loss_ranking_v, loss_ranking_a, loss_cont/2
            # return loss_ranking_cont

    def save_model(self, PATH):
        torch.save(self.feature_learner.state_dict(), PATH)
        
    def load_model(self, PATH):
        self.feature_learner.load_state_dict(torch.load(PATH))