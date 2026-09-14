from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from .segment_anything import build_sam_vit_h
from .unilm.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .configuration_evf import EvfConfig

from .DecoupledMambaFusion import *
def printDataShape(sam_img,beit3_img,mask,input_ids):
    print("\n🔍 [Debug] tensor shape and dtype check:")

    def get_info(tensor):
        if not isinstance(tensor, torch.Tensor):
            return f"Type: {type(tensor)} (Not a Tensor)"

        # Collect basic tensor metadata.
        shape_str = str(list(tensor.shape))
        dtype_str = str(tensor.dtype).replace('torch.', '')
        device_str = str(tensor.device)

        # Try to print min/max values, which is useful for masks and images.
        try:
            # Convert to float before printing to avoid half-precision errors.
            min_val = tensor.min().float().item()
            max_val = tensor.max().float().item()
            range_str = f"Range: [{min_val:.2f}, {max_val:.2f}]"
        except:
            range_str = "Range: N/A"

        return f"Shape: {shape_str:<20} | {dtype_str:<8} | {range_str:<20} | {device_str}"

    print(f"   🖼️  SAM Image : {get_info(sam_img)}")
    print(f"   🖼️  BEiT Image: {get_info(beit3_img)}")
    print(f"   🎭 Mask      : {get_info(mask)}")
    print(f"   📝 Input IDs : {get_info(input_ids)}")
    print("-" * 80)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss



class DAPMambaModel(PreTrainedModel):
    config_class = EvfConfig
    def __init__(
        self,
        config,
        **kwargs
    ):
        super(DAPMambaModel, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.encoder_pretrained = kwargs.get("encoder_pretrained", None)
        self.dice_loss_weight = kwargs.get("dice_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.get("bce_loss_weight", 1.0)
        self.train_mask_decoder = True #kwargs.get("train_mask_decoder", False)
        self.train_prompt_encoder = True #kwargs.get("train_prompt_encoder", False)
        self.initialize_evf_modules(config)
    def postprocess_heatmap(
            self,
            heatmap: torch.Tensor,
            input_size: tuple,
            original_size: tuple,
    ) -> torch.Tensor:
        """
         mask ， padding 

        Args:
            heatmap: [T, H, W] ( T x 64 x 64)
            input_size: (H_valid, W_valid)  Padding  ( 1024, 576)
            original_size: (H_orig, W_orig)  ( 1920, 1080)
        """
        heatmap = F.interpolate(
            heatmap.unsqueeze(1),
            size=(self.visual_model.image_encoder.img_size, self.visual_model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        heatmap = heatmap[..., : input_size[0], : input_size[1]]

        heatmap = F.interpolate(
            heatmap,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        # [T, 1, H, W] -> [T, H, W]
        return heatmap.squeeze(1)

    def initialize_evf_modules(self, config):
        # SAM
        if config.sam_scale=="huge":
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
        else:
            raise NotImplementedError

        for param in self.visual_model.parameters():
            param.requires_grad = False
        if self.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        if self.train_prompt_encoder:
            self.visual_model.prompt_encoder.no_mask_embed.requires_grad_(True)

        # beit-3
        if self.config.mm_extractor_scale == "base":
            beit_config = _get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = _get_large_config()
        else:
            raise AttributeError(f"model config should contain key 'mm_extractor_scale', with value 'base' or 'large'.")

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained)["model"]
            self.mm_extractor.load_state_dict(
                beit_state_dict,
                strict=False
            )

        for param in self.mm_extractor.parameters():
            param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        assert in_dim==beit_config.encoder_embed_dim, \
            f"projection layer dim {in_dim} mismatch with mm_extractor dim {beit_config.encoder_embed_dim}"
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # self.temporal_fusion = nn.MultiheadAttention(
        #     embed_dim=1024,
        #     num_heads=1,
        #     batch_first=True
        # )
        self.temporal_fusion = DecoupledMambaFusion(d_model=256)

        # Ensure Mamba modules participate in training.
        for param in self.temporal_fusion.parameters():
            param.requires_grad = True

        # Interaction layer.
        self.prompt_interaction = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            batch_first=True
        )
        for param in self.prompt_interaction.parameters():
            param.requires_grad = True

        # Projection layer.
        self.weight_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        for param in self.weight_proj.parameters():
            param.requires_grad = True

        self.temporal_pe = nn.Embedding(2048, 256)
        for param in self.temporal_pe.parameters():
            param.requires_grad = True

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(
        self,
        images: torch.FloatTensor,
        images_evf: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        num_frames: int = 1,
        sampled_indices: torch.Tensor = None,
        **kwargs,
    ):

        # printDataShape(sam_img=images,beit3_img=images_evf,mask=masks_list,input_ids=input_ids)


        # image_embeddings = self.get_visual_embs(images)
        image_embeddings = self.visual_model.image_encoder(images)

        totalBatchsize = image_embeddings.shape[0]
        batch_size = totalBatchsize // num_frames
        #assert batch_size == len(offset) - 1


        multimask_output = False
        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,       #(1,3,224,224)
            textual_tokens=input_ids,       #(1,10)
            text_padding_position=torch.zeros_like(input_ids)
            )

        feat = output["encoder_out"][:, :1, ...]  #(1,1,1024)

        feat = self.text_hidden_fcs[0](feat)  ##(1,1,1024)

        B = totalBatchsize // num_frames

        C = feat.shape[-1]
        vid_len=len(images_evf)
        feat = feat.view(B, vid_len, C)
        # feat = feat + self.attention(feat, feat, feat)[0]
        feat = self.temporal_fusion(feat)
        # Global context features.
        global_context = feat.mean(dim=1, keepdim=True)
        if sampled_indices is not None:
            sampled_indices = sampled_indices.to(device=feat.device, dtype=torch.long)
        batch_idx = torch.arange(B).view(B, 1).to(feat.device)  # Create batch indices for alignment.
        feat_sampled = feat[batch_idx, sampled_indices]  # Output shape: (B, 5, C)

        # Fuse global context.
        gate = torch.sigmoid(self.weight_proj(feat_sampled))
        feat_sampled = gate * feat_sampled + (1 - gate) * global_context
        # Cross-frame interaction.
        pos_embed = self.temporal_pe(sampled_indices)  # [B, N, C]
        feat_sampled = feat_sampled + pos_embed
        feat_sampled = self.prompt_interaction(feat_sampled)

        feat=feat_sampled.reshape(-1, 1, C)   # Output shape: (B*T, 1, C)

        # print(f'After temporal fusion: {feat.shape}')
        pred_masks = []
        for i in range(totalBatchsize):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat[i].unsqueeze(0), # Requires a 3D feature tensor (1, 1, 256).
            )
            sparse_embeddings = sparse_embeddings.to(feat[i].dtype)
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),  # Requires a 4D image embedding tensor.
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            if multimask_output:
                sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
                low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]

            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list,
                original_size=label_list,
            )
            pred_masks.append(pred_mask[:, 0])   #(1,1,720,1280)


        gt_masks = masks_list   #(1,720,1280)

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx].unsqueeze(0)
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            #import pdb;pdb.set_trace()
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = mask_loss

        return {
            "loss": loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def inference(
            self,
            images,
            images_evf,
            input_ids,
            resize_list,
            original_size_list,
            multimask_output=False,
            num_frames:int=1,
            return_attn: bool = False,
            sampled_indices: torch.Tensor = None,
        ):

        image_embeddings = self.get_visual_embs(images)
        multimask_output = multimask_output

        totalBatchsize = image_embeddings.shape[0]

        output = self.mm_extractor.beit3(visual_tokens=images_evf, textual_tokens=input_ids, text_padding_position=torch.zeros_like(input_ids))

        feat_raw = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat_raw)  # Per-frame features before temporal fusion (T, 1, C).
        B = totalBatchsize // num_frames

        C = feat.shape[-1]
        vid_len = len(images_evf)
        feat = feat.view(B, vid_len, C)
        # feat = feat + self.attention(feat, feat, feat)[0]
        feat = self.temporal_fusion(feat)
        # Global context features.
        global_context = feat.mean(dim=1, keepdim=True)
        if sampled_indices is not None:
            sampled_indices = sampled_indices.to(device=feat.device, dtype=torch.long)
        batch_idx = torch.arange(B).view(B, 1).to(feat.device)  # Create batch indices for alignment.
        feat_sampled = feat[batch_idx, sampled_indices]  # Output shape: (B, 5, C)

        # Fuse global context.
        gate = torch.sigmoid(self.weight_proj(feat_sampled))
        feat_sampled = gate * feat_sampled + (1 - gate) * global_context
        # Cross-frame interaction.
        pos_embed = self.temporal_pe(sampled_indices)  # [B, N, C]
        feat_sampled = feat_sampled + pos_embed
        feat_sampled = self.prompt_interaction(feat_sampled)

        feat = feat_sampled.reshape(-1, 1, C)  # Output shape: (B*T, 1, C)


        pred_masks=[]
        iou_scores=[]
        for i in range(totalBatchsize):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat[i].unsqueeze(0),  # Requires 3D input.
            )

            sparse_embeddings = sparse_embeddings.to(feat.dtype)
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            iou_scores.append(iou_predictions[0])
            if multimask_output:
                sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
                low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]

            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[0],
                original_size=original_size_list[0],
            )
            pred_masks.append(pred_mask[:, 0])

        return pred_masks, iou_scores


AutoConfig.register("evf", EvfConfig)
AutoModelForCausalLM.register(EvfConfig, DAPMambaModel)
