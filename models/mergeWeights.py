import torch
from collections import OrderedDict
from safetensors.torch import load_file  # required for reading safetensors
import os,sys
# 1. absolute path of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. parent directory (project root directory)
parent_dir = os.path.dirname(current_dir)
# 3. parent directorypath
sys.path.append(parent_dir)
from utils import checkWeights


def merge_with_mapping(
        evfsam_bin_path,
        hqsam_safetensors_path,
        output_path,
        my_model_prefix="visual_model."  # prefix before mask_decoder in this project
):
    print(f"\n🚀 Starting dictionary-mapping weight merge...")

    # =======================================================
    # ： DAP-Mamba  ( module. )
    # =======================================================
    print(f"1️⃣  Loading DAP-Mamba (.bin): {evfsam_bin_path}")
    if not os.path.exists(evfsam_bin_path):
        print(f"❌ File not found: {evfsam_bin_path}");
        return

    #  pytorch， weights_only=True 
    state_dict = torch.load(evfsam_bin_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    #  'module.' 
    final_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        final_state_dict[name] = v

    print(f"   ✅ Base checkpoint loaded with {len(final_state_dict)} parameters。")

    # =======================================================
    # ： HQ-SAM  (.safetensors)
    # =======================================================
    print(f"2️⃣  Loading HQ-SAM (.safetensors): {hqsam_safetensors_path}")
    if not os.path.exists(hqsam_safetensors_path):
        print(f"❌ File not found: {hqsam_safetensors_path}");
        return

    hq_state_dict = load_file(hqsam_safetensors_path)

    # =======================================================
    # ：Define mapping rules (core section)
    # left side: layer name in the official weights (after removing mask_decoder.)
    # right side: corresponding nn.Sequential layer name in this project
    # =======================================================
    mapping_rules = {
        # --- 1. HQ Token & MLP ---
        "hq_token": "hf_token",
        "hq_mask_mlp.proj_in":  "hf_mlp.layers.0",
        "hq_mask_mlp.layers.0": "hf_mlp.layers.1",
        "hq_mask_mlp.proj_out": "hf_mlp.layers.2",

        # --- 2. ViT  (1280 -> 256) ---
        "compress_vit_conv1": "compress_vit_feat.0",  # conv
        "compress_vit_norm": "compress_vit_feat.1",  # Norm
        # .2  GELU (no parameters)
        "compress_vit_conv2": "compress_vit_feat.3",  # conv

        # --- 3. Encoder  (256 -> 64) ---
        "encoder_conv1": "embedding_encoder.0",
        "encoder_norm": "embedding_encoder.1",
        "encoder_conv2": "embedding_encoder.3",

        # --- 4. Mask  (32 -> 64) ---
        "mask_conv1": "embedding_maskfeature.0",
        "mask_norm": "embedding_maskfeature.1",
        "mask_conv2": "embedding_maskfeature.3",
    }

    # =======================================================
    # ：Iterate and merge
    # =======================================================
    print("\n3️⃣  Key mapping table:")
    print("=" * 120)
    print(f"{'HQOriginal Key (Original)':<55}  --->  {'Merged new Key (New)'}")
    print("=" * 120)
    merged_count = 0

    for hq_key, hq_val in hq_state_dict.items():
        # Only keep parameters under mask_decoder.
        if not hq_key.startswith("mask_decoder."):
            continue

        # 1. Extract the core layer name.
        # hq_key example: "mask_decoder.compress_vit_conv1.weight"
        # remove the prefix -> "compress_vit_conv1.weight"
        content = hq_key.replace("mask_decoder.", "")

        matched = False

        # 2. Find a match in the mapping table.
        for src_layer, target_layer in mapping_rules.items():
            # Check whether content starts with src_layer.
            # : "compress_vit_conv1.weight" startswith "compress_vit_conv1"
            if content.startswith(src_layer):
                # 3. Build the new key.
                # Get the suffix. ( .weight  .bias)
                suffix = content.replace(src_layer, "")

                # Concatenate:  + mask_decoder. +  + 
                # Result: visual_model.mask_decoder.compress_vit_feat.0.weight
                new_key = f"{my_model_prefix}mask_decoder.{target_layer}{suffix}"

                # 4. Write into the final state dict.
                final_state_dict[new_key] = hq_val
                merged_count += 1
                matched = True
                print(f"{hq_key:<55}  --->  {new_key}")
                # print(f"   [] {src_layer} -> {target_layer}") # for debugging
                break

        # if not matched:
        # print(f"   [Skip] {hq_key} (not a new parameter)")

    # =======================================================
    # ：Result
    # =======================================================
    print(f"   ✅ Successfully mapped and injected {merged_count} HQ-SAM parameters。")
    print(f"4️⃣  Saving to: {output_path}")
    torch.save(final_state_dict, output_path)
    print("🎉 Merge complete. You can load this file with model.load_state_dict().")


if __name__ == "__main__":
    # ---------------- Configuration (edit here) ----------------
    # DAP-Mamba weights (.bin file)
    # evf_sam_path = "models/pretrainedModels/DAP-Mamba-multitask/pytorch_model.bin"
    #
    # # HQ-SAM weights (.safetensors file)
    # hq_sam_path = "models/pretrainedModels/hq_sam/model.safetensors"
    # #
    # # # output path
    # out_path = "weights/hq_sam/evfsam_hq_merged_mapped.pth"
    #
    # # model prefix (keep visual_model. unchanged)
    # prefix = "visual_model."
    #
    # merge_with_mapping(evf_sam_path, hq_sam_path, out_path, prefix)




    # hqsam_path="models/pretrainedModels/hq_sam/model.safetensors"
    model_path="weights/hq_sam_0002.pth"
    checkWeights(model_path)