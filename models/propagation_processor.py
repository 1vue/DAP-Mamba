import os

import torch

from transformers import Sam2VideoModel, Sam2VideoProcessor


class Sam2Processor:
    def __init__(self, device):
        super().__init__()

        self.device = device
        local_path = "weights/sam2/sam2.1-hiera-large"
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model path not found: {local_path}")
        self.model = Sam2VideoModel.from_pretrained(local_path, dtype=torch.bfloat16).to(device)
        self.processor = Sam2VideoProcessor.from_pretrained(local_path)
        # self.model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large", cache_dir='../huggingface').to(device, dtype=torch.bfloat16)
        # self.processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-large")

    def propagate(self, frames, obj_ids, mask, ann_idx):
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            dtype=torch.bfloat16)

        self.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=ann_idx,
            obj_ids=obj_ids,
            input_masks=mask
        )

        outputs = self.model(
            inference_session=session,
            frame_idx=ann_idx,
        )

        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks], original_sizes=[[session.video_height, session.video_width]], binarize=True
        )[0]

        masks = []
        # forward pass
        for i, output in enumerate(self.model.propagate_in_video_iterator(session)):
            mask = self.processor.post_process_masks([output.pred_masks],
                                                     original_sizes=[[session.video_height, session.video_width]],
                                                     binarize=True)[0][0][0]
            masks.append(mask)

        for i, output in enumerate(self.model.propagate_in_video_iterator(session, reverse=True)):
            mask = self.processor.post_process_masks([output.pred_masks],
                                                     original_sizes=[[session.video_height, session.video_width]],
                                                     binarize=True)[0][0][0]
            if i != 0:
                masks.insert(0, mask)

        return masks
