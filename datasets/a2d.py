import os
import json
import h5py
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video

from datasets.transforms_image import ResizeLongestSide


class A2DDataset(Dataset):
    """
    A dataset class for the A2D-Sentences dataset (single-frame annotations).
    Matches the interface and outputs of YTVOSDataset/MeViSDataset:
    returns (imgs_sam, imgs_beit, target).
    """

    def __init__(
        self,
        subset_type: str = "train",
        dataset_path: str = "../../dataset/a2d_sentences",
        tf=None,
        return_masks: bool = True,
        num_frames: int = 1,
        max_skip: int = 0,
    ):
        assert subset_type in ["train", "test"], "subset_type must be 'train' or 'test'"
        self.subset_type = subset_type
        self.dataset_path = dataset_path
        self._transforms = tf
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.max_skip = max_skip
        self.max_beit_frames = 14
        # Align with other datasets: preprocessing for BeIT branch
        self.transforms_beit = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=3, antialias=None),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        # A2D-Sentences paths
        self.videos_dir = os.path.join(self.dataset_path, "Release", "clips320H")
        self.mask_annotations_dir = os.path.join(
            self.dataset_path, "text_annotations", "a2d_annotation_with_instances"
        )
        self.ann_file = os.path.join(
            self.dataset_path, f"a2d_sentences_single_frame_{subset_type}_annotations.json"
        )

        self.text_annotations = self._load_text_annotations()

        print(f"\n{subset_type} sample num: ", len(self.text_annotations))
        print("\n")

    def _load_text_annotations(self):
        with open(self.ann_file, "r") as f:
            # list of [text_query, video_id, frame_idx, instance_id]
            return [tuple(a) for a in json.load(f)]

    @staticmethod
    def _bounding_box(mask_np: np.ndarray):
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]] if rows.any() else (0, 0)
        x_min, x_max = np.where(cols)[0][[0, -1]] if cols.any() else (0, 0)
        return x_min, y_min, x_max, y_max

    def __len__(self):
        return len(self.text_annotations)

        # Read a mask from the extracted H5 annotation.
    def _read_h5_mask(self, video_id, frame_idx, instance_id):
        anno_path = os.path.join(
            self.mask_annotations_dir, str(video_id), f"{int(frame_idx):05d}.h5"
        )
        with h5py.File(anno_path, "r") as f:
            instances = list(f["instance"])
            instance_masks = np.array(f["reMask"])
            if len(instances) == 1:
                instance_masks = instance_masks[np.newaxis, ...]
            if instance_masks.ndim == 3 and instance_masks.shape[1] != instance_masks.shape[2]:
                instance_masks = np.transpose(instance_masks, (0, 2, 1))

            instances_py = [int(x) for x in instances]
            tgt_idx = instances_py.index(int(instance_id))
            mask_np = (instance_masks[tgt_idx] > 0).astype(np.float32)
        return mask_np

    def __getitem__(self, idx):
        instance_ok = False
        while not instance_ok:
            text_query, video_id, frame_idx, instance_id = self.text_annotations[idx]
            text_query = " ".join(text_query.lower().split())

            # A2D annotations are 1-indexed for frames
            valid_frame_id = int(frame_idx) - 1

            # Read the video frames
            video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
            video_frames, _, _ = read_video(video_path, pts_unit="sec")
            vid_len = len(video_frames)

            sample_indices = [valid_frame_id]
            if self.num_frames > 1:
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_idx = [max(0, valid_frame_id - sample_id_before),
                             min(vid_len - 1, valid_frame_id + sample_id_after)]
                sample_indices.extend(local_idx)

                # Deduplicate once to avoid repeated edge frames from local sampling.
                sample_indices = list(set(sample_indices))

                # Fill missing samples when deduplication leaves too few frames.
                if len(sample_indices) < self.num_frames:
                    all_inds = list(range(vid_len))
                    remain_inds = [i for i in all_inds if i not in sample_indices]
                    global_n = self.num_frames - len(sample_indices)
                    if len(remain_inds) >= global_n:
                        sample_indices.extend(random.sample(remain_inds, global_n))
                    else:
                        sample_indices.extend(random.choices(all_inds, k=global_n))

                # Trim to num_frames while keeping the annotated frame.
                if len(sample_indices) > self.num_frames:
                    # Randomly drop extra non-annotated frames.
                    sample_indices.remove(valid_frame_id)
                    sample_indices = random.sample(sample_indices, self.num_frames - 1)
                    sample_indices.append(valid_frame_id)

            sample_indices.sort()
            # Record the annotated frame position within sample_indices.
            valid_indices_in_sample = sample_indices.index(valid_frame_id)

            # --- 2. BeIT branch sampling ---
            if vid_len <= self.max_beit_frames:
                beit_indx = list(range(vid_len))
            else:
                required_inds = sample_indices.copy()
                remaining_count = self.max_beit_frames - len(required_inds)
                available_inds = [i for i in range(vid_len) if i not in required_inds]
                step_idx = np.linspace(0, len(available_inds) - 1, max(1, remaining_count)).astype(int)
                extra_inds = [available_inds[i] for i in step_idx]
                beit_indx = sorted(list(set(required_inds + extra_inds)))[:self.max_beit_frames]

            # Record each SAM sample position in the BeIT sequence.
            sampled_relative_indices = [beit_indx.index(i) if i in beit_indx else 0 for i in sample_indices]

            # --- 3. Image and target processing ---
            imgs_sam, imgs_beit, labels, boxes, masks, valid = [], [], [], [], [], []

            # Process the SAM branch.
            for s_idx in sample_indices:
                frame_tensor = video_frames[s_idx]
                img_np = frame_tensor.numpy()

                img_s, resize = sam_preprocess(img_np)
                imgs_sam.append(img_s)

                if s_idx == valid_frame_id:
                    # Read the mask only for the annotated frame.
                    mask_np = self._read_h5_mask(video_id, frame_idx, instance_id)

                    if (mask_np > 0).any():
                        x1, y1, x2, y2 = self._bounding_box(mask_np)  # Fixed coordinate unpacking order.
                        boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float))
                        valid.append(1)
                    else:
                        boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float))
                        valid.append(0)

                    masks.append(torch.from_numpy(mask_np))
                else:
                    # Use empty targets for non-annotated frames.
                    boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float))
                    masks.append(torch.zeros(frame_tensor.shape[:2], dtype=torch.float32))
                    valid.append(0)

                labels.append(torch.tensor(0, dtype=torch.long))

            # Process the BeIT branch.
            for b_idx in beit_indx:
                img_b = video_frames[b_idx].numpy()
                imgs_beit.append(self.transforms_beit(img_b))

            # --- 4. Pack the target dictionary ---
            h, w = video_frames[0].shape[:2]
            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            masks = torch.stack(masks, dim=0)

            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            target = {
                "frames_idx": torch.tensor(sample_indices),
                "sampled_indices": torch.tensor(sampled_relative_indices),
                "valid_indices": torch.tensor([valid_indices_in_sample]),  # Use the correct relative index.
                "labels": labels,
                "boxes": boxes,
                "masks": masks,
                "valid": torch.tensor(valid),
                "caption": text_query,
                "orig_size": torch.as_tensor([int(h), int(w)]),
                "resize": resize,
                "vid_len": len(beit_indx)
            }

            imgs_sam = torch.stack(imgs_sam, dim=0)
            imgs_beit = torch.stack(imgs_beit, dim=0)

            if torch.any(target["valid"] == 1):
                instance_ok = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs_sam, imgs_beit, target


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size: int = 1024,
):
    # Normalize colors
    x = ResizeLongestSide(img_size).apply_image(x)
    h, w = resize_shape = x.shape[:2]
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    x = (x - pixel_mean) / pixel_std

    # Pad to square
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x, [resize_shape]
