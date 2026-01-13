# bridge_dataset_process_depth_refactor_all.py
# 目标：
# 1) 每个 episode 下的 images0/images1/... (或 image0/image1/...) -> 输出到 output_dir/{episode_id}/images{sid}/rgb.mp4
# 2) caption / segmentation 全部只读 output_dir/{episode_id}/images{sid}/rgb.mp4
# 3) 输出 masks: output_dir/{episode_id}/images{sid}/frame_XXXX.npz + result.gif
# 4) 兼容 imagesK / imageK 命名差异（优先 imagesK）

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import random
import fnmatch
import shutil
import inspect
from typing import Optional, List, Dict, Tuple
from collections import Counter, defaultdict
from itertools import chain
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import cv2
import decord

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from transformers.generation import GenerationConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import supervision as sv
from supervision.draw.color import ColorPalette, Color

# SAM2 imports
from thirdparty.grounded_sam_2.sam2.build_sam import (
    build_sam2_video_predictor,
    build_sam2,
)
from thirdparty.grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from thirdparty.grounded_sam_2.sam2.sam2_video_predictor import SAM2VideoPredictor


# -------------------------
# Colors (for visualization)
# -------------------------
STATIC_COLORS_60: List[tuple[int, int, int]] = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
    (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 0, 128), (128, 0, 0), (0, 128, 0),
    (0, 128, 128), (0, 0, 128), (184, 134, 11), (34, 139, 34), (30, 144, 255),
    (255, 99, 71), (218, 112, 214), (0, 191, 255), (255, 140, 0), (46, 139, 87),
    (188, 143, 143), (123, 104, 238), (72, 61, 139), (199, 21, 133), (0, 206, 209),
    (176, 196, 222), (255, 182, 193), (205, 92, 92), (135, 206, 235), (154, 205, 50),
    (233, 150, 122), (250, 128, 114), (186, 85, 211), (107, 142, 35), (95, 158, 160),
    (100, 149, 237), (160, 82, 45), (70, 130, 180), (105, 105, 105), (0, 0, 0),
]


# -------------------------
# Utils: video IO
# -------------------------
def extract_frame_from_video(video_path: str, frame_idx: int = 0, temp_dir: Optional[str] = None) -> str:
    if temp_dir is None:
        temp_dir = os.path.dirname(video_path)
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"无法从视频中提取第 {frame_idx} 帧: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_image_path = os.path.join(temp_dir, f"{video_name}_frame_{frame_idx}.png")
    cv2.imwrite(temp_image_path, frame)
    return temp_image_path


def create_video_from_images(image_dir: str, video_path: str, fps: int = 30) -> None:
    """
    读取 image_dir 下的 im_XXXX.jpg 按序写入 mp4
    """
    image_files = []
    for file in os.listdir(image_dir):
        if file.startswith("im_") and file.endswith(".jpg"):
            num_str = file[3:-4]
            try:
                num = int(num_str)
                image_files.append((num, file))
            except ValueError:
                continue

    image_files.sort(key=lambda x: x[0])
    image_paths = [os.path.join(image_dir, f) for _, f in image_files]
    if not image_paths:
        raise ValueError(f"目录中没有找到有效的图片文件: {image_dir} (需要 im_XXXX.jpg)")

    first = cv2.imread(image_paths[0])
    if first is None:
        raise ValueError(f"无法读取图片: {image_paths[0]}")
    h, w = first.shape[:2]

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    written = 0
    for p in image_paths:
        fr = cv2.imread(p)
        if fr is None:
            print(f"[WARN] 无法读取图片 {p}，跳过")
            continue
        vw.write(fr)
        written += 1
    vw.release()

    if written == 0:
        raise RuntimeError(f"写入 0 帧，视频无效: {video_path}")

    print(f"[OK] 视频已创建: {video_path}, 帧数: {written}")


def copy_non_image_folders(src_dir: str, dst_dir: str) -> None:
    """
    复制非 image/images 开头的目录到 output episode 目录
    """
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        if os.path.isdir(src_path) and (item.startswith("image") or item.startswith("images")):
            continue

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


# -------------------------
# Step A: scan episodes & build output/{episode}/images{sid}/rgb.mp4
# -------------------------
_EP_RE = re.compile(r"^\d+$")
_IMG_DIR_RE = re.compile(r"^(images|image)(\d+)$")  # images0/images1... or image0/image1...

def list_episodes(input_dir: str, max_videos: Optional[int] = None) -> List[Tuple[int, str]]:
    eps = []
    for name in os.listdir(input_dir):
        if _EP_RE.match(name):
            p = os.path.join(input_dir, name)
            if os.path.isdir(p):
                eps.append((int(name), p))
    eps.sort(key=lambda x: x[0])
    if max_videos is not None:
        eps = eps[:max_videos]
    return eps


def find_all_image_stream_dirs(ep_dir: str) -> Dict[int, str]:
    """
    返回 {stream_id: 绝对路径}
    优先 imagesK，其次 imageK（如果两者都存在，优先 imagesK）
    """
    found: Dict[int, str] = {}

    # 先扫 imagesK
    for sub in os.listdir(ep_dir):
        m = _IMG_DIR_RE.match(sub)
        if not m:
            continue
        prefix, sid_str = m.group(1), m.group(2)
        sid = int(sid_str)
        subp = os.path.join(ep_dir, sub)
        if not os.path.isdir(subp):
            continue
        if prefix == "images":
            found[sid] = subp

    # 再补 imageK（仅当 imagesK 不存在）
    for sub in os.listdir(ep_dir):
        m = _IMG_DIR_RE.match(sub)
        if not m:
            continue
        prefix, sid_str = m.group(1), m.group(2)
        sid = int(sid_str)
        subp = os.path.join(ep_dir, sub)
        if not os.path.isdir(subp):
            continue
        if prefix == "image" and sid not in found:
            found[sid] = subp

    return dict(sorted(found.items(), key=lambda x: x[0]))


def build_rgb_videos_all_streams(
    input_dir: str,
    output_dir: str,
    max_videos: Optional[int] = None,
    fps: int = 30,
) -> List[Tuple[str, int, str]]:
    """
    为每个 episode 的每个 stream 生成：
      output/{episode_id}/images{sid}/rgb.mp4

    返回列表：[(episode_id, sid, sample_id), ...]
    sample_id = "{episode_id}_s{sid}" 用于 captions/seg 的唯一键
    """
    episodes = list_episodes(input_dir, max_videos=max_videos)
    samples: List[Tuple[str, int, str]] = []

    print("=" * 50)
    print("[Step A] Building rgb.mp4 under output/{episode}/images{sid}/ ...")
    print("=" * 50)

    for ep_int, ep_dir in tqdm(episodes, desc="Build rgb videos"):
        episode_id = str(ep_int)
        streams = find_all_image_stream_dirs(ep_dir)
        if not streams:
            print(f"[WARN] episode {ep_int:05d} has no imagesK/imageK: {ep_dir}")
            continue

        for sid, img_dir in streams.items():
            out_img_dir = os.path.join(output_dir, episode_id, f"images{sid}")
            os.makedirs(out_img_dir, exist_ok=True)

            rgb_mp4 = os.path.join(out_img_dir, "rgb.mp4")
            if os.path.exists(rgb_mp4) and os.path.getsize(rgb_mp4) > 0:
                sample_id = f"{episode_id}_s{sid}"
                samples.append((episode_id, sid, sample_id))
                continue

            try:
                create_video_from_images(img_dir, rgb_mp4, fps=fps)
                sample_id = f"{episode_id}_s{sid}"
                samples.append((episode_id, sid, sample_id))
            except Exception as e:
                print(f"[WARN] failed episode={episode_id} stream={sid}: {e}")

    print(f"[OK] built rgb.mp4 for {len(samples)} (episode,stream) pairs")
    return samples


# -------------------------
# Step 1: captions from output/{episode}/images{sid}/rgb.mp4
# -------------------------
@torch.no_grad()
def get_captions_from_stream_videos(
    output_dir: str,
    samples: List[Tuple[str, int, str]],  # (episode_id, sid, sample_id)
    extract_frame_idx: int = 0,
    device: str = "cuda:0",
) -> None:
    device_obj = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat-Int4",
        device_map=device_obj,
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)

    # 规避 past_key_values NoneType
    model.generation_config.use_cache = False

    # pad_token==eos_token 警告：pad 对齐 eos
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        model.generation_config.pad_token_id = eos_id
        model.config.pad_token_id = eos_id

    captions_dir = os.path.join(output_dir, "captions")
    os.makedirs(captions_dir, exist_ok=True)
    save_path = os.path.join(captions_dir, "rank_0.jsonl")
    temp_dir = os.path.join(captions_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    results = []
    print("=" * 50)
    print("[Step 1] Generating captions from output/{episode}/images{sid}/rgb.mp4 ...")
    print("=" * 50)

    for episode_id, sid, sample_id in tqdm(samples, desc="Captioning"):
        video_path = os.path.join(output_dir, episode_id, f"images{sid}", "rgb.mp4")
        if not os.path.exists(video_path):
            print(f"[WARN] missing rgb.mp4: {video_path}")
            continue

        try:
            frame_path = extract_frame_from_video(video_path, extract_frame_idx, temp_dir)
        except Exception as e:
            print(f"[WARN] extract frame failed {video_path}: {e}")
            continue

        query = tokenizer.from_list_format([
            {"image": frame_path},
            {"text": "List the main object classes in the image, with only one word for each class (no more than ten):"},
        ])

        response, _ = model.chat(
            tokenizer,
            query=query,
            history=None,
            generation_config=model.generation_config,
        )

        labels = [x.lower().strip() for x in response.strip(".").split(",") if x.strip()]
        labels = list(set(labels))
        if len(labels) > 10:
            continue

        results.append({
            "episode_id": sample_id,  # ✅ 唯一键 = episode+stream
            "split": "video",
            "raw_labels": labels,
            "meta": {"episode": episode_id, "stream": sid},
        })

    if results:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in sorted(results, key=lambda x: x["episode_id"])))

    print(f"[OK] captions saved to: {save_path}, count={len(results)}")


# -------------------------
# Step 2: postprocess captions (基本保持你的逻辑)
# -------------------------
@torch.no_grad()
def postprocess_captions(data_dir: str, save_dir: str = None):
    if save_dir is None:
        save_dir = os.path.dirname(data_dir) if os.path.basename(data_dir) == "captions" else data_dir

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")

    caption_files = list(sorted(fnmatch.filter(os.listdir(data_dir), "rank*.jsonl")))
    all_captions = []

    if len(caption_files) > 0:
        for file in tqdm(caption_files, desc="Loading captions ..."):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                r = [json.loads(line.strip()) for line in f if line.strip()]
                all_captions.extend(r)
        print(f"Loaded captions for all {len(all_captions)} episodes.")
    else:
        with open(os.path.join(data_dir, "all_captions.jsonl"), "r", encoding="utf-8") as f:
            all_captions = [json.loads(line.strip()) for line in f if line.strip()]

    all_labels = list(chain(*[
        caption["raw_labels"] for caption in tqdm(all_captions, desc="Extracting labels ...")
    ]))
    print(f"Have {len(all_labels)} raw labels.")

    # sort by episode_id string (sample_id like "0_s0") - keep stable
    all_captions = list(sorted(all_captions, key=lambda x: x["episode_id"]))
    all_labels = list(chain(*[
        caption["raw_labels"] for caption in tqdm(all_captions, desc="Re-extracting labels ...")
    ]))

    pattern = re.compile(r"^[A-Za-z ]+$")
    for i in tqdm(range(len(all_captions)), desc="Filter out outliers ..."):
        raw_labels = all_captions[i]["raw_labels"]
        track_labels = list(filter(pattern.match, raw_labels))
        all_captions[i]["track_labels"] = track_labels
    all_labels = list(filter(pattern.match, all_labels))
    print(f"Have {len(all_labels)} labels after filtering out outliers.")

    label_counts = Counter(all_labels)
    frequency = np.array(list(label_counts.values()))
    threshold = max(1, np.percentile(frequency, 10))

    for i in tqdm(range(len(all_captions)), desc="Filter out repetitions ..."):
        track_labels = all_captions[i]["track_labels"]
        track_labels = list(set(track_labels))
        all_captions[i]["track_labels"] = track_labels

    all_labels = list(set(all_labels))
    extra_labels = ["gripper", "countertop", "otherproperty", "background"]
    all_labels.extend(extra_labels)
    print(f"Have {len(all_labels)} labels after filtering out repetitions.")

    raw_labels_file_path = os.path.join(save_dir, "raw_labels.txt")
    os.makedirs(save_dir, exist_ok=True)
    with open(raw_labels_file_path, "w", encoding="utf-8") as f:
        f.writelines(lbl + "\n" for lbl in all_labels)
    print(f"Raw labels are saved to {raw_labels_file_path}.")

    # text embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tok2 = AutoTokenizer.from_pretrained(model_name)
    enc_model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    batch_size = 128
    all_labels = list(sorted(all_labels))
    for i in tqdm(range(0, len(all_labels), batch_size), desc="Get text embeddings ..."):
        batch_labels = all_labels[i: i + batch_size]
        tokens = tok2(batch_labels, padding=True, truncation=True, return_tensors="pt").to(device)
        output = enc_model(**tokens)
        token_embeddings = output[0]
        input_mask_expanded = tokens["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        emb = (torch.sum(token_embeddings * input_mask_expanded, 1) /
               torch.clamp(input_mask_expanded.sum(1), min=1e-9)).cpu()
        embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"Embeddings shape: {embeddings.shape}")

    n_samples, n_features = embeddings.shape
    n_components = min(128, n_samples, n_features)
    if n_components < n_features:
        embeddings = PCA(n_components=n_components).fit_transform(embeddings)
        print(f"Applied PCA: {n_features} -> {n_components}")

    num_clusters = min(51, len(all_labels))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=50)
    cluster_ids = kmeans.fit_predict(embeddings)

    cluster_to_labels = [[] for _ in range(num_clusters)]
    for label, cid in zip(all_labels, cluster_ids):
        cluster_to_labels[cid].append(label)

    top_labels = {}
    for cid, cluster_labels in enumerate(tqdm(cluster_to_labels, desc="Getting top labels ...")):
        valid = list(filter(
            lambda lbl: ((label_counts[lbl] > threshold and len(lbl.split(" ")) == 1) or lbl in extra_labels),
            cluster_labels
        ))
        if not valid:
            continue

        idxs = [all_labels.index(lbl) for lbl in valid]
        cluster_embeds = embeddings[idxs]
        center = cluster_embeds.mean(axis=0)
        dist = np.linalg.norm(cluster_embeds - center, axis=1)
        top_label = valid[np.argmin(dist)]
        top_labels[cid] = top_label

    labels = list(top_labels.values())
    if "background" not in labels:
        labels.append("background")

    top_labels_file_path = os.path.join(save_dir, "labels.txt")
    with open(top_labels_file_path, "w", encoding="utf-8") as f:
        f.writelines(lbl + "\n" for lbl in labels)
    print(f"Finalized top labels are saved to {top_labels_file_path}.")

    # map labels to top labels
    label_map = {}
    for cid, cluster_labels in enumerate(tqdm(cluster_to_labels, desc="Building label map ...")):
        top_label = top_labels.get(cid, "background")
        for lbl in cluster_labels:
            label_map[lbl] = top_label

    # save clusters
    label_clusters: Dict[str, List[str]] = defaultdict(list)
    for k, v in label_map.items():
        label_clusters[v].append(k)
    label_clusters_ = [{k: list(sorted(label_clusters[k]))} for k in label_clusters.keys()]
    label_clusters_ = list(sorted(label_clusters_, key=lambda x: list(x.keys())[0]))
    with open(os.path.join(save_dir, "label_clusters.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(map(json.dumps, label_clusters_)))

    # update episodes
    for i in tqdm(range(len(all_captions)), desc="Updating labels for each episodes ..."):
        caption = all_captions[i]
        new_labels = [label_map[label] for label in caption["track_labels"]]
        caption["labels"] = new_labels
        caption["label_ids"] = [labels.index(label) for label in new_labels]
        all_captions[i] = caption

    # keep stable sort
    all_captions = list(sorted(all_captions, key=lambda x: x["episode_id"]))
    with open(os.path.join(save_dir, "all_captions.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(map(json.dumps, all_captions)))

    print(f"[OK] 后处理完成，结果已保存到: {save_dir}")


# -------------------------
# Step 3: segmentation from output/{episode}/images{sid}/rgb.mp4
# -------------------------
@torch.no_grad()
def get_labels_from_stream_videos(
    output_dir: str,
    samples: List[Tuple[str, int, str]],  # (episode_id, sid, sample_id)
    all_captions_file_path: str,
    all_labels_file_path: str,
    box_th: float = 0.25,
    text_th: float = 0.3,
):
    device = torch.device("cuda:0")

    with open(all_captions_file_path, "r", encoding="utf-8") as f:
        all_captions_list = [json.loads(line.strip()) for line in f if line.strip()]
    with open(all_labels_file_path, "r", encoding="utf-8") as f:
        all_labels = [line.strip() for line in f if line.strip()]

    # sample_id -> captions
    all_captions = {
        ep["episode_id"]: {"track_labels": ep["track_labels"], "label_ids": ep["label_ids"], "meta": ep.get("meta", {})}
        for ep in all_captions_list
    }

    # optional extra label
    all_labels.append("black robot gripper")

    # SAM2
    sam2_checkpoint = "thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l"
    video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)  # not strictly required, but keep for consistency

    # GroundingDINO
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def postprocess_grounding(outputs, input_ids, target_sizes):
        fn = processor.post_process_grounded_object_detection
        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs = {"target_sizes": target_sizes}

        # transformers version differences
        if "box_threshold" in params:
            kwargs["box_threshold"] = box_th
        elif "threshold" in params:
            kwargs["threshold"] = box_th
        elif "score_threshold" in params:
            kwargs["score_threshold"] = box_th

        if "text_threshold" in params:
            kwargs["text_threshold"] = text_th

        return fn(outputs, input_ids, **kwargs)

    print("=" * 50)
    print("[Step 3] Generating masks under output/{episode}/images{sid}/ ...")
    print("=" * 50)

    for episode_id, sid, sample_id in tqdm(samples, desc="Segmenting"):
        traj = all_captions.get(sample_id, None)
        if traj is None:
            print(f"[WARN] Not found captions for sample_id={sample_id}")
            continue

        video_path = os.path.join(output_dir, episode_id, f"images{sid}", "rgb.mp4")
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            print(f"[WARN] Missing/empty rgb.mp4: {video_path}")
            continue

        traj_labels = list(traj["track_labels"])
        label_ids = list(traj["label_ids"])

        # 额外补一个 robot arm
        traj_labels.append("robot arm")
        label_ids.append(len(all_labels) - 1)

        text = f"{', '.join(traj_labels)}."
        print(f"Processing {sample_id} with text={text}")

        save_dir = os.path.join(output_dir, episode_id, f"images{sid}")
        os.makedirs(save_dir, exist_ok=True)

        # 读视频
        try:
            vr = decord.VideoReader(uri=video_path, num_threads=2)
        except Exception as e:
            print(f"[WARN] decord cannot open {video_path}: {e}")
            continue

        batch = vr.get_batch(range(len(vr)))
        frames = batch.asnumpy() if hasattr(batch, "asnumpy") else np.asarray(batch)
        frames = frames.astype(np.uint8)

        existing = fnmatch.filter(os.listdir(save_dir), "frame_*.npz")
        if len(existing) == len(frames) and len(frames) > 0:
            print(f"[SKIP] {sample_id} already done")
            continue

        try:
            inference_state = video_predictor.init_state(video_path=video_path)

            init_frame = Image.fromarray(frames[0])
            inputs = processor(images=init_frame, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = postprocess_grounding(
                outputs=outputs,
                input_ids=inputs.input_ids,
                target_sizes=[init_frame.size[::-1]],
            )

            input_boxes = results[0]["boxes"].detach().cpu().numpy()
            OBJECTS = results[0]["labels"]

            if len(input_boxes) == 0:
                print(f"[WARN] {sample_id}: GroundingDINO found 0 boxes")
                continue

            valid = [i for i, obj in enumerate(OBJECTS) if obj in traj_labels]
            if len(valid) == 0:
                print(f"[WARN] {sample_id}: no detected objects match traj_labels")
                continue

            input_boxes = np.asarray([input_boxes[i] for i in valid])
            OBJECTS = [OBJECTS[i] for i in valid]

            global_ids = np.array([label_ids[traj_labels.index(obj)] for obj in OBJECTS], dtype=np.uint8)
            ID_TO_OBJECT = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

            # add box prompts
            for object_id, box in enumerate(input_boxes, start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    box=box,
                )

            # propagate
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # ✅ 强制保存 gif，确保你“看得到 mask”
            result_frames = []
            for frame_idx, segs in video_segments.items():
                img = frames[frame_idx]
                object_ids = list(segs.keys())
                masks_arr = np.concatenate(list(segs.values()), axis=0)

                np.savez_compressed(
                    os.path.join(save_dir, f"frame_{frame_idx:04d}.npz"),
                    masks=masks_arr.astype(np.bool_),
                    track_labels=np.array(OBJECTS, dtype=object),
                    object_ids=np.array(object_ids, dtype=np.uint8),
                    label_ids=global_ids.astype(np.uint8),
                    sample_id=sample_id,
                )

                dets = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks_arr),
                    mask=masks_arr,
                    class_id=np.array(object_ids, dtype=np.int32),
                )
                labels_local = []
                for oid in object_ids:
                    obj = ID_TO_OBJECT.get(oid, "obj")
                    gid = int(global_ids[oid - 1]) if (oid - 1) < len(global_ids) else 255
                    labels_local.append(f"{gid}:{obj}")

                annotated = sv.MaskAnnotator().annotate(scene=img.copy(), detections=dets)
                annotated = sv.LabelAnnotator().annotate(annotated, detections=dets, labels=labels_local)
                result_frames.append(Image.fromarray(annotated))

            if result_frames:
                gif_path = os.path.join(save_dir, "result.gif")
                result_frames[0].save(gif_path, save_all=True, append_images=result_frames[1:], duration=100, loop=0)
                print(f"[OK] saved gif: {gif_path}")

        except Exception as e:
            print(f"[WARN] Get labels failed {sample_id}: {e}")
            continue


# -------------------------
# Step 4: label visualization postprocess (将 frame_*.npz 再写 annotated_frame_color/index)
# -------------------------
def generate_colors() -> List[tuple[int, int, int]]:
    return STATIC_COLORS_60


def _postprocess_labels_one_dir(mask_annotator: sv.MaskAnnotator, save_dir: str, global_step: int):
    frames = list(sorted(fnmatch.filter(os.listdir(save_dir), "frame_*.npz")))
    annotated_frames = []

    for frame in (pbar := tqdm(frames, leave=False, desc=f"Post frames ({os.path.basename(save_dir)})")):
        pbar.set_postfix(frame=frame)
        fp = os.path.join(save_dir, frame)
        try:
            data = np.load(fp, allow_pickle=True)
            labels = dict(data)
            data.close()
        except Exception as e:
            print(f"[WARN] load npz failed: {fp} -> {e}")
            continue

        masks = labels["masks"].astype(np.bool_)
        label_ids = labels["label_ids"]

        if ("annotated_frame_color" in labels) and ("annotated_frame_index" in labels):
            continue

        dets = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(label_ids, dtype=np.int32),
        )

        annotated = mask_annotator.annotate(
            scene=np.zeros((*masks.shape[-2:], 3), dtype=np.uint8),
            detections=dets
        )
        annotated = sv.LabelAnnotator().annotate(
            annotated, detections=dets, labels=[str(i) for i in label_ids]
        )

        idx_map = np.zeros(masks.shape[-2:], dtype=np.int32) - 1
        for det_mask, lid in zip(dets.mask, label_ids):
            idx_map[det_mask] = lid
        idx_map = idx_map.astype(np.uint8)

        labels["annotated_frame_color"] = annotated
        labels["annotated_frame_index"] = idx_map
        np.savez_compressed(fp, **labels)
        annotated_frames.append(annotated)

    if (np.random.rand() < 0.5 or global_step < 20) and len(annotated_frames) > 0:
        imgs = [Image.fromarray(a) for a in annotated_frames]
        imgs[0].save(
            os.path.join(save_dir, "result2.gif"),
            save_all=True,
            append_images=imgs[1:],
            duration=100,
            loop=0,
        )


def postprocess_labels(output_dir: str):
    colors60_list = generate_colors()
    colors60_list[-1] = (0, 0, 0)
    colors60 = ColorPalette([Color(*c) for c in colors60_list])
    mask_annotator = sv.MaskAnnotator(color=colors60, opacity=1.0)

    # 遍历 output_dir/{episode}/images{sid}/
    episodes = [d for d in os.listdir(output_dir) if d.isdigit() and os.path.isdir(os.path.join(output_dir, d))]
    episodes.sort(key=lambda x: int(x))

    tasks = []
    for ep in episodes:
        ep_dir = os.path.join(output_dir, ep)
        for sub in os.listdir(ep_dir):
            if sub.startswith("images") and os.path.isdir(os.path.join(ep_dir, sub)):
                tasks.append(os.path.join(ep_dir, sub))

    print(f"[Step 4] Found {len(tasks)} stream dirs for postprocess.")
    for i, sd in enumerate(tqdm(tasks, desc="Postprocess streams")):
        _postprocess_labels_one_dir(mask_annotator, sd, i)


# -------------------------
# Main pipeline
# -------------------------
def main(
    input_dir: str,
    output_dir: str,
    max_videos: Optional[int] = None,
    extract_frame_idx: int = 0,
    device: str = "cuda:0",
):
    os.makedirs(output_dir, exist_ok=True)

    # Step 0: copy non-image folders into output episode folders
    print("=" * 50)
    print("[Step 0] Copying non-image folders...")
    print("=" * 50)

    episodes = list_episodes(input_dir, max_videos=max_videos)
    for ep_int, ep_path in episodes:
        episode_id = str(ep_int)
        dst_dir = os.path.join(output_dir, episode_id)
        try:
            copy_non_image_folders(ep_path, dst_dir)
        except Exception as e:
            print(f"[WARN] copy failed {ep_path}: {e}")

    # Step A: build rgb.mp4 under output/{episode}/images{sid}/
    samples = build_rgb_videos_all_streams(
        input_dir=input_dir,
        output_dir=output_dir,
        max_videos=max_videos,
        fps=30
    )
    if not samples:
        raise RuntimeError("No rgb.mp4 built. Check imagesK/imageK folders and im_XXXX.jpg naming.")

    # Step 1: captions
    get_captions_from_stream_videos(
        output_dir=output_dir,
        samples=samples,
        extract_frame_idx=extract_frame_idx,
        device=device,
    )

    # Step 2: postprocess captions
    print("=" * 50)
    print("[Step 2] Postprocessing captions...")
    print("=" * 50)
    caption_dir = os.path.join(output_dir, "captions")
    postprocess_captions(data_dir=caption_dir, save_dir=output_dir)

    # Step 3: segmentation
    print("=" * 50)
    print("[Step 3] Generating labels (segmentation)...")
    print("=" * 50)
    all_captions_file_path = os.path.join(output_dir, "all_captions.jsonl")
    all_labels_file_path = os.path.join(output_dir, "labels.txt")
    get_labels_from_stream_videos(
        output_dir=output_dir,
        samples=samples,
        all_captions_file_path=all_captions_file_path,
        all_labels_file_path=all_labels_file_path,
    )

    # Step 4: postprocess labels (write annotated_frame_color/index and optional gif)
    print("=" * 50)
    print("[Step 4] Postprocessing labels...")
    print("=" * 50)
    postprocess_labels(output_dir)

    print("=" * 50)
    print("[DONE] All processing completed!")
    print("=" * 50)


if __name__ == "__main__":
    input_dir = "/media/xuran-yao/WM/bridge_dataset/bridge_depth"
    output_dir = "/media/xuran-yao/WM/bridge_dataset/bridge_wc"
    max_videos = 10
    extract_frame_idx = 0
    device = "cuda:0"

    main(
        input_dir=input_dir,
        output_dir=output_dir,
        max_videos=max_videos,
        extract_frame_idx=extract_frame_idx,
        device=device,
    )