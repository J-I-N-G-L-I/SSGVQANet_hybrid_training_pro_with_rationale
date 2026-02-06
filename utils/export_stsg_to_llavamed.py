#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export STSG temporal QA into LLaVA-Med style JSONL + multi-frame image grids."
    )
    parser.add_argument(
        "--stsg_qa_root",
        required=True,
        help="Root directory of STSG temporal QA, e.g. /mnt/scratch/.../STSG_QA_Pro_8_Classes"
    )
    parser.add_argument(
        "--video_root",
        required=True,
        help="Root directory of raw frames, e.g. /mnt/scratch/.../CholecT45/data"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Where to save generated grids and question JSONL"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="List of video IDs, e.g. VID01 VID22 VID74 ..."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=16,
        help="Max number of frames to use per QA (will be sampled evenly if more)"
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=4,
        help="Number of rows in the frame grid"
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="Number of cols in the frame grid"
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=320,
        help="Width of each tile in the grid"
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        default=180,
        help="Height of each tile in the grid"
    )
    parser.add_argument(
        "--frame_ext",
        default=".png",
        help="Frame image extension (e.g. .png or .jpg)"
    )
    parser.add_argument(
        "--allowed_categories",
        nargs="+",
        default=[
            "count", "duration", "ordering", "extreme",
            "boundary", "motion", "concurrency", "phase_transition",
        ],
        help="Subset of categories to export"
    )
    parser.add_argument(
        "--output_questions",
        default="stsg_temporal_eval_questions.jsonl",
        # default="stsg_temporal_eval_questions_with_rationale.jsonl",
        help="Filename of question JSONL under output_root"
    )
    parser.add_argument(
        "--use_rationale",
        action="store_true",
        help="If set, append available English rationales to the prompt text."
    )
    parser.add_argument(
        "--vocabs_json",
        required=True,
        help="Path to vocabs_train.json (used to build closed-set prompts for text tasks)"
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    

def even_sample(sequence: List[int], max_len: int) -> List[int]:
    """Evenly sample up to `max_len` elements from a non-empty list."""
    if len(sequence) == 0:
        raise ValueError("even_sample() received an empty sequence.")
    if len(sequence) <= max_len:
        return list(sequence)

    indices = []
    n = len(sequence)
    for i in range(max_len):
        pos = int(round(i * (n - 1) / max(max_len - 1, 1)))
        indices.append(sequence[pos])
    return indices


def extract_frame_range_and_sample(
    qa_item: Dict[str, Any],
    max_frames: int
) -> List[int]:
    """
    Extract a frame range from evidence.scope.frames or evidence.keyframes,
    then evenly sample up to max_frames frame indices.
    """
    evidence = qa_item.get("evidence", {})
    scope = evidence.get("scope", {})
    keyframes = evidence.get("keyframes", None)

    start_f = None
    end_f = None

    # 1) Prefer explicit scope.frames = [start, end]
    frames = scope.get("frames", None)
    if isinstance(frames, list) and len(frames) == 2:
        try:
            start_f = int(frames[0])
            end_f = int(frames[1])
        except Exception:
            start_f = None
            end_f = None

    # 2) If scope.frames is not available, fall back to min/max of keyframes
    if (start_f is None or end_f is None) and isinstance(keyframes, list) and len(keyframes) > 0:
        kfs = [int(x) for x in keyframes]
        start_f = min(kfs)
        end_f = max(kfs)

    if start_f is None or end_f is None:
        raise ValueError("Cannot determine frame range from QA evidence.")

    if end_f < start_f:
        start_f, end_f = end_f, start_f

    all_frames = list(range(start_f, end_f + 1))
    sampled = even_sample(all_frames, max_frames)
    return sampled


def load_and_resize(img_path: Path, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


def build_grid(
    frame_paths: List[Path],
    out_path: Path,
    rows: int,
    cols: int,
    tile_size: Tuple[int, int],
):
    """Build a grid image from list of frame paths and save it."""
    if len(frame_paths) == 0:
        raise ValueError("Empty frame_paths when building grid image.")

    total_cells = rows * cols
    if len(frame_paths) < total_cells:
        frame_paths = frame_paths + [frame_paths[-1]] * (total_cells - len(frame_paths))
    else:
        frame_paths = frame_paths[:total_cells]

    tile_w, tile_h = tile_size
    grid_w = cols * tile_w
    grid_h = rows * tile_h

    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

    for idx, frame_path in enumerate(frame_paths):
        r = idx // cols
        c = idx % cols
        try:
            tile = load_and_resize(frame_path, (tile_w, tile_h))
        except FileNotFoundError:
            # If a frame is missing, reuse the previous tile or fill with black
            if idx > 0:
                tile = grid.crop(
                    (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
                )
            else:
                tile = Image.new("RGB", (tile_w, tile_h), color=(0, 0, 0))
        grid.paste(tile, (c * tile_w, r * tile_h))

    grid.save(out_path, format="JPEG", quality=95)


def normalize_category(cat: str) -> str:
    """Normalize category string to match training schema."""
    if cat is None:
        return "unknown"
    cat = str(cat).lower()
    if cat in ["phase", "phase_transition"]:
        return "phase_transition"
    return cat


def infer_choice_indices_from_question(question_text: str) -> List[int]:
    """
    Infer choice indices from patterns like '1)' '2)' '3)' in the question text.
    """
    if not isinstance(question_text, str):
        return []
    indices = set()
    for m in re.finditer(r"\b(\d+)\)", question_text):
        try:
            indices.add(int(m.group(1)))
        except Exception:
            continue
    return sorted(indices)


def question_looks_like_choice(question_text: str) -> bool:
    """
    Heuristic: detect whether this question is a multiple-choice numeric question.
    """
    if not isinstance(question_text, str):
        return False
    s = question_text
    # If it lists options like "1) ... 2) ...", we consider it a choice question.
    has_opt1 = re.search(r"\b1\)", s) is not None
    has_opt2 = re.search(r"\b2\)", s) is not None
    if has_opt1 and has_opt2:
        return True
    # Optional: also check for explicit instruction
    if "answer with a number" in s.lower():
        return True
    return False


def infer_label_type_and_task(
    category: str,
    raw_answer_type: str,
    question_text: str
) -> Tuple[str, str, Optional[List[int]]]:
    """
    Map (category, raw_answer_type, question text) to a refined label_type and task name.
    label_type in {
        'count', 'seconds', 'bool', 'text', 'choice_index'
    }
    task is a string aligned with your multi-head schema
    (e.g. 'count', 'duration', 'boundary_bool', 'concurrency_choice', etc.).
    Returns (label_type, task, choice_indices or None).
    """
    cat = (category or "unknown").lower()
    raw = (raw_answer_type or "unknown").lower()
    q = question_text or ""

    # Default values
    label_type = "text"
    task = "unknown"
    choice_indices: Optional[List[int]] = None

    # Count
    if cat == "count":
        if raw == "numeric":
            label_type = "count"
            task = "count"
            return label_type, task, choice_indices

    # Duration
    if cat == "duration":
        if raw == "numeric":
            label_type = "seconds"
            task = "duration"
            return label_type, task, choice_indices

    # Motion
    if cat == "motion":
        # Usually string class labels
        label_type = "text"
        task = "motion_text"
        return label_type, task, choice_indices

    # Boundary
    if cat == "boundary":
        if raw == "boolean":
            label_type = "bool"
            task = "boundary_bool"
            return label_type, task, choice_indices
        if raw == "numeric":
            label_type = "seconds"
            task = "boundary_reg"
            return label_type, task, choice_indices
        if raw == "string":
            label_type = "text"
            task = "boundary_text"
            return label_type, task, choice_indices

    # Concurrency
    if cat == "concurrency":
        if raw == "boolean":
            label_type = "bool"
            task = "concurrency_bool"
            return label_type, task, choice_indices
        if raw == "numeric":
            # numeric concurrency can be 'reg' or 'choice'
            if question_looks_like_choice(q):
                label_type = "choice_index"
                task = "concurrency_choice"
                choice_indices = infer_choice_indices_from_question(q)
            else:
                label_type = "seconds"
                task = "concurrency_reg"
            return label_type, task, choice_indices

    # Extreme
    if cat == "extreme":
        if raw == "boolean":
            label_type = "bool"
            task = "extreme_bool"
            return label_type, task, choice_indices
        if raw == "numeric":
            label_type = "seconds"
            task = "extreme_reg"
            return label_type, task, choice_indices
        if raw == "string":
            label_type = "text"
            task = "extreme_text"
            return label_type, task, choice_indices

    # Ordering
    if cat == "ordering":
        if raw == "boolean":
            label_type = "bool"
            task = "ordering_bool"
            return label_type, task, choice_indices
        if raw == "string":
            label_type = "text"
            task = "ordering_text"
            return label_type, task, choice_indices
        if raw == "numeric":
            # ordering choice questions use numeric options 1..K
            label_type = "choice_index"
            task = "ordering_choice"
            choice_indices = infer_choice_indices_from_question(q)
            return label_type, task, choice_indices

    # Phase / Phase transition
    if cat in ["phase", "phase_transition"]:
        if raw == "boolean":
            label_type = "bool"
            task = "phase_bool"
            return label_type, task, choice_indices
        if raw == "numeric":
            if question_looks_like_choice(q):
                label_type = "choice_index"
                task = "phase_choice"
                choice_indices = infer_choice_indices_from_question(q)
            else:
                label_type = "seconds"
                task = "phase_reg"
            return label_type, task, choice_indices
        if raw == "string":
            label_type = "text"
            task = "phase_text"
            return label_type, task, choice_indices

    # Fallbacks
    if raw == "boolean":
        label_type = "bool"
        task = f"{cat}_bool"
    elif raw == "numeric":
        label_type = "seconds"
        task = f"{cat}_reg"
    elif raw == "string":
        label_type = "text"
        task = f"{cat}_text"
    else:
        label_type = "text"
        task = "unknown"

    return label_type, task, choice_indices


def extract_question_text(qa_item: Dict[str, Any]) -> str:
    """Extract question string."""
    if "question" in qa_item and isinstance(qa_item["question"], str):
        return qa_item["question"]
    if "q" in qa_item and isinstance(qa_item["q"], str):
        return qa_item["q"]
    raise ValueError("Cannot find question text in QA item.")


def extract_gt_answer(qa_item: Dict[str, Any]) -> Any:
    """
    Extract ground-truth answer from QA item.
    For now we simply read qa_item['answer'].
    """
    if "answer" in qa_item:
        return qa_item["answer"]
    for key in ["answer_str", "answer_text", "ans", "label"]:
        if key in qa_item:
            return qa_item[key]
    raise ValueError("Cannot find ground-truth answer in QA item.")


RATIONALE_KEYS = [
    "rationale_count_v1_en",
    "rationale_duration_v1_en",
    "rationale_motion_v1_en",
    "rationale_ordering_v1_en",
    "rationale_extreme_v1_en",
    "rationale_concurrency_v1_en",
    "rationale_boundary_v1_en",
    "rationale_phase_transition_v1_en",
]


def extract_rationale_text(qa_item: Dict[str, Any]) -> Optional[str]:
    """
    从 QA item 中抽取英文 rationale 文本。
    按照已知的一组 key 依次查找，找到第一个非空字符串就返回。
    没有就返回 None。
    注意：这里是从 qa_item 顶层找，不是 evidence 里。
    """
    for key in RATIONALE_KEYS:
        val = qa_item.get(key, None)
        if isinstance(val, str):
            val = val.strip()
            if val:
                return val
    return None


def build_prompt(
    question_text: str,
    label_type: str,
    choice_indices: Optional[List[int]],
    category: str,
    vocabs: Dict[str, List[str]],
    rationale_text: Optional[str] = None,
) -> str:
    """
    Build the LLaVA-Med prompt text.
    We use label_type to specify the answer format, and for text-type labels
    we further specialize by category using the closed-set vocabs.

    - motion_text: explicit 8-class options (closed set)
    - boundary_text / extreme_text / phase_text: list full vocab as options
    - ordering_text: enforce "<event_1> before <event_2>" pattern

    At the end, we always require the model to output "FINAL_ANSWER: <answer>"
    so that the evaluator can reliably parse the prediction.
    """
    header = (
        "You are a medical video analysis assistant.\n"
        "The 4x4 image grid shows frames from a laparoscopic cholecystectomy video.\n"
        "Frames are ordered chronologically from left to right, top to bottom.\n\n"
    )

    # 先拼 question
    body = f"Question: {question_text}\n"

    # 如果有 rationale，就附加在问题后面
    if rationale_text:
        body += (
            "\nYou are also given the following reasoning hint derived from the temporal evidence for this question.\n"
            "Use it only as an additional explanation and still rely on the visual content to answer correctly.\n"
            f"{rationale_text}\n"
        )

    cat = (category or "").lower()
    lt = (label_type or "text").lower()

    text_options_block = ""
    format_hint = ""

    # ---------- 非文本类 ----------
    if lt == "bool":
        format_hint = (
            "Answer with a single word: 'yes' or 'no'. "
            "Do not repeat the question or these instructions."
        )

    elif lt == "seconds":
        format_hint = (
            "Answer with a single non-negative integer number representing seconds. "
            "Do not output any units or extra words. "
            "Do not repeat the question or these instructions."
        )

    elif lt == "count":
        format_hint = (
            "Answer with a single non-negative integer number representing the count. "
            "Do not output any units or extra words. "
            "Do not repeat the question or these instructions."
        )

    elif lt == "choice_index":
        if choice_indices:
            min_idx = min(choice_indices)
            max_idx = max(choice_indices)
            format_hint = (
                f"Select the best option and answer with a single integer between "
                f"{min_idx} and {max_idx}. "
                "Do not output any extra words. "
                "Do not repeat the question or these instructions."
            )
        else:
            format_hint = (
                "Select the best option and answer with a single integer index. "
                "Do not output any extra words. "
                "Do not repeat the question or these instructions."
            )

    # ---------- 文本类 ----------
    else:  # lt == "text"
        # 1) motion_text: 8 类 closed-set
        if cat == "motion" and "motion" in vocabs:
            motion_labels = [x for x in vocabs["motion"] if x != "<UNK>"]
            text_options_block = (
                "Possible motion labels (closed set) are:\n"
                + "\n".join(f"- {lab}" for lab in motion_labels)
                + "\n"
            )
            format_hint = (
                "Select exactly ONE motion label from the list above and answer "
                "with that label only, without any extra words. "
                "Do not repeat the question or these instructions."
            )

        # 2) boundary_text / extreme_text / phase_text：列出整个 vocab
        elif cat in {"boundary", "extreme", "phase_transition"} and cat in vocabs:
            raw_labels = vocabs[cat]
            vocab_labels = [x for x in raw_labels if x != "<UNK>"]

            if cat == "phase_transition":
                pretty_name = "phase / phase-transition"
            else:
                pretty_name = cat

            text_options_block = (
                f"Possible {pretty_name} answer labels (closed set) are:\n"
                + "\n".join(f"- {lab}" for lab in vocab_labels)
                + "\n"
            )
            format_hint = (
                "Select exactly ONE label from the list above and answer with that "
                "label only, without any extra words. Many labels have the structure "
                "'<instrument> <verb> on <target>', e.g. 'hook dissect on gallbladder'. "
                "Do not repeat the question or these instructions."
            )

        # 3) ordering_text：只约束输出格式，不给全 vocab
        elif cat == "ordering":
            format_hint = (
                "Your answer must be a single short phrase in the format "
                "'<event_1> before <event_2>'. "
                "<event_1> and <event_2> must be short phrases taken directly from the "
                "question, describing surgical tool-action-target events. For example: "
                "'scissors cut liver before bipolar coagulate liver'. "
                "Do NOT repeat these instructions. Do NOT say 'the answer format is'. "
                "Do NOT explain your reasoning. Output ONLY the final phrase."
            )

        # 4) 其他文本类
        else:
            format_hint = (
                "Answer with a short phrase describing the correct answer, "
                "without any extra explanation. "
                "Do not repeat the question or these instructions."
            )

    # 统一添加 FINAL_ANSWER 协议
    format_hint = (
        format_hint
        + " Finally, output your final answer after the string 'FINAL_ANSWER:'. "
          "Do not write anything else after 'FINAL_ANSWER:'."
    )

    # 拼最终 prompt
    prompt = "<image>\n" + header + body

    if text_options_block:
        prompt += "\n" + text_options_block + "\n"

    prompt += f"\nAnswer format: {format_hint}\n"
    return prompt


def main():
    args = parse_args()

    stsg_root = Path(args.stsg_qa_root)
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    images_root = output_root / "images"
    ensure_dir(output_root)
    ensure_dir(images_root)

    # 加载 vocabs_train.json
    with open(args.vocabs_json, "r", encoding="utf-8") as vf:
        vocabs = json.load(vf)

    question_path = output_root / args.output_questions
    allowed_cats = set(args.allowed_categories)

    question_id_counter = 0    # 会成为 answers 里的 question_id
    num_samples = 0

    with question_path.open("w", encoding="utf-8") as qf:
        for vid in args.videos:
            vid = str(vid)
            qa_file = stsg_root / vid / "temporal_qa.json"
            if not qa_file.is_file():
                print(f"[WARN] QA file not found for {vid}: {qa_file}")
                continue

            with qa_file.open("r", encoding="utf-8") as f:
                try:
                    qa_items = json.load(f)
                except Exception as e:
                    print(f"[ERROR] Failed to load {qa_file}: {e}")
                    continue

            print(f"[INFO] Processing {vid}, {len(qa_items)} QA examples")

            for local_idx, qa_item in enumerate(qa_items):
                try:
                    # ---- basic fields from STSG QA ----
                    raw_cat = qa_item.get("category", None)
                    if raw_cat is None:
                        evidence = qa_item.get("evidence", {})
                        raw_cat = evidence.get("category", "unknown")
                    category = normalize_category(raw_cat)

                    if category not in allowed_cats:
                        continue

                    raw_answer_type = qa_item.get("answer_type", "unknown")
                    question_text = extract_question_text(qa_item)
                    gt_answer = extract_gt_answer(qa_item)
                    evidence = qa_item.get("evidence", {})

                    # ---- infer label_type & task (aligned with STSG-Net heads) ----
                    label_type, task_name, choice_indices = infer_label_type_and_task(
                        category, raw_answer_type, question_text
                    )

                    # ---- sample frame ids within the temporal window ----
                    sampled_frames = extract_frame_range_and_sample(
                        qa_item, args.max_frames
                    )

                    # absolute frame paths for grid construction
                    frame_paths = [
                        video_root / vid / f"{fid:06d}{args.frame_ext}"
                        for fid in sampled_frames
                    ]

                    # ---- extract rationale text (if available) ----
                    rationale_text = extract_rationale_text(qa_item)
                    if not getattr(args, "use_rationale", False):
                        rationale_text = None

                    # ---- build LLaVA-Med prompt text ----
                    prompt_text = build_prompt(
                        question_text=question_text,
                        label_type=label_type,
                        choice_indices=choice_indices,
                        category=category,
                        vocabs=vocabs,
                        rationale_text=rationale_text,
                    )

                    # ---- build grid image and paths ----
                    question_id = question_id_counter
                    question_id_counter += 1

                    image_name = f"{vid}_q_{local_idx:06d}.jpg"
                    image_rel_path = f"images/{image_name}"
                    image_abs_path = images_root / image_name

                    build_grid(
                        frame_paths=frame_paths,
                        out_path=image_abs_path,
                        rows=args.grid_rows,
                        cols=args.grid_cols,
                        tile_size=(args.tile_width, args.tile_height),
                    )

                    # ---- pack rich metadata for later evaluation ----
                    metadata = {
                        "question": question_text,
                        "answer": gt_answer,
                        "answer_type": raw_answer_type,
                        "category": category,
                        "raw_category": raw_cat,
                        "evidence": evidence,
                        "label_type": label_type,
                        "task": task_name,
                        "video_id": vid,
                        "local_idx": local_idx,
                        "frame_ids_sampled": sampled_frames,
                        "keyframes": evidence.get("keyframes", []),
                        "scope": evidence.get("scope", {}),
                        "choice_indices": choice_indices,
                    }

                    record = {
                        "question_id": question_id,
                        "image": image_rel_path,
                        "text": prompt_text,   # will be read as "prompt" by LLaVA-Med
                        "metadata": metadata,
                    }

                    qf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_samples += 1

                except Exception as e:
                    print(f"[WARN] Skip one QA in {vid} due to error: {e}")
                    continue

    print(f"[DONE] Exported {num_samples} QA examples to {question_path}")
    print(f"[INFO] Images saved under {images_root}")


if __name__ == "__main__":
    main()
