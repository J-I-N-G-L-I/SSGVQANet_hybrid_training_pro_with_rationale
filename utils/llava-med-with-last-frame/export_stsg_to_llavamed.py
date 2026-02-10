#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from utils.export_stsg_to_llavamed import (
    extract_gt_answer,
    extract_question_text,
    extract_rationale_text,
    infer_label_type_and_task,
    normalize_category,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export STSG temporal QA into LLaVA-Med style JSONL using only the last frame."
    )
    parser.add_argument(
        "--stsg_qa_root",
        required=True,
        help="Root directory of STSG temporal QA, e.g. /mnt/scratch/.../STSG_QA_Pro_8_Classes",
    )
    parser.add_argument(
        "--video_root",
        required=True,
        help="Root directory of raw frames, e.g. /mnt/scratch/.../CholecT45/data",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Where to save generated frames and question JSONL",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="List of video IDs, e.g. VID01 VID22 VID74 ...",
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=320,
        help="Width of the exported frame image",
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        default=180,
        help="Height of the exported frame image",
    )
    parser.add_argument(
        "--frame_ext",
        default=".png",
        help="Frame image extension (e.g. .png or .jpg)",
    )
    parser.add_argument(
        "--allowed_categories",
        nargs="+",
        default=[
            "count",
            "duration",
            "ordering",
            "extreme",
            "boundary",
            "motion",
            "concurrency",
            "phase_transition",
        ],
        help="Subset of categories to export",
    )
    parser.add_argument(
        "--output_questions",
        default="stsg_temporal_eval_questions_last_frame.jsonl",
        help="Filename of question JSONL under output_root",
    )
    parser.add_argument(
        "--use_rationale",
        action="store_true",
        help="If set, append available English rationales to the prompt text.",
    )
    parser.add_argument(
        "--vocabs_json",
        required=True,
        help="Path to vocabs_train.json (used to build closed-set prompts for text tasks)",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_last_frame_id(qa_item: Dict[str, Any]) -> int:
    """
    Extract the last frame index for a QA item.

    Priority:
    1) evidence.scope.frames = [start, end] -> use end
    2) evidence.keyframes -> use max(keyframes)
    """
    evidence = qa_item.get("evidence", {})
    scope = evidence.get("scope", {})
    keyframes = evidence.get("keyframes", None)

    frames = scope.get("frames", None)
    if isinstance(frames, list) and len(frames) == 2:
        try:
            return int(frames[1])
        except Exception:
            pass

    if isinstance(keyframes, list) and len(keyframes) > 0:
        return int(max(int(x) for x in keyframes))

    raise ValueError("Cannot determine last frame from QA evidence.")


def load_and_resize(img_path: Path, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


def save_single_frame(frame_path: Path, out_path: Path, size: Tuple[int, int]):
    try:
        img = load_and_resize(frame_path, size)
    except FileNotFoundError:
        img = Image.new("RGB", size, color=(0, 0, 0))
    img.save(out_path, format="JPEG", quality=95)


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
    This variant uses a single frame image instead of a 4x4 grid.
    """
    header = (
        "You are a medical video analysis assistant.\n"
        "The image shows a single frame from a laparoscopic cholecystectomy video.\n\n"
    )

    body = f"Question: {question_text}\n"

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
    else:
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
        else:
            format_hint = (
                "Answer with a short phrase describing the correct answer, "
                "without any extra explanation. "
                "Do not repeat the question or these instructions."
            )

    format_hint = (
        format_hint
        + " Finally, output your final answer after the string 'FINAL_ANSWER:'. "
        "Do not write anything else after 'FINAL_ANSWER:'."
    )

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

    with open(args.vocabs_json, "r", encoding="utf-8") as vf:
        vocabs = json.load(vf)

    question_path = output_root / args.output_questions
    allowed_cats = set(args.allowed_categories)

    question_id_counter = 0
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

                    label_type, task_name, choice_indices = infer_label_type_and_task(
                        category, raw_answer_type, question_text
                    )

                    last_frame_id = extract_last_frame_id(qa_item)
                    frame_paths = [
                        video_root / vid / f"{last_frame_id:06d}{args.frame_ext}"
                    ]

                    rationale_text = extract_rationale_text(qa_item)
                    if not getattr(args, "use_rationale", False):
                        rationale_text = None

                    prompt_text = build_prompt(
                        question_text=question_text,
                        label_type=label_type,
                        choice_indices=choice_indices,
                        category=category,
                        vocabs=vocabs,
                        rationale_text=rationale_text,
                    )

                    question_id = question_id_counter
                    question_id_counter += 1

                    image_name = f"{vid}_q_{local_idx:06d}.jpg"
                    image_rel_path = f"images/{image_name}"
                    image_abs_path = images_root / image_name

                    save_single_frame(
                        frame_path=frame_paths[0],
                        out_path=image_abs_path,
                        size=(args.tile_width, args.tile_height),
                    )

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
                        "frame_ids_sampled": [last_frame_id],
                        "keyframes": evidence.get("keyframes", []),
                        "scope": evidence.get("scope", {}),
                        "choice_indices": choice_indices,
                    }

                    record = {
                        "question_id": question_id,
                        "image": image_rel_path,
                        "text": prompt_text,
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
