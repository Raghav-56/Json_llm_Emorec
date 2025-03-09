"""
Data Preparation Tool for Emotion Recognition Dataset

This script transforms JSON training data into formats suitable for vision language models,
including ShareGPT and standard vision-LM format.
"""

import json
import ast
import argparse
import numpy as np
from pathlib import Path
import logging
import sys
import os


from config.logger_config import logger
from config.defaults import (
    JSON_INDENT,
    ELEMENT_SCORE_THRESHOLD,
    TRANSFORMED_DATA_FILENAME,
    VISION_LM_FILENAME,
    SHAREGPT_FILENAME,
    DEFAULT_ARGS,
    IMAGE_PATH_VERIFICATION,
)


def load_json_data(file_path):
    """
    Load JSON data from a file and transform it into the required structure.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict: Loaded and transformed JSON data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        # Transform data into requested structure
        structured_data = {"root": {}}

        for idx, item in enumerate(raw_data):
            structured_data["root"][str(idx)] = {
                "conversation": {
                    "0": {
                        "from": "human",
                        "value": f"<image>What is the emotion of person in image among these 6 emotions: [Anger, Disgust, Fear, Happy, Neutral, Sad]?",
                    },
                    "1": {
                        "from": "gpt",
                        "value": str(item.get("total_score", "3")),
                    },
                },
                "images": {"0": item.get("img_path", "")},
            }

        logger.info(f"Successfully loaded and transformed data from {file_path}")
        return structured_data
    except Exception as e:
        logger.error(f"Failed to load or transform JSON from {file_path}: {e}")
        raise


def transform_json(input_data):
    """
    Transform the input JSON data structure.

    Args:
        input_data (list): List of input data dictionaries

    Returns:
        list: Transformed data with simplified scores
    """
    logger.info("Transforming JSON data...")
    new_data = []

    for item in input_data:
        new_entry = {
            "prompt": item["prompt"],
            "img_path": item["img_path"],
            "total_score": int(round(np.mean(item["total_score"]), 2)),
            "element_score": {},
        }

        element_scores = ast.literal_eval(item["element_score"])

        for element, scores in element_scores.items():
            new_entry["element_score"][element] = (
                1 if sum(scores) >= ELEMENT_SCORE_THRESHOLD else 0
            )

        new_data.append(new_entry)

    logger.info(f"Successfully transformed {len(new_data)} entries")
    return new_data


def save_json_data(data, output_path):
    """
    Save JSON data to a file.

    Args:
        data: The data to save
        output_path (str): Path to save the JSON file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=JSON_INDENT)
        logger.info(f"Successfully saved data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {output_path}: {e}")
        raise


def create_vision_lm_format(train_data):
    """
    Create a data format suitable for vision language models training.

    Args:
        train_data (list): List of training data entries

    Returns:
        list: Data formatted for vision language model training
    """
    logger.info("Creating vision-LM format...")
    vision_lm_data = []

    for item in train_data:
        total_score_entry = {
            "messages": [
                {
                    "content": f"<image>What is the overall quality score of this image on a scale of 1-5?",
                    "role": "user",
                },
                {"content": str(item["total_score"]), "role": "assistant"},
            ],
            "images": [item["img_path"]],
        }
        vision_lm_data.append(total_score_entry)

        for element, score in item["element_score"].items():
            element_entry = {
                "messages": [
                    {
                        "content": f"<image>Does this image contain {element}?",
                        "role": "user",
                    },
                    {"content": "Yes" if score == 1 else "No", "role": "assistant"},
                ],
                "images": [item["img_path"]],
            }
            vision_lm_data.append(element_entry)

    logger.info(f"Created {len(vision_lm_data)} vision-LM format entries")
    return vision_lm_data


def create_sharegpt_format(train_data):
    """
    Create data in ShareGPT format for training.

    Args:
        train_data (list): List of training data entries

    Returns:
        list: Data formatted for ShareGPT
    """
    logger.info("Creating ShareGPT format...")
    sharegpt_data = []

    for item in train_data:
        total_score_entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>What is the overall quality score of this image on a scale of 1-5?",
                },
                {"from": "gpt", "value": str(item["total_score"])},
            ],
            "images": [item["img_path"]],
        }
        sharegpt_data.append(total_score_entry)

        for element, score in item["element_score"].items():
            element_entry = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>Does this image contain {element}?",
                    },
                    {"from": "gpt", "value": "Yes" if score == 1 else "No"},
                ],
                "images": [item["img_path"]],
            }
            sharegpt_data.append(element_entry)

    logger.info(f"Created {len(sharegpt_data)} ShareGPT format entries")
    return sharegpt_data


def verify_image_paths(data, frames_dir=None, required=False):
    """
    Verify that image paths in the data exist and update them if needed.

    Args:
        data (list): List of data entries with img_path fields
        frames_dir (str, optional): Base directory for frame images
        required (bool): If True, raise error for missing images

    Returns:
        list: Data with verified and possibly updated image paths
        int: Count of valid images found
    """
    logger.info("Verifying image paths...")
    valid_count = 0
    invalid_paths = []

    for item in data:
        original_path = item["img_path"]
        path_obj = Path(original_path)

        if path_obj.exists():
            valid_count += 1
            continue

        if frames_dir:
            frame_dir_path = Path(frames_dir)
            potential_paths = [
                frame_dir_path / path_obj.name,
                frame_dir_path / original_path,
                frame_dir_path / path_obj.parent.name / path_obj.name,
            ]

            found = False
            for p in potential_paths:
                if p.exists():
                    item["img_path"] = str(p)
                    valid_count += 1
                    found = True
                    break

            if not found:
                invalid_paths.append(original_path)
        else:
            invalid_paths.append(original_path)

    logger.info(f"Found {valid_count} valid images out of {len(data)}")
    if invalid_paths:
        message = f"Could not locate {len(invalid_paths)} images"
        max_to_show = IMAGE_PATH_VERIFICATION["max_invalid_paths_to_show"]
        if required:
            logger.error(message)
            logger.error(f"First few missing: {invalid_paths[:max_to_show]}")
            raise FileNotFoundError(message)
        else:
            logger.warning(message)
            logger.warning(f"First few missing: {invalid_paths[:max_to_show]}")

    return data, valid_count


def process_data(input_file, output_dir=None, frames_dir=None, require_images=False):
    """
    Process data from input file and generate transformed formats.

    Args:
        input_file (str): Path to the input JSON file
        output_dir (str, optional): Directory to save output files. Defaults to same directory as input.
        frames_dir (str, optional): Base directory containing extracted video frames
        require_images (bool): If True, raise error if images not found

    Returns:
        tuple: Paths to the created files
    """
    input_path = Path(input_file)
    if not output_dir:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    data = load_json_data(input_path)
    transformed_data = transform_json(data)

    if frames_dir:
        logger.info(f"Checking image paths against frames directory: {frames_dir}")
        transformed_data, _ = verify_image_paths(
            transformed_data, frames_dir, require_images
        )

    transformed_path = output_dir / TRANSFORMED_DATA_FILENAME
    save_json_data(transformed_data, transformed_path)

    vision_lm_data = create_vision_lm_format(transformed_data)
    vision_lm_path = output_dir / VISION_LM_FILENAME
    save_json_data(vision_lm_data, vision_lm_path)

    sharegpt_data = create_sharegpt_format(transformed_data)
    sharegpt_path = output_dir / SHAREGPT_FILENAME
    save_json_data(sharegpt_data, sharegpt_path)

    logger.info("Data processing complete.")
    return transformed_path, vision_lm_path, sharegpt_path


def main():
    """
    Main entry point for the data preparation script.
    """
    parser = argparse.ArgumentParser(
        description="Transform training data into various formats for vision language models"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input JSON file containing training data"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for transformed data (default: same as input file)",
    )
    parser.add_argument(
        "--frames-dir",
        "-f",
        help="Directory containing extracted video frames from main.py",
        default=DEFAULT_ARGS["frames_dir"],
    )
    parser.add_argument(
        "--require-images",
        "-r",
        action="store_true",
        default=DEFAULT_ARGS["require_images"],
        help="Require all referenced images to exist",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=DEFAULT_ARGS["verbose"],
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        output_files = process_data(
            args.input, args.output, args.frames_dir, args.require_images
        )
        print("\nProcessing Summary:")
        print("------------------")
        print(f"Input file: {args.input}")
        print(
            f"Frames directory: {args.frames_dir if args.frames_dir else 'Not specified'}"
        )
        print("Output files:")
        for path in output_files:
            print(f"  - {path}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
