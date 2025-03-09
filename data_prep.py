"""
Data Preparation Tool for Emotion Recognition Dataset

This script transforms video frame directories into JSON training data formats suitable
for vision language models, including ShareGPT and standard vision-LM format.
"""

import json
import argparse
import logging
import sys
from pathlib import Path
import os

from config.logger_config import logger
from config.defaults import (
    JSON_INDENT,
    TRANSFORMED_DATA_FILENAME,
    VISION_LM_FILENAME,
    SHAREGPT_FILENAME,
    DEFAULT_ARGS,
)


def process_directory(root_dir):
    """
    Process a directory structure containing emotion categories and video frames
    to create training data for a Vision-Language Model.

    Args:
        root_dir (str): Path to the root directory containing emotion categories

    Returns:
        dict: Dictionary of processed data with frame paths and emotion labels
    """
    logger.info(f"Processing directory structure from: {root_dir}")
    root_path = Path(root_dir)

    # Check if the root directory exists
    if not root_path.exists():
        logger.error(f"Root directory {root_dir} does not exist.")
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    # Dictionary to store all processed data
    all_data = {}
    video_count = 0
    frame_count = 0

    # Iterate through all emotion category folders
    for emotion_dir in root_path.iterdir():
        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name
        logger.info(f"Processing emotion category: {emotion}")

        # Iterate through all video folders in the emotion category
        for video_dir in emotion_dir.iterdir():
            if not video_dir.is_dir():
                continue

            video_name = video_dir.name
            logger.info(f"Processing video: {video_name}")

            # Create a dictionary for this video
            video_data = {}
            video_frame_count = 0

            # Process all frames in the video folder
            for frame_path in sorted(video_dir.glob("frame_*.jpg")):
                frame_num = int(frame_path.stem.split("_")[1])
                frame_entry = {
                    "conversation": {
                        "0": {
                            "from": "human",
                            "value": "<image>What is the emotion of the person in the image?",
                        },
                        "1": {
                            "from": "gpt",
                            "value": f"Emotion of the person in this image is: {emotion}",
                        },
                    },
                    "images": {str(frame_num): str(frame_path)},
                }

                # Add frame entry to the video data
                video_data[str(frame_num)] = frame_entry
                video_frame_count += 1

            # Add the video data to the all_data dictionary using a unique identifier
            video_id = f"{emotion}_{video_name}"
            all_data[video_id] = video_data

            logger.info(f"Processed {video_frame_count} frames for video {video_name}")
            frame_count += video_frame_count
            video_count += 1

    logger.info(
        f"Completed processing {video_count} videos with {frame_count} total frames"
    )
    return all_data


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


def create_vision_lm_format(processed_data):
    """
    Create a data format suitable for vision language models training.

    Args:
        processed_data (dict): Dictionary containing processed video frames data

    Returns:
        list: Data formatted for vision language model training
    """
    logger.info("Creating vision-LM format...")
    vision_lm_data = []

    for video_id, video_data in processed_data.items():
        for frame_idx, frame_data in video_data.items():
            conversation = frame_data["conversation"]
            image_path = list(frame_data["images"].values())[0]

            entry = {
                "messages": [
                    {
                        "content": conversation["0"]["value"],
                        "role": "user",
                    },
                    {"content": conversation["1"]["value"], "role": "assistant"},
                ],
                "images": [image_path],
            }
            vision_lm_data.append(entry)

    logger.info(f"Created {len(vision_lm_data)} vision-LM format entries")
    return vision_lm_data


def create_sharegpt_format(processed_data):
    """
    Create data in ShareGPT format for training.

    Args:
        processed_data (dict): Dictionary containing processed video frames data

    Returns:
        list: Data formatted for ShareGPT
    """
    logger.info("Creating ShareGPT format...")
    sharegpt_data = []

    for video_id, video_data in processed_data.items():
        for frame_idx, frame_data in video_data.items():
            conversation = frame_data["conversation"]
            image_path = list(frame_data["images"].values())[0]

            entry = {
                "conversations": [
                    {
                        "from": conversation["0"]["from"],
                        "value": conversation["0"]["value"],
                    },
                    {
                        "from": conversation["1"]["from"],
                        "value": conversation["1"]["value"],
                    },
                ],
                "images": [image_path],
            }
            sharegpt_data.append(entry)

    logger.info(f"Created {len(sharegpt_data)} ShareGPT format entries")
    return sharegpt_data


def process_data(input_dir, output_dir=None):
    """
    Process data from input directory and generate transformed formats.

    Args:
        input_dir (str): Path to the input directory containing emotion categories
        output_dir (str, optional): Directory to save output files. Defaults to same directory as input.

    Returns:
        tuple: Paths to the created files
    """
    input_path = Path(input_dir)
    if not output_dir:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Process directory to get structured data
    processed_data = process_directory(input_path)

    # Save the transformed data
    transformed_path = output_dir / TRANSFORMED_DATA_FILENAME
    save_json_data(processed_data, transformed_path)

    # Create and save vision LM format
    vision_lm_data = create_vision_lm_format(processed_data)
    vision_lm_path = output_dir / VISION_LM_FILENAME
    save_json_data(vision_lm_data, vision_lm_path)

    # Create and save ShareGPT format
    sharegpt_data = create_sharegpt_format(processed_data)
    sharegpt_path = output_dir / SHAREGPT_FILENAME
    save_json_data(sharegpt_data, sharegpt_path)

    logger.info("Data processing complete.")
    return transformed_path, vision_lm_path, sharegpt_path


def main():
    """
    Main entry point for the data preparation script.
    """
    parser = argparse.ArgumentParser(
        description="Transform video frames into JSON formats for vision language models"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing emotion categories and video frames",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for generated data (default: same directory as input)",
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
        output_files = process_data(args.input, args.output)
        print("\nProcessing Summary:")
        print("------------------")
        print(f"Input directory: {args.input}")
        print("Output files:")
        for path in output_files:
            print(f"  - {path}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
