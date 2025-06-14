# Preprocessing Script for Medical Image Classification

import os
import pandas as pd
from PIL import Image
import argparse

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)
    # Resize the image to a fixed size (e.g., 224x224)
    image = image.resize((224, 224))
    # Convert the image to RGB (if not already in that format)
    image = image.convert('RGB')
    return image

def preprocess_data(train_data_dir, val_data_dir, test_data_dir, train_metadata_path, val_metadata_path, test_metadata_path):
    # Load metadata
    train_metadata = load_metadata(train_metadata_path)
    val_metadata = load_metadata(val_metadata_path)
    test_metadata = load_metadata(test_metadata_path)

    # Preprocess training images
    for index, row in train_metadata.iterrows():
        image_path = os.path.join(train_data_dir, row['filename'])
        image = preprocess_image(image_path)
        # Save the preprocessed image (you can choose a different format or location)
        image.save(os.path.join(train_data_dir, 'processed', row['filename']))

    # Preprocess validation images
    for index, row in val_metadata.iterrows():
        image_path = os.path.join(val_data_dir, row['filename'])
        image = preprocess_image(image_path)
        image.save(os.path.join(val_data_dir, 'processed', row['filename']))

    # Preprocess test images
    for index, row in test_metadata.iterrows():
        image_path = os.path.join(test_data_dir, row['filename'])
        image = preprocess_image(image_path)
        image.save(os.path.join(test_data_dir, 'processed', row['filename']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess image data for melanoma classification.')
    parser.add_argument('--train_data', required=True, help='Path to training data directory')
    parser.add_argument('--val_data', required=True, help='Path to validation data directory')
    parser.add_argument('--test_data', required=True, help='Path to test data directory')
    parser.add_argument('--train_metadata', required=True, help='Path to training metadata CSV file')
    parser.add_argument('--val_metadata', required=True, help='Path to validation metadata CSV file')
    parser.add_argument('--test_metadata', required=True, help='Path to test metadata CSV file')

    args = parser.parse_args()

    preprocess_data(args.train_data, args.val_data, args.test_data, args.train_metadata, args.val_metadata, args.test_metadata)