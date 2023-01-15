#!/bin/bash

# Set environment variables
export REST_API_URL=https://api.example.com/data
export RAW_DATA_PATH=data/raw/raw_data.json
export TRANSFORMED_DATA_PATH=data/transformed/transformed_data.parquet
export VALIDATION_CONFIG_PATH=config/validation_config.json

# Run pipeline
python pipeline/elt_pipeline.py
