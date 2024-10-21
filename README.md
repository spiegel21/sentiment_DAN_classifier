# Sentiment Analysis Models

This repository contains Python scripts for training and evaluating sentiment analysis models using PyTorch. The models include Bag-of-Words (BOW) and Deep Averaging Networks (DAN), with the capability to handle and process text data for binary sentiment classification.

## Files Overview

- `DANmodels.py`: Contains the implementation of Deep Averaging Network (DAN) models for sentiment analysis.
- `main.py`: The main script used to run the training and evaluation processes for the models.
- `BOWmodels.py`: Includes implementations of simple Bag-of-Words models for sentiment classification.
- `sentiment_data.py`: Manages data reading and preprocessing. This script handles the loading and vectorization of sentiment data.
- `utils.py`: Provides utility functions including data indexing and embedding operations.

## Setup and Installation

To run the scripts, you will need Python 3 and PyTorch installed. The recommended way to set up your environment is through Anaconda:

```bash
conda create -n sentiment_analysis python=3.8
conda activate sentiment_analysis
conda install pytorch torchvision -c pytorch

python main.py --model BOW  # For Bag-of-Words model
python main.py --model DAN  # For Pre-trained Deep Averaging Network
python main.py --model RandDAN  # For Randomly Initialized Deep Averaging Network

