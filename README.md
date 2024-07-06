# WodnesdaegNLP

This repository contains the model training, testing, inference, and data prep code that support the models on my similarly named website: [Wodnesdaeg](https://www.wodnesdaeg.com).

## Overview

### Config

WodnesdaegNLP uses its own custom yaml config format which models what it must run as a series of steps (`pipes`) in a `pipeline`.
Each `pipe` is a class in the WodnesdaegNLP codebase and has 1 or more `execution_steps` which are functions contained in the class.
Each `execution_step` has its own arguments (`args`) and `expected_output`.
1 output from 1 `execution_step` can then be exposed to the rest of the `pipeline` as the main output for that `pipe`.
This allows other downstream `pipes` to use that output as input to 1 or more of their `execution_steps`.

Check `config/examples` to see how configs are set up.

### Data Prep

WodnesdaegNLP's `CorpusExtractor` class is the primary way by which raw data is preprocessed in preparation for model training and testing.
A few of its functions are used directly, but it's mostly used as a parent class for dataset-specific extractors designed to handle specific corpora:

Below is a list of the currently supported extractors.
Each has an associated dataset that's readily discoverable through the CLTK repository:

|Extractor Name| Language           |
|-|--------------------|
|IceCorpusExtractor| Old Norse          |
|LatinTreeBankPerseusCorpusExtractor| Latin              |
|REMXMLCorpusExtractor| Middle High German |
|ISWOCTreebankCorpusExtractor| Old English |

All extractors process data into a standard 3-column TSV format containing the foloowing:
1. Original sentence
2. Per-token part of speech (POS) tags
3. Lemmatized sentence

### Model Training

WodnesdaegNLP is currently capable of training 2 types of models through its `HuggingFacePytorchModelFineTuner` class:

1. Part of speech (POS) taggers
2. Lemmatizers

POS tagger models use HuggingFace's `AutoModelForTokenClassification` framing the POS tagging task as simple token labeling.

Lemmatizers models use HuggingFace's `AutoModelForSeq2SeqLM` framing lemmatization as text generation.

Most of HuggingFace's `Trainer` arguments and parameters are exposed in WodnesdaegNLP's custom yaml config format.

After model training, WodnesdaegNLP can also produce training and validation plots to help visualize how well training went (if configured, not yet supported for lemmatizer models).

### Model Inference

Model inference is handled through the `HuggingFacePytorchModelPredictor` class and supports all the same models `HuggingFacePytorchModelFineTuner` does.

POS tagger inference reports the POS tags per token and associated model confidence scores.

Lemmatizer inference requires POS tags first, and so input must first run through a POS tagger or already have POS tags.

Input to inference can be either a file (using `FlatFileReader`) or interactive at the command line (using `InteractiveInputReader`).

## Installation

**NOTE: WodnesdaegNLP was developed on an Apple Silicon Mac. The installation instructions are for this kind of machine. If you do not have one of these computers, you'll need to adjust accordingly, including the package versions in `requirements.txt`**

1. Install [Anaconda](https://www.anaconda.com/download/)
2. Create a conda environment using:
    - `python 3.11`
    - The `requirements.txt` file included in this repo
    - The following conda channels in the exact listed order (can be set using a `.condarc` file)
      1. `conda-forge`
      2. `pytorch-nightly`
      3. `defaults`
3. Set your `PYTHONPATH` to `WodnesdaegNLP/src`