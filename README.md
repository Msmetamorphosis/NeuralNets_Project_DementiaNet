# Speech Based Dementia Classification with Wav2Vec2 and SpecAugment

## Project Overview

This project implements and extends a speech based dementia classification pipeline using a fine tuned Wav2Vec2 model. The objective is to reproduce a research baseline and introduce a controlled augmentation experiment to evaluate performance improvements under limited data conditions.

The original framework was provided as part of a structured baseline workflow consisting of:

1. Data preparation  
2. Model fine tuning  
3. Evaluation  

This repository documents the full pipeline, required modernization steps for current deep learning libraries, and an experimental extension using SpecAugment.

---

## Research Context

Automatic dementia detection through speech analysis is an active research area within:

- Medical AI  
- Computational linguistics  
- Low resource learning  
- Clinical decision support systems  

Cognitive decline often manifests in speech patterns through:

- Reduced lexical richness  
- Increased pauses  
- Fluency degradation  
- Prosodic changes  

This project approaches the problem as a supervised binary classification task:

**Healthy Control vs Dementia**

A pre trained Wav2Vec2 speech encoder is fine tuned for this downstream classification objective.

---

## Baseline Reproduction

The initial goal was to successfully reproduce the provided baseline pipeline. This required:

- Preparing audio data into model ready format  
- Configuring the Wav2Vec2 backbone  
- Implementing a custom speech classification head  
- Executing fine tuning for a fixed number of epochs  
- Running evaluation to establish baseline metrics  

The original notebook was written against an earlier version of the Hugging Face Transformers library. In order to execute the baseline successfully under the current library stack, several compatibility updates were required:

- Updating the custom Trainer implementation to align with the Transformers version 5 API  
- Refactoring deprecated mixed precision handling  
- Resolving weight tying compatibility changes  
- Adjusting training step behavior to use Accelerator based backpropagation  

These updates ensured that the baseline could be reproduced faithfully under the modern training framework.

---

## Model Architecture

The model consists of:

1. Pre trained Wav2Vec2 encoder  
2. Custom classification head  
3. Temporal pooling layer  
4. Cross entropy loss for binary classification  

### Classification Head

- Linear projection  
- Dropout  
- Non linear activation  
- Output projection to label space  

Pooling across the time dimension is performed using a configurable merge strategy, enabling aggregation of frame level representations into utterance level predictions.

---

## Training Configuration

Fine tuning was executed using:

- Hugging Face Trainer  
- Accelerator managed mixed precision  
- Gradient accumulation  
- Step based evaluation  
- Fixed training epochs as defined in the baseline  

The baseline was trained for **22 epochs**, consistent with the original configuration. The intent was to establish a stable reference point prior to experimentation.

---

## Experimental Extension: SpecAugment

After establishing the baseline, a controlled experiment was introduced using SpecAugment.

SpecAugment is a speech data augmentation technique that improves robustness by masking:

- Time segments  
- Frequency bands  

Rather than modifying the raw waveform, SpecAugment operates on spectral representations, encouraging the model to generalize beyond narrow acoustic patterns.

### Experimental Conditions

**Baseline Condition**  
Wav2Vec2 fine tuned without augmentation.

**Augmented Condition**  
Wav2Vec2 fine tuned with SpecAugment enabled.

### Experimental Objective

The goal of the experiment is to evaluate whether augmentation improves:

- Validation loss  
- Classification accuracy  
- Generalization stability  

This is particularly relevant in small dataset scenarios common in medical speech applications.

---

## Evaluation

Evaluation is performed using:

- Validation loss  
- Classification accuracy  
- F1 score  

Metrics are computed at defined evaluation intervals during training and at final model convergence.

Comparative analysis between the baseline and augmented models enables assessment of augmentation effectiveness.

---

## Repository Structure

- 01_prepare_data: 
Audio preprocessing and dataset construction

- 02_finetune: 
Model definition, Trainer configuration, and training loop

- 03_evaluate: 
Metric computation and final evaluation


### Custom Components

- `Wav2Vec2ForSpeechClassification`  
- Updated `CTCTrainer` compatible with Transformers v5  
- SpecAugment integration  

---

## Key Contributions of This Implementation

- Successful reproduction of a speech based dementia classification baseline  
- Modernization of legacy training code for current Transformers infrastructure  
- Integration of SpecAugment as a controlled experimental variable  
- Structured experiment design for baseline vs augmentation comparison  
- Reproducible training and evaluation pipeline  

---

## Future Extensions

Potential next steps include:

- Early stopping integration  
- Cross validation across folds  
- Additional augmentation strategies  
- Alternative pooling mechanisms  
- Hyperparameter search  
- Model scaling experiments  

---

## Conclusion

This project establishes a reproducible speech based dementia classification baseline and extends it through structured experimentation with spectral augmentation.

It provides an end to end deep learning workflow including:

- Data preparation  
- Model architecture customization  
- Training framework modernization  
- Experimental augmentation design  
- Performance evaluation  

The repository provides a clean and extensible foundation for further research in speech based cognitive impairment detection.
