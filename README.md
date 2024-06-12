# Text as Any-Modality for Zero-Shot Classification by Consistent Prompt Tuning

cache: including train data, train labels, and test files.

configs: training config.

DATA: training and testing dataset. Kinetic-400/600/700, MSCOCO, VOC2007, ESC50, etc, from official websites.

data_gen: text training data generation by LLaMA-2-7B

dataset: class labels for all modalities, dataset

models: main framework, ViCLIP, CLIP, CLAP, etc.

pretrained_weights: saving pretrained weights of multimodal models.

process_label: processing text and label.

training data: saving text training data generated from LLaMA-2-7B.

utils: python files.

# Training and Testing

**python train_multimodal_clip.py**. # modifying hyp-parameters in config to perform training for different datasets.