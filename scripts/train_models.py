import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
import os
import sys
MGIT_PATH=os.path.dirname(os.getcwd())
sys.path.append(MGIT_PATH)

from utils import model_utils

tasks = ["wnli", "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
adapter = False
all_perturbations = ["word_order", "char_lettercase", "char_delete", "char_replace",
                     "addtypos", "char_swap", "char_misspelledword", "char_insert",
                     "char_repetition"]

for task in tasks:
    perturbations = []
    init_filepath = f"../models/roberta-base_vanilla-finetune_{task}"
    for version_number, perturbation in enumerate(all_perturbations):
        next_version_filepath = f"../models/roberta-base_vanilla-finetune_{task}_v{version_number+1}"
        os.mkdir(next_version_filepath)
        perturbations.append(perturbation)
        model_utils.train_and_save(save_path=next_version_filepath,
                                   init_mdckpt=init_filepath,
                                   epochs=1,
                                   ds="glue",
                                   task=task,
                                   adapter=adapter,
                                   perturbations=perturbations)
        init_filepath = next_version_filepath
