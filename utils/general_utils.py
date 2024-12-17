import random
import os
import string

from omegaconf import OmegaConf
from optuna.trial import TrialState
from pycocoevalcap.bleu.bleu import Bleu
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
from transformers import get_linear_schedule_with_warmup

from dataset.constants import POSITION_MAPPING


def random_choice(prob: float = 0.5):
    # Return True with the prob assigned, otherwise false
    number = random.randint(1, 100)
    treshold = prob * 100
    if number <= treshold:
        return True
    else:
        return False


def create_directory_if_not_exists(path: str):
    """
    Check if saving directory exists, and in case create it before saving
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_study_statistics(study, file_path):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    with open(file_path, "w") as f:
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials: {len(complete_trials)}\n")

        f.write("Best trial:\n")
        trial = study.best_trial

        f.write(f"  Value: {trial.value}\n")

        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

    print("Study statistics saved to:", file_path)


def tokens_to_text(tokens, tokenizer):
    decoded_tokens = tokenizer.batch_decode(tokens, skip_special_tokens=True)

    decoded_tokens = [
        " ".join(sentence.split(" ")[1:]).strip() for sentence in decoded_tokens
    ]  # To remove the starting word

    return decoded_tokens


def b_score(gts, res):
    bleu_scorer = Bleu(n=4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)

    return bleu_score


def get_scheduler(optimizer, config, dataset_len, scheduler_type=0):

    if scheduler_type == 0:
        scheduler = ConstantLR(
            optimizer=optimizer,
            factor=config.trainer.schedulers.constant_lr.factor,
            total_iters=config.trainer.schedulers.constant_lr.total_iters,
        )
    elif scheduler_type == 1:
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=0.1,
            epochs=config.trainer.epochs,
            steps_per_epoch=int(dataset_len / config.trainer.batch_size),
            anneal_strategy="linear",
        )
    elif scheduler_type == 2:
        train_steps = dataset_len * config.trainer.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.trainer.schedulers.linear_schedule_with_warmup.warmup_percentage
            * train_steps,
            num_training_steps=train_steps,
        )
    else:
        raise ValueError("Invalid scheduler type")

    return scheduler


def remove_punctuation(annotations):
    if isinstance(annotations, list):
        return [
            annotation.translate(str.maketrans("", "", string.punctuation))
            for annotation in annotations
        ]
    elif isinstance(annotations, str):
        return annotations.translate(str.maketrans("", "", string.punctuation))
    else:
        raise TypeError("Input should be either a string or a list of strings.")


def remove_special_characters(value, remove_and=False):
    if remove_and:
        return (
            value.replace(":", " ")
            .replace("_", " ")
            .replace("-", " ")
            .replace("&", " ")
            .strip()
        )

    return value.replace(":", " ").replace("_", " ").replace("-", " ").strip()


def clean_content(content):
    clean_osm_content = []
    for osm_features in content:
        temp_clean_content = []
        for key, value in osm_features.items():
            # Tokenize the 'v' and append directly
            key = remove_special_characters(key, remove_and=True)

            key_tokens = [remove_punctuation(word) for word in key.split()]
            temp_clean_content.extend(key_tokens)

            # Tokenize and append 'value' based on length
            value = remove_special_characters(value, remove_and=True)

            value_tokens = (
                [remove_punctuation(word) for word in value.split()]
                if len(value) > 1
                else [remove_punctuation(value)]
            )
            temp_clean_content.extend(value_tokens)
        
        clean_osm_content.append(temp_clean_content)

    return clean_osm_content


def unique_objects(clean_content):
    clean_content = [" ".join(tokens) for tokens in clean_content]
    # Create a set and then a list to get unique objects
    unique_content = list(set(clean_content))
    # Split again for compatibility
    unique_content = [object.split(" ") for object in unique_content]

    return unique_content


def clean_annotation(annotation):
    if isinstance(annotation, list):
        return [remove_punctuation(annot) for annot in annotation]
    else:
        return [remove_punctuation(word) for word in annotation.split()]


def get_unique_words_and_lower_case(clean_content):
    # lower case all words and remove duplicates due to case sensitivity
    return list(set(word.lower() for word in clean_content))


def lower_case_words(clean_text):
    if isinstance(clean_text[0], list):
        # If it's a list of lists, lowercase each word in each list
        return [[word.lower() for word in sublist] for sublist in clean_text]
    else:
        # If it's a flattened list, lowercase each word
        return [word.lower() for word in clean_text]


def save_config(data, path):
    OmegaConf.save(data, path)


# TODO: After merge check on this method as im unsure if still used
def tokenize_text(text, tokenizer):
    tokenized_text = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False,
    )

    return tokenized_text


def update_metric(metric: float, best_metric: float, minimize: bool = True):
    return metric < best_metric if minimize else metric > best_metric


def modify_position_value(content):
    for element in content:
        if "position" in element:
            element["position"] = POSITION_MAPPING[element["position"]]

    return content
