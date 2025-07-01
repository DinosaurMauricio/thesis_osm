import json
from torch.utils.data.dataset import random_split

TRAIN_PERCENTAGE = 0.9
VALIDATION_PERCENTAGE = 0.1
JSON_PATH = "./annotations.json"


if __name__ == "__main__":
    with open(JSON_PATH) as f:
        json_file = json.load(f)
    
    keys = list(json_file.keys())

    train_keys, test_keys = random_split(keys, [TRAIN_PERCENTAGE, (1-TRAIN_PERCENTAGE)])
    
    train_keys, val_keys = random_split(train_keys, [TRAIN_PERCENTAGE, VALIDATION_PERCENTAGE])
    
    # Control
    for key in test_keys:
        if key in val_keys or key in train_keys:
            print("error")

    for key in train_keys:
        json_file[key]["split"] = "train"
    for key in test_keys:
        json_file[key]["split"] = "test"
    for key in val_keys:
        json_file[key]["split"] = "val"
        
    # verify that all the json has a split key
    for key in json_file.keys():
        assert "split" in json_file[key], f"Key {key} does not have a split key"

    with open("annotations_splitted.json", "w") as f:
        json.dump(json_file, f, indent=4)

    print("Dataset split successfully")
    print(f"Train size: {len(train_keys)}")
    print(f"Validation size: {len(val_keys)}")
    print(f"Test size: {len(test_keys)}")

    f.close()
