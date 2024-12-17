from .general_utils import random_choice


def get_annotation_type(
    osm_data_input: bool, prob_osm_target: float, force_augmented=False
) -> str:
    """
    Get General or OSM-augmented tag. General annotations use only visual cues,
    while OSM-augmented include OSM data. Randomly excludes OSM data, but can be forced for testing.
    """

    if osm_data_input and random_choice(prob=prob_osm_target) or force_augmented:
        return "osm"
    else:
        return "general_annot"


def get_annotation(sample: dict, annot_type: str) -> str:
    """
    Return an annotation prefixed with a special keyword:
    'aug ' for OSM-augmented, 'gen ' for general captions,
    followed by the text and an end token.
    """
    if annot_type == "osm":
        special_keyword = "aug "
    else:
        special_keyword = "gen "

    annotation = special_keyword + sample[annot_type] + "<|endoftext|>"

    return annotation
