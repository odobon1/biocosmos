import random

from utils.data import gen_text


def test_gen_text_fills_known_tokens() -> None:
    random.seed(3)
    template = [
        ["$COM$"],
        [" butterfly"],
        ["$POS$"],
    ]

    text = gen_text(
        {
            "species": "Danaus_plexippus",
            "common_name": "monarch",
        },
        template,
        dataset="lepid",
        meta={"pos": "dorsal", "sex": "female"},
    )

    assert text == "monarch butterfly, dorsal view"


def test_gen_text_falls_back_when_common_name_missing() -> None:
    random.seed(0)
    template = [["$COM$"]]

    text = gen_text(
        {
            "species": "Danaus_plexippus",
            "common_name": None,
        },
        template,
        dataset="lepid",
    )

    assert text in {
        "Danaus plexippus",
        "animalia arthropoda insecta lepidoptera nymphalidae Danaus plexippus",
    }


def test_gen_text_uses_indefinite_article_token() -> None:
    random.seed(1)
    template = [
        ["$AAN$ "],
        ["$SCI$"],
    ]

    text = gen_text(
        {
            "species": "Erebia_epipsodea",
            "common_name": None,
        },
        template,
        dataset="lepid",
    )

    assert text == "a Erebia epipsodea"


def test_gen_text_uses_dataset_specific_taxonomy() -> None:
    template = [["$TAX$"]]

    cub_text = gen_text(
        {
            "order": "Passeriformes",
            "family": "Corvidae",
            "genus": "Cyanocitta",
            "species": "Cyanocitta_cristata",
            "common_name": "blue jay",
        },
        template,
        dataset="cub",
    )

    assert cub_text == "animalia chordata aves Passeriformes Corvidae Cyanocitta Cyanocitta cristata"


def test_gen_text_bryo_works_without_species_key() -> None:
    template = [["$SCI$"]]

    text = gen_text(
        {
            "family": "smittinidae",
            "genus": "smittoidea",
            "common_name": None,
        },
        template,
        dataset="bryo",
    )

    assert text == "smittoidea"