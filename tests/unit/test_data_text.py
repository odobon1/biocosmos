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
        "animalia arthropoda insecta lepidoptera Danaus plexippus",
    }


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

    assert cub_text == "animalia chordata aves Passeriformes Corvidae Cyanocitta cristata"


def test_gen_text_taxonomy_does_not_repeat_genus_for_species() -> None:
    template = [["$TAX$"]]

    nymph_text = gen_text(
        {
            "subfamily": "nymphalinae",
            "genus": "tegosa",
            "species": "tegosa_guatemalena",
            "common_name": None,
        },
        template,
        dataset="nymph",
    )

    assert nymph_text == (
        "animalia arthropoda insecta lepidoptera nymphalidae "
        "nymphalinae tegosa guatemalena"
    )


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