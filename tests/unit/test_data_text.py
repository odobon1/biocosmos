from __future__ import annotations

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
        "Danaus_plexippus",
        template,
        pos="dorsal",
        sex="female",
        common_name="monarch",
    )

    assert text == "monarch butterfly, dorsal view"


def test_gen_text_falls_back_when_common_name_missing() -> None:
    random.seed(0)
    template = [["$COM$"]]

    text = gen_text("Danaus_plexippus", template, common_name=None)

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

    text = gen_text("Erebia_epipsodea", template)

    assert text == "a Erebia epipsodea"
