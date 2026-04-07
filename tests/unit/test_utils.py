from __future__ import annotations

from utils.utils import RunningMean, get_text_template, shuffle_list


def test_running_mean_tracks_average() -> None:
    mean = RunningMean()

    for value in [2.0, 4.0, 6.0, 8.0]:
        mean.update(value)

    assert mean.n == 4
    assert mean.value() == 5.0


def test_shuffle_list_is_seeded_and_non_mutating() -> None:
    values = [1, 2, 3, 4, 5]

    shuffled_a = shuffle_list(values, seed=17)
    shuffled_b = shuffle_list(values, seed=17)

    assert shuffled_a == shuffled_b
    assert values == [1, 2, 3, 4, 5]
    assert shuffled_a != values


def test_get_text_template_returns_known_templates() -> None:
    train_template = get_text_template("train")
    bioclip_template = get_text_template("bioclip_sci")

    assert isinstance(train_template, list)
    assert train_template[0] == ["", "a photo of "]
    assert bioclip_template == [["a photo of $SCI$"]]
