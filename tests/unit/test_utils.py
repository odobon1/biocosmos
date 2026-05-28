from utils.utils import RunningMean, PrintLog, get_text_template, shuffle_list


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
    lepid_template = get_text_template("train", dataset="lepid")
    bioclip_template = get_text_template("sci", dataset="cub")

    assert isinstance(train_template, list)
    assert train_template[0] == ["", "a photo of "]
    assert lepid_template[-1] == ["", " butterfly"]
    assert bioclip_template == [["a photo of $SCI$"]]


def test_printlog_eval_handles_missing_loss_key() -> None:
    class _EvalPipe:
        partition_names = ["id"]
        nshot_bucket_names = []
        bucket_partition_name = None
        best_comp_map = None
        best_i2i_map = None
        best_full_set_comp_map = None
        best_full_set_i2i_map = None

    scores_eval = {
        "id": {
            "standard": {
                "map": {"i2t": 0.1, "i2i": 0.2, "t2i": 0.3},
                "acc": {"i2t": 0.4},
                "full_set": {
                    "map": {"i2t": 0.11, "i2i": 0.21, "t2i": 0.31},
                    "acc": {"i2t": 0.41},
                },
            },
            "per_class": {
                "map": {"i2t": 0.12, "i2i": 0.22, "t2i": 0.32},
                "acc": {"i2t": 0.42},
                "full_set": {
                    "map": {"i2t": 0.13, "i2i": 0.23, "t2i": 0.33},
                    "acc": {"i2t": 0.43},
                },
            },
        },
        "comp": {
            "standard": {
                "map": {"all": 0.2, "i2i": 0.2, "id": 0.2},
                "full_set": {
                    "map": {"all": 0.21, "i2i": 0.21, "id": 0.21},
                },
            },
            "per_class": {
                "map": {"all": 0.22, "i2i": 0.22, "id": 0.22},
                "full_set": {
                    "map": {"all": 0.23, "i2i": 0.23, "id": 0.23},
                },
            },
        },
    }

    PrintLog.eval(scores_eval, _EvalPipe())