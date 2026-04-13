from utils.data import assemble_indexes, truncate_subspecies, sid_to_genus


def test_assemble_indexes_assigns_consistent_class_encodings() -> None:
    data_index = {
        "rfpaths": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "sids": [
            "Danaus_plexippus",
            "Danaus_plexippus",
            "Heliconius_erato",
            "Danaus_gilippus",
        ],
        "pos": ["dorsal", "ventral", "dorsal", "ventral"],
        "sex": ["male", "female", "male", "female"],
    }

    index_data, sid_to_class = assemble_indexes(data_index)

    assert index_data["class_encs"] == [0, 0, 1, 2]
    assert sid_to_class == {
        "Danaus_plexippus": 0,
        "Heliconius_erato": 1,
        "Danaus_gilippus": 2,
    }
    assert index_data["rfpaths"] == data_index["rfpaths"]


def test_sid_helpers_extract_expected_parts() -> None:
    assert sid_to_genus("Danaus_plexippus") == "Danaus"
    assert truncate_subspecies("Danaus_plexippus_formA") == "Danaus_plexippus"
    assert truncate_subspecies("Danaus_plexippus") == "Danaus_plexippus"