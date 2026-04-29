from utils.data import truncate_subspecies, species_to_genus


def test_sid_helpers_extract_expected_parts() -> None:
    assert species_to_genus("Danaus_plexippus") == "Danaus"
    assert truncate_subspecies("Danaus_plexippus_formA") == "Danaus_plexippus"
    assert truncate_subspecies("Danaus_plexippus") == "Danaus_plexippus"