"""
metadata nymph / gbif --> labels data
Takes metadata/tax/nymph structure and generates 1. base_labels/nymph_sci and 2. base_labels/nymph_tax
(base labels in that they don't have "a photo of " prepended to them)
"""

from utils import paths, read_pickle, write_pickle


tax_nymph = read_pickle(paths["metadata_o"] / "tax/nymph.pkl")

base_labels_nymph_sci = {}
base_labels_nymph_tax = {}

for sid in tax_nymph["found"].keys():

    tax_sid = tax_nymph["found"][sid]["tax"]

    sci_name = f"{tax_sid['genus'].capitalize()} {tax_sid['species']}"
    tax_name = f"Animalia Arthropoda Insecta Lepidoptera Nymphalidae {sci_name}"

    base_labels_nymph_sci[sid] = sci_name
    base_labels_nymph_tax[sid] = tax_name

write_pickle(base_labels_nymph_sci, paths["metadata_o"] / "base_labels/nymph_sci.pkl")
write_pickle(base_labels_nymph_tax, paths["metadata_o"] / "base_labels/nymph_tax.pkl")
