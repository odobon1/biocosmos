"""
metadata nymph / gbif --> labels data
Takes metadata/tax/nymph structure and generates 1. base_labels/nymph_sci and 2. base_labels/nymph_tax
(base labels in that they don't have "a photo of " prepended to them)
"""

from utils import dirpaths, read_pickle, write_pickle


tax_nymph = read_pickle(dirpaths["repo_oli"] / "metadata/tax/nymph.pkl")

base_labels_nymph_sci = {}
base_labels_nymph_tax = {}

for s in tax_nymph["found"].keys():

    tax_s = tax_nymph["found"][s]["tax"]

    sci_name = f"{tax_s['genus'].capitalize()} {tax_s['species']}"
    tax_name = f"Animalia Arthropoda Insecta Lepidoptera Nymphalidae {sci_name}"

    base_labels_nymph_sci[s] = sci_name
    base_labels_nymph_tax[s] = tax_name

write_pickle(base_labels_nymph_sci, dirpaths["repo_oli"] / "metadata/base_labels/nymph_sci.pkl")
write_pickle(base_labels_nymph_tax, dirpaths["repo_oli"] / "metadata/base_labels/nymph_tax.pkl")
