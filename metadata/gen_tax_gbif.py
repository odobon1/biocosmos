from pygbif import species

from utils import read_pickle, write_pickle, dirpaths

import pdb


def get_tax_metadata_species(sci_name):

    try:
        # rank="species" to narrow the search to species-level matches, strict=False to enable fuzzy matches (default)
        result = species.name_backbone(name=sci_name, rank="species", strict=False)
    except Exception as e:
        print(f"Error querying GBIF: {e}")
        return {}

    # if no match or multiple matches are found
    if not result or result.get("matchType") == "NONE":
        print("No unambiguous match found for the given name")
        return {}  # return empty dict to indicate failure
    
    return result

def get_tax_metadata(img_dir_names, verbose=False):
    """
    List["<genus>_<species>"] --> pygbif metadata dictionary
    """

    """
    found ----- full tax tree retrieved from GBIF
    missing --- nothing retrieved from GBIF
    partial --- partial hierarchy retrieved from GBIF (discarded for now)
    (if you return to exploring the partial tax structures, keep in mind that it is also possible for GBIF to return tax info that contains lower ranks but is missing some intermediary 
    ranks e.g. GBIF data for Mallika jacksoni contains genus + species, but is missing family)
    """
    metadata = {
        "found" : {},
        "partial" : {},
        "missing" : set(),
    }

    ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    for idx, img_dir_name in enumerate(img_dir_names):

        if verbose:
            print("------------------------------")
            print(idx)
            print(img_dir_name)

        genus, species_epithet = img_dir_name.split("_")
        genus = genus.capitalize()
        sci_name = f"{genus} {species_epithet}"
        result = get_tax_metadata_species(sci_name)

        if result:
            partial = False

            tax = {}
            for rank in ranks:
                if rank in result:
                    if rank == "species":
                        tax[rank] = result[rank].split()[-1]
                    else:
                        tax[rank] = result[rank]
                else:
                    partial = True
                    tax[rank] = None

            meta = {
                "status" : result["status"],
                "confidence" : result["confidence"],
                "matchType" : result["matchType"],
                "synonym" : result["synonym"],
            }
            species_metadata = {
                "tax" : tax,
                "meta" : meta,
            }

            if partial:
                metadata["partial"][img_dir_name] = species_metadata
            else:
                metadata["found"][img_dir_name] = species_metadata

            if verbose:
                if partial:
                    print("PARTIAL")
                print(
                    f"1. {tax['kingdom']} "
                    f"2. {tax['phylum']} "
                    f"3. {tax['class']} "
                    f"4. {tax['order']} "
                    f"5. {tax['family']} "
                    f"6. {tax['genus']} "
                    f"7. {tax['species']}"
                )
                print(
                    f"status: {meta['status']}; "
                    f"confidence: {meta['confidence']}; "
                    f"matchType: {meta['matchType']}; "
                    f"synonym: {meta['synonym']}; "
                )

        else:
            if verbose:
                print("SPECIES NOT FOUND")
            metadata["missing"].add(img_dir_name)

    return metadata

def main():

    img_dirs = read_pickle(dirpaths["repo_oli"] / "metadata/img_dirs/known.pkl")
    metadata = get_tax_metadata(img_dirs, verbose=True)
    write_pickle(metadata, dirpaths["repo_oli"] / "metadata/tax/gbif.pkl")

if __name__ == "__main__":
    main()
