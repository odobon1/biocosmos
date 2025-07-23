from pygbif import species

from utils import load_pickle, save_pickle, paths

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

def get_tax_metadata(sids, verbose=False):
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
        "found": {},
        "partial": {},
        "missing": set(),
    }

    ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    for idx, sid in enumerate(sids):

        if verbose:
            print("------------------------------")
            print(idx)
            print(sid)

        genus, species_epithet = sid.split("_")
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
                "status": result["status"],
                "confidence": result["confidence"],
                "matchType": result["matchType"],
                "synonym": result["synonym"],
            }
            species_metadata = {
                "tax": tax,
                "meta": meta,
            }

            if partial:
                metadata["partial"][sid] = species_metadata
            else:
                metadata["found"][sid] = species_metadata

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
            metadata["missing"].add(sid)

    return metadata

def main():

    sids = load_pickle(paths["metadata_o"] / "species_ids/known.pkl")
    metadata = get_tax_metadata(sids, verbose=True)
    save_pickle(metadata, paths["metadata_o"] / "tax/gbif.pkl")

if __name__ == "__main__":
    main()
