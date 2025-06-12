from Bio import Entrez
import sys
import time
from tqdm import tqdm

from utils import read_pickle, write_pickle, dirpath_repo_oli

Entrez.email = "odobon2@gmail.com"
Entrez.api_key = "310aae4d36e96379c856bde5db7b7e5d6209"


# searches for species in genus
def species_search(genus_tax_id, species_epithet):
    """
    Search for a specific species within a genus using NCBI taxonomy links.
    
    Args:
        - genus_tax_id [str] --- genus taxonomy ID
        - species_name [str] --- species epithet to search for
        
    Returns:
        [dict] --- Taxonomy record for the species if found, None otherwise
    """
    try:
        # get all species in genus
        link_handle = Entrez.elink(dbfrom="taxonomy", db="taxonomy", id=genus_tax_id, linkname="taxonomy_taxonomy_lower")
        links = Entrez.read(link_handle)
        link_handle.close()
        
        if not links or not links[0].get("LinkSetDb"):
            return None, False
            
        # get linked taxonomy IDs (should be species under genus)
        linked_ids = []
        for linksetdb in links[0]["LinkSetDb"]:
            if linksetdb["LinkName"] == "taxonomy_taxonomy_lower":
                linked_ids = [link["Id"] for link in linksetdb["Link"]]
                break
        
        if not linked_ids:
            return None, False
        
        # check each species for name matches
        for batch_start in range(0, len(linked_ids), 20):  # process in batches of 20
            batch_ids = linked_ids[batch_start:batch_start + 20]
            
            fetch_handle = Entrez.efetch(db="taxonomy", id=",".join(batch_ids))
            species_records = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            for record in species_records:
                sci_name_record = record.get("ScientificName", "")
                rank = record.get("Rank", "")
                
                if rank == "species":
                    # check scientific name
                    if species_epithet.lower() in sci_name_record.lower():
                        # found potential match
                        return record, False
                    
                    # check synonyms and other names
                    other_names = record.get("OtherNames", {})
                    synonyms = other_names.get("Synonym", [])
                    
                    for synonym in synonyms:
                        if species_epithet.lower() in synonym.lower():
                            # found synonym match
                            return record, True
        
        return None, False
        
    except Exception as e:
        print(f"Error searching species in genus: {e}")
        return None, False

def get_tax_metadata_species(genus, species_epithet):
    """
    Get detailed taxonomic hierarchy with proper ranks using taxonomy ID lookup.
    Includes synonym searching and species-level resolution.
    
    Args:
        - genus [str]
        - species_epithet [str]
    """

    meta = {
        "direct_rank" : None,
        "direct_sci_name" : None,
        "direct_match_exact" : False,
        "direct_match_inexact" : False,
        "genus_search_exact" : False,
        "genus_search_broad" : False,
        "synonym" : False,
        "species_match" : False,
        "genus_match" : False,
        "genus_not_found" : False,
        "partial_match" : False,
        "num_partial_matches" : 0,
        "partial_name" : None,
        "partial_rank" : None,
        "final_name" : None,
        "final_rank" : None,
        "genus_level_only" : False,
        "inferred" : False,
        "no_tax_info" : False,
    }

    try:
        
        sci_name = f"{genus} {species_epithet}".lower()
        
        # search NCBI taxonomy DB
        search_handle = Entrez.esearch(db="taxonomy", term=sci_name)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        best_match = None
        
        if search_results["IdList"]:
            # check direct search results first
            tax_ids = search_results["IdList"]
            
            for tax_id in tax_ids[:5]:
                fetch_handle = Entrez.efetch(db="taxonomy", id=tax_id)
                taxonomy_records = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                if taxonomy_records:
                    record = taxonomy_records[0]
                    sci_name_record = record.get("ScientificName", "").lower()
                    rank = record.get("Rank", "")
                    meta["direct_rank"] = rank
                    meta["direct_sci_name"] = sci_name_record
                    
                    # prefer exact species matches
                    if sci_name_record == sci_name and rank == "species":
                        best_match = record
                        meta["direct_match_exact"] = True
                        break
                    elif rank == "species" and species_epithet.lower() in sci_name_record:
                        best_match = record
                        meta["direct_match_inexact"] = True
                        break
        
        # if no species found, try searching within the genus
        if not best_match or best_match.get("Rank") != "species":
            
            genus_found = False
            genus_tax_id = None
            
            # genus search strategy 1: exact genus name with rank filter
            genus_search = Entrez.esearch(db="taxonomy", term=f"{genus}[Scientific Name] AND genus[Rank]")
            genus_results = Entrez.read(genus_search)
            genus_search.close()
            
            if genus_results["IdList"]:
                genus_tax_id = genus_results["IdList"][0]
                genus_found = True
                meta["genus_search_exact"] = True
            else:
                # genus search strategy 2: broader genus search without rank filter
                # exact genus search failed, trying broader search
                genus_search = Entrez.esearch(db="taxonomy", term=f"{genus}")
                genus_results = Entrez.read(genus_search)
                genus_search.close()
                
                if genus_results["IdList"]:
                    # check each result for genus-rank match
                    for gid in genus_results["IdList"][:5]:
                        fetch_handle = Entrez.efetch(db="taxonomy", id=gid)
                        temp_records = Entrez.read(fetch_handle)
                        fetch_handle.close()
                        
                        if temp_records:
                            temp_record = temp_records[0]
                            temp_name = temp_record.get("ScientificName", "")
                            temp_rank = temp_record.get("Rank", "")
                            
                            if temp_rank == "genus" and genus.lower() in temp_name.lower():
                                genus_tax_id = gid
                                genus_found = True
                                meta["genus_search_broad"] = True
                                break
            
            if genus_found and genus_tax_id:
                # search for species within genus
                species_match, synonym = species_search(genus_tax_id, species_epithet)
                meta["synonym"] = synonym
                if species_match:
                    best_match = species_match
                    meta["species_match"] = True
                else:
                    # if still no species, get genus info for hierarchy
                    fetch_handle = Entrez.efetch(db="taxonomy", id=genus_tax_id)
                    genus_records = Entrez.read(fetch_handle)
                    fetch_handle.close()
                    if genus_records:
                        best_match = genus_records[0]
                        meta["genus_match"] = True
            else:
                meta["genus_not_found"] = True
                
                # try partial matching as last resort
                partial_search = Entrez.esearch(db="taxonomy", term=f"{genus}* OR *{genus}*")
                partial_results = Entrez.read(partial_search)
                partial_search.close()
                
                if partial_results["IdList"]:
                    meta["num_partial_matches"] = len(partial_results['IdList'])

                    for pid in partial_results["IdList"][:10]:  # show up to 10
                        fetch_handle = Entrez.efetch(db="taxonomy", id=pid)
                        partial_records = Entrez.read(fetch_handle)
                        fetch_handle.close()
                        
                        if partial_records:
                            p_record = partial_records[0]
                            p_name = p_record.get("ScientificName", "")
                            p_rank = p_record.get("Rank", "")
                            meta["partial_name"] = p_name
                            meta["partial_rank"] = p_rank
                            
                            # use first reasonable match if we don't have anything
                            if not best_match and genus.lower() in p_name.lower():
                                best_match = p_record
                                meta["partial_match"] = True
        
        if not best_match:
            # no taxonomic information found
            meta["no_tax_info"] = True
            return [], meta
            
        final_name = best_match.get("ScientificName", f"{genus} {species_epithet}")
        final_rank = best_match.get("Rank", "species")
        meta["final_name"] = final_name
        meta["final_rank"] = final_rank
        
        # if we found only genus-level info, try to construct species name
        if final_rank == "genus" and final_name.lower() != sci_name:
            # only genus-level info available, species may not exist in NCBI taxonomy
            meta["genus_level_only"] = True
        
        # get lineage with proper taxonomy IDs and ranks
        lineage_ex = best_match.get("LineageEx", [])

        # build complete hierarchy
        hierarchy = []
        # add all lineage levels
        for lineage_item in lineage_ex:
            rank = lineage_item.get("Rank", "no rank")
            name = lineage_item.get("ScientificName", "")
            if name and rank != "cellular root":
                hierarchy.append((rank, name))
        
        # add the matched organism itself
        hierarchy.append((final_rank, final_name))
        
        # if we only found genus info
        if final_rank == "genus" and species_epithet.lower() != genus.lower():
            hierarchy.append(("species", species_epithet))
            meta["inferred"] = True
        
        return hierarchy, meta
        
    except Exception as e:
        print(f"Error retrieving taxonomy data: {e}")
        return [], meta
    
def get_tax_metadata(img_dirs):
    
    metadata = {
        "found" : {},
        "missing" : set(),
    }

    for img_dir in tqdm(img_dirs):

        genus, species_epithet = img_dir.split("_")
        tax, meta = get_tax_metadata_species(genus, species_epithet)

        if tax:
            metadata["found"][img_dir] = {
                "tax" : tax, 
                "meta" : meta
            }
        else:
            metadata["missing"].add(img_dir)

    return metadata

def main():

    img_dirs = read_pickle(dirpath_repo_oli / "tax_tree/metadata/img_dirs/known.pkl")
    metadata = get_tax_metadata(img_dirs)
    write_pickle(metadata, dirpath_repo_oli / "tax_tree/metadata/tax/ncbi.pkl")

if __name__ == "__main__":
    main()
