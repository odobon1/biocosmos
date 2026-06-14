from collections import defaultdict

from preprocessing.common.split_gen import GenSplitDataManager


def normalize_cub_rfpath(raw_path: str) -> str:
    raw_path = str(raw_path)
    idx_images = raw_path.find("images/")
    return raw_path[idx_images + len("images/"):]

def _class_dir_to_common_name(class_dir: str) -> str:
    _, raw_name = class_dir.split(".", 1)
    return raw_name.lower()

def _build_classdir_to_cid(class_data):
    classdir_to_cid = {}
    for cid, cid_data in class_data.items():
        species = cid_data.get("species")
        common_name = cid_data.get("common_name", cid)
        classdir_to_cid[common_name] = species
    return classdir_to_cid

def build_img_ptrs(rfpaths_all, class_data=None):
    if class_data is None:
        class_data = GenSplitDataManager.class_data
    classdir_to_cid = _build_classdir_to_cid(class_data)

    img_ptrs = defaultdict(dict)
    rfpath_2_skey = {}
    cid_offsets = defaultdict(int)

    for rfpath in rfpaths_all:
        parts = rfpath.split("/")
        class_dir = parts[0]
        class_name = _class_dir_to_common_name(class_dir)

        cid = classdir_to_cid[class_name]
        samp_idx = cid_offsets[cid]
        cid_offsets[cid] += 1

        img_ptrs[cid][samp_idx] = rfpath
        rfpath_2_skey[rfpath] = (cid, samp_idx)

    return img_ptrs, rfpath_2_skey, sorted(img_ptrs.keys())