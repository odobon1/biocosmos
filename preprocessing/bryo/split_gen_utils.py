

def build_img_ptrs_bryo(genera):
    from utils.utils import paths

    img_ptrs = {}
    for genus in sorted(genera):
        dpath_imgs_genus = paths["imgs"]["bryo"] / genus
        if not dpath_imgs_genus.is_dir():
            continue

        ffpaths_jpg = sorted(
            fpath
            for fpath in dpath_imgs_genus.iterdir()
            if fpath.is_file() and fpath.suffix.lower() == ".jpg"
        )

        img_ptrs[genus] = {
            idx: f"{genus}/{fpath.name}"
            for idx, fpath in enumerate(ffpaths_jpg)
        }

    return img_ptrs

def build_data_indexes_bryo(genera, skeys_partitions, cid2enc, img_ptrs=None):
    if img_ptrs is None:
        img_ptrs = build_img_ptrs_bryo(genera)

    def build_partition_index(partition):
        data_index = []

        for cid, samp_idx in sorted(skeys_partitions[partition]):
            data_index.append(
                {
                    "cid": cid,
                    "class_enc": cid2enc[cid],
                    "rfpath": img_ptrs[cid][samp_idx],
                    "meta": None,
                }
            )
        return data_index

    return {
        "train": build_partition_index("train"),
        "trainval": build_partition_index("trainval"),
        "val": {
            "id": build_partition_index("id_val"),
            "ood": build_partition_index("ood_val"),
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood": build_partition_index("ood_test"),
        },
        "whole": build_partition_index("whole"),
    }