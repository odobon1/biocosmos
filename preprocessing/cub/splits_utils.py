from typing import Iterable
import matplotlib.pyplot as plt  # type: ignore[import]


def build_data_indexes_cub(
    cids: Iterable[str],
    skeys_partitions,
    img_ptrs,
):
    cid_set = set(cids)
    missing = cid_set - set(img_ptrs.keys())
    if missing:
        raise KeyError(f"Image pointers missing for {len(missing)} cids")

    def build_partition_index(partition_name):
        data_index = []
        cid2enc = {}
        for cid, samp_idx in sorted(skeys_partitions[partition_name]):
            if cid not in cid2enc:
                cid2enc[cid] = len(cid2enc)
            data_index.append({
                "cid": cid,
                "class_enc": cid2enc[cid],
                "rfpath": str(img_ptrs[cid][samp_idx]),
                "meta": None,
            })
        return data_index

    return {
        "train": build_partition_index("train"),
        "trainval": build_partition_index("trainval"),
        "validation": {
            "id": build_partition_index("id_val"),
            "ood": build_partition_index("ood_val"),
        },
        "test": {
            "id": build_partition_index("id_test"),
            "ood": build_partition_index("ood_test"),
        },
    }