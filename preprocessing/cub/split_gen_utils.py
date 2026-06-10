from typing import Iterable
import matplotlib.pyplot as plt


def build_data_indexes_cub(skeys_partitions, img_ptrs, cid2enc):

    def build_partition_index(partition):
        data_index = []
        for cid, samp_idx in sorted(skeys_partitions[partition]):
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