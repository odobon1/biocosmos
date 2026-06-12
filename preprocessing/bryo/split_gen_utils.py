
def build_img_ptrs(genera):
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
