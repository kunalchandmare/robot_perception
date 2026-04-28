from pathlib import Path
import robotathome as rh
import numpy as np
import os
import pandas as pd
import cv2
from robotathome import RobotAtHome
from robotathome import logger, log, set_log_level
from robotathome import time_win2unixepoch, time_unixepoch2win
from robotathome import get_labeled_img, plot_labeled_img
log.set_log_level('INFO')  # SUCCESS is the default
import matplotlib.pyplot as plt

data_path = "data"
local_files_path = data_path + "/files"
rgbd = "rgbd"
scene = "scene"

DATASETS = [
    {
        "url": "https://zenodo.org/record/7811795/files/Robot@Home2_db.tgz",
        "filename": "Robot@Home2_db.tgz",
        "md5": "d34fb44c01f31c87be8ab14e5ecd0767",
        "extract_to": data_path,
    },
    {
        "url": "https://zenodo.org/record/7811795/files/Robot@Home2_files.tgz",
        "filename": "Robot@Home2_files.tgz",
        "md5": "36faa2ffdd936a14455b2d1f3075e6ca",
        "extract_to": local_files_path,
    },
]

def ask_yes_no(question: str, default: bool = False) -> bool:
    prompt = " [Y/n]: " if default else " [y/N]: "
    reply = input(question + prompt).strip().lower()

    if not reply:
        return default

    return reply in {"y", "yes"}

def download_rh(out_dir, extract_root=None, force_download=None):
    out_dir = Path(out_dir).expanduser()
    extract_root = Path(extract_root).expanduser() if extract_root else Path.home()

    out_dir.mkdir(parents=True, exist_ok=True)

    for item in DATASETS:
        archive_path = out_dir / item["filename"]
        extract_path = extract_root / item["extract_to"]

        should_download = True

        if archive_path.exists():
            print(f"Found existing file: {archive_path}")

            if force_download is None:
                should_download = ask_yes_no(
                    f"Do you want to force re-download {item['filename']}?",
                    default=False
                )
            else:
                should_download = force_download

            if should_download:
                print(f"Re-downloading {item['filename']}...")
            else:
                print(f"Reusing existing file: {item['filename']}")

        if should_download:
            rh.download(item["url"], str(out_dir))

        print(f"Verifying {item['filename']}...")
        md5_actual = rh.get_md5(str(archive_path))

        if md5_actual != item["md5"]:
            print(
                f"MD5 mismatch for {item['filename']} "
                f"(expected {item['md5']}, got {md5_actual})."
            )

            re_download = ask_yes_no(
                f"Integrity check failed. Re-download {item['filename']} now?",
                default=True
            )

            if re_download:
                rh.download(item["url"], str(out_dir))
                md5_actual = rh.get_md5(str(archive_path))

                if md5_actual != item["md5"]:
                    raise ValueError(
                        f"MD5 mismatch persists for {item['filename']} after re-download."
                    )
            else:
                raise ValueError(
                    f"Cannot continue with corrupted file: {item['filename']}"
                )

        extract_path.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {item['filename']} to {extract_path}...")
        rh.uncompress(str(archive_path), str(extract_path))

    print("Done.")


def align_all_masks(mask_list, img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    aligned_masks = []

    for mask in mask_list:
        # Check if shape already matches
        if (mask.shape[0], mask.shape[1]) == (h, w):
            aligned_masks.append(mask)
            continue

        # Rotate if dimensions are swapped
        if (h, w) == (mask.shape[1], mask.shape[0]):
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        # Resize if still mismatching
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        aligned_masks.append(mask)

    return aligned_masks

def query_sample_annotation():

    rgbd_path = Path(local_files_path).joinpath(rgbd).resolve()
    scene_path = Path(local_files_path).joinpath(scene).resolve()

    # 1. Initialize the toolbox with your dataset path
    try:
        rh_db = RobotAtHome(rh_path=Path(data_path).resolve(),rgbd_path=rgbd_path,scene_path=scene_path)
    except Exception as e:
        print(f"Error initializing RobotAtHome: {e}")
        return
    # The full dataset is returned by default
    lblrgbd = rh_db.get_sensor_observations('lblrgbd')
    print(lblrgbd.head())
    print(lblrgbd.columns)
    rng = np.random.default_rng()
    sample_index = rng.integers(0, len(lblrgbd))
    sample_id = int(lblrgbd.iloc[sample_index]["id"]) if "id" in lblrgbd.columns else int(lblrgbd.index[0])
    print("sample_id:", sample_id)
    print(f"# Labeled RGBD set: {len(lblrgbd)} observations with {len(lblrgbd.columns)} fields")
    # 2. Query a specific observation (one image + annotation)
    # Example: get the 100th observation from session 'alma-s1'
    [rgb_img, depth_img] = rh_db.get_RGBD_files(sample_id)
    logger.info("Sensor observation {} files\n RGB file   : {}\n Depth file : {}", sample_id, rgb_img, depth_img)


    # 4. Retrieve the per-pixel annotation (this is the ground truth)
    # The annotation provides the semantic class label for every pixel
    annotation = rh_db.get_RGBD_labels(id=sample_id)  # This returns an array of the same shape as your images
    #print(annotation.head())
    logger.info("\nlabels: \n{}", annotation.columns)
    # 5. Visualize
    # 2. LOAD the images using OpenCV
    rgb_img_loaded = cv2.imread(rgb_img)
    depth_img_loaded = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(rgb_img_loaded, cv2.COLOR_BGR2RGB))
    ax[0].set_title("RGB Image")
    ax[1].imshow(depth_img_loaded, cmap='gray')
    ax[1].set_title("Depth Image")

    aligned_masks = align_all_masks(annotation['mask'], rgb_img)
    annotation['mask'] = aligned_masks
    [labeled_img, _] = get_labeled_img(annotation, rgb_img)
    ax[2].imshow(labeled_img)  # Semantic labels are color-coded
    ax[2].set_title("Per-pixel Annotation")

    plt.show()

if __name__ == "__main__":
    #download_rh("C:/Users/fixc9dv/Downloads",extract_root=Path.cwd(),force_download=False)

    query_sample_annotation()