from pathlib import Path
import random
import shutil

src_root = "yolo"
split_root = "yolo_split"

def split_yolo_dataset(
    source_root,
    output_root,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    image_exts=(".jpg", ".jpeg", ".png")
):
    """
       Split a YOLO dataset into train/val/test and copy files into split folders.

       Expected source layout: source_root/
               images/
               labels/

       Output layout:  output_root/
               images/train, images/val, images/test
               labels/train, labels/val, labels/test

       Args:
           source_root (str | Path): Root folder containing `images/` and `labels/`.
           output_root (str | Path): Destination root for split dataset.
           train_ratio (float): Fraction of images assigned to train split.
           val_ratio (float): Fraction of images assigned to val split.
           test_ratio (float): Fraction of images assigned to test split.
           seed (int): Random seed for deterministic shuffling.
           image_exts (tuple[str, ...]): Image file extensions to include.

       Behavior:
           - Recursively finds images under `source_root/images`.
           - Matches each image to a label by relative path and `.txt` suffix.
           - Copies images/labels into split-specific folders.
           - Prints warnings for images with missing labels.
           - Prints total and per-split image counts.
        Note: currently not considering Integrity of dataset while splitting or class distributions. Initial appraoch is
        simple as poc phase

       Returns:
           None
       """
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    images_root = source_root / "images"
    labels_root = source_root / "labels"

    image_files = []
    for ext in image_exts:
        image_files.extend(images_root.rglob(f"*{ext}"))

    image_files = sorted(image_files)
    random.Random(seed).shuffle(image_files)

    total = len(image_files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:]
    }

    for split_name, files in splits.items():
        for img_path in files:
            rel_path = img_path.relative_to(images_root)
            label_path = labels_root / rel_path.with_suffix(".txt")

            out_img = output_root / "images" / split_name / rel_path
            out_lbl = output_root / "labels" / split_name / rel_path.with_suffix(".txt")

            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(img_path, out_img)

            if label_path.exists():
                shutil.copy2(label_path, out_lbl)
            else:
                print(f"Missing label for: {img_path}")

    print(f"Total images: {total}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")



if __name__ == "__main__":

    split_yolo_dataset(
        source_root=r"C:\Data\Python Projects\robot_perception\yolo",
        output_root=r"C:\Data\Python Projects\robot_perception\yolo_split",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
