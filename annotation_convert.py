from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
from robotathome import RobotAtHome

from utilis import ensure_dir, align_all_masks, plot_image, plot_mask_overlay, plot_yolo_bboxes, align_all_masks_image


def prepare_binary_mask(mask):
    """Convert mask to 2D binary uint8."""
    if mask is None:
        return None

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]

    m = (m > 0).astype(np.uint8)
    return m if m.sum() > 0 else None


def mask_to_yolo_polygon(mask, class_id, img_w, img_h, epsilon_ratio=0.002):
    """Convert one aligned object mask to one YOLO segmentation line."""
    m = prepare_binary_mask(mask)
    if m is None:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    points = approx.reshape(-1, 2)

    if len(points) < 3:
        return None

    coords = []
    for x, y in points:
        coords.append(x / float(img_w))
        coords.append(y / float(img_h))

    return f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in coords)


def get_yolo_lines_for_observation(rh_db, obs_id, epsilon_ratio):
    """
    Fetches labels, aligns masks to image, and returns YOLO lines for an obs_id.
    """
    # 1. Fetch files and image
    rgb_path, _ = rh_db.get_RGBD_files(obs_id)
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None, []
    # Check alignment: mask is 320x240, so we expect h=320, w=240
    # If your image is 240 height and 320 width , you MUST rotate
    img_h, img_w = image.shape[:2]
    if img_h < img_w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_h, img_w= image.shape[:2]  # Now h=240, w=320

    # 2. Get labels and align masks
    labels_with_masks = rh_db.get_RGBD_labels(obs_id)
    if labels_with_masks.empty:
        return image, rgb_path, []

    aligned_masks = align_all_masks_image(labels_with_masks["mask"], image)

    # 3. Convert aligned masks to YOLO format
    label_lines = []
    for (_, row), aligned_mask in zip(labels_with_masks.iterrows(), aligned_masks):
        line = mask_to_yolo_polygon(
            mask=aligned_mask,
            class_id=row["object_type_id"],
            img_w=img_w,
            img_h=img_h,
            epsilon_ratio=epsilon_ratio
        )
        if line:
            label_lines.append(line)

    return image, rgb_path, label_lines

def convert_df_to_yolo_seg(rh_db, output_root,rgbd_root, epsilon_ratio=0.002):
    """
    Export Robot@Home annotations to YOLO segmentation dataset.

    Args:
        db:
            Loaded Robot@Home database object.
            Must provide:
            - db.get_RGBD_annotations()
            - db.get_RGBD_files(obs_id)

        output_root:
            Output directory where images/ and labels/ will be created.

        epsilon_ratio:
            Polygon simplification factor for cv2.approxPolyDP().

    Returns:
        None
    """
    output_root = Path(output_root)
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"


    observtions_df = rh_db.get_sensor_observations('lblrgbd')

    grouped = observtions_df.groupby("id")

    for obs_id, group in grouped:
        # rgb_path, depth_path = rh_db.get_RGBD_files(obs_id)
        #
        # image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        # if image is None:
        #     continue
        #
        # labels_with_masks = rh_db.get_RGBD_labels(obs_id)
        # masks = labels_with_masks["mask"]
        # aligned_masks = align_all_masks(masks, str(rgb_path))
        #
        # img_h, img_w = image.shape[:2]
        # label_lines = []
        #
        # for (_, row), aligned_mask in zip(labels_with_masks.iterrows(), aligned_masks):
        #     class_id = row["object_type_id"]
        #     class_name = rh_db.id2name(class_id)
        #
        #     yolo_line = mask_to_yolo_polygon(
        #         mask=aligned_mask,
        #         class_id=class_id,
        #         img_w=img_w,
        #         img_h=img_h,
        #         epsilon_ratio=epsilon_ratio,
        #     )
        #
        #     if yolo_line is not None:
        #         label_lines.append(yolo_line)
        image, rgb_path, label_lines = get_yolo_lines_for_observation(rh_db, obs_id, epsilon_ratio)

        relative_dir = Path(rgb_path).parent.relative_to(rgbd_root)
        final_img_dir = images_dir / relative_dir
        final_label_dir = labels_dir / relative_dir

        ensure_dir(final_img_dir)
        ensure_dir(final_label_dir)

        image_out = final_img_dir /  f"{obs_id}.jpg"
        label_out = final_label_dir / f"{obs_id}.txt"

        cv2.imwrite(str(image_out), image)

        with open(label_out, "w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

def replace_class_ids_with_names(label_lines, rh_db):
    """
    Replace first token in each YOLO label line from class id to class name.

    Args:
        label_lines: list[str]
            Example:
            ["5 0.12 0.34 0.56 0.78", "7 0.11 0.22 0.33 0.44"]

        class_id_to_name: dict
            Example:
            {5: "toilet", 7: "window"}

    Returns:
        list[str]
            Example:
            ["toilet 0.12 0.34 0.56 0.78", "window 0.11 0.22 0.33 0.44"]
    """
    replaced_lines = []

    for line in label_lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        class_id = int(float(parts[0]))
        class_name = rh_db.id2name(class_id, 'o')
        parts[0] = class_name

        replaced_lines.append(" ".join(parts))

    return replaced_lines


data_path = "data"
local_files_path = data_path + "/files"
rgbd = "rgbd"
scene = "scene"


def test_observation_visualization(rh_db, obs_id, epsilon_ratio=0.002):
    # 1. Get raw data and labels
    image, _, label_lines = get_yolo_lines_for_observation(rh_db, obs_id, epsilon_ratio)
    if image is None: return

    # 2. Prepare masks for overlay
    labels_with_masks = rh_db.get_RGBD_labels(obs_id)
    aligned_masks = align_all_masks_image(labels_with_masks["mask"], image)

    # 3. Create 3-panel plot and call separate plotting functions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_image(image, title="1. Original RGB", ax=axes[0])
    plot_mask_overlay(image, aligned_masks, title="2. DB Mask Overlay", ax=axes[1],rotate_90_ccw=False)

    label_lines = replace_class_ids_with_names(label_lines, rh_db)
    plot_yolo_bboxes(image, label_lines, title="3. YOLO BBoxes", ax=axes[2], rotate_90_ccw=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    rgbd_path = Path(local_files_path).joinpath(rgbd).resolve()
    scene_path = Path(local_files_path).joinpath(scene).resolve()

    # 1. Initialize the toolbox with your dataset path
    try:
        db = RobotAtHome(rh_path=Path(data_path).resolve(),rgbd_path=rgbd_path,scene_path=scene_path)
    except Exception as e:
        print(f"Error initializing RobotAtHome: {e}")
        exit(1)

    observations = db.get_sensor_observations()

    print(f"Total observations: {len(observations)}")
    print(observations.head())

    #convert_df_to_yolo_seg(rh_db=db, output_root="yolo", rgbd_root = rgbd_path)
    id = 100000
    test_observation_visualization(rh_db=db, obs_id=id, epsilon_ratio=0.002)
    