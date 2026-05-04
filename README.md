# Robot Perception

A robotics perception project focused on building a model that detects and understands household objects using the **Robot@Home** dataset.

This repository is part of a larger robotics effort whose long-term goal is to create a robot that can move through an unknown house, build a map using **SLAM**, and understand its surroundings semantically through perception.

---

## Overview

The broader vision behind this project is to develop a robot that can:

- explore indoor household environments,
- localize itself while moving,
- build a map of an unknown world,
- detect and understand household objects,
- combine spatial and semantic information into a richer world model.

This repository focuses on the **perception module** of that larger system.

In the full robotics stack:

- **SLAM** answers:  
  *Where am I?* and *What does the environment look like geometrically?*

- **Perception** answers:  
  *What objects are visible?*  
  *What type of room or scene is this?*  
  *How can semantic information improve mapping and navigation?*

---

## Project Goal

The immediate goal of this repository is to create a perception pipeline trained on the **Robot@Home** dataset for indoor household object understanding.

The perception system is intended to help a robot:

- detect household objects,
- interpret indoor scenes,
- understand labeled RGB-D observations,
- support future semantic mapping and navigation tasks.

This project is therefore a building block toward a robot that can both **map** and **understand** a home environment.

---

## Dataset

This project uses the **Robot@Home** dataset, which contains indoor robot-acquired data such as:

- RGB images
- depth images
- scene observations
- semantic labels / annotations

The dataset itself is **not stored in this repository**.

Instead, dataset files are expected to be handled locally by the project code.

---

## Data Management

The `data/` directory is intentionally excluded from version control.

That means:

- `data/` is created locally,
- Robot@Home archives are downloaded locally,
- extracted dataset files are stored locally,
- large data files are not committed to Git.

This behavior matches the repository setup in `.gitignore`, where `data/` is ignored.

So this repository contains the **code to work with the dataset**, but not the dataset itself.

---

## Current Functionality

The current code supports the full perception preparation workflow:

1. download and prepare Robot@Home data locally,
2. convert annotations to YOLO-compatible format,
3. split the dataset into train/val/test (including stratified splitting),
4. train and evaluate a YOLO segmentation model,
5. inspect results and class-distribution reports.

---

## Function description

- `download.py`  
  Handles dataset download, archive integrity checks, extraction, and initial data exploration/visualization.

- `annotation_convert.py`  
  Converts Robot@Home RGB-D annotations and masks into YOLO segmentation labels and prepares image/label export structure.

- `yolo_data_split.py`  
  Creates train/val/test splits for YOLO data, supports random and stratified strategies, and can generate split review reports.

- `training.py`  
  Builds YOLO training config (`data.yaml`), runs segmentation training, performs test evaluation, and saves metrics/plots.

- `utilis.py`  
  Provides shared utility helpers for file operations, JSON I/O, plotting, mask alignment, and GPU info reporting.

---

## Repository Structure

```text
robot_perception/
├── annotation_convert.py
├── download.py
├── training.py
├── utilis.py
├── yolo_data_split.py
├── environment.yml
├── README.md
└── data/                      # created locally, ignored by Git