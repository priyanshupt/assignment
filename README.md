# Brain MRI Metastasis Segmentation

This project implements **Nested U-Net** and **Attention U-Net** architectures for segmenting metastasis regions in brain MRI images. The objective is to demonstrate proficiency in computer vision techniques and deploy a segmentation model via a web application.

---

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Preprocessing](#preprocessing)
- [Model Architectures](#model-architectures)
  - Nested U-Net
  - Attention U-Net
- [Model Training & Evaluation](#model-training--evaluation)
- [Web Application](#web-application)
- [Installation & Setup](#installation--setup)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## Objective

The goal of this project is to:
- Implement two deep learning architectures (Nested U-Net and Attention U-Net) to accurately segment metastasis regions from brain MRI scans.
- Develop a web application using **FastAPI** for model inference and **Streamlit** for visualization, allowing users to upload images and view segmentation results.

---

## Dataset

You will be working with a dataset of brain MRI images and their corresponding metastasis segmentation masks. The dataset has the following structure:
- **Images**: MRI scans of the brain.
- **Masks**: Binary masks indicating the metastasis regions.

The dataset has been split into 80% training and 20% testing sets.

---

## Project Structure

