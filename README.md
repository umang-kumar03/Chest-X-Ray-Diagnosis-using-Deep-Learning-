# Chest X-Ray Diagnosis using Deep Learning

An educational project for COVID-19 classification from chest
X-ray images using a custom TensorFlow/Keras CNN and Streamlit.

## Project Status

- Python environment and package setup completed.
- Dataset labels and image paths checked.
- Class distribution graph generated.
- Image quality checks, training, evaluation, and app development pending.

## Dataset

Source: [COVIDx on Kaggle](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2)

Selected release: Kaggle Version 7.

Verified counts from the attached label files:

| Official split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Train | 13,992 | 15,994 | 29,986 |
| Test | 200 | 200 | 400 |
| Total | 14,192 | 16,194 | 30,386 |

A patient-wise validation subset will be created from the official
training split. The official test split will be reserved for final
evaluation.

### Initial Checks

- All labeled image paths exist.
- No duplicate label rows or repeated filenames within either split.
- No shared filenames or patient ID strings between train and test.
- Image readability and duplicate image content checks are pending.

## Dataset Exploration

![Class distribution](artifacts/results/class_distribution.png)

[View dataset summary](artifacts/results/dataset_summary.csv)

## Tools

Python, TensorFlow/Keras, NumPy, pandas, OpenCV, scikit-learn,
Matplotlib, Seaborn, and Streamlit.

Development is performed in VS Code. Model training is planned
on Kaggle using an available GPU.

## Results

Model training has not been completed. Accuracy, precision,
recall, F1 score, and confusion matrix results will be added
after evaluation.

## Intended Use

For education and research only. This project is not validated
for clinical diagnosis and must not be used for medical decisions.