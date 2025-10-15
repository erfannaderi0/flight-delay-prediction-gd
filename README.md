# ✈️ Flight Delay & Cancellation Prediction (2019–2023)

> Predicting flight delays and cancellations using real-world U.S. flight data (2019–2023)  
> Built from scratch — including manual feature engineering and gradient descent implementation.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Motivation](#-motivation)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results (to-be-added)](#-results-to-be-added)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🚀 Overview

Air travel delays and cancellations cost the industry billions each year and inconvenience millions of passengers.  
This project analyzes **millions of U.S. flight records (2019–2023)** to:

- Understand the **patterns behind delays** (airline, weather, time, route)
- Build a **predictive model** from scratch using **gradient descent**
- Compare the hand-coded model to **Scikit-Learn baselines**

---

## 💡 Motivation

Most ML projects rely on libraries like `scikit-learn`.  
This project aims to **implement the learning algorithm itself** (gradient descent, feature scaling, etc.)  
to deeply understand how models train and optimize — not just use black-box APIs.

---

## 🧠 Dataset

**Source:** [Flight Delay and Cancellation Dataset (2019–2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023) by *patrickzel* on Kaggle.

**Size:** ~15 million rows (1.5 GB+ compressed)

**Key features include:**
- `FL_DATE` – Flight date  
- `OP_UNIQUE_CARRIER` – Airline code  
- `ORIGIN` / `DEST` – Airport codes  
- `DEP_DELAY`, `ARR_DELAY` – Departure/arrival delays (in minutes)  
- `CANCELLED`, `CANCELLATION_CODE` – Cancellation details  
- Weather and airport metadata (optional merging)

---

## 🧩 Project Structure
```
flight-delay-prediction/
│
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_exploration.ipynb
│ └── 02_preprocessing.ipynb
├── src/
│ ├── gradient_descent.py
│ └── utils.py
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```