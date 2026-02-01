# wildfire-ignition-loads-WUS
Daily predictive modeling of wildfire ignitions across the Western US (WUS). This research quantifies ignition potential versus actual loads, highlighting the roles of stochasticity, human drivers, and expanding climatic pressures.

# Potential versus Actual Wildfire Ignition Loads in the Western United States

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17535757.svg)](https://doi.org/10.5281/zenodo.17535757)

## Overview
This repository hosts the research and predictive models analyzing daily fire ignitions across the Western United States (WUS). By integrating biological, physical, and human-related variables, this project distinguishes between **Actual Ignition Load** (yearly ignition frequency) and **Potential Ignition Load** (spatiotemporal opportunity for ignition).

## Key Research Findings
* **Predictive Performance:** Models achieved varying success based on ignition cause, with F1-scores ranging from **0.8% (Power-started)** to **77.4% (Natural fires)**.
* **Geospatial Insights:** While California dominates in *actual* ignition numbers, **Washington** exhibits a higher *potential* ignition load due to expansive agricultural-wildland interfaces.
* **Predictability Limits:** The study identifies inherent stochasticity in certain ignition types (e.g., equipment-caused) and data gaps in others (e.g., lack of infrastructure data for power-caused fires).
* **Seasonality:** Ignition potential is a near year-round phenomenon in the Southwest where social and biophysical drivers converge, whereas weather governs seasonality at higher latitudes.

## Data & Variables
The models utilize a diverse suite of drivers:
- **Biophysical:** Fuel types, climate, and weather patterns.
- **Anthropogenic:** Agricultural interfaces, population density, and social drivers.

## Technical Stack & Methodology:
* This project utilizes a high-performance machine learning pipeline designed for imbalanced spatio-temporal datasets.
* Modeling: Developed using XGBoost and Scikit-learn for robust classification of ignition events.
* Optimization: Hyperparameter tuning conducted via Bayesian Optimization to maximize F1-scores across diverse ignition causes.
* Spatial Data: Processed using GeoPandas and ArcPy to integrate 267 physical and anthropogenic attributes.
* Data Handling: Large-scale feature engineering with Pandas and NumPy, managing over 20 years of climate and social variable data.

## Citation
> Pourmohamad, Y. (2026). Potential versus Actual Wildfire Ignition Loads in the Western United States. Zenodo. https://doi.org/10.5281/zenodo.17535757
