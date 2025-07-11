# Lake Chlorophyll Prediction using Gaussian Processes

This project uses Gaussian Process Regression (GPR) to model and predict chlorophyll concentration in a lake based on GPS coordinates and water temperature. It consists of a two-stage pipeline:

1. Predicting temperature from spatial coordinates `(x, y)`
2. Predicting chlorophyll from the estimated temperature

The system supports data input via CSV, generates predictive chlorophyll maps, and can be extended for sampling optimization and environmental monitoring.

---
