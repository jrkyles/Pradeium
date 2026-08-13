# Praedium

Praedium is a commercial real estate risk-analysis application that explores loan performance data and estimates the probability of a loan becoming 60+ days delinquent.

## What it demonstrates

- End-to-end data preparation with pandas and NumPy
- Mixed numeric and categorical feature pipelines with scikit-learn
- Logistic-regression probability scoring with class balancing
- Interactive portfolio insights and risk visualizations
- Flask API and server-rendered workflows
- React, TypeScript, Vite, Axios, and Recharts frontend

## Core workflows

- Review portfolio-level statistics such as LTV, DSCR, note rate, loan size, and property age
- Explore how individual features relate to delinquency
- Enter property and loan attributes to generate a probability-of-delinquency estimate
- Translate predicted probability into a simple risk grade

## Repository structure

- `MAIA Project.py` — current Flask application and modeling pipeline
- `frontend/` — React and TypeScript interface
- `templates/` — server-rendered Flask templates
- `Praedium_dataset_20251124.csv` — project dataset
- `test.py` — application tests
- `Praedium.py` — earlier prototype retained for reference

## Run locally

### Backend

Create a Python virtual environment and install the libraries imported by `MAIA Project.py`, including Flask, Flask-CORS, pandas, NumPy, matplotlib, seaborn, and scikit-learn. Then run:

```bash
python "MAIA Project.py"
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Responsible use

Praedium is an educational portfolio project, not a production underwriting system. Predictions depend on the included dataset and should not be used as the sole basis for lending or investment decisions.

## Author

[James Kyles](https://github.com/jrkyles)
