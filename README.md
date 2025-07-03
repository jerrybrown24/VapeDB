
# VaporIQ Analytics Dashboard (Streamlit)

A multi‑tab Streamlit app that lets you explore, model, and download insights from the *VaporIQ* consumer‑survey dataset.

## Features

| Tab | Highlights |
|-----|------------|
| **Data Visualisation** | 10+ interactive charts (histograms, heat‑map, pairplot, facet exploration) with sidebar filters |
| **Classification** | KNN, Decision Tree, Random Forest, Gradient Boosted Trees; confusion‑matrix toggle; combined ROC curve; CSV upload → predictions download |
| **Clustering** | K‑means with slider (k=2‑10), elbow + silhouette plots, persona summary table, labelled‑data download |
| **Association Rules** | Apriori on flavour & channel columns; user‑defined support / confidence; top‑10 rules |
| **Regression** | Linear, Ridge, Lasso, Decision‑Tree Regressor with hyper‑parameter tuning; r², RMSE, residual plots |

## File Layout
```
.
├── app.py
├── vaporiq_synthetic_named.csv
├── requirements.txt
└── README.md
```

## Quick Start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Create a GitHub repo and push these files.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repo, choose **app.py** as the entry point, and deploy.
3. Make sure **requirements.txt** is in the repo root so dependencies install automatically.

Enjoy exploring your data!
