
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             r2_score, mean_squared_error, silhouette_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import io
import seaborn as sns

st.set_page_config(page_title="VaporIQ Dashboard", layout="wide")

# ---- Helpers ----
@st.cache_data
def load_data():
    return pd.read_csv("vaporiq_synthetic_named.csv")

def binary_target(df):
    return (df["SubscribeLikelihood"] >= 7).astype(int)

def get_feature_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Encode categoricals via one-hot
    X = pd.get_dummies(X, drop_first=True)
    return X, y

# ---- Sidebar filters ----
df = load_data()

with st.sidebar:
    st.header("Global Filters")
    age_range = st.slider("Age range", int(df.Age.min()), int(df.Age.max()),
                          (18, 60))
    income_range = st.slider("Income (USD)", int(df.IncomeUSD.min()),
                             int(df.IncomeUSD.max()),
                             (int(df.IncomeUSD.min()), int(df.IncomeUSD.max())))
    gender_filter = st.multiselect("Gender", df.Gender.unique().tolist(),
                                   default=df.Gender.unique().tolist())
    st.markdown("---")
    st.caption("Filters apply to Visualisation & Clustering tabs.")
# Apply filters
mask = ((df.Age.between(*age_range)) &
        (df.IncomeUSD.between(*income_range)) &
        (df.Gender.isin(gender_filter)))
dff = df.loc[mask].reset_index(drop=True)

# ---- Tabs ----
tabs = st.tabs(["📊 Data Visualisation", "🤖 Classification",
                "🌀 Clustering", "🔗 Association Rules",
                "📈 Regression"])

# ======================= 1. Data Visualisation =========================
with tabs[0]:
    st.subheader("Dataset Overview")
    st.write(f"Filtered rows: **{len(dff)}** / {len(df)}")
    st.dataframe(dff.head())

    st.subheader("Key Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(dff["Age"], kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(dff["IncomeUSD"], kde=True, ax=ax)
        ax.set_title("Income Distribution")
        st.pyplot(fig)

    # 10+ insights …
    st.subheader("Correlation Heat‑Map")
    numeric_cols = dff.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_cols.corr(), cmap="viridis", annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Flavour Family Popularity")
    flat = dff["FlavourFamilies"].str.get_dummies(sep=",")
    st.bar_chart(flat.sum().sort_values(ascending=False))

    st.markdown("*(Scroll sidebar for more filters. Switch tabs for advanced analysis.)*")

# ======================= 2. Classification ============================
with tabs[1]:
    st.header("Binary Subscription Intent Classifier")
    st.markdown("`SubscribeIntent = 1` if SubscribeLikelihood ≥ 7")

    model_name = st.selectbox("Choose algorithm",
                              ("KNN", "Decision Tree", "Random Forest",
                               "Gradient Boosting"))
    test_size = st.slider("Test set %", 10, 40, 25, step=5)

    X, y = get_feature_target(df.copy(), "SubscribeLikelihood")
    y_bin = (y >= 7).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=test_size/100, stratify=y_bin, random_state=42)

    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=6, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = GradientBoostingClassifier(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_table = pd.DataFrame({
        "Accuracy": [accuracy_score(y_test, y_pred)],
        "Precision": [precision_score(y_test, y_pred)],
        "Recall": [recall_score(y_test, y_pred)],
        "F1‑score": [f1_score(y_test, y_pred)]
    })
    st.table(metrics_table.style.format("{:.3f}"))

    if st.checkbox("Show confusion matrix"):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    # ROC curves for all models
    st.subheader("ROC Curves (all algorithms)")
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42)
    }
    fig, ax = plt.subplots()
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_prob = mdl.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # Upload & predict
    st.markdown("---")
    st.subheader("Predict on new data")
    up_file = st.file_uploader("Upload CSV (same columns, no SubscribeLikelihood)", type="csv")
    if up_file:
        new_df = pd.read_csv(up_file)
        new_X = pd.get_dummies(new_df, drop_first=True)
        new_X = new_X.reindex(columns=X_train.columns, fill_value=0)
        preds = model.predict(new_X)
        out = new_df.copy()
        out["PredictedIntent"] = preds
        st.dataframe(out.head())
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

# ======================= 3. Clustering ================================
with tabs[2]:
    st.header("Customer Segmentation (K‑Means)")
    k = st.slider("Number of clusters (k)", 2, 10, 4)
    sub_cols = ["Age", "IncomeUSD", "PodsPerWeek", "NicotineStrength_mgml",
                "SweetLike", "MentholLike", "NewFlavourFreq"]
    scaler = MinMaxScaler()
    X_clust = scaler.fit_transform(df[sub_cols])
    # Elbow inertia
    inertias = []
    for i in range(2, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init="auto")
        km.fit(X_clust)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertias, "o-")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Method")
    st.pyplot(fig)

    # Silhouette
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_clust)
    sil_score = silhouette_score(X_clust, labels)
    st.success(f"Silhouette score for k={k}: **{sil_score:.3f}**")

    df_clusters = df.copy()
    df_clusters["Cluster"] = labels
    st.subheader("Cluster Persona Summary")
    persona = df_clusters.groupby("Cluster")[sub_cols].mean().round(1)
    st.dataframe(persona)

    csv_lab = df_clusters.to_csv(index=False).encode("utf-8")
    st.download_button("Download data with clusters", csv_lab,
                       "vaporiq_clustered.csv", "text/csv")

# ======================= 4. Association Rules =========================
with tabs[3]:
    st.header("Association Rule Mining")
    st.markdown("Running Apriori on flavour & channel columns")
    min_supp = st.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
    min_conf = st.number_input("Min confidence", 0.05, 1.0, 0.3, 0.05)

    # Prepare basket (one-hot)
    basket = df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool)
    basket["Online-direct"] = (df["PurchaseChannel"]=="Online-direct")
    basket["Retail shop"]  = (df["PurchaseChannel"]=="Retail shop")
    frequent = apriori(basket, min_support=min_supp, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support",
                        "confidence","lift"]])

# ======================= 5. Regression ===============================
with tabs[4]:
    st.header("Spend Prediction (Regression)")
    y = df["MaxMonthlySpendUSD"]
    X = df.drop(columns=["MaxMonthlySpendUSD", "SubscribeLikelihood"])
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    reg_name = st.selectbox("Model", ("Linear", "Ridge", "Lasso", "Decision Tree"))
    scaler_choice = st.selectbox("Scaler", ("MinMax", "Robust", "None"))
    if scaler_choice == "MinMax":
        scaler = MinMaxScaler()
    elif scaler_choice == "Robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    if reg_name == "Linear":
        model = LinearRegression()
        params = {}
    elif reg_name == "Ridge":
        model = Ridge()
        params = {"alpha":[0.1,1,10,20]}
    elif reg_name == "Lasso":
        model = Lasso(max_iter=10000)
        params = {"alpha":[0.001,0.01,0.1,1]}
    else:
        model = DecisionTreeRegressor(random_state=42)
        params = {"max_depth":[3,4,5,6,8]}

    if params:
        grid = GridSearchCV(model, params, cv=5, scoring="r2")
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
    else:
        model.fit(X_train_scaled, y_train)
        best_model = model

    y_pred = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.metric("R²", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:.2f} USD")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    ax.set_xlabel("Actual spend"); ax.set_ylabel("Predicted spend")
    st.pyplot(fig)

    st.info("The closer the scatter lies to the diagonal, the better the model.")
