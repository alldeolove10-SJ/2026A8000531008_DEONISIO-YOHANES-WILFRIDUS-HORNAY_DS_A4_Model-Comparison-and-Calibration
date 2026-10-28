from pathlib import Path
import json, zipfile, textwrap, hashlib, sys, platform, os, subprocess, importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

BASE = Path(__file__).resolve().parents[1]
for d in ["data", "src", "notebooks", "results", "figures", "report"]:
    (BASE/d).mkdir(parents=True, exist_ok=True)

# Freeze a built-in public educational dataset locally as a reproducible CSV.
bunch = load_breast_cancer(as_frame=True)
data = bunch.frame.copy()
# Explicitly rename target to make semantics readable. sklearn target 0=malignant, 1=benign.
data = data.rename(columns={"target":"target_benign"})
data["diagnosis"] = np.where(data["target_benign"].eq(1), "benign", "malignant")
data = data.drop(columns=["target_benign"])
csv_path = BASE/"data"/"wdbc_sklearn_frozen.csv"
data.to_csv(csv_path, index=False, float_format="%.10g")
sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
(BASE/"data"/"DATA_PROVENANCE.md").write_text(f"""# Frozen dataset provenance

- Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), distributed by scikit-learn's `load_breast_cancer`.
- Local frozen CSV: `wdbc_sklearn_frozen.csv`
- Rows: {len(data)}
- Predictors: 30 numeric features; target: `diagnosis` (benign/malignant)
- Local CSV SHA-256: `{sha}`
- Dataset source described by scikit-learn: UCI Machine Learning Repository, Wisconsin Diagnostic Breast Cancer dataset.
- Source page: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Upstream repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Download/creation method: generated locally from the installed scikit-learn dataset loader; no manual rows added.
- Important: verify current source/license terms and describe this as an educational benchmark, not a clinical tool. The SHA-256 verifies this local file's bytes only.
""", encoding="utf-8")

# Define deterministic pipeline and actual run for genuine illustrative outputs.
df = pd.read_csv(csv_path)
target = (df["diagnosis"] == "malignant").astype(int)  # positive class explicitly malignant
X = df.drop(columns=["diagnosis"])
X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.20, random_state=42, stratify=target
)
pre = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                           ("scale", StandardScaler())]), X.columns.tolist())])
model_defs = {
    "Prior baseline": DummyClassifier(strategy="prior"),
    "Logistic regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=42)
}
pipes = {name: Pipeline([("preprocess", clone(pre)), ("model", clone(m))]) for name,m in model_defs.items()}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def ece(y, p, bins=10):
    y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); out=0.
    for i in range(bins):
        mask=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if mask.any(): out += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(out)

def metrics(y, p, t=.5):
    pred=(np.asarray(p)>=t).astype(int)
    return {"ROC_AUC":roc_auc_score(y,p),
            "Average_Precision":average_precision_score(y,p),
            "Brier":brier_score_loss(y,p),
            "ECE_10_equal_width_bins":ece(y,p,10),
            "Accuracy_at_0.5":accuracy_score(y,pred),
            "Precision_at_0.5":precision_score(y,pred,zero_division=0),
            "Recall_at_0.5":recall_score(y,pred,zero_division=0),
            "F1_at_0.5":f1_score(y,pred,zero_division=0)}
cv_probs={}
cv_rows=[]
for name, pipe in pipes.items():
    probs=cross_val_predict(pipe,X_train,y_train,cv=cv,method="predict_proba")[:,1]
    cv_probs[name]=probs
    cv_rows.append({"Model":name,**metrics(y_train,probs)})
cv_table=pd.DataFrame(cv_rows).set_index("Model")
test_probs={}
test_rows=[]
fitted={}
for name,pipe in pipes.items():
    fitted[name]=clone(pipe).fit(X_train,y_train)
    p=fitted[name].predict_proba(X_test)[:,1]
    test_probs[name]=p
    test_rows.append({"Model":name,**metrics(y_test,p)})
test_table=pd.DataFrame(test_rows).set_index("Model")
cv_table.to_csv(BASE/"results"/"cv_model_comparison.csv")
test_table.to_csv(BASE/"results"/"test_model_comparison.csv")
preds=pd.DataFrame({"y_true_train_oof":y_train.to_numpy()})
for n,p in cv_probs.items(): preds["cv_p_"+n]=p
preds.to_csv(BASE/"results"/"paired_cv_predictions.csv",index=False)
testpred=pd.DataFrame({"y_true":y_test.to_numpy()})
for n,p in test_probs.items(): testpred["p_"+n]=p
testpred.to_csv(BASE/"results"/"paired_test_predictions.csv",index=False)

# Paired bootstrap AP difference on same OOF rows; CI is case-resampling uncertainty only.
def paired_boot(y, pa, pb, n_boot=1000, seed=42):
    rng=np.random.default_rng(seed); y=np.asarray(y); pa=np.asarray(pa); pb=np.asarray(pb); vals=[]
    for _ in range(n_boot):
        ix=rng.integers(0,len(y),len(y))
        if np.unique(y[ix]).size==2:
            vals.append(average_precision_score(y[ix],pa[ix])-average_precision_score(y[ix],pb[ix]))
    return {"comparison":"Logistic regression minus Prior baseline (OOF AP)",
            "difference":average_precision_score(y,pa)-average_precision_score(y,pb),
            "ci_low":float(np.quantile(vals,.025)), "ci_high":float(np.quantile(vals,.975)),
            "valid_bootstrap_replicates":len(vals)}
boot=paired_boot(y_train,cv_probs["Logistic regression"],cv_probs["Prior baseline"])
pd.DataFrame([boot]).to_csv(BASE/"results"/"paired_bootstrap_interval.csv",index=False)

# threshold sensitivity grid: demonstrate tradeoff under different hypothetical cost ratios,
# not a final threshold recommendation.
thresholds=np.linspace(.05,.95,91)
sens=[]
for model_name,p in cv_probs.items():
    for ratio in [1,2,5,10]:
        for cap in [.10,.20,.30,.50]:
            for t in thresholds:
                pred=(p>=t).astype(int)
                tn,fp,fn,tp=confusion_matrix(y_train,pred,labels=[0,1]).ravel()
                sens.append({"model":model_name,"FN_cost_to_FP_cost_ratio":ratio,"review_capacity_fraction":cap,
                             "threshold":t,"review_fraction":float(pred.mean()),
                             "FP":fp,"FN":fn,"cost_per_case":float((fp+ratio*fn)/len(y_train)),
                             "within_capacity":bool(pred.mean()<=cap)})
sens_df=pd.DataFrame(sens)
sens_df.to_csv(BASE/"results"/"threshold_sensitivity_grid.csv",index=False)

# Pre-defined illustrative feature slice: quartiles of mean radius on training distribution.
radius_col="mean radius"
q_edges=np.unique(np.quantile(X_train[radius_col],[0,.25,.5,.75,1]))
# Use training-derived cutpoints for test slices, evaluate logistic test predictions.
test_slice=pd.cut(X_test[radius_col].reset_index(drop=True), bins=q_edges, include_lowest=True, duplicates="drop")
slice_rows=[]
lp=test_probs["Logistic regression"]
y_test_reset = y_test.reset_index(drop=True)
for label in test_slice.cat.categories:
    ix=np.where(test_slice.astype(str).to_numpy() == str(label))[0]
    if len(ix)==0: continue
    yy=y_test_reset.iloc[ix].to_numpy(); pp=lp[ix]; pr=(pp>=.5).astype(int)
    slice_rows.append({"feature":"mean radius","slice":str(label),"n":len(ix),
                       "positive_rate":float(yy.mean()),"ROC_AUC":float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else np.nan,
                       "Brier":float(brier_score_loss(yy,pp)),"Recall_at_0.5":float(recall_score(yy,pr,zero_division=0)),
                       "FP":int(((pr==1)&(yy==0)).sum()),"FN":int(((pr==0)&(yy==1)).sum())})
slice_df=pd.DataFrame(slice_rows)
slice_df.to_csv(BASE/"results"/"illustrative_feature_slice_analysis.csv",index=False)

# Make plots, green/pink palette.
GREEN="#16803c"; PINK="#ec168c"; DARKGREEN="#006b2d"; HOTPINK="#ff4fa3"; GRAY="#777777"
plt.rcParams.update({"font.size":9, "axes.titlesize":12, "axes.labelsize":10, "legend.fontsize":8})
colorset=[GREEN,PINK,DARKGREEN,HOTPINK]
# ROC
fig,ax=plt.subplots(figsize=(7.2,5.2))
for (name,p),c in zip(cv_probs.items(),colorset):
    fpr,tpr,_=roc_curve(y_train,p); ax.plot(fpr,tpr,color=c,lw=2,label=f"{name} (AUC {roc_auc_score(y_train,p):.3f})")
ax.plot([0,1],[0,1],"--",color=GRAY,lw=1); ax.set(xlabel="False-positive rate",ylabel="True-positive rate",title="ROC Curves — Paired 5-Fold CV")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(BASE/"figures"/"01_roc_curves.png",dpi=220); plt.close(fig)
# PR
fig,ax=plt.subplots(figsize=(7.2,5.2))
for (name,p),c in zip(cv_probs.items(),colorset):
    prc,rec,_=precision_recall_curve(y_train,p); ax.plot(rec,prc,color=c,lw=2,label=f"{name} (AP {average_precision_score(y_train,p):.3f})")
ax.set(xlabel="Recall",ylabel="Precision",title="Precision–Recall Curves — Paired 5-Fold CV"); ax.legend(loc="best"); fig.tight_layout()
fig.savefig(BASE/"figures"/"02_precision_recall.png",dpi=220); plt.close(fig)
# calibration
fig,ax=plt.subplots(figsize=(7.2,5.2))
for (name,p),c in zip(cv_probs.items(),colorset):
    obs,mp=calibration_curve(y_train,p,n_bins=10,strategy="uniform"); ax.plot(mp,obs,marker="o",ms=4,color=c,lw=1.8,label=name)
ax.plot([0,1],[0,1],"--",color=GRAY,lw=1,label="Ideal"); ax.set(xlabel="Mean predicted probability",ylabel="Observed positive fraction",title="Calibration Curves — Paired 5-Fold CV")
ax.legend(loc="best"); fig.tight_layout(); fig.savefig(BASE/"figures"/"03_calibration_curves.png",dpi=220); plt.close(fig)
# metric panel
metric_cols=["ROC_AUC","Average_Precision","Brier","ECE_10_equal_width_bins"]
fig,axs=plt.subplots(2,2,figsize=(9,6))
for ax,col in zip(axs.flat,metric_cols):
    vals=cv_table[col]; barcols=[GREEN if "Brier" not in col and "ECE" not in col else PINK for _ in vals.index]
    ax.bar(range(len(vals)),vals.values,color=barcols)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals.index,rotation=30,ha="right")
    ax.set_title(col.replace("_"," ")); ax.grid(axis="y",alpha=.2)
fig.suptitle("Cross-Validation Metrics (threshold-free metrics and calibration)",y=1.01); fig.tight_layout()
fig.savefig(BASE/"figures"/"04_metric_comparison.png",dpi=220,bbox_inches="tight"); plt.close(fig)
# threshold cost sensitivity: logistic, illustrative ratios and capacity, normalized costs
fig,axs=plt.subplots(1,2,figsize=(10,4.5))
for ratio,c in zip([1,2,5,10],colorset):
    sub=sens_df[(sens_df.model=="Logistic regression")&(sens_df.FN_cost_to_FP_cost_ratio==ratio)&(sens_df.review_capacity_fraction==.30)]
    axs[0].plot(sub.threshold,sub.cost_per_case,color=c,label=f"FN:FP cost={ratio}:1")
axs[0].set(xlabel="Threshold",ylabel="Illustrative cost per case",title="Cost Sensitivity (30% capacity scenario)"); axs[0].legend()
for cap,c in zip([.10,.20,.30,.50],colorset):
    sub=sens_df[(sens_df.model=="Logistic regression")&(sens_df.FN_cost_to_FP_cost_ratio==5)&(sens_df.review_capacity_fraction==cap)]
    eligible=sub[sub.within_capacity]
    if len(eligible): axs[1].plot(eligible.threshold,eligible.cost_per_case,color=c,label=f"Capacity ≤ {cap:.0%}")
axs[1].set(xlabel="Threshold meeting capacity",ylabel="Illustrative cost per case",title="Capacity Sensitivity (FN:FP=5:1)"); axs[1].legend()
fig.tight_layout(); fig.savefig(BASE/"figures"/"05_threshold_sensitivity.png",dpi=220); plt.close(fig)
# feature slice plot
fig,ax=plt.subplots(figsize=(7.5,4.8))
x=np.arange(len(slice_df)); ax.bar(x,slice_df["Recall_at_0.5"],color=GREEN,label="Recall at 0.5")
ax.set_xticks(x); ax.set_xticklabels([f"Q{i+1}" for i in range(len(slice_df))])
ax.set_ylim(0,1.05); ax.set(xlabel="Training-derived mean-radius quartile (test set)",ylabel="Recall",title="Illustrative Critical-Slice Check — Logistic Regression")
for i,row in slice_df.iterrows(): ax.text(i,row["Recall_at_0.5"]+.02,f"n={int(row['n'])}",ha="center",fontsize=8)
ax.legend(); fig.tight_layout(); fig.savefig(BASE/"figures"/"06_critical_slice.png",dpi=220); plt.close(fig)

env={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,
     "scikit_learn":importlib.import_module("sklearn").__version__,"seed":42,"test_size":0.2,"cv_folds":5,
     "dataset_csv":"data/wdbc_sklearn_frozen.csv","dataset_sha256":sha,
     "positive_class":"malignant","positive_class_encoding":"1=malignant; 0=benign",
     "threshold_cost_scenarios":"illustrative sensitivity grid only; not a final decision"}
(BASE/"results"/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")

print("Created frozen data and analysis outputs.")
print("SHA256:",sha)
print("CV metrics:\n",cv_table.round(4).to_string())
print("Test metrics:\n",test_table.round(4).to_string())
print("Slice:\n",slice_df.round(3).to_string(index=False))