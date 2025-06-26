<!-- Project title banner (optional) -->
<h1 align="center">
  🏥 Mortality-Risk Prediction for Heart-Failure ICU Patients
</h1>
<p align="center">
  <em>Machine-learning pipeline built on the MIMIC-III database to predict
  in-hospital mortality and identify key clinical risk factors for
  heart-failure patients admitted to intensive-care units.</em>
</p>

---

## 📌 Motivation
Accurately predicting the **risk of in-hospital death** for heart-failure
(HF) patients on ICU admission helps clinicians:

* prioritise high-risk cases for aggressive management  
* allocate limited critical-care resources  
* discuss prognosis with families early  
* design decision-support tools that go beyond traditional severity scores
  such as APACHE-II/IV and SOFA

---

## 🗃️ Data Source
This work uses the publicly available **MIMIC-III v1.4** database:

| Item | Value |
|------|-------|
| ICU admissions in MIMIC-III | 58 ,976 |
| HF admissions screened (ICD-9 codes 398.91, 428.xx) | 13 ,389 |
| Final cohort after cleaning (age ≥ 18, first ICU stay, <10 % missing) | **≈ 1 ,200** patients |

> **Note** Direct access to raw MIMIC-III requires completing the
> PhysioNet credentialing process.

---

## Results

# train Gradient-Boost baseline
python scripts/main_classification.py

| Classifier                | Precision | Recall | **F1 Score** |
|---------------------------|-----------|--------|-------------|
| Logistic Regression       | 0.34      | **0.65** | **0.65** |
| Linear SVC                | 0.31      | 0.62   | 0.63 |
| Random Forest             | **0.43**  | 0.34   | 0.64 |
| Support Vector Machine    | 0.42      | 0.20   | 0.59 |
| Adaptive Boosting         | 0.38      | 0.17   | 0.57 |
| Decision Tree             | 0.25      | 0.34   | 0.57 |
| K-Nearest Neighbor (KNN)  | 0.21      | 0.48   | 0.54 |


```bash
# clone
git clone https://github.com/AkshatP0285/Prediction-of-Mortality-Rate-of-Heart-Failure-Patients-Admitted-to-ICU.git
cd Prediction-of-Mortality-Rate-of-Heart-Failure-Patients-Admitted-to-ICU

# set up env
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

