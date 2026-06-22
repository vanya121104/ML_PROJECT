# HR Attrition Analysis & Business Recommendations Report

## Executive Summary
This report analyzes employee attrition within our organization using historical data. The primary objective is to identify key drivers of attrition and offer data-backed, actionable recommendations to improve employee retention, work-life satisfaction, and operational efficiency. 

By applying predictive modeling (including logistic regression, random forests, and gradient boosting classifiers) and SHAP (SHapley Additive exPlanations), we have isolated the most significant risk indicators.

---

## 1. Attrition Demographics & Imbalance
Our dataset comprises **1,470 employees**, with a current attrition rate of **16.12%** (237 departed, 1,233 retained). While 16% might seem manageable, voluntary departures generate substantial costs in recruitment, onboarding, lost productivity, and diminished team morale.

---

## 2. Key Drivers of Attrition

### ⚠️ A. Overtime Work (The Strongest Behavioral Predictor)
Employees who work overtime exhibit a **significantly higher attrition rate** compared to those who do not:
* **Overtime Employees Attrition Rate:** **~30.5%**
* **Non-Overtime Employees Attrition Rate:** **~10.4%**
* **Insight:** Constant overtime leads directly to burnout. The data shows that the probability of exit triples when overtime is regularly demanded.

### 💰 B. Compensation (Monthly Income)
We observe a clear inverse correlation between monthly income and attrition:
* The median monthly income of employees who left is **~$3,200**, compared to **~$5,200** for those who stayed.
* **Insight:** Entry-level staff and individuals in lower salary tiers (Job Level 1 & 2) are highly susceptible to market poaching. Compensation adjustments at these lower bands yield the highest retention returns.

### 🚗 C. Travel Frequency
* Employees who **Travel Frequently** have an attrition rate of **~24.9%**, compared to only **~15.0%** for those who travel rarely and **~8.0%** for non-travelers.
* **Insight:** Heavy travel schedules compromise work-life balance and increase job stress.

### 🧠 D. Job Satisfaction & Work-Life Balance
* Employees rating their **Work-Life Balance** as "Bad" (1 out of 4) experience a **~31.2% attrition rate**.
* Similarly, low **Job Satisfaction** (1 out of 4) corresponds to a **~22.8% attrition rate**, which falls to **~11.3%** for employees who rate it as high (4 out of 4).
* **Insight:** A supportive environment with flexible hours and engaging roles is a critical safety net against attrition.

### ⏳ E. Tenure & Manager Relationships
* Attrition is heavily concentrated in the first **1–3 years** of tenure. 
* Employees working with the same manager for **less than a year** are highly vulnerable.
* **Insight:** The transition period for new hires and alignment with managers are critical milestones where attrition risk is elevated.

---

## 3. Policy & Intervention Recommendations

Based on these findings, we recommend that HR and Executive Leadership implement the following targeted initiatives:

1. **Overtime Auditing & Compensation Adjustments:**
   * Audit departments with high overtime rates (specifically R&D laboratory technicians and Sales representatives).
   * Put caps on consecutive overtime weeks.
   * Shift from standard overtime requests to rotating on-call structures where possible.

2. **Targeted Career Pathways & Micro-Raises for Low Job Levels:**
   * Establish structured, transparent 18-month career progression plans for Job Levels 1 and 2.
   * Conduct salary reviews to ensure lower-band roles align with competitive local market rates.

3. **Travel Compensation & Hybrid Work Accommodation:**
   * Define a "high-travel threshold" and provide rest days or work-from-home options immediately following frequent travel periods.
   * Compensate heavy travel roles with supplemental bonuses or extra paid time off.

4. **Manager Training and Early Onboarding Check-ins:**
   * Implement manager feedback surveys at the 3-month and 6-month milestones for new hires.
   * Train managers on empathy, work allocation, and burnout detection.
