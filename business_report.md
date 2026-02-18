# HR Employee Attrition — Business Report

**Analyst:** Tanmay Patil  
**Dataset:** IBM HR Analytics | 1,470 Employees  
**Tools:** Python, SQL, Scikit-Learn, Tableau, Excel

---

## Executive Summary

This report analyzes 1,470 employee records to understand why employees
leave, which factors drive attrition, and how the company can reduce
turnover. A predictive model was built to identify at-risk employees
before they resign.

**Key Takeaway:** The company loses an estimated **$2.7M+ annually**
due to attrition. The top drivers are **overtime, low income, poor
satisfaction, and lack of career growth**. Targeted interventions
could reduce attrition by 30-50%, saving **$800K–$1.3M per year**.

---

## Section 1: Attrition Overview

| Metric               | Value         |
| -------------------- | ------------- |
| Total Employees      | 1,470         |
| Employees Who Left   | 237 (16.1%)   |
| Employees Who Stayed | 1,233 (83.9%) |

### By Department

| Department             | Employees | Attrition Rate |
| ---------------------- | --------- | -------------- |
| Sales                  | 446       | 20.6%          |
| Human Resources        | 63        | 19.0%          |
| Research & Development | 961       | 13.8%          |

### By Job Role (Top 5 Highest)

| Job Role              | Attrition Rate |
| --------------------- | -------------- |
| Sales Representative  | 39.8%          |
| Laboratory Technician | 23.8%          |
| Human Resources       | 23.1%          |
| Sales Executive       | 17.5%          |
| Research Scientist    | 16.1%          |

**Insight:** Sales Representatives have nearly **4x the average
attrition rate** — this single role needs urgent attention.

---

## Section 2: Root Cause Analysis

### 2.1 Overtime (STRONGEST FACTOR)

| Overtime | Attrition Rate |
| -------- | -------------- |
| No       | 10.4%          |
| Yes      | 30.5%          |

> **Employees working overtime are 3x more likely to leave.**

### 2.2 Income

| Group   | Avg Income (Stayed) | Avg Income (Left) | Gap  |
| ------- | ------------------- | ----------------- | ---- |
| Overall | $6,833              | $4,787            | -33% |

> **Employees who left earned 33% less** than those who stayed.

### 2.3 Tenure

| Tenure     | Attrition Rate |
| ---------- | -------------- |
| 0-1 year   | 34.5%          |
| 2-3 years  | 16.2%          |
| 4-5 years  | 11.8%          |
| 6-10 years | 9.5%           |
| 10+ years  | 8.2%           |

> **New employees (0-1 year) have 4x higher attrition** than veterans.

### 2.4 Job Satisfaction

| Level         | Attrition Rate |
| ------------- | -------------- |
| Low (1)       | 22.8%          |
| Medium (2)    | 16.3%          |
| High (3)      | 16.5%          |
| Very High (4) | 11.3%          |

### 2.5 Work-Life Balance

| Level      | Attrition Rate |
| ---------- | -------------- |
| Bad (1)    | 31.3%          |
| Good (2)   | 16.8%          |
| Better (3) | 15.1%          |
| Best (4)   | 13.2%          |

### 2.6 Business Travel

| Travel            | Attrition Rate |
| ----------------- | -------------- |
| Non-Travel        | 8.0%           |
| Travel Rarely     | 14.9%          |
| Travel Frequently | 24.9%          |

---

## Section 3: Predictive Model

### Model Comparison

| Model               | Accuracy | AUC-ROC |
| ------------------- | -------- | ------- |
| Logistic Regression | ~84%     | ~0.82   |
| Random Forest       | ~87%     | ~0.84   |

### Top 5 Predictive Features

1. **OverTime** — Strongest predictor
2. **MonthlyIncome** — Lower income = higher risk
3. **Age** — Younger employees leave more
4. **TotalWorkingYears** — Less experience = higher risk
5. **JobSatisfaction** — Low satisfaction drives exits

### Risk Distribution

| Risk Level  | Employees | % of Workforce |
| ----------- | --------- | -------------- |
| Low Risk    | ~1,050    | ~71%           |
| Medium Risk | ~280      | ~19%           |
| High Risk   | ~140      | ~10%           |

---

## Section 4: Financial Impact

### Cost of Attrition

| Metric                            | Value              |
| --------------------------------- | ------------------ |
| Avg Annual Salary (Left)          | ~$57,400           |
| Replacement Cost (100% of salary) | ~$57,400           |
| Total Annual Attrition Cost       | ~$2.7M             |
| Cost if Attrition Reduced by 30%  | Saves ~$810K/year  |
| Cost if Attrition Reduced by 50%  | Saves ~$1.35M/year |

---

## Section 5: Recommendations

### Priority 1: Overtime Management

- **Problem:** 30.5% attrition with OT vs 10.4% without
- **Action:**
  - Implement mandatory overtime caps (max 10 hrs/week)
  - Hire additional staff for overloaded teams
  - Offer comp time instead of extended overtime
- **Impact:** Could reduce attrition by ~8% overall

### Priority 2: Compensation Review

- **Problem:** Employees who left earned 33% less
- **Action:**
  - Market salary benchmarking for all roles
  - Immediate 10-15% adjustment for underpaid roles
  - Annual salary reviews tied to performance
  - Retention bonuses for high-risk employees
- **Impact:** Reduce income-related attrition by ~25%

### Priority 3: New Employee Retention

- **Problem:** 0-1 year employees = 34.5% attrition
- **Action:**
  - 90-day structured onboarding program
  - Assign mentors for first 6 months
  - Monthly check-ins during first year
  - "Stay interview" at 6-month mark
- **Impact:** Reduce early attrition by ~40%

### Priority 4: Career Development

- **Problem:** No promotion >5 years = high exit risk
- **Action:**
  - Clear promotion criteria and timelines
  - Individual development plans for each employee
  - Cross-training and lateral move opportunities
  - Skills-based advancement tracks
- **Impact:** Improve tenure by 1-2 years on average

### Priority 5: Work-Life Balance

- **Problem:** Bad WLB = 31% attrition vs 13% with good WLB
- **Action:**
  - Flexible working hours / hybrid model
  - Mental health support programs
  - Mandatory PTO utilization
  - Manager training on team wellbeing
- **Impact:** Reduce burnout-related exits by ~30%

---

## Department-Specific Actions

### Sales Department (20.6% attrition — HIGHEST)

- Review quota pressure and commission structures
- Sales Representative role needs immediate salary adjustment
- Reduce mandatory travel for frequently traveling reps
- Implement team-based incentives alongside individual targets

### Human Resources (19.0% attrition)

- Small team = each exit has outsized impact
- Focus on job enrichment and cross-functional projects
- Consider role restructuring for variety

### Research & Development (13.8% attrition)

- Relatively healthy but still above 10% benchmark
- Focus on Lab Technician retention (23.8% rate)
- Provide research autonomy and publication opportunities

---

## Implementation Roadmap

| Timeline  | Action                                | Owner           |
| --------- | ------------------------------------- | --------------- |
| Month 1-2 | Overtime audit & caps                 | HR + Managers   |
| Month 1-3 | Salary benchmarking & adjustments     | HR + Finance    |
| Month 2-4 | Onboarding program redesign           | HR + L&D        |
| Month 3-6 | Career development framework          | HR + Leadership |
| Month 4-6 | Work-life balance programs            | HR + Wellness   |
| Ongoing   | Monthly risk monitoring with ML model | HR Analytics    |

---

## Success Metrics (Track Quarterly)

| Metric                   | Current | 6-Month Target | 1-Year Target |
| ------------------------ | ------- | -------------- | ------------- |
| Attrition Rate           | 16.1%   | 13%            | 10%           |
| Avg Tenure               | 7.0 yr  | 7.5 yr         | 8.0 yr        |
| Job Satisfaction         | 2.73/4  | 3.0/4          | 3.2/4         |
| Overtime Rate            | ~28%    | 20%            | 15%           |
| New Hire Retention (1yr) | 65%     | 75%            | 85%           |

---

_Report by: [Tanmay Patil]_ |
_Full code : [https://github.com/tanmay121-hub/hr-attrition-analytics]_
