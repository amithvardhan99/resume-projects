# Instructor Performance and Course Quality Evaluation on EduPro

## Project Overview

This project builds a data-driven framework to evaluate instructor effectiveness and course quality on the EduPro online learning platform.

## Files in this Folder

| File | Description |
|------|-------------|
| `teachers.csv` | Instructor details (60 records) |
| `courses.csv` | Course details (60 records) |
| `transactions.csv` | Enrolment / payment records (10,000 rows) |
| `users.csv` | Learner details (3,000 records) |
| `notebook.ipynb` | Full Exploratory Data Analysis (EDA) |
| `app.py` | Streamlit interactive dashboard |
| `Research_Paper.docx` | Research paper with EDA, insights and recommendations |
| `Executive_Summary.docx` | Short summary for stakeholders / government |
| `Introduction.docx` | Background introduction document |
| `requirements.txt` | Python packages needed for the dashboard |

## How to Run the Streamlit Dashboard

1. Open a terminal in this folder.
2. Install the required packages (if not already installed):

```bash
pip install -r requirements.txt
```

3. Start the dashboard:

```bash
streamlit run app.py
```

4. The dashboard will open in your browser. Use the left sidebar to filter by:
   - Instructor Expertise
   - Course Category
   - Course Level
   - Teacher Rating range

## Dashboard Modules

1. **Key Performance Indicators (KPIs)** – Average Teacher Rating, Average Course Rating, Rating Consistency Index, Experience Impact Score, Enrolment Influence Ratio
2. **Instructor Performance Leaderboard** – Ranked list of instructors with top and bottom performers
3. **Experience vs Rating Analysis** – Scatter plots of experience against teacher rating and course rating
4. **Course Quality Evaluation** – Bar charts and heatmap by category and level
5. **Expertise-wise Performance Comparison** – Summary table and charts by expertise domain
6. **Instructor Impact on Enrolments** – Comparison of High / Mid / Low rating tiers

## Key Analytical Questions Answered

1. What is the overall distribution of instructor ratings?
2. Do instructors with more experience receive higher ratings?
3. Is there a relationship between TeacherRating and CourseRating?
4. Which expertise areas consistently deliver high-quality courses?
5. Are highly rated instructors associated with higher enrolments?

## Main Insights (Short)

- Average Teacher Rating ≈ 3.12; ratings are spread across the full scale.
- Experience and Teacher Rating have a moderate positive correlation (≈ 0.60).
- Teacher Rating and Course Rating are almost unrelated (correlation ≈ –0.03).
- High-rated instructors tend to attract more enrolments.
- Marketing and Programming have only one instructor each – staffing gap.
- Machine Learning and Cybersecurity need special attention on teaching quality.

## Code Style

All Python code (especially `app.py`) is written in a simple, easy-to-read style suitable for learners. Functions are kept short and comments explain each major step.
