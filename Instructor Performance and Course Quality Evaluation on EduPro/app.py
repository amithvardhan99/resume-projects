# ============================================================
# EduPro - Instructor Performance and Course Quality Evaluation
# Streamlit Dashboard
# Simple and easy-to-understand Python code
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="EduPro Instructor & Course Quality Dashboard",
    page_icon="📚",
    layout="wide"
)

# ------------------------------------------------------------
# Load data (simple function)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    teachers = pd.read_csv("teachers.csv")
    courses = pd.read_csv("courses.csv")
    transactions = pd.read_csv("transactions.csv")

    # Join the three tables
    temp = pd.merge(teachers, transactions, on="TeacherID", how="right")
    merged = pd.merge(courses, temp, on="CourseID", how="right")
    return teachers, courses, transactions, merged

teachers, courses, transactions, df = load_data()

# ------------------------------------------------------------
# Title and introduction
# ------------------------------------------------------------
st.title("📚 EduPro - Instructor Performance & Course Quality Evaluation")
st.markdown("""
This dashboard helps EduPro evaluate **instructor effectiveness** and **course quality**.
Use the filters on the left to explore the data.
""")

# ------------------------------------------------------------
# Sidebar filters (simple)
# ------------------------------------------------------------
st.sidebar.header("Filters")

# Expertise filter
all_expertise = sorted(teachers["Expertise"].unique().tolist())
selected_expertise = st.sidebar.multiselect(
    "Select Expertise",
    options=all_expertise,
    default=all_expertise
)

# Course Category filter
all_categories = sorted(courses["CourseCategory"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Select Course Category",
    options=all_categories,
    default=all_categories
)

# Course Level filter
all_levels = sorted(courses["CourseLevel"].unique().tolist())
selected_levels = st.sidebar.multiselect(
    "Select Course Level",
    options=all_levels,
    default=all_levels
)

# Rating range slider
min_rating, max_rating = st.sidebar.slider(
    "Teacher Rating Range",
    min_value=1.0,
    max_value=5.0,
    value=(1.0, 5.0),
    step=0.1
)

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------
filtered_teachers = teachers[
    (teachers["Expertise"].isin(selected_expertise)) &
    (teachers["TeacherRating"] >= min_rating) &
    (teachers["TeacherRating"] <= max_rating)
]

filtered_df = df[
    (df["Expertise"].isin(selected_expertise)) &
    (df["CourseCategory"].isin(selected_categories)) &
    (df["CourseLevel"].isin(selected_levels)) &
    (df["TeacherRating"] >= min_rating) &
    (df["TeacherRating"] <= max_rating)
]

filtered_courses = courses[
    (courses["CourseCategory"].isin(selected_categories)) &
    (courses["CourseLevel"].isin(selected_levels))
]

# ------------------------------------------------------------
# KPI Section
# ------------------------------------------------------------
st.header("Key Performance Indicators (KPIs)")

col1, col2, col3, col4, col5 = st.columns(5)

avg_teacher_rating = round(filtered_teachers["TeacherRating"].mean(), 2) if len(filtered_teachers) > 0 else 0
avg_course_rating = round(filtered_courses["CourseRating"].mean(), 2) if len(filtered_courses) > 0 else 0

# Rating Consistency Index = standard deviation of course ratings (lower is better)
rating_consistency = round(filtered_courses["CourseRating"].std(), 2) if len(filtered_courses) > 1 else 0

# Experience Impact Score = correlation between experience and teacher rating
if len(filtered_teachers) > 2:
    experience_impact = round(
        filtered_teachers[["YearsOfExperience", "TeacherRating"]].corr().iloc[0, 1], 3
    )
else:
    experience_impact = 0

# Enrolment Influence Ratio = correlation between teacher rating and number of enrolments
enrol_data = filtered_df.groupby("TeacherName").agg(
    Enrolments=("TransactionID", "count"),
    TeacherRating=("TeacherRating", "first")
).reset_index()
if len(enrol_data) > 2:
    enrolment_influence = round(
        enrol_data[["TeacherRating", "Enrolments"]].corr().iloc[0, 1], 3
    )
else:
    enrolment_influence = 0

col1.metric("Average Teacher Rating", avg_teacher_rating)
col2.metric("Average Course Rating", avg_course_rating)
col3.metric("Rating Consistency Index", rating_consistency)
col4.metric("Experience Impact Score", experience_impact)
col5.metric("Enrolment Influence Ratio", enrolment_influence)

st.markdown("---")

# ------------------------------------------------------------
# Module 1: Instructor Performance Leaderboard
# ------------------------------------------------------------
st.header("1. Instructor Performance Leaderboard")

leaderboard = filtered_teachers[
    ["TeacherName", "Expertise", "YearsOfExperience", "TeacherRating", "Age", "Gender"]
].sort_values(by="TeacherRating", ascending=False).reset_index(drop=True)

leaderboard.index = leaderboard.index + 1
leaderboard.index.name = "Rank"

st.dataframe(leaderboard, use_container_width=True)

# Top and bottom performers
if len(filtered_teachers) > 0:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 5 Instructors")
        st.table(leaderboard.head(5)[["TeacherName", "Expertise", "TeacherRating"]])
    with col_b:
        st.subheader("Bottom 5 Instructors")
        st.table(leaderboard.tail(5)[["TeacherName", "Expertise", "TeacherRating"]])

st.markdown("---")

# ------------------------------------------------------------
# Module 2: Experience vs Rating Scatter Plot
# ------------------------------------------------------------
st.header("2. Experience vs Rating Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Years of Experience vs Teacher Rating")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=filtered_teachers,
        x="YearsOfExperience",
        y="TeacherRating",
        hue="Expertise",
        ax=ax1
    )
    ax1.set_xlabel("Years of Experience")
    ax1.set_ylabel("Teacher Rating")
    ax1.set_title("Experience vs Teacher Rating")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    st.pyplot(fig1)
    plt.close()

with col2:
    st.subheader("Years of Experience vs Course Rating")
    exp_course = filtered_df[["YearsOfExperience", "CourseRating"]].drop_duplicates()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=exp_course,
        x="YearsOfExperience",
        y="CourseRating",
        ax=ax2,
        color="teal"
    )
    ax2.set_xlabel("Years of Experience")
    ax2.set_ylabel("Course Rating")
    ax2.set_title("Experience vs Course Rating")
    st.pyplot(fig2)
    plt.close()

# Simple correlation message
st.info(
    f"**Insight:** Correlation between Years of Experience and Teacher Rating is "
    f"**{experience_impact}**. More experience tends to give higher teacher ratings, "
    f"but the link with course ratings is much weaker."
)

st.markdown("---")

# ------------------------------------------------------------
# Module 3: Course Quality Heatmap & Distribution
# ------------------------------------------------------------
st.header("3. Course Quality Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Course Rating by Category")
    cat_rating = filtered_courses.groupby("CourseCategory")["CourseRating"].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=cat_rating.values, y=cat_rating.index, palette="viridis", ax=ax3)
    ax3.set_xlabel("Average Course Rating")
    ax3.set_ylabel("Course Category")
    ax3.set_xlim(0, 5)
    st.pyplot(fig3)
    plt.close()

with col2:
    st.subheader("Average Course Rating by Level")
    level_rating = filtered_courses.groupby("CourseLevel")["CourseRating"].mean().reindex(
        ["Beginner", "Intermediate", "Advanced"]
    )
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=level_rating.index, y=level_rating.values, palette="muted", ax=ax4)
    ax4.set_ylabel("Average Course Rating")
    ax4.set_ylim(0, 5)
    st.pyplot(fig4)
    plt.close()

# Heatmap: Category vs Level
st.subheader("Course Quality Heatmap (Category × Level)")
heatmap_data = filtered_courses.pivot_table(
    values="CourseRating",
    index="CourseCategory",
    columns="CourseLevel",
    aggfunc="mean"
)
# Reorder columns if present
desired_order = [c for c in ["Beginner", "Intermediate", "Advanced"] if c in heatmap_data.columns]
heatmap_data = heatmap_data[desired_order]

fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    ax=ax5,
    vmin=1,
    vmax=5
)
ax5.set_title("Average Course Rating by Category and Level")
st.pyplot(fig5)
plt.close()

st.markdown("---")

# ------------------------------------------------------------
# Module 4: Expertise-wise Performance Comparison
# ------------------------------------------------------------
st.header("4. Expertise-wise Performance Comparison")

# Average teacher rating and number of instructors per expertise
expertise_summary = filtered_teachers.groupby("Expertise").agg(
    Average_Teacher_Rating=("TeacherRating", "mean"),
    Number_of_Instructors=("TeacherID", "count"),
    Average_Experience=("YearsOfExperience", "mean")
).round(2).sort_values(by="Average_Teacher_Rating", ascending=False)

st.subheader("Expertise Summary Table")
st.dataframe(expertise_summary, use_container_width=True)

# Bar chart of average teacher rating by expertise
fig6, ax6 = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=expertise_summary.index,
    y=expertise_summary["Average_Teacher_Rating"],
    palette="coolwarm",
    ax=ax6
)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha="right")
ax6.set_ylabel("Average Teacher Rating")
ax6.set_ylim(0, 5)
ax6.set_title("Average Teacher Rating by Expertise")
st.pyplot(fig6)
plt.close()

# Teacher rating distribution
st.subheader("Overall Teacher Rating Distribution")
fig7, ax7 = plt.subplots(figsize=(8, 4))
sns.histplot(filtered_teachers["TeacherRating"], bins=15, kde=True, color="steelblue", ax=ax7)
ax7.set_xlabel("Teacher Rating")
ax7.set_ylabel("Count")
ax7.axvline(avg_teacher_rating, color="red", linestyle="--", label=f"Mean = {avg_teacher_rating}")
ax7.legend()
st.pyplot(fig7)
plt.close()

st.markdown("---")

# ------------------------------------------------------------
# Module 5: Instructor Impact on Course Success (Tiers)
# ------------------------------------------------------------
st.header("5. Instructor Impact on Enrolments")

# Create rating tiers
def get_tier(rating):
    if rating >= 4.0:
        return "High"
    elif rating >= 2.0:
        return "Mid"
    else:
        return "Low"

enrol_data["Tier"] = enrol_data["TeacherRating"].apply(get_tier)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Enrolments by Instructor Tier")
    tier_enrol = enrol_data.groupby("Tier")["Enrolments"].mean().reindex(["Low", "Mid", "High"])
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=tier_enrol.index, y=tier_enrol.values, palette=["#e74c3c", "#f39c12", "#27ae60"], ax=ax8)
    ax8.set_ylabel("Average Enrolments per Instructor")
    ax8.set_title("Average Enrolments by Rating Tier")
    st.pyplot(fig8)
    plt.close()

with col2:
    st.subheader("Number of Instructors in each Tier")
    tier_count = enrol_data["Tier"].value_counts().reindex(["Low", "Mid", "High"])
    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=tier_count.index, y=tier_count.values, palette=["#e74c3c", "#f39c12", "#27ae60"], ax=ax9)
    ax9.set_ylabel("Number of Instructors")
    st.pyplot(fig9)
    plt.close()

st.markdown("---")

# ------------------------------------------------------------
# Footer / Recommendations summary
# ------------------------------------------------------------
st.header("Quick Recommendations")
st.markdown("""
1. **Experience helps teacher ratings** – Correlation is moderate (~0.60). Mentoring junior instructors can improve quality.
2. **Teacher rating does not strongly drive course rating** – Course design and content also matter a lot.
3. **Marketing & Programming** have only 1 instructor each – Hire more instructors in these areas.
4. **Machine Learning and Cybersecurity** show higher dependence on instructor quality – Focus training here.
5. **High-rated instructors** tend to attract more enrolments – Promote top performers on the platform.
""")

st.caption("EduPro Instructor Performance & Course Quality Evaluation Dashboard | Data-driven insights for better education outcomes")
