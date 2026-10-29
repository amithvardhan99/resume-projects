def return_expertise_vs_rating(df_1,display_graph=False,**kwargs):
    df = df_1.copy()
    dic = kwargs.copy()
    for i in dic:
        df = df[df[i].isin(dic[i])]

    expertise_vs_rating = df[["Expertise","CourseRating"]].drop_duplicates(ignore_index=True)
    expertise_vs_rating["Number"] = 0
    for i in expertise_vs_rating["Expertise"].unique():
        pu = expertise_vs_rating[expertise_vs_rating["Expertise"] == i]
        ctr = 0
        for j in pu.index:
            expertise_vs_rating.loc[j,"Number"] = ctr
            ctr += 1

    each_expertise = pd.DataFrame(columns=["Expertise","LeastRating","HighestRating","AverageRating"])
    each_expertise["Expertise"] = df["Expertise"].unique()
    least_ratings = []
    highest_ratings = []
    avg_ratings = []
    ctr = 0
    for i in df["Expertise"].unique():
        least_rating = df[df["Expertise"] == i]["CourseRating"].min()
        highest_rating = df[df["Expertise"] == i]["CourseRating"].max()
        avg_rating = df[df["Expertise"] == i]["CourseRating"].mean()

        least_ratings.append(least_rating)
        highest_ratings.append(highest_rating)
        avg_ratings.append(avg_rating)

    each_expertise["LeastRating"] = least_ratings
    each_expertise["HighestRating"] = highest_ratings
    each_expertise["AverageRating"] = avg_ratings

    if display_graph == True:
        sns.pointplot(x=level_vs_rating["Number"],y=level_vs_rating["CourseRating"],hue=level_vs_rating["Expertise"])
        plt.show()

    return expertise_vs_rating, each_expertise, df



#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

nrows = 4
ncols = len(df["CourseCategory"].unique())//nrows

list_of_course_categories = df["CourseCategory"].unique()

# fig, ax = plt.subplots(nrows=len(list_of_course_categories),ncols=1,figsize=(20,10))
fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(20,18))
fig.subplots_adjust(hspace=0.8, wspace=0.5)

ctr = 0
for (i,j) in list(itertools.product(range(nrows),range(ncols))):
    course_category = list_of_course_categories[ctr]
    lf = df[df["CourseCategory"]==course_category]["CourseRating"].reset_index(drop=True)
    sns.lineplot(lf,ax=ax[i,j],drawstyle="steps-post",lw=4)
    ax[i,j].set_title(course_category)
    ax[i,j].set_xticks([])

    #ax[ctr] = sns.pointplot(lf)
    #
    ctr += 1


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################


total_enrolments = pd.DataFrame(columns=["Tier","Total Enrolments","Average Per Teacher"])
total_enrolments.loc[0,"Tier"] = "Low"
total_enrolments.loc[0,"Total Enrolments"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Low"]["Enrolments"].sum()
total_enrolments.loc[0,"Average Per Teacher"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Low"]["Enrolments"].mean().round(2)
total_enrolments.loc[0,"First Quartile"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Low"]["Enrolments"].quantile(0.25)
total_enrolments.loc[0,"Median Enrolments"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Low"]["Enrolments"].median()
total_enrolments.loc[0,"Third Quartile"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Low"]["Enrolments"].quantile(0.75)



total_enrolments.loc[1,"Tier"] = "Mid"
total_enrolments.loc[1,"Total Enrolments"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Mid"]["Enrolments"].sum()
total_enrolments.loc[1,"Average Per Teacher"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Mid"]["Enrolments"].mean().round(2)
total_enrolments.loc[1,"First Quartile"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Mid"]["Enrolments"].quantile(0.25)
total_enrolments.loc[1,"Median Enrolments"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Mid"]["Enrolments"].median()
total_enrolments.loc[1,"Third Quartile"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="Mid"]["Enrolments"].quantile(0.75)


total_enrolments.loc[2,"Tier"] = "High"
total_enrolments.loc[2,"Total Enrolments"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="High"]["Enrolments"].sum()
total_enrolments.loc[2,"Average Per Teacher"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="High"]["Enrolments"].mean().round(2)
total_enrolments.loc[2,"First Quartile"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="High"]["Enrolments"].quantile(0.25)
total_enrolments.loc[2,"Median Enrolments"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="High"]["Enrolments"].median()
total_enrolments.loc[2,"Third Quartile"] = teacher_names_with_ratings[teacher_names_with_ratings["Tier"]=="High"]["Enrolments"].quantile(0.75)



sns.boxplot(x=teacher_names_with_ratings_1["Tier"],y=teacher_names_with_ratings_1["Enrolments"])
total_enrolments

