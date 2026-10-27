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