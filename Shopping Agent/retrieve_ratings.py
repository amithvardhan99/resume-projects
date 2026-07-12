import sqlite3

conn = sqlite3.connect("store.db")
cursor = conn.cursor()

def fetch_product_rating(id):
    cursor.execute("SELECT product_id, avg(rating) as avg_rating, count(*) as review_count FROM reviews WHERE product_id = ? group by product_id", (id,))
    row = cursor.fetchall()[0]
    if row:
        avg_rating = round(row[1],2) if row[1] else 0.0
        review_count = row[2]  if row[2] else 0
    else:
        avg_rating = 0.0
        review_count = 0

    dic = {"product_id": id, "avg_rating": avg_rating, "review_count": review_count}
    return dic

def ratings_for_different_products(ids):
    ratings = []
    for id in ids:
        ratings.append(fetch_product_rating(id))
    return ratings

