from datetime import datetime
import sqlite3
from dotenv import load_dotenv
load_dotenv()

def search_products(name, category, description="", max_price=None, is_organic=None):
    conn = sqlite3.connect("../store.db")
    cursor = conn.cursor()
    sql_query = "select name, category, description, price, is_organic from products where 1=1"
    all_parameters = []

    sql_query += " AND (name LIKE ? OR category LIKE ? OR description LIKE ?)"
    all_parameters.extend([name, category, description])

    if max_price is not None:
        sql_query += " AND price <= ?"
        all_parameters.append(max_price)

    if is_organic is not None:
        sql_query += " AND organic = ?"
        all_parameters.append(is_organic)

    cursor.execute(sql_query, all_parameters)
    all_rows = cursor.fetchall()

    result_list = []

    for i in all_rows:
        result_dict = {
            "name": i[0],
            "category": i[1],
            "description": i[2] if i[2] else "",
            "price": i[3] if i[3] else 0.0,
            "is_organic": bool(i[4])
        }
        result_list.append(result_dict)
    conn.close()

    return result_list

def fetch_product_rating(id):
    conn = sqlite3.connect("../store.db")
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, avg(rating) as avg_rating, count(*) as review_count FROM reviews WHERE product_id = ? group by product_id", (id,))
    row = cursor.fetchall()[0]
    if row:
        avg_rating = round(row[1],2) if row[1] else 0.0
        review_count = row[2]  if row[2] else 0
    else:
        avg_rating = 0.0
        review_count = 0
    conn.close()

    dic = {"product_id": id, "avg_rating": avg_rating, "review_count": review_count}
    return dic

def ratings_for_different_products(ids):
    ratings = []
    for id in ids:
        ratings.append(fetch_product_rating(id))
    return ratings

def fetch_current_time():
    y = lambda x: f"0{x}" if x < 10 else str(x)

    # Get current date and time
    now = datetime.now()
    current_date = now.date()
    current_time = now.time()

    year = y(current_date.year)
    month = y(current_date.month)
    day = y(current_date.day)

    hour = y(current_time.hour)
    minute = y(current_time.minute)
    second = y(current_time.second)

    total_time = f"{year}-{month}-{day} {hour}:{minute}:{second}"
    return total_time

def checkout(id):
    conn = sqlite3.connect("../store.db")
    cursor = conn.cursor()

    cursor.execute("select id, name, price from products where id = ?", (id,))
    purchased_product = cursor.fetchall()[0]

    product_id = purchased_product[0]
    product_name = purchased_product[1]
    product_price = purchased_product[2]

    cursor.execute("select id from orders")
    new_id = cursor.fetchall()[-1][0] + 1
    current_time = fetch_current_time()

    if purchased_product:
        cursor.execute("insert into orders (id,product_id, product_name, price,ordered_at) values (?, ?, ?, ?, ?)", (new_id, product_id, product_name, product_price, current_time))
        return_dict = {
            "status": "success",
            "message": f"Order placed successfully for '{product_name}'.",
            "product_id": product_id,
            "product_name": product_name,
            "price": product_price,
            "ordered_at": current_time
        }
    else:
        return_dict = {
            "status": "error",
            "message": f"No product {product_name} found. Please verify the product ID and try again."
        }
    conn.close()
    return return_dict