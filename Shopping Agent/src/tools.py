from langchain.tools import tool
import sqlite3
from dotenv import load_dotenv
load_dotenv()

@tool
def search_products_tool(query, max_price=None, is_organic=None):
    """
    Search for products in the product catalog.

    Use this tool whenever the user is looking for products based on keywords, category, description, price, or organic status.

    Arguments:
        query:
            A keyword or phrase to search for. The search is performed when either product name or category or description is provided.

        max_price:
            Optional maximum price filter. Only products with a price less than or equal to this value are returned.

        is_organic:
            Optional organic filter.
            - True: return only organic products.
            - False: return only non-organic products.
            - None: do not filter by organic status.

    Returns:
        A list of matching products. Each product contains:
        - name
        - category
        - description
        - price
        - is_organic

    Use this tool for requests such as:
    - "Find honey products."
    - "Show organic honey."
    - "Find snacks under $10."
    - "Search for herbal tea."
    - "Show beverages below $5."
    - "Find products containing almonds."

    Do not use this tool when the user is asking about product reviews, ratings, availability, inventory, or order status.
    """
    conn = sqlite3.connect("../store.db")
    cursor = conn.cursor()
    sql_query = "select name, category, description, price, is_organic from products where 1=1"
    all_parameters = []

    if query:
        sql_query += " AND (name LIKE ? OR category LIKE ? OR description LIKE ?)"
        query_param = f"%{query}%"
        all_parameters.extend([query_param,query_param,query_param])

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
            "name" : i[0],
            "category" : i[1],
            "description" : i[2] if i[2] else "",
            "price" : i[3] if i[3] else 0.0,
            "is_organic" : bool(i[4])
        }
        result_list.append(result_dict)
    conn.close()

    return result_list

@tool
def fetch_product_rating_tool(id):
    """
    Use this tool to retrieve the average customer rating and total review count for a specific product based on its product ID.
    The tool returns a dictionary containing the product ID, average rating (rounded to two decimal places), and the number of reviews.
    If the product has no reviews, the average rating is returned as 0.0 and the review count as 0.
    """
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


@tool
def checkout_tool(id):
    """
    Places an order for a product using its unique product ID.

    Use this tool when the user explicitly confirms that they want to buy,
    purchase, order, or check out a specific product. The input must be the
    product's ID, not its name. This tool creates a new order record with the
    current timestamp and returns a confirmation message if the order is placed
    successfully. Do not use this tool to search for products or retrieve
    product information.
    """
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
        cursor.execute("insert into orders (id,product_id, product_name, price,ordered_at) values (?, ?, ?, ?, ?)",
                       (new_id, product_id, product_name, product_price, current_time))
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