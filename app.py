import psycopg2
import numpy as np
import pandas as pd
import faiss
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
import re


# === Config ===
DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': '123456789',
    'port': '5432'
}

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDINGS_FILE = 'embeddings.npy'
INDEX_FILE = 'faiss.index'
MATERIALS_FILE = 'materials.pkl'
CLUSTERS_FILE = 'clusters.pkl'
EMBEDDING_DIM = 384
NUM_CLUSTERS = 50

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def generate_material_summary():
    conn = connect_db()
    cur = conn.cursor()

    query = """
    SELECT 
        material_name,
        SUM(quantity) AS total_ordered_qty,
        MAX(quantity) AS stock_left,
        MAX(purchase_date) AS last_order_date,
        MAX(total_price) AS last_total_price
    FROM requests
    GROUP BY material_name
    ORDER BY total_ordered_qty DESC;
    """

    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("detailed_material_summary.csv", index=False)

    try:
        cur.execute("DELETE FROM materialsummary")
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO materialsummary (
                    material_name, total_ordered_qty,
                    stock_left, last_order_date, last_total_price
                ) VALUES (%s, %s, %s, %s, %s)
            """, tuple(row))
        conn.commit()
        print("✅ Summary table updated.")
    except Exception as e:
        print("❌ Failed to update materialsummary table:", e)

    cur.close()
    conn.close()
    return df

def clean_description(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_materials():
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute('SELECT MTL_DESC, PSL_UNIT, STORE_CODE, RATE, ITEM_DESCRIPTION FROM data')
        rows = cur.fetchall()
        materials = []
        for idx, row in enumerate(rows):
            materials.append({
                'id': str(idx),                          # Unique ID for FAISS search
                'name': str(row[0]),                     # MTL_DESC (used for display / request)
                'psl_unit': str(row[1]),                 # PSL_UNIT (optional)
                'store_code': str(row[2]),               # STORE_CODE
                'rate': str(row[3]) if row[3] is not None else '0',  # RATE
                'description': str(row[4]),              # ITEM_DESCRIPTION (used for vector search)
                'current_stock': 0                     # Placeholder stock value
            })
        cur.close()
        conn.close()
        return materials
    except Exception as e:
        print("❌ Failed to load materials:", e)
        return []

def prepare_index():
    print("🔍 Preparing FAISS index and clustering...")
    materials = load_materials()
    if not materials:
        print("❌ No materials found.")
        return

    # Step 1: Create composite and cleaned descriptions
    descriptions = [
        clean_description(
            f"{m['name']} {m['description']} {m['psl_unit']} {m['store_code']} rate {m['rate']}"
        )
        for m in materials
    ]

    # Step 2: Encode descriptions using the model
    vectors = model.encode(descriptions, show_progress_bar=True).astype("float32")

    # Step 3: Build and save FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)
    np.save(EMBEDDINGS_FILE, vectors)

    # Step 4: Apply clustering
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=1000, random_state=42)
    labels = kmeans.fit_predict(vectors)

    # Step 5: Assign cluster labels to materials
    for i, label in enumerate(labels):
        materials[i]['cluster'] = int(label)

    # Step 6: Save materials and cluster labels
    with open(MATERIALS_FILE, 'wb') as f:
        pickle.dump(materials, f)
    with open(CLUSTERS_FILE, 'wb') as f:
        pickle.dump(labels, f)

    print(f"✅ Indexed {len(materials)} materials with {NUM_CLUSTERS} clusters.")



def semantic_search(query, top_k=10000, threshold=1.2, cluster_id=None):
    try:
        # Step 1: Encode the query
        query_vec = model.encode([query])[0]  # Single embedding

        # Step 2: Load FAISS index and material data
        index = faiss.read_index(INDEX_FILE)
        with open(MATERIALS_FILE, 'rb') as f:
            materials = pickle.load(f)

        # Step 3: Perform vector search
        D, I = index.search(np.array([query_vec]), top_k)

        # Step 4: Filter results by distance threshold
        results = [materials[i] for i, d in zip(I[0], D[0]) if d < threshold]

        # Optional: Cluster filter
        if cluster_id is not None:
            results = [r for r in results if r.get('cluster') == cluster_id]

        return results

    except Exception as e:
        print("❌ Semantic search error:", e)
        return []


def save_request(material, quantity, store_code):
    try:
        quantity = int(quantity)
        rate = float(material['rate'])
        total_price = rate * quantity

        conn = connect_db()
        cur = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        notification = "N/A"

        cur.execute("""
            INSERT INTO requests (material_name, quantity, store_code, purchase_date, rate, notification, total_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            material['name'],
            quantity,
            store_code,
            timestamp,
            rate,
            notification,
            total_price
        ))

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Request saved.")
    except Exception as e:
        print("❌ Failed to save request:", e)

def add_material_to_database(material_dict):
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO data (MTL_CODE, MTL_DESC, PSL_UNIT, STORE_CO, RATE)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            material_dict['MTL_CODE'], material_dict['MTL_DESC'],
            material_dict['PSL_UNIT'], material_dict['STORE_CO'],
            material_dict['RATE']
        ))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ New material added to database.")
        prepare_index()
    except Exception as e:
        print("❌ Failed to add material:", e)

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare_index()
        return

    print("=== 🧠 Semantic Material Assistant ===")
    while True:
        print("\nOptions:")
        print("1. Semantic search")
        print("2. Prepare Index")
        print("3. Generate Summary")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == '1':
            query = input("Enter material-related query: ")
            results = semantic_search(query, top_k=10)
            if not results:
                print("No results found.")
                continue

            for i, r in enumerate(results, 1):
                print(f"{i}. {r['name']} | Stock: {r['current_stock']} | Store: {r['store_code']} | Rate: {r['rate']} | Cluster: {r.get('cluster', -1)}")

            pick = input("Enter material number to request (or 'q' to skip): ")
            if pick.isdigit():
                index = int(pick) - 1
                if 0 <= index < len(results):
                    qty = input("Enter quantity: ")
                    store = input("Enter your store code: ")
                    # Construct expected material dict structure for save_request()
                    material = {
                        "name": results[index]['name'],
                        "rate": results[index]['rate']
                    }
                    save_request(material, qty, store)

        elif choice == '2':
            prepare_index()

        elif choice == '3':
            df = generate_material_summary()
            print(df.head(10))

        elif choice == '4':
            print("👋 Exiting...")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()