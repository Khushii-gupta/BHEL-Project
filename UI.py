import pandas as pd
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import messagebox, simpledialog, ttk
import psycopg2
import pickle
import faiss
import numpy as np
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Frame as TkFrame
import app


# === Config ===
DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': '123456789',
    'port': '5432'
}

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
INDEX_FILE = 'faiss.index'
MATERIALS_FILE = 'materials.pkl'
EMBEDDINGS_FILE = 'embeddings.npy'


model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

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


def save_request(material_name, quantity, store_code, rate):
    try:
        total_price = float(rate) * int(quantity)
        conn = connect_db()
        cur = conn.cursor()
        purchase_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        notification = "N/A"

        cur.execute("""
            INSERT INTO requests (material_name, quantity, store_code, purchase_date, rate, notification, total_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (material_name, quantity, store_code, purchase_date, rate, notification, total_price))

        conn.commit()
        cur.close()
        conn.close()
        messagebox.showinfo("Success", f"{quantity} units of '{material_name}' requested from store {store_code}.")
    except Exception as e:
        messagebox.showerror("Request Error", f"{e}")

def show_summary():
    df = app.generate_material_summary()
    if df.empty:
        messagebox.showinfo("Summary", "No summary data available.")
        return

    def show_table():
        for widget in summary_window.winfo_children():
            widget.destroy()
        tree = ttk.Treeview(summary_window, columns=list(df.columns), show="headings")
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(fill="both", expand=True)

    def show_graph():
        for widget in summary_window.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(10, 5))
        df_sorted = df.sort_values(by='total_ordered_qty', ascending=False).head(10)
        ax.bar(df_sorted['material_name'], df_sorted['total_ordered_qty'], color='skyblue')
        ax.set_xlabel('Material')
        ax.set_ylabel('Total Ordered Quantity')
        ax.set_title('Top 10 Materials by Ordered Quantity')
        ax.tick_params(axis='x', rotation=45)
        canvas = FigureCanvasTkAgg(fig, master=summary_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    summary_window = tb.Toplevel()
    summary_window.title("Material Summary Report")
    summary_window.geometry("1000x600")

    view_option = ttk.Combobox(summary_window, values=["Table View", "Graph View"])
    view_option.set("Table View")
    view_option.pack(pady=10)

    def update_view(event):
        if view_option.get() == "Table View":
            show_table()
        else:
            show_graph()

    view_option.bind("<<ComboboxSelected>>", update_view)
    show_table()

def add_material_window():
    form = tb.Toplevel()
    form.title("Add New Material")
    form.geometry("600x600")

    fields = {
        "MTL_CODE": "",
        "MTL_DESC": "",
        "PSL_UNIT": "",
        "STORE_CO": "",
        "RATE": ""
    }
    entries = {}

    for i, (label, _) in enumerate(fields.items()):
        tb.Label(form, text=label, font=("Segoe UI", 10)).pack(pady=5)
        entry = tb.Entry(form)
        entry.pack(fill="x", padx=20)
        entries[label] = entry

    def save():
        values = {k: v.get().strip() for k, v in entries.items()}
        if any(v == "" for v in values.values()):
            messagebox.showerror("Error", "All fields are required.")
            return
        try:
            app.add_material_to_database(values)
            messagebox.showinfo("Success", "Material added and index refreshed successfully.")
            form.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add material:\n{e}")

    tb.Button(form, text="Save Material", bootstyle="success", command=save).pack(pady=10)

def run_gui():
    from tkinter import Frame as TkFrame
    global result_msg_label

    root = tb.Window(themename="flatly")
    root.title("Smart Material Assistant")
    root.geometry("1440x850")

    # === Header ===
    header = tb.Label(
        root,
        text="📦 SMART MATERIAL SEARCH DASHBOARD",
        font=("Segoe UI", 24, "bold"),
        foreground="#003366",
        background="#f5f5f5",
        anchor="center",
        padding=20
    )
    header.pack(fill="x")

    # === Main Split Frame ===
    main_frame = tb.Frame(root)
    main_frame.pack(fill="both", expand=True)

    # === Left Panel (30%) ===
    left_panel = tb.Frame(main_frame, width=400, bootstyle="light")
    left_panel.pack(side="left", fill="y")
    left_panel.pack_propagate(False)

    # === Right Panel (70%) ===
    right_panel = tb.Frame(main_frame)
    right_panel.pack(side="right", fill="both", expand=True)

    # === Search Bar and Buttons on Left ===
    tb.Label(left_panel, text="Material Controls", font=("Segoe UI", 12, "bold"),
            background="#e1e1e1", anchor="w", padding=10).pack(fill="x", pady=(20, 10))

    tb.Label(left_panel, text="Search Query:", font=("Segoe UI", 11)).pack(padx=15, anchor="w")

    canvas_frame = TkFrame(left_panel, background="#dddddd", borderwidth=1, relief="solid")
    canvas_frame.pack(padx=15, pady=(0, 10), fill="x")
    query_entry = tb.Entry(canvas_frame, font=("Segoe UI", 10))
    query_entry.pack(ipady=5, padx=5, pady=2, fill="x")

    tb.Button(left_panel, text="Search", bootstyle="success", command=lambda: perform_search()).pack(padx=15, pady=5, fill="x")
    tb.Button(left_panel, text="Request Selected", bootstyle="warning", command=lambda: on_request()).pack(padx=15, pady=5, fill="x")
    tb.Button(left_panel, text="Add Material", bootstyle="primary", command=add_material_window).pack(padx=15, pady=5, fill="x")
    tb.Button(left_panel, text="Summary Table View", bootstyle="info", command=show_summary).pack(padx=15, pady=5, fill="x")

    result_msg_label = tb.Label(left_panel, text="", font=("Segoe UI", 9), padding=(10, 5))
    result_msg_label.pack(padx=15, pady=10, anchor="w")

    stock_status_label = tb.Label(left_panel, text="", font=("Segoe UI", 10), wraplength=350, justify="left")
    stock_status_label.pack(padx=15, anchor="w")

    tb.Button(left_panel, text="❌ Exit", bootstyle="danger", command=root.quit).pack(side="bottom", pady=30, padx=15, fill="x")

    # === Results Table on Right Panel ===
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"), background="#336699", foreground="white")
    style.configure("Treeview", font=("Segoe UI", 10), rowheight=32, background="white", foreground="black", fieldbackground="white")

    columns = ("S.No", "Name", "Stock", "Store", "Rate", "Cluster")
    results_tree = tb.Treeview(right_panel, columns=columns, show="headings", style="Treeview")
    for col in columns:
        results_tree.heading(col, text=col)
        results_tree.column(col, anchor="center", width=140 if col != "Name" else 260)
    results_tree.pack(fill="both", expand=True, padx=10, pady=10)

    scrollbar = tb.Scrollbar(results_tree, orient="vertical", command=results_tree.yview)
    results_tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    # === Functions ===
    def perform_search(event=None):
        query = query_entry.get().strip()
        results_tree.delete(*results_tree.get_children())
        stock_status_label.config(text="")  # Clear old notification

        if not query:
            return

        results = semantic_search(query)

        if len(results) > 1000:
            result_msg_label.config(
                text=f"{len(results)} items found. Showing top 1000. Refine your query for more options.")
        else:
            result_msg_label.config(text=f"{len(results)} items found.")

        for i, r in enumerate(results[:1000], 1):
            stock = int(r['current_stock'])
            tag = "low" if stock < 10 else "normal"
            cluster_display = str(r.get('cluster', ''))

            results_tree.insert("", "end", values=(
                i,  # Serial number
                r['name'],
                stock,
                r['store_code'],
                r['rate'],
                cluster_display
            ), tags=(tag,))

        results_tree.tag_configure("low", background="#ffe5e5", font=("Segoe UI", 10, "bold"))
        results_tree.tag_configure("normal", background="white")

    def on_table_select(event):
        selected = results_tree.focus()
        if not selected:
            return
        values = results_tree.item(selected, 'values')
        stock = int(values[2])
        if stock < 10:
            stock_status_label.config(
                text=f"⚠️ Low Stock Alert!\nOnly {stock} items left.",
                foreground="#cc0000",
                font=("Segoe UI", 10, "bold")
            )
        else:
            stock_status_label.config(
                text=f"✅ {stock} items available in stock.",
                foreground="#006600",
                font=("Segoe UI", 10)
            )

    def on_request():
        selected = results_tree.focus()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a material first.")
            return

        values = results_tree.item(selected, 'values')
        if len(values) < 6:
            messagebox.showerror("Error", "Invalid selection format.")
            return

        sno, name, stock, store_code, rate, cluster = values

        store = simpledialog.askstring("Store Code", "Enter your store code:", parent=root)
        if not store:
            return

        qty = simpledialog.askstring("Quantity", f"Enter quantity for '{name}':", parent=root)
        if qty and qty.isdigit():
            save_request(name, qty, store, rate)
        else:
            messagebox.showerror("Invalid Input", "Please enter a valid quantity.")

    results_tree.bind("<<TreeviewSelect>>", on_table_select)
    query_entry.bind("<Return>", perform_search)
    root.mainloop()

try:
    run_gui()
except Exception as e:
    print("🔥 UI Error:", e)
