import json

nb_path = "notebooks/01_eda_odm_zoning.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix the problematic f-string by changing outer quotes to double, removing backslashes
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "Per Section 2.4, above 10?" in source:
            source = source.replace(
                "print(f'Imbalance ratio (max/min): {imbalance_ratio:.2f}. Per Section 2.4, above 10? {\\'Yes\\' if imbalance_ratio > 10 else \\'No\\'}')",
                'print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}. Per Section 2.4, above 10? {\'Yes\' if imbalance_ratio > 10 else \'No\'}")'
            )
            cell["source"] = source.splitlines(True)
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Fixed syntax error in notebook.")
