import json

nb_path = "notebooks/01_eda_odm_zoning.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        # Join lines into one string, do replacements, split back
        source = "".join(cell["source"])
        source = source.replace(
            "data/raw/odm_statements_raw.csv",
            "../data/raw/odm_statements_raw.csv"
        ).replace(
            "reports/figures",
            "../reports/figures"
        )
        # Replace the cell source (as a list of lines again)
        cell["source"] = source.splitlines(True)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Paths updated in notebook.")
