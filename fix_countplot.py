import json

nb_path = "notebooks/01_eda_odm_zoning.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "order=['Support', 'Oppose', 'Neutral', None]" in src:
            src = src.replace(
                "order=['Support', 'Oppose', 'Neutral', None]",
                "order=['Support', 'Oppose', 'Neutral']"
            )
            cell["source"] = src.splitlines(True)
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Fixed countplot order.")
