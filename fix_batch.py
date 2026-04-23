import json

nb_path = "src/deployment/batch_predict.py"
with open(nb_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line 'logger.info("Feature matrix prepared.")' and add the guard after it
new_lines = []
for line in lines:
    new_lines.append(line)
    if 'logger.info("Feature matrix prepared.")' in line:
        # Insert the empty‑check block after this line
        new_lines.append("""
# ---------- 6a. Guard against empty input ----------
if feature_df.empty:
    logger.warning("No new statements provided this week. Saving empty output.")
    empty_out = pd.DataFrame(columns=['statement_id','text','leader_name','county','rank','date','source',
                                        'year','month','day_of_week','quarter','is_weekend','is_august','is_december',
                                        'text_length','word_count','contains_zoning','contains_coalition','contains_stronghold',
                                        'contains_nationwide','contains_ugatuzi','contains_ukabila','contains_mgombea','contains_urais',
                                        'len_x_weekend','words_x_zoning','leader_statement_count','leader_avg_text_length',
                                        'leader_zoning_keyword_pct','predicted_label_encoded','predicted_stance'])
    empty_out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Empty predictions saved. Exiting.")
    exit(0)
""")

with open(nb_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Guard added to batch_predict.py")
