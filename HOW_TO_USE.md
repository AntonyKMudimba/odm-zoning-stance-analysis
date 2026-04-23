# HOW TO USE THE ODM ZONING STANCE MODEL

> This guide is for anyone. You do NOT need to know Python, coding, or anything technical.

---

## What this tool does

It reads statements from ODM leaders and tells you whether they **Support**, **Oppose**, or are **Neutral** on zoning.

You give it a list of statements (in an Excel/CSV file), and it gives you back the same list with a **predicted stance** and a **confidence score**.

---

## What you need before you start

1. A Windows laptop (the same one where the project was set up).
2. Two shortcuts on your desktop that the tech person already created:
   - **ODM Stance Predictor** (runs the model)
   - **Open This Week's Results** (opens the results file)

---

## Weekly routine (3 steps - 5 minutes total)

### Step 1 - Prepare the statements

1. Open the **statements_template.csv** file.
2. Fill in one row for each new statement.
3. Save the file exactly as new_statements.csv in the data/raw folder.

### Step 2 - Run the model

1. Double-click the **ODM Stance Predictor** shortcut.
2. A black window appears and then closes.

### Step 3 - Open the results

1. Double-click the **Open This Week's Results** shortcut.
2. Excel opens weekly_stance_predictions.csv.
3. Look at the columns: leader_name, text, predicted_stance, and the three probability columns.

---

## Example

You enter:

| leader_name | text |
|-------------|------|
| Raila Odinga | We must protect our strongholds with zoning. |
| James Orengo | Zoning is not our tradition. Let the people decide. |
| Gladys Wanga | The party is still consulting. |

The results show:

| leader_name | predicted_stance | probability_Support |
|-------------|------------------|---------------------|
| Raila Odinga | Support | 0.94 |
| James Orengo | Oppose | 0.95 |
| Gladys Wanga | Neutral | 0.80 |

---

## Important note

The model must be trained on real labelled data before the predictions become trustworthy.

---

## What else can this model do? (10+ other uses)

The same engine can be used for any topic where you have labelled data:

1. Coalition partner sentiment
2. Policy support
3. Internal party faction detection
4. Corruption allegation monitoring
5. Youth agenda support
6. Health policy stance
7. Election preparedness
8. Media sentiment analysis
9. Constituency needs classification
10. Fraud risk in procurement
11. Donor report summarisation
12. WhatsApp group sentiment

**How to use for any of these:** Just collect labelled data for the new topic, retrain with the Socrato scripts, and start predicting. The whole pipeline remains identical; only the data changes.

---

**Socrato | The Bigger Picture**
