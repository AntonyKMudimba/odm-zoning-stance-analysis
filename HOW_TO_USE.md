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


---

## Frequently Asked Questions

### 1. After you train the model on quality data, how does it work on real, unseen statements?
The model learns from the labelled examples you gave it.  
When you feed it **new, unlabelled statements**, it reads the text, finds patterns it has seen before (e.g., “protect strongholds” usually means Support), and predicts the stance.  
**You do not need to label the new statements.** Just put them in the input file, run the predictor, and the model labels them for you.

### 2. What if I have too many real statements to work on?
You don’t need to label them all. The model processes hundreds of statements in seconds.  
All you do is **paste all the raw statements into the input CSV**, run the predictor, and get back a spreadsheet with predictions for every single row.  
You can then sort or filter the results (e.g., show only “Oppose” or only high‑confidence predictions) to find the most important signals.

### 3. What does the final output look like, and how does it help me make decisions?
The output is a simple Excel/CSV file with columns like:
- leader_name, 	ext, predicted_stance, and three probability columns.

**Example of a real decision:**  
30 new statements from the week show 18 Support, 7 Oppose, 5 Neutral.  
Support is strongest in Nyanza, while opposition comes mainly from the Coast.  
Neutral leaders from Homa Bay and Kisumu could be swayed.  
The NEC decides to propose **partial zoning** (only for certain regions) rather than a blanket demand – keeping both wings united.  
All this came from a single spreadsheet, no guessing.

### 4. What do the probabilities mean?
Each prediction has three numbers (e.g., probability_Support = 0.94, probability_Oppose = 0.02, probability_Neutral = 0.04).  
These add up to 1.0 and tell you how confident the model is.  
A 0.94 Support means “the model is 94% sure this statement supports zoning”.  
You can trust predictions with high confidence (>0.85) more than those around 0.5, which are borderline and should be checked by a human.

### 5. Can the model make mistakes?
Yes. No model is perfect.  
If it gives a wrong label, it means either the statement was phrased in a way the model hasn’t seen before, or the training data didn’t cover that style.  
That’s why the system tracks performance over time (using the monitoring script in Phase 9) and alerts you if the model starts drifting.  
Always treat the predictions as **a guide**; the final decision should involve human judgment.

### 6. What should I do if I think a prediction is wrong?
First, check the original statement.  
If you are sure it’s misclassified, you can add that statement to the training data with the correct label and retrain the model later.  
Over time, the model improves as you feed it more corrected examples.

### 7. How often should I retrain the model?
- In a fast‑changing political environment, retrain **monthly**.  
- In a stable period, retrain **quarterly**.  
The monitoring script will remind you when it’s time.  
If you notice that predictions are consistently off (e.g., many high‑confidence errors), retrain immediately.

### 8. Can I use this model for something other than zoning?
Yes! The same engine works for any text‑classification task.  
You only need to collect labelled data for the new topic (e.g., “policy support”, “corruption accusation”, “election readiness”) and retrain the model.  
The entire pipeline stays the same; only the data changes.  
See the section “What else can this model do?” above for 12 examples.

### 9. Is my data safe? Where is it stored?
All data stays on the laptop where you run the project. Nothing is sent to the internet.  
The model runs completely offline.  
If you add it to GitHub, be careful not to upload sensitive internal party data – our .gitignore already excludes data files by default.

### 10. What if the black window shows an error?
Take a screenshot or copy the text and send it to your technical support person.  
Common errors include:
- Forgetting to save the input file as 
ew_statements.csv in the correct folder.
- Missing columns in the template.
- Running out of disk space (very rare).

### 11. Can I use this on a phone?
Currently, the system requires a Windows laptop because of the one‑time setup.  
However, the **results** (the CSV file) can be opened on any phone with Excel or Google Sheets, so you can view the intelligence on the go.

### 12. Who built this, and can I get help?
This system was built by **Socrato**, using the same standards as McKinsey, Deloitte, and Google Brain, but adapted for the Kenyan context.  
For support or to request new features, contact the project maintainer.

---

**Socrato | The Bigger Picture**
