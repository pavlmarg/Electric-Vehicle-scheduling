# Οδηγίες Εκτέλεσης — EV Fleet Simulator

Αυτός ο οδηγός περιγράφει βήμα προς βήμα πώς να στήσεις και να τρέξεις το σύστημα αφού λάβεις το αρχείο `ppo_model.zip` μέσω email.

---

## Απαιτήσεις Συστήματος

- **Python** 3.10 ή νεότερη
- **Node.js** 20 ή νεότερη (μαζί με npm)
- **Git** (προαιρετικό, αν κάνεις clone από repository)

---

## Βήμα 1 — Τοποθέτηση του Μοντέλου

Μετά τη λήψη του email, αποθήκευσε το αρχείο `ppo_model.zip` **στον φάκελο root του project**, δηλαδή εκεί που βρίσκονται τα `environments/`, `baselines/`, `server/` κ.λπ.

```
Electric-Vehicle-scheduling/
├── ppo_model.zip        ← εδώ
├── environments/
├── baselines/
├── server/
└── ev-visualizer/
```

> **Σημείωση:** Το αρχείο δεν χρειάζεται να αποσυμπιεστεί. Το Stable-Baselines3 το φορτώνει αυτόματα ως `.zip`.

---

## Βήμα 2 — Δημιουργία Virtual Environment (Python)

Άνοιξε τερματικό στον φάκελο root του project και εκτέλεσε:

```bash
python -m venv venv
```

Ενεργοποίησε το virtual environment:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## Βήμα 3 — Εγκατάσταση Python Εξαρτήσεων

```bash
pip install -r requirements.txt
```

Αν δεν υπάρχει ήδη `requirements.txt`, εγκατάστησε τα πακέτα χειροκίνητα:

```bash
pip install gymnasium stable-baselines3 numpy fastapi uvicorn pydantic torch
```

---

## Βήμα 4 — Εκτέλεση του Greedy Baseline (main_simulation.py)

Αυτό το script τρέχει τον ευρετικό αλγόριθμο και στα 10 προφίλ ζήτησης και εκτυπώνει τα αποτελέσματα στο τερματικό.

```bash
python baselines/main_simulation.py
```

**Αναμενόμενη έξοδος:**

```
Profile         | Profit (€)   | Service %  | Wait(m)
-------------------------------------------------------
Normal          |   ΧΧΧΧX.XX€ |     XX.X%  |    X.Xm
Commuter        |   ΧΧΧΧX.XX€ |     XX.X%  |    X.Xm
...
```

Η εκτέλεση διαρκεί συνήθως **1–2 λεπτά** (10 προφίλ × simulation 1 ημέρας).

---

## Βήμα 5 — Αξιολόγηση του PPO Μοντέλου (evaluate_model.py)

Αυτό το script φορτώνει το `ppo_model.zip` και αξιολογεί τον PPO agent επίσης στα 10 προφίλ.

```bash
python reinforcement_learning/evaluate_model.py
```

**Αναμενόμενη έξοδος:**

```
Profile         | Profit (€)   | Service %  | Wait(m) | Overrides
-----------------------------------------------------------------
Normal          |   ΧΧΧΧX.XX€ |     XX.X%  |    X.Xm |      XX
Commuter        |   ΧΧΧΧX.XX€ |     XX.X%  |    X.Xm |      XX
...
```

> **Σημείωση:** Αν το αρχείο `ppo_model.zip` δεν βρεθεί στον σωστό φάκελο, το script θα εμφανίσει μήνυμα σφάλματος. Βεβαιώσου ότι βρίσκεται στον φάκελο root (όχι μέσα στο `reinforcement_learning/`).

Η εκτέλεση διαρκεί συνήθως **3–8 λεπτά** ανάλογα με τον υπολογιστή σου.

---

## Βήμα 6 — Εκκίνηση του FastAPI Backend

Σε **νέο τερματικό** (με ενεργοποιημένο το venv), εκτέλεσε:

```bash
python server/api_server.py
```

Αν το μοντέλο φορτωθεί επιτυχώς, θα δεις:

```
✓ AI model loaded successfully.
INFO:     Started server process [...]
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Μπορείς να επαληθεύσεις ότι τρέχει ανοίγοντας στον browser:

```
http://127.0.0.1:8000/health
```

Αναμενόμενη απάντηση:
```json
{"status": "ok", "ai_model_loaded": true}
```

> **Κράτα αυτό το τερματικό ανοιχτό** — το backend πρέπει να τρέχει για να δουλεύει το frontend.

---

## Βήμα 7 — Εκκίνηση του React Frontend (Visualizer)

Σε **άλλο νέο τερματικό**, μετάβηκε στον φάκελο `ev-visualizer/`:

```bash
cd ev-visualizer
```

Εγκατάστησε τις εξαρτήσεις Node.js (μόνο την πρώτη φορά):

```bash
npm install
```

Εκκίνησε τον development server:

```bash
npm run dev
```

Άνοιξε τον browser στη διεύθυνση:

```
http://localhost:5173
```

---

## Βήμα 8 — Χρήση του Visualizer

Μόλις ανοίξεις το `http://localhost:5173`, μπορείς να:

1. **Επιλέξεις αλγόριθμο** — `Greedy` (ευρετικός) ή `PPO AI` (νευρωνικό δίκτυο)
2. **Επιλέξεις προφίλ ζήτησης** (0–9) από το dropdown
3. **Ορίσεις seed** (προαιρετικό — αφήνοντας κενό επιλέγεται τυχαίο)
4. Πατήσεις **Run Simulation** και περιμένεις 10–30 δευτερόλεπτα
5. Χρησιμοποιήσεις τα **controls αναπαραγωγής** (play/pause, βέλη, ταχύτητα)
6. Κάνεις **zoom/pan** στον χάρτη με scroll και drag
7. Παρακολουθήσεις τα **live metrics** στο δεξί panel (SoC, ουρές, κατανομή καταστάσεων)

**Πλήκτρα:**

| Πλήκτρο | Λειτουργία |
|---|---|
| `Space` | Play / Pause |
| `→` / `←` | Επόμενο / Προηγούμενο frame |
| `Home` | Πρώτο frame |
| `End` | Τελευταίο frame |

---

## Σύνοψη — Σειρά Εκτέλεσης

```
[Τερματικό 1 — venv ενεργό]
  python baselines/main_simulation.py         # Βήμα 4
  python reinforcement_learning/evaluate_model.py  # Βήμα 5
  python server/api_server.py                 # Βήμα 6 (κρατάς ανοιχτό)

[Τερματικό 2]
  cd ev-visualizer
  npm install   (μόνο πρώτη φορά)
  npm run dev                                 # Βήμα 7 (κρατάς ανοιχτό)

[Browser]
  http://localhost:5173                       # Βήμα 8
```

---

## Συνηθισμένα Προβλήματα

| Πρόβλημα | Λύση |
|---|---|
| `ppo_model.zip not found` | Βεβαιώσου ότι βρίσκεται στον φάκελο root, **όχι** μέσα στο `reinforcement_learning/` |
| `ModuleNotFoundError` | Βεβαιώσου ότι το venv είναι ενεργοποιημένο και έχεις τρέξει `pip install` |
| `Connection refused` στο frontend | Βεβαιώσου ότι ο FastAPI server τρέχει στο τερματικό 1 |
| `ai_model_loaded: false` στο `/health` | Το μοντέλο δεν φορτώθηκε — έλεγξε το path του `ppo_model.zip` |
| `CORS error` στον browser | Βεβαιώσου ότι ο API server τρέχει στο `http://127.0.0.1:8000` |