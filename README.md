## Αναγνώριση Οντοτήτων σε Κείμενα Χρησιμοποιώντας Μεγάλα Γλωσσικά Μοντέλα (LLMS)

## Περιγραφή

Αυτό το script συγκρίνει την απόδοση διαφορετικών μοντέλων Επεξεργασίας Φυσικής Γλώσσας (NLP) στην αναγνώριση οντοτήτων (Named Entity Recognition - NER) σε διάφορα γνωστά datasets:

- CoNLL2003
- Financial Phrasebank
- WNUT17
- NCBI Disease

Χρησιμοποιούνται τόσο κλασικά εργαλεία NLP όπως spaCy και HuggingFace Pipelines όσο και μεγάλα γλωσσικά μοντέλα (LLMs) μέσω του Langchain και του Ollama.

---

Απαιτήσεις

- Python 3.8 ή νεότερη
- Εγκατάσταση βιβλιοθηκών:

```bash
pip install pandas datasets scikit-learn spacy transformers tqdm matplotlib langchain_ollama

- Για τη χρήση LLMs (π.χ. llama3, mistral, gemma2), απαιτείται Ollama εγκατεστημένο και σε λειτουργία τοπικά.

Τρόπος Χρήσης

1.Εγκαταστήστε τις απαιτούμενες βιβλιοθήκες και τα μοντέλα (π.χ. en_core_web_trf του spaCy).
2.Εκτελέστε το script:
python script.py
3.Το script:
- φορτώνει τα datasets,
- εφαρμόζει 4 μοντέλα NER (spacy, llama3, gemma2:latest, mistral),
- αξιολογεί την απόδοση (precision, recall, F1),
- δημιουργεί διαγράμματα,
- αποθηκεύει αποτελέσματα στον φάκελο results/.

Παραγόμενα Αρχεία

-results/{dataset}_metrics.csv: Αποτελέσματα αξιολόγησης

-results/{dataset}_comparison_plot.png: Γραφήματα σύγκρισης μοντέλων

Υποστηριζόμενα Μοντέλα

| Όνομα     | Περιγραφή                                          |
| --------- | -------------------------------------------------- |
| spacy   | en_core_web_trf (transformer-based NER μοντέλο)      |
| gemma2  | LLM μέσω Ollama                                      |
| mistral | LLM μέσω Ollama                                      |
| llama3  | LLM μέσω Ollama                                      |

Αξιολόγηση

Το script υπολογίζει για κάθε μοντέλο και dataset:

-Precision
-Recall
-F1-score
-Μέσο χρόνο επεξεργασίας ανά δείγμα

Παραμετροποίηση

Μπορείτε να αλλάξετε τον αριθμό δειγμάτων ανά dataset (default: 300) από τη συνάρτηση:

process_dataset(dataset_name, num_samples=100)
