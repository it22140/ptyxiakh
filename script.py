import os
import random
import time
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from langchain_ollama import OllamaLLM
import matplotlib.pyplot as plt
import re
import spacy
from transformers import pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Χαρτες μετατροπής ---
conll2003_map = {
    'PERSON': 'B-PER', 'ORG': 'B-ORG', 'LOC': 'B-LOC', 'GPE': 'B-LOC',
    'DATE': 'B-MISC', 'FAC': 'B-MISC', 'MONEY': 'B-MISC',
    'PERCENT': 'B-MISC', 'CARDINAL': 'B-MISC', 'PRODUCT': 'B-ORG', 'GOVORG': 'B-ORG'
}

financial_phrasebank_map = {
    'SENTIMENT_POSITIVE': 'B-SENT', 'SENTIMENT_NEGATIVE': 'B-SENT',
    'SENTIMENT_NEUTRAL': 'B-SENT', 'O': 'O'
}

wnut17_map = {
    'B-ENT': 'B-ORG', 'I-ENT': 'I-ORG', 'O': 'O'
}

ncbi_disease_map = {
    'B-DISEASE': 'B-MISC', 'I-DISEASE': 'I-MISC', 'O': 'O'
}


def get_label_map(dataset_name):
    label_maps = {
        "conll2003": conll2003_map,
        "financial_phrasebank": financial_phrasebank_map,
        "wnut17": wnut17_map,
        "ncbi_disease": ncbi_disease_map
    }
    return label_maps.get(dataset_name, {'O': 'O'})



def parse_llm_response(response_text, tokens):
    entity_lines = re.findall(r'[-\*\u2022]?\s*(.+?)\s*\(([^)]+)\)', response_text)

    if not entity_lines:
        print(" Δεν βρέθηκαν entities στο LLM output.")
        return [(token, 'O') for token in tokens]

    label_map = {
        'GPE': 'LOC', 'PERSON': 'PER', 'ORGANIZATION': 'ORG',
        'DATE': 'MISC', 'TIME': 'MISC', 'MONEY': 'MISC',
        'PERCENT': 'MISC', 'FACILITY': 'MISC', 'NORP': 'MISC',
        'LANGUAGE': 'MISC', 'LOCATION': 'LOC', 'PRODUCT': 'ORG',
        'GOVORG': 'ORG'
    }

    predictions = [(token, 'O') for token in tokens]

    for phrase, ent_type in entity_lines:
        ent_type = label_map.get(ent_type.upper(), 'O')
        phrase_tokens = phrase.strip().split()

        for i in range(len(tokens) - len(phrase_tokens) + 1):
            window = tokens[i:i+len(phrase_tokens)]
            if [w.lower() for w in window] == [w.lower() for w in phrase_tokens]:
                for j in range(len(phrase_tokens)):
                    predictions[i+j] = (tokens[i+j], f"B-{ent_type}")
    return predictions



def call_model(model_name, tokens):
    print(f"Κλήση του μοντέλου: {model_name}")
    text = " ".join(tokens)

    if model_name == "spacy":
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(text)
        predictions = []
        for token in doc:
            entity = token.ent_type_ if token.ent_type_ else 'O'
            entity = conll2003_map.get(entity, 'O')
            predictions.append((token.text, entity))
        return predictions

    elif model_name in ["gemma2:latest", "mistral"]:
        model_pipeline = pipeline("ner", model=("dslim/bert-base-NER" if model_name == "gemma2:latest" else "Jean-Baptiste/roberta-large-ner-english"), aggregation_strategy="simple")
        entities = model_pipeline(text)
        predictions = [(token, 'O') for token in tokens]

        for entity in entities:
            for i, token in enumerate(tokens):
                if entity['word'].lower() in token.lower():
                    predictions[i] = (token, f"B-{entity['entity_group']}")
        return predictions

    else:
        llm = OllamaLLM(model=model_name)
        prompt = f"""Extract named entities from the following text.
Return results as bullet points in the format: * Entity (TYPE)

Text:
{text}

Entities:"""
        try:
            response = llm.invoke(prompt)
            print(f" LLM Response:\n{response}")
            return parse_llm_response(response, tokens)
        except Exception as e:
            print(f"Σφάλμα LLM parsing: {e}")
            return [(token, 'O') for token in tokens]


# --- Datasets ---
def load_custom_dataset(dataset_name):
    print(f" Φόρτωση dataset: {dataset_name}")
    dataset_map = {
        "conll2003": "eriktks/conll2003",
        "financial_phrasebank": ("financial_phrasebank", "sentences_allagree"),
        "wnut17": "wnut_17",
        "ncbi_disease": "ncbi_disease"
    }

    ds_info = dataset_map.get(dataset_name)
    if isinstance(ds_info, tuple):
        return load_dataset(ds_info[0], ds_info[1], trust_remote_code=True)
    else:
        return load_dataset(ds_info, trust_remote_code=True)


def normalize_label(label, valid_labels):
    return label if label in valid_labels else 'O'


def precision_recall_f1(true_labels, pred_labels, label_map):
    labels = list(set(label_map.values())) + ['O']
    pred_labels_norm = [normalize_label(l, labels) for l in pred_labels]
    true_int = [labels.index(l) for l in true_labels]
    pred_int = [labels.index(l) for l in pred_labels_norm]

    precision, recall, f1, _ = precision_recall_fscore_support(true_int, pred_int, average='macro', zero_division=0)
    return precision, recall, f1


def save_results(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    for dataset, metrics in results.items():
        pd.DataFrame(metrics).T.to_csv(f"{output_dir}/{dataset}_metrics.csv")
        print(f" Αποθηκεύτηκαν: {output_dir}/{dataset}_metrics.csv")


def plot_comparison(results):
    print("Δημιουργία γραφημάτων σύγκρισης...")
    os.makedirs("results", exist_ok=True)
    for dataset_name, metrics in results.items():
        df = pd.DataFrame(metrics).T
        ax = df[['precision', 'recall', 'f1']].plot(kind='bar', figsize=(12, 6), title=f"Model Comparison on {dataset_name}")
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()

        # Αποθήκευση του γραφήματος
        plot_filename = f"results/{dataset_name}_comparison_plot.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"✅ Αποθηκεύτηκε στο: {plot_filename}")


# --- Κύρια Επεξεργασία Dataset ---
def process_dataset(dataset_name, num_samples=100):
    dataset = load_custom_dataset(dataset_name)
    label_map = get_label_map(dataset_name)
    sample_indices = random.sample(range(len(dataset['train'])), num_samples)
    models = ["spacy", "llama3", "gemma2:latest", "mistral"]
    model_results = {}

    for model_name in tqdm(models, desc=f"Models on {dataset_name}"):
        total_p = total_r = total_f1 = 0
        start = time.time()

        for idx in tqdm(sample_indices, desc=f"{model_name} Samples", leave=False):
            sample = dataset['train'][idx]
            tokens = sample.get('tokens') or sample.get('words') or sample.get('sentence').split()
            true_ner = sample.get('ner_tags') or [0]*len(tokens)
            true_labels = [label_map.get(tag, 'O') for tag in true_ner]

            predicted = call_model(model_name, tokens)
            pred_labels = [l for _, l in predicted]

            if len(pred_labels) != len(tokens):
                print(f" Length mismatch! {len(tokens)} tokens vs {len(pred_labels)} preds.")
                if len(pred_labels) > len(tokens):
                    pred_labels = pred_labels[:len(tokens)]
                else:
                    pred_labels += ['O'] * (len(tokens) - len(pred_labels))

            p, r, f1 = precision_recall_f1(true_labels, pred_labels, label_map)
            total_p += p
            total_r += r
            total_f1 += f1

        elapsed = time.time() - start
        avg_time = elapsed / len(sample_indices)

        model_results[model_name] = {
            'precision': total_p/len(sample_indices),
            'recall': total_r/len(sample_indices),
            'f1': total_f1/len(sample_indices),
            'time_per_sample': avg_time
        }

    print(f"\n===== ΟΛΟΚΛΗΡΩΣΗ Dataset: {dataset_name} =====")
    for model, metrics in model_results.items():
        print(f"Model: {model}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Μέσος Χρόνος ανά δείγμα: {metrics['time_per_sample']:.4f} sec\n")
        print("="*40)

    return model_results


# --- Main ---
def main():
    datasets = ["financial_phrasebank", "conll2003", "wnut17", "ncbi_disease"]
    results = {}

    for ds in datasets:
        res = process_dataset(ds)
        results[ds] = res

    plot_comparison(results)
    save_results(results)

if __name__ == "__main__":
    main()
