# # NLP on Notes / Symptoms / Diagnosis


# """
# NLP module:
#   - Word clouds + top tokens
#   - N-gram analysis
#   - Symptom co-occurrence
#   - Sentiment (rule-based, works offline)
# """

# import os
# import re
# import warnings
# warnings.filterwarnings("ignore")

# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# from itertools import combinations

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.util import ngrams

# try:
#     from wordcloud import WordCloud
#     HAS_WC = True
# except ImportError:
#     HAS_WC = False

# CHART_DIR = "charts"
# REPORT_DIR = "reports"
# os.makedirs(CHART_DIR, exist_ok=True)
# os.makedirs(REPORT_DIR, exist_ok=True)

# STOP = set(stopwords.words("english")) | {
#     "patient", "pt", "nil", "na", "no", "yes", "not", "also"
# }
# LEM = WordNetLemmatizer()


# def _save(fig, name):
#     path = os.path.join(CHART_DIR, name)
#     fig.savefig(path, dpi=110, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  💾 {path}")


# def clean_text(t):
#     t = str(t).lower()
#     t = re.sub(r"[^a-z\s]", " ", t)
#     tokens = [LEM.lemmatize(w) for w in t.split()
#               if w not in STOP and len(w) > 2]
#     return tokens


# def top_tokens(tokens, n=25):
#     return Counter(tokens).most_common(n)


# def plot_wordcloud(tokens, name, color="viridis"):
#     if not HAS_WC:
#         print("  ℹ️  wordcloud not installed – skipping.")
#         return
#     text = " ".join(tokens)
#     if not text.strip():
#         return
#     wc = WordCloud(width=1200, height=600, background_color="white",
#                     colormap=color, max_words=120).generate(text)
#     fig, ax = plt.subplots(figsize=(13, 6))
#     ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
#     ax.set_title(f"Word Cloud – {name}")
#     plt.tight_layout()
#     _save(fig, f"nlp_wordcloud_{name}.png")


# def plot_top_tokens(tokens, name, n=20):
#     top = top_tokens(tokens, n)
#     if not top:
#         return
#     words, counts = zip(*top)
#     fig, ax = plt.subplots(figsize=(11, 6))
#     sns.barplot(x=list(counts), y=list(words), palette="mako", ax=ax)
#     ax.set_title(f"Top {n} tokens – {name}")
#     plt.tight_layout()
#     _save(fig, f"nlp_top_tokens_{name}.png")


# def plot_ngrams(tokens, name, n=2, k=15):
#     grams = list(ngrams(tokens, n))
#     if not grams:
#         return
#     counter = Counter(grams).most_common(k)
#     labels = [" ".join(g) for g, _ in counter]
#     counts = [c for _, c in counter]
#     fig, ax = plt.subplots(figsize=(11, 6))
#     sns.barplot(x=counts, y=labels, palette="flare", ax=ax)
#     ax.set_title(f"Top {k} {n}-grams – {name}")
#     plt.tight_layout()
#     _save(fig, f"nlp_{n}grams_{name}.png")


# def cooccurrence(tokens, name, top_k=12):
#     counts = Counter(tokens)
#     common = [w for w, _ in counts.most_common(top_k)]
#     pairs = Counter()
#     for a, b in combinations(sorted(set(tokens)), 2):
#         if a in common and b in common:
#             pairs[(a, b)] += 1
#     if not pairs:
#         return
#     df = pd.DataFrame(
#         [(a, b, c) for (a, b), c in pairs.most_common(30)],
#         columns=["A", "B", "CoOccurs"])
#     pivot = df.pivot_table(index="A", columns="B", values="CoOccurs", fill_value=0)
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(pivot, cmap="Reds", annot=False, ax=ax)
#     ax.set_title(f"Co-occurrence – {name}")
#     plt.tight_layout()
#     _save(fig, f"nlp_cooccurrence_{name}.png")


# def analyze_field(df, field, label):
#     if field not in df.columns:
#         return
#     s = df[field].dropna().astype(str)
#     if s.empty:
#         return
#     print(f"\n  🔎 Analyzing '{field}' ({len(s)} records)")
#     tokens = []
#     for t in s:
#         tokens.extend(clean_text(t))
#     if not tokens:
#         return
#     plot_wordcloud(tokens, label)
#     plot_top_tokens(tokens, label)
#     plot_ngrams(tokens, label, n=2)
#     plot_ngrams(tokens, label, n=3)

#     # Save top tokens CSV
#     top = pd.DataFrame(top_tokens(tokens, 100), columns=["token", "count"])
#     top.to_csv(os.path.join(REPORT_DIR, f"nlp_top_tokens_{label}.csv"), index=False)


# def run(consultations, nursing, lab_tests):
#     print("\n=== NLP MODULE ===")
#     analyze_field(consultations, "Symptoms", "symptoms")
#     analyze_field(consultations, "Diagnosis", "diagnosis")
#     analyze_field(consultations, "Treatment Plan", "treatment")
#     analyze_field(nursing, "Notes", "nursing_notes")
#     analyze_field(lab_tests, "Other Tests", "lab_other")

#     # Co-occurrence across symptoms + diagnosis
#     if not consultations.empty:
#         blob = " ".join(
#             consultations.get("Symptoms", pd.Series(dtype=str)).dropna().astype(str).tolist() +
#             consultations.get("Diagnosis", pd.Series(dtype=str)).dropna().astype(str).tolist()
#         )
#         tokens = clean_text(blob)
#         cooccurrence(tokens, "symptoms_diagnosis", top_k=15)


# if __name__ == "__main__":
#     import main_analysis as analysis
#     d = analysis.clean_all(analysis.load_data())
#     run(d["consultations"], d["nursing"], d["lab_tests"])
    
"""
NLP analysis on clinical notes/symptoms/diagnosis:
  - Word clouds + top tokens
  - N-gram analysis (bi/tri-grams)
  - Symptom co-occurrence heatmap
  - Top diagnosis text patterns
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations

CHART_DIR = "charts"
REPORT_DIR = "reports"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --- NLTK (with graceful fallback) ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.util import ngrams

    # Download required corpora on first run
    for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    STOP = set(stopwords.words("english")) | {
        "patient", "pt", "nil", "na", "no", "yes", "not", "also", "may", "one"
    }
    LEM = WordNetLemmatizer()
    HAS_NLTK = True
except Exception as e:
    print(f"⚠️  NLTK unavailable: {e}")
    HAS_NLTK = False
    STOP = set()
    def LEM(x): return x

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False
    print("ℹ️  wordcloud not installed — run: pip install wordcloud")


# ============================================================
# Helpers
# ============================================================
def _save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 {path}")


def clean_text(t):
    if pd.isna(t):
        return []
    t = str(t).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = [LEM.lemmatize(w) for w in t.split()
              if w not in STOP and len(w) > 2]
    return tokens


def top_tokens(tokens, n=25):
    return Counter(tokens).most_common(n)


# ============================================================
# Plotting
# ============================================================
def plot_wordcloud(tokens, name, color="viridis"):
    if not HAS_WC or not tokens:
        return
    text = " ".join(tokens)
    if not text.strip():
        return
    try:
        wc = WordCloud(width=1200, height=600, background_color="white",
                       colormap=color, max_words=120).generate(text)
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {name}", fontsize=16,
                     fontweight="bold", color="#0b4a6f")
        plt.tight_layout()
        _save(fig, f"nlp_wordcloud_{name}.png")
    except Exception as e:
        print(f"  ⚠️  Wordcloud failed: {e}")


def plot_top_tokens(tokens, name, n=20):
    top = top_tokens(tokens, n)
    if not top:
        return
    words, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(x=list(counts), y=list(words), palette="mako", ax=ax)
    ax.set_title(f"Top {n} Tokens — {name}", fontsize=15,
                 fontweight="bold", color="#0b4a6f")
    plt.tight_layout()
    _save(fig, f"nlp_top_tokens_{name}.png")


def plot_ngrams(tokens, name, n=2, k=15):
    if len(tokens) < n:
        return
    grams = list(ngrams(tokens, n))
    if not grams:
        return
    counter = Counter(grams).most_common(k)
    labels = [" ".join(g) for g, _ in counter]
    counts = [c for _, c in counter]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(x=counts, y=labels, palette="flare", ax=ax)
    ax.set_title(f"Top {k} {n}-grams — {name}", fontsize=15,
                 fontweight="bold", color="#0b4a6f")
    plt.tight_layout()
    _save(fig, f"nlp_{n}grams_{name}.png")


def cooccurrence(tokens, name, top_k=12):
    if not tokens:
        return
    counts = Counter(tokens)
    common = [w for w, _ in counts.most_common(top_k)]
    pairs = Counter()
    for a, b in combinations(sorted(set(tokens)), 2):
        if a in common and b in common:
            pairs[(a, b)] += 1
    if not pairs:
        return
    df = pd.DataFrame(
        [(a, b, c) for (a, b), c in pairs.most_common(30)],
        columns=["A", "B", "CoOccurs"]
    )
    pivot = df.pivot_table(index="A", columns="B", values="CoOccurs",
                           fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="Reds", annot=False, ax=ax)
    ax.set_title(f"Token Co-occurrence — {name}", fontsize=15,
                 fontweight="bold", color="#0b4a6f")
    plt.tight_layout()
    _save(fig, f"nlp_cooccurrence_{name}.png")


# ============================================================
# Field analyzer
# ============================================================
def analyze_field(df, field, label):
    if df is None or df.empty or field not in df.columns:
        print(f"  ℹ️  Skipping '{field}' (column missing or empty).")
        return
    s = df[field].dropna().astype(str)
    s = s[s.str.strip().ne("")]
    if s.empty:
        print(f"  ℹ️  No data in '{field}'.")
        return

    print(f"\n  🔎 Analyzing '{field}' ({len(s)} records)")
    tokens = []
    for t in s:
        tokens.extend(clean_text(t))

    if not tokens:
        print(f"  ⚠️  No tokens extracted from '{field}'.")
        return

    print(f"     → {len(tokens)} tokens, {len(set(tokens))} unique")
    plot_wordcloud(tokens, label)
    plot_top_tokens(tokens, label)
    plot_ngrams(tokens, label, n=2)
    plot_ngrams(tokens, label, n=3)

    # Save top tokens CSV
    top = pd.DataFrame(top_tokens(tokens, 100), columns=["token", "count"])
    top.to_csv(os.path.join(REPORT_DIR, f"nlp_top_tokens_{label}.csv"), index=False)


# ============================================================
# Main
# ============================================================
def run(consultations, nursing, lab_tests):
    print("\n=== NLP MODULE ===")

    analyze_field(consultations, "Symptoms", "symptoms")
    analyze_field(consultations, "Diagnosis", "diagnosis")
    analyze_field(consultations, "Treatment Plan", "treatment")
    analyze_field(nursing, "Notes", "nursing_notes")
    analyze_field(lab_tests, "Other Tests", "lab_other")

    # Co-occurrence across symptoms + diagnosis
    if consultations is not None and not consultations.empty:
        blob_parts = []
        for col in ["Symptoms", "Diagnosis"]:
            if col in consultations.columns:
                blob_parts.extend(
                    consultations[col].dropna().astype(str).tolist()
                )
        if blob_parts:
            tokens = []
            for t in blob_parts:
                tokens.extend(clean_text(t))
            if tokens:
                cooccurrence(tokens, "symptoms_diagnosis", top_k=15)

    print("\n✅ NLP module complete.")


if __name__ == "__main__":
    import main_analysis as analysis
    d = analysis.clean_all(analysis.load_data())
    run(d["consultations"], d["nursing"], d["lab_tests"])