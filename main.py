import argparse
from src.classifier import LanguageClassifier

clf = LanguageClassifier()

parser = argparse.ArgumentParser()
parser.add_argument("text", nargs="?", help="Text to classify")
parser.add_argument("--metric", default="cosine", choices=["cosine","euclidean"])
parser.add_argument("--eval", action="store_true", help="Run full evaluation")
args = parser.parse_args()

if args.eval:
    from src.evaluator import evaluate, accuracy_vs_length
    evaluate(clf); accuracy_vs_length(clf)
elif args.text:
    scores = clf.classify(args.text, args.metric)
    print(f"Detected: {list(scores.keys())[0]}")
    for lang, s in list(scores.items())[:3]:
        print(f"  {lang}: {s:.4f}")
else:
    parser.print_help()