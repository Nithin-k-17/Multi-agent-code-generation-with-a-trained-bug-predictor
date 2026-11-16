"""Quick check that the trained model loads and scores code:  python main.py demo"""
from .predictor import BugPredictor

BUGGY = {
    "Division by zero / off-by-one":
        "def compute_average(items):\n    return sum(items) / len(items) + 1",
    "Mutable default argument & missing key":
        "def append_and_lookup(val, store={}, key='id'):\n    store['items'].append(val)\n    return store[key]",
    "Wrong variable swap":
        "def swap_and_min(a, b):\n    min_val = min(a, b)\n    a = b\n    b = b\n    return a, b, min_val",
}
CLEAN = {
    "Safe average with guard":
        "def compute_average(items):\n    if not items:\n        return 0.0\n    return float(sum(items)) / len(items)",
    "Safe dictionary lookup":
        "def safe_lookup(mapping, key, default=None):\n    if not isinstance(mapping, dict):\n        return default\n"
        "    return mapping.get(key, default)",
    "Correct value swap":
        "def swap_values(a, b):\n    return b, a",
}
MULTI_FUNCTION = "def clean_helper(x):\n    return x * 2\n\ndef buggy_helper(xs):\n    return xs[100] / 0\n"


def main(device="cpu"):
    bp = BugPredictor(device=device)
    print("=" * 60)
    print(f"Bug predictor  device={bp.device}  threshold={bp.threshold:.2f}  max_len={bp.max_length}")
    print(f"Model: {bp.path}")
    print("=" * 60)
    for title, samples in (("BUGGY SNIPPETS", BUGGY), ("CLEAN SNIPPETS", CLEAN)):
        print(f"\n--- {title} ---")
        for name, code in samples.items():
            print(f"{name:42s} score={bp.score(code):.4f}  risky={bp.is_risky(code)}")
    print("\n--- PER-FUNCTION SCORES ---")
    for fn in bp.score_functions(MULTI_FUNCTION):
        print(f"{fn['name']:15s} lines {fn['line_start']}-{fn['line_end']}  score={fn['score']:.4f}  risky={fn['is_risky']}")


if __name__ == "__main__":
    main()
