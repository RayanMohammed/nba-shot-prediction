import warnings

# On Apple-chip Macs, numpy's math library raises false "divide by zero / overflow / invalid value
# encountered in matmul" warnings while scikit-learn fits models. The results are unaffected (checked
# Oct 2026: identical numbers to a Linux run), so hide only those specific warnings.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
