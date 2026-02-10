import numpy as np


def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Arrays must have the same length.")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    if denominator == 0:
        return 0  # To handle cases where the denominator might be zero

    return numerator / denominator


# Example usage
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

r = pearson_correlation(x, y)
print(f"Pearson Correlation Coefficient: {r}")
