import numpy as np
import matplotlib.pyplot as plt
import os


def plot_regression_bands(
        x: np.ndarray,
        y: np.ndarray,
        y_est: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        filename: str = None
):
    """
    Plot regression confidence band
    """

    n = len(x)
    n0 = len(y)

    if not (len(y_est) == n and len(lower_bound) == n and len(upper_bound) == n):
        raise ValueError("Arrays x, y_est, lower_bound, and upper_bound must have the same length n.")

    if n0 > n:
        raise ValueError("Length of y array must be less or equal than the length of other arrays.")

    plt.figure(figsize=(10, 6))

    plt.scatter(x[:n0], y, color="red", marker="+", label="y", zorder=5)
    plt.plot(x, y_est, color="black", linestyle="-", label="y_est")
    plt.plot(x, upper_bound, color="c", linestyle="-", label="y_est + K * u(y_est)")
    plt.plot(x, lower_bound, color="g", linestyle="-", label="y_est - K * u(y_est)")
    plt.title("confidence band")
    plt.xlabel("x")
    plt.ylabel("y")
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_family("monospace")

    plt.grid(True, linestyle=':', alpha=0.6)

    if filename:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(filename, format="png", bbox_inches="tight")
        print(f"the plot saved to: {os.path.abspath(filename)}")
    else:
        plt.show()

    plt.close()
