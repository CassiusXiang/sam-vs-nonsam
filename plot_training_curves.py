import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from CSV files.")
    parser.add_argument("--csv_files", nargs="+", required=True, help="List of CSV file names (relative to root_dir).")
    parser.add_argument("--labels", nargs="*", default=None, help="Legend labels for each CSV file. Defaults to file names if not provided.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size used in training (to estimate batches per epoch).")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory containing the CSV files. Defaults to current directory.")
    parser.add_argument("--output", type=str, default=None, help="Output file to save the plot (e.g., plot.png). If not provided, display the plot.")
    return parser.parse_args()

def plot_training_curves(csv_files, labels, batch_size, root_dir="", output=None):
    # Ensure Matplotlib works in Jupyter
    # try:
    #     get_ipython()  # Check if running in Jupyter
    #     %matplotlib inline
    # except NameError:
    #     pass  # Not in Jupyter, proceed normally

    # Validate inputs
    full_paths = [os.path.join(root_dir, csv_file) for csv_file in csv_files]
    for full_path in full_paths:
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"CSV file not found: {full_path}")
    
    if labels is not None and len(labels) != len(csv_files):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of CSV files ({len(csv_files)}).")

    # Default labels to file names (without path or extension) if not provided
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    # Estimate batches per epoch for CIFAR-10 (50,000 samples)
    num_batches_per_epoch = 50000 // batch_size

    # Initialize figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Process each CSV file
    for full_path, label in zip(full_paths, labels):
        # Read CSV
        df = pd.read_csv(full_path)
        required_columns = ["epoch", "batch_idx", "loss", "accuracy"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file {full_path} missing required columns: {required_columns}")

        # Compute x-axis: epoch + batch_idx / num_batches_per_epoch
        x = df["epoch"] + df["batch_idx"] / num_batches_per_epoch

        # Plot loss
        ax1.plot(x, df["loss"], label=label)

        # Plot accuracy
        ax2.plot(x, df["accuracy"], label=label)

    # Customize loss plot
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    # Customize accuracy plot
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save or display plot
    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        plt.show()

if __name__ == "__main__":
    args = parse_args()
    plot_training_curves(args.csv_files, args.labels, args.batch_size, args.root_dir, args.output)