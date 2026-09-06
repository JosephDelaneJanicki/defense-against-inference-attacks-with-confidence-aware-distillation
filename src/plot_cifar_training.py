import json
import os
import matplotlib.pyplot as plt

BASE_DIR = "results/cifar"

def plot_history(history, title, save_path):
    epochs = [h["epoch"] for h in history]

    # ===== Loss =====
    plt.figure()
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")

    if "val_loss" in history[0]:
        plt.plot(epochs, [h["val_loss"] for h in history], label="Val Loss")

    plt.plot(epochs, [h["test_loss"] for h in history], label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.savefig(save_path + "_loss.png")
    plt.close()

    # ===== Accuracy =====
    plt.figure()
    plt.plot(epochs, [h["train_acc"] for h in history], label="Train Acc")

    if "val_acc" in history[0]:
        plt.plot(epochs, [h["val_acc"] for h in history], label="Val Acc")

    plt.plot(epochs, [h["test_acc"] for h in history], label="Test Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.savefig(save_path + "_accuracy.png")
    plt.close()


def main():
    for model_name in ["overfit_teacher", "realistic_teacher"]:
        history_path = os.path.join(BASE_DIR, model_name, "history.json")

        if not os.path.exists(history_path):
            print(f"Missing: {history_path}")
            continue

        with open(history_path, "r") as f:
            history = json.load(f)

        save_path = os.path.join(BASE_DIR, model_name, model_name)

        plot_history(history, model_name, save_path)

        print(f"Saved plots for {model_name}")


if __name__ == "__main__":
    main()