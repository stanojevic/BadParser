import matplotlib.pyplot as plt
import pandas as pd

def boxplotSwaps(datafile):
    df = pd.read_csv(datafile, delimiter=" ")
    df.boxplot("swap", "num_words")
    plt.show()
    df.plot()
    plt.close()


def meanSwapsErrorBars(datafile):
    df = pd.read_csv(datafile, delimiter=" ")
    grouped = df.groupby("num_words", as_index=True)
    idx = grouped.groups.keys()
    plt.errorbar(idx, grouped.mean()["swap"], yerr=grouped.std(ddof=0)["swap"], fmt='o')
    plt.show()
    plt.close()

def meanSwapsShadowLine(datafile, legend=""):

    #Process data
    df = pd.read_csv(datafile, delimiter=" ")
    grouped = df.groupby("num_words", as_index=True)
    idx = grouped.groups.keys()
    means=grouped.mean()['swap']
    stds=grouped.std(ddof=0)['swap']

    #Plot
    plt.xlabel("Sentence Length", fontsize=15)
    plt.ylabel("Mean Number of Swaps", fontsize=15)
    plt.plot(idx, means, 'r', marker="p", lw=1.5, label=legend)
    plt.fill_between(idx, means+stds, means-stds, color="r", alpha=0.2)
    plt.legend(loc="upper left")
    #plt.show()
    plt.savefig("swaps%s.pdf"%legend)
    plt.close()

if __name__ == "__main__":
    datafile = "transition_counts_training_set_log.csv"
    meanSwapsShadowLine(datafile, legend="Eager")