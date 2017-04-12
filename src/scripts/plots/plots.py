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

def meanSwapsShadowedLine(datafile, prefix, ylabel, xmax=None):

    #Process data
    df = pd.read_csv(datafile, delimiter=",")
    grouped = df.groupby("words", as_index=True)
    idx = grouped.groups.keys()


    all_means=grouped.mean()
    all_stds=grouped.std(ddof=0)


    #Plot
    cols = [prefix+laziness for laziness in ('Eager', 'Lazy', 'Lazier')]
    labels = ('Eager', 'Lazy', 'Lazier')
    markers = ('p', '^', '8')
    color = ('aquamarine', 'gold', 'purple')
    if xmax is not None:
        plt.xlim(0,xmax+2)
    for i in range(len(cols)):
        means = all_means[cols[i]][:xmax]
        stds = all_stds[cols[i]][:xmax]
        idx = idx[:xmax]
        plt.xlabel("Sentence Length", fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.plot(idx, means, color[i], marker=markers[i], lw=1.5, label=labels[i])
        plt.fill_between(idx, means+stds, means-stds, color=color[i], alpha=0.5*1/(i+1))
    plt.legend(loc="upper left")
    #plt.show()
    plt.savefig("%sswaps.pdf"%prefix)
    plt.close()




if __name__ == "__main__":
    datafile = "all_stats.csv"
    meanSwapsShadowedLine(datafile, prefix="swaps", ylabel="swaps", xmax=90)
    meanSwapsShadowedLine(datafile, prefix="avgAltBlockSize", ylabel="avg alt block size", xmax=90)
    meanSwapsShadowedLine(datafile, prefix="avgBlockSize", ylabel=" avgBlockSize", xmax=90)