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

def meanSwapsShadowedLine(datafile, prefix, ylabel, xmax=None, ylim=None):

    #Process data
    df = pd.read_csv(datafile, delimiter=",")

    #Filter out sentences with 0 swaps
    df = df.loc[df[prefix+"Eager"]>0]
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
        plt.xlim(2,xmax+0.5)
    if ylim is not None:
	plt.ylim(ylim) 

    for i in range(len(cols)):
        means = all_means[cols[i]][:xmax]
        stds = all_stds[cols[i]][:xmax]
        idx = idx[:xmax]
        plt.xlabel("Sentence Length", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.plot(idx, means, color[i], marker=markers[i], lw=1.5, label=labels[i])
        plt.fill_between(idx, means+stds, means-stds, color=color[i], alpha=0.5*1/(i+1))
    plt.legend(loc="upper left")
    #plt.show()
    plt.savefig("%sswaps.pdf"%prefix)
    plt.close()

def plotBeam(datafile):
    df = pd.read_csv(datafile, delimiter=",")
    plt.plot(df['beam_size'], df['all'], label='all', marker='o', color="gold", lw=2)
    plt.plot(df['beam_size'], df['disco'], label='disco', marker='v', color="darkred", lw=2)
    plt.legend(loc='lower right')
    plt.ylim(30,90)
    plt.xlabel("Beam size")
    plt.ylabel("F-score")
    plt.savefig("beam_fscore.png")
#    plt.show()
    plt.close()


def meansOfMeans(datafile):


    df = pd.read_csv(datafile, delimiter=",")
    df = df.loc[df["swapsEager"]>0]
    grouped = df.groupby("words", as_index=True)
    idx = grouped.groups.keys()

    all_means=grouped.mean()
    mean_of_means = all_means.mean()
    std_of_means = all_means.std()

    print "& Average number of swaps & Average jump size \\\\"
    print "\hline"
    for laziness in ("Eager", "Lazy", "Lazier"):
	    
    	print "{} & {}({}) & {}({})\\\\".format(laziness, mean_of_means["swaps{}".format(laziness)], std_of_means["swaps%s"%laziness], mean_of_means["avgAltBlockSize%s"%laziness], std_of_means["avgAltBlockSize%s"%laziness])


def stats(datafile):

    df = pd.read_csv(datafile, delimiter=",")
    df = df.loc[df["swapsEager"]>0]
    grouped = df.groupby("words", as_index=True)
    idx = grouped.groups.keys()


    #Biggest difference in any sentence
    print 'biggest difference in number of swaps'
    difflazy = abs(df["swapsEager"] - df["swapsLazy"])
    difflazier = abs(df["swapsEager"] - df["swapsLazier"])
    print "Eager - Lazy, nswaps: ", difflazy.max()
    print "Eager - Lazier, nswaps: ", difflazier.max()

    print 'biggest difference in jump size'
    difflazy = abs(df["avgAltBlockSizeEager"] - df["avgAltBlockSizeLazy"])
    difflazier = abs(df["avgAltBlockSizeEager"] - df["avgAltBlockSizeLazier"])
    print "Eager - Lazy, jump size: ", difflazy.max()
    print "Eager - Lazier, jump size: ", difflazier.max()


    df["diffAltEagerLazy"] = df.apply(lambda row: abs(row["avgAltBlockSizeEager"] - row["avgAltBlockSizeLazy"]), axis=1)

    #print df[df["diffAltEagerLazy"]==62]

    print df["avgAltBlockSizeLazier"].max()



    #df.apply(df["swapsEager"] - df["swapsLazy"], axis=1)


if __name__ == "__main__":
    datafile = "all_stats.csv"

    meansOfMeans(datafile)

    #meanSwapsShadowedLine(datafile, prefix="swaps", ylabel="Mean number of swaps", xmax=80, ylim=(-35, 205))
    #meanSwapsShadowedLine(datafile, prefix="avgAltBlockSize", ylabel="Mean size of the swap jump", xmax=80, ylim=(-13, 55))
    #meanSwapsShadowedLine(datafile, prefix="avgBlockSize", ylabel=" avgBlockSize", xmax=90)

    # plotBeam('beam_data.csv')
