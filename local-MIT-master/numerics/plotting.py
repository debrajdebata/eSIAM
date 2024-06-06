from matplotlib import pyplot as plt
####    MODULE FOR PLOTTING FIGURES     ####

plt.style.use('ggplot')
markers = ["o", "X", "P", "p", "*"]
linestyles = ["-", "--", ":"]
cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

def please_plot(x_data, y_data, legends, ylabel, xlabel, savename, logx=False, logy=False):
    if len(legends) == 0:
        plt.scatter(x_data, y_data, marker=markers[0], color=cols[0])
    else:
        for y_datum, label, marker, col in zip(y_data, legends, markers, cols):
            plt.scatter(x_data, y_datum, label=label, marker=marker, color=col)
            plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    if savename: plt.savefig(savename, bbox_inches='tight')
    plt.show()
