import matplotlib.pyplot as plt

def plot_loss(x, which, loc, fname):
    
    plt.figure(figsize=(20, 7))
    plt.title(which)
    plt.plot(x)
    plt.xlabel("epoch")
    plt.ylabel(fname)
    plt.savefig(loc + fname + ".png")
    plt.close()