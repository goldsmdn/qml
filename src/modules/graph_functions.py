#graph functions
import matplotlib.pyplot as plt

def plot_simple_scatter(x:list, y:list, labels:list):
    """plot a simple scatter graph with coloured labels"""
    markers = ['v', '^', 'o']

    for i in range(len(x)):
        plt.scatter(x[i], 
                    y[i], 
                    label=labels[i],
                    marker=markers[i],
                    )    

    plt.title('Price against cabin for Titanic project')
    plt.xlabel('Price paid')
    plt.ylabel('Cabin')
    plt.grid(True)

    plt.legend(title='Data points')
    plt.show