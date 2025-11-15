#graph functions
import matplotlib.pyplot as plt

marker_dict = {0: 'v',
               1: '^',
               '': '.'}

text_dict = {0: 'Died',
             1: 'Survived',
             '': 'Test'}

def plot_simple_scatter(x:list, y:list, labels:list, survived:list):
    """plot a simple scatter graph with coloured labels"""
    for i in range(len(x)):
        marker = marker_dict[survived[i]]
        text = text_dict[survived[i]]
        label = f'{labels[i]} - {text}'
        plt.scatter(x[i], 
                    y[i], 
                    label=label,
                    marker=marker,
                    )    

    plt.title('Price against cabin for Titanic project')
    plt.xlabel('Price paid')
    plt.xlim(left=0)
    plt.xlim(right=1.0)
    plt.ylim(bottom=0)
    plt.ylim(top=1.0)
    plt.ylabel('Cabin')
    plt.grid(True)

    plt.legend(title='Data points')
    plt.show