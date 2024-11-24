import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from Z_HELPERS.DM_communication import get_docs_VM

#function to calculate mean and std
def calculate_stats(transmission_data, normalization_factor):
    normalized_data = np.array(transmission_data) / normalization_factor
    mean = normalized_data.mean()
    std = normalized_data.std()
    return mean, std

#function to plot data with ellipses indicating the spread
#stats_dict: dictionary where keys are bacteria types and values are lists of (mean, std) tuples
def plot_data(stats_dict, title, display_names):
    fig, ax = plt.subplots()
    colors = {
        'bs134': 'b', 'coli': 'r', 'li142': 'g', 'ns139': 'y', 'pp46': 'orange',
        'se26': 'cyan', 'se9': 'magenta', 'ys134': 'black'
    }

    #assign a unique color to each bacteria type and create the ellipses
    for bacteria, stats in stats_dict.items():
        if bacteria == "pp6":
            continue
        means, stds = zip(*stats)

        color = colors.get(bacteria, 'gray')  # Default color if not found
        ax.scatter(means, stds, label=bacteria, color=color)

        #2 sigma ellipse for data spread
        ellipse_2sigma = Ellipse((np.mean(means), np.mean(stds)), 2 * np.std(means), 2 * np.std(stds),
                                 edgecolor=color, facecolor='none', linestyle='--')
        #3 sigma ellipse
        ellipse_3sigma = Ellipse((np.mean(means), np.mean(stds)), 3 * np.std(means), 3 * np.std(stds),
                                 edgecolor=color, facecolor='none', linestyle=':')
        ax.add_patch(ellipse_2sigma)
        ax.add_patch(ellipse_3sigma)

        display_name = display_names.get(bacteria, bacteria)
        ax.text(np.mean(means), np.mean(stds), display_name, color=color, fontsize=13, ha='right')

    ax.set_xlabel('Mean (μ)')
    ax.set_ylabel('Standard Deviation (σ)')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    query = {}
    #retrieves a sample of documents from the database using the query
    sample_docs = get_docs_VM(db_name='Optical_Trapping_ML_DB', coll_name='tseries_Gram_analysis', query=query)

    #stores stats for each bacteria type
    stats_dict = {}
    display_names = {
        'bs134': 'B. subtilis',
        'coli': 'E. coli',
        'li142': 'L. innocua',
        'ns139': 'N. sicca',
        'pp46': 'P. putida',
        'pp6': 'Pseudomonas putida',
        'se26': 'S. epidermidis B',
        'se9': 'S. epidermidis A',
        'ys134': 'Y. ruckeri'
    }

    for doc in sample_docs:
        normalization_factor = doc.get('normalization_factor')
        bacteria = doc.get('bacteria')
        bacteria_name1 = doc.get('name')
        data = doc.get('data')
        if bacteria not in stats_dict:
            stats_dict[bacteria] = []
        #calculates and stores mean and std for each bacteria type
        if normalization_factor and data:
            transmission_data = data.get('transmission', [])
            if transmission_data:
                mean, std = calculate_stats(transmission_data, normalization_factor)
                print(bacteria_name1, mean)
                stats_dict[bacteria].append((mean, std))

    if stats_dict:
        plot_data(stats_dict, "Bacterial Transmission Data", display_names)
    else:
        print("No valid data found to plot.")
