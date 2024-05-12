from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import matplotlib.colors as mcolors
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

def save_multi_image(filename):
   pp = PdfPages('../Results/'+filename)
   fig_nums = plt.get_fignums()
   figs = [plt.figure(n) for n in fig_nums]
   for fig in figs:
      fig.savefig(pp, format='pdf',bbox_inches='tight')
   pp.close()

def green_Cmap(num_shades):
    """
    Creates an inverted colormap with specified number of green shades,
    where 0 corresponds to light green and the highest number to dark green,
    but avoiding black for the darkest shade.

    :param num_shades: Number of green shades in the colormap
    :return: A matplotlib colormap object
    """
    # Define the range of green colors (from light to dark, avoiding black)
    green_colors = [(0, i/num_shades, 0) for i in range(num_shades)]
    # Create an inverted colormap
    green_cmap = mcolors.LinearSegmentedColormap.from_list('inverted_custom_green', green_colors, N=num_shades)

    return green_cmap

def generate_color_map(base_color, num_colors):
    """
    Generate a color map with different shades of the base color by adjusting lightness.
    """
    # Convert base color name to RGB and then to HLS for lightness adjustment
    base_rgb = mcolors.hex2color(mcolors.CSS4_COLORS.get(base_color, '#0000ff'))  # Default to blue if not found
    base_hls = colorsys.rgb_to_hls(*base_rgb)
    lightness_steps = np.linspace(0.7, 0.2, num=num_colors)  # Adjust this range to control shade variation

    # Generate colors by modifying the lightness in the HLS space
    color_list = [colorsys.hls_to_rgb(base_hls[0], l, base_hls[2]) for l in lightness_steps]
    return LinearSegmentedColormap.from_list("shade_cmap", color_list, N=num_colors)

def plot_optimal_policy_2d(states, optimal_policy, A, seuil, r1, r2, r3, moy):
    """
    Plots two 2D heatmaps of the optimal policy for Day and Night where color indicates the action.
    
    :param states: List of states in the form [(x1, h1, 'Day'), (x2, h2, 'Night'), ...]
    :param optimal_policy: List of optimal actions corresponding to each state
    """
    
    # Define a custom color map with exactly A colors (5 in your case)
    #cmap = ListedColormap(['#ff6666', '#fcff66', '#66ff66', '#6666ff', '#ff66ff'])
    cmap = generate_color_map('blue', A)
    boundaries = np.arange(0, A + 1) - 0.5  # Create boundaries for each action
    norm = plt.Normalize(boundaries.min(), boundaries.max())
    
    # Separate states by Day or Night with their respective actions
    day_states = [(state[0], state[1]) for state in states if state[2] == 1]
    night_states = [(state[0], state[1]) for state in states  if state[2] == 0]
    day_actions = [m for state, m in zip(states, optimal_policy) if state[2] == 1]
    night_actions = [m for state, m in zip(states, optimal_policy) if state[2] == 0]
    
    # Determine the maximum energy value for consistent scale across plots
    all_energies = [state[0] for state in states]
    max_energy = int(max(all_energies)) + 1

    # Helper function to plot each heatmap
    def plot_heatmap(ax, states, actions, time_of_day):
        if states:  # Check if the list is not empty
            energy = [state[0] for state in states]  # Energy
            time = [state[1] for state in states]    # Time
            
            # Create a grid for plotting
            max_time = int(max(time)) + 1
            grid = np.full((max_energy, max_time), np.nan)

            for state, action in zip(states, actions):
                grid[state[0], state[1]] = action

            heatmap = ax.imshow(grid, cmap=cmap, norm=norm, origin='lower')
            
            # Annotate each cell with the action
            for i in range(max_energy):
                for j in range(max_time):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f'a{int(grid[i, j]+1)}', ha="center", va="center", color="w", fontsize=13)
            
            # Setting the time axis to align with actual data
            ax.set_xticks(range(max_time))  # Set x-ticks to be at every integer step
            ax.set_xticklabels(range(max_time))  # Label x-ticks with corresponding time values


            # Add a vertical line for the energy threshold
            ax.axhline(y=seuil, color='r', linestyle='--')

            # Labels and title
            ax.set_xlabel('Time (h)', fontsize=15)
            ax.set_ylabel('Energy (x)', fontsize=15)
            ax.set_title(f'Optimal Policy for {time_of_day}, r1={r1}, r2={r2}, r3={r3}: $\\rho^{{(\pi*)}} ={round(moy,4)}$', fontsize=13)  # Set a smaller font size for subplot titles

            return heatmap
        else:
            print(f"No data for {time_of_day}")

    fig, axs = plt.subplots(1, 2, figsize=(15, 15))  # Setup subplot with 1 row and 2 columns
    #fig, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'wspace': 0.00})

    # Plot for Day
    heatmap_day = plot_heatmap(axs[0], day_states, day_actions, 'Day')

    # Plot for Night
    heatmap_night = plot_heatmap(axs[1], night_states, night_actions, 'Night')
    cbar = plt.colorbar(heatmap_night, ax=axs[1], boundaries=boundaries, ticks=np.arange(A), spacing='proportional')
    cbar.ax.set_yticklabels([f'a{i+1}' for i in range(A)],fontsize=15)  # Set custom labels

    # Adjust layout to minimize space
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0)
    plt.tight_layout()  # Optimize layout

    # Save the plot as a PDF
    #plt.savefig("Resultats_Day_Night_{}_r2_{}.pdf".format(len(states),r2), bbox_inches='tight')
    save_multi_image("Battery_Day_Night_{}_r1_{}_r3_{}.pdf".format(len(states),r1,r3))

def plot_optimal_rewards_3d(data):
    # Extraction des données
    r1 = [row[0] for row in data]
    r2 = [row[1] for row in data]
    r3 = [row[2] for row in data]
    moyennes = [row[3] for row in data]

    # Création du graphique 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Nuage de points avec la couleur représentant la valeur 'moyenne'
    scatter = ax.scatter(r1, r2, r3, c=moyennes, cmap=green_Cmap(10))

    # Ajout de la barre de couleur
    color_bar = plt.colorbar(scatter)
    color_bar.set_label('Moyenne')

    # Étiquettes des axes
    ax.set_xlabel('r1+')
    ax.set_ylabel('r2-')
    ax.set_zlabel('r3-')

    # Titre du graphique
    plt.title('Graphique 3D avec la couleur représentant la moyenne')
    plt.tight_layout()  # Optimize layout

    # Save the plot as a PDF
    plt.savefig("../Results/Avg_Rewards_r1_r2_r3.pdf", bbox_inches='tight')
    plt.show()
