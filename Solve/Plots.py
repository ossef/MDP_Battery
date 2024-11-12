from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import colorsys
import matplotlib.colors as mcolors
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

def save_multi_image(filename, folder1, folder2):
   pp = PdfPages('../Results/'+folder1+'/'+folder2+'/'+filename)
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

def plot_optimal_policy_2d_WiMob24(states, optimal_policy, A, seuil, r1, r2, r3, moy):
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
    save_multi_image("Battery_Day_Night_{}_r1_{}_r3_{}.pdf".format(len(states),r1,r3), "WiMob24", "HeatMaps")

def plot_optimal_policy_2d_ComCom25(states, optimal_policy, A, Thr, Buffer, number, city, start_hour, end_hour, r1, r2, r3, moy, DATA_TYPE) :
    """
    Plots two 2D heatmaps of the optimal policy for Day and Night where color indicates the action.
    
    :param states: List of states in the form [(x1, h1, 'Day'), (x2, h2, 'Night'), ...]
    :param optimal_policy: List of optimal actions corresponding to each state
    :param start_hour: The hour at which the plot should start (default is 7h)
    :param end_hour: The hour at which the plot should end (default is 18h)
    """

    # Define a custom color map with exactly A colors (5 in your case)
    cmap = generate_color_map('blue', A+1)
    boundaries = np.arange(0, A + 2) - 0.5  # Create boundaries for each action
    norm = plt.Normalize(boundaries.min(), boundaries.max())

    # Adaptation of drawing: Last hour the decision is automaticaly to release battery, therefore last action is A-1
    optimal_policy = [A if state[1] == end_hour else m for state, m in zip(states, optimal_policy)]

    # Separate states by Day or Night with their respective actions
    day_states = [(state[0], state[1]) for state in states if state[2] == 1]
    night_states = [(state[0], state[1]) for state in states if state[2] == 0]
    day_actions = [m for state, m in zip(states, optimal_policy) if state[2] == 1]
    night_actions = [m for state, m in zip(states, optimal_policy) if state[2] == 0]

    # Determine the maximum energy value for consistent scale across plots
    all_energies = [state[0] for state in states]
    max_energy = max(all_energies) + 1 #Buffer+1

    # Helper function to plot each heatmap
    def plot_heatmap(ax, states, actions, time_of_day):
        if states:  # Check if the list is not empty
            energy = [state[0] for state in states]  # Energy
            time = [state[1] for state in states]    # Time
            
            # Filter to include only the hours from start_hour to end_hour
            filtered_states = [(e, t) for (e, t) in states if start_hour <= t <= end_hour]
            filtered_actions = [a for (s, a) in zip(states, actions) if start_hour <= s[1] <= end_hour]

            energy_filtered = [state[0] for state in filtered_states]
            time_filtered = [state[1] for state in filtered_states]

            # Create a grid for plotting
            max_time = end_hour - start_hour + 1
            grid = np.full((max_energy, max_time), np.nan)

            for state, action in zip(filtered_states, filtered_actions):
                grid[state[0], state[1] - start_hour] = action  # Subtract start_hour to shift the time

            heatmap = ax.imshow(grid, cmap=cmap, norm=norm, origin='lower', aspect='auto')

            # Annotate each cell with the action
            for i in range(max_energy):
                for j in range(max_time):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f'a{int(grid[i, j]+1)}', ha="center", va="center", color="w", fontsize=10)
            
            # Set the time axis to start from start_hour
            ax.set_xticks(range(max_time))  # Set x-ticks to be at every integer step
            ax.set_xticklabels(range(start_hour, end_hour + 1))  # Label x-ticks with corresponding time values

            # Add a vertical line for the energy threshold
            ax.axhline(y=Thr, color='r', linewidth=4, linestyle='--', label="Energy threshold")

            # Labels and title
            ax.set_xlabel('Day Time (h)', fontsize=15)
            ax.set_ylabel('Energy (x)', fontsize=15)

            """# Calcul de l'exposant pour la partie absolue
            r2_exp = int(np.log10(abs(r2)))
            r3_exp = int(np.log10(abs(r3)))

            # Gestion de l'affichage avec signe négatif si nécessaire
            r2_str = f"-10^{{{r2_exp}}}" if r2 < 0 else f"10^{{{r2_exp}}}"
            r3_str = f"-10^{{{r3_exp}}}" if r3 < 0 else f"10^{{{r3_exp}}}"
            """

            # Titre avec LaTeX pour afficher les exposants
            ax.set_title(f'PV-{time_of_day} states, (r1={r1},r2={r2},r3={r3}) : avg={round(moy,2)}', fontsize=13)

            ax.legend(loc='upper left', fontsize="12.5")
            return heatmap
        else:
            print(f"No data for {time_of_day}")

    # Création d'une grille de spécifications pour contrôler les proportions et l'emplacement de la colorbar
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.1)  # La dernière colonne est réservée pour la colorbar

    # Plot for Day
    ax0 = plt.subplot(gs[0])
    heatmap_day = plot_heatmap(ax0, day_states, day_actions, 'ON')

    # Plot for Night
    ax1 = plt.subplot(gs[1])
    heatmap_night = plot_heatmap(ax1, night_states, night_actions, 'OFF')
    ax1.set_ylabel("")

    # Create a single color bar on the right, aligned with both heatmaps
    cbar_ax = plt.subplot(gs[2])  # Utilisation de la troisième colonne pour la colorbar
    cbar = fig.colorbar(heatmap_night, cax=cbar_ax, boundaries=boundaries, ticks=np.arange(A+1), spacing='proportional')
    #cbar_ax.set_aspect(1)
    # Set custom labels for the colorbar
    cbar.ax.set_yticklabels([f'a{i+1}' for i in range(A+1)], fontsize=15)

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PDF
    if DATA_TYPE == 1:
        save_multi_image("ComCom25_{}_{}_Thr_{}_B_{}_r1_{}_r2_{}_r3_{}.pdf".format(city, number, Thr, Buffer, r1, r2, r3), "ComCom25/NSRDB", "HeatMaps")
    if DATA_TYPE == 2:
        save_multi_image("ComCom25_{}_{}_Thr_{}_B_{}_r1_{}_r2_{}_r3_{}.pdf".format(city, number, Thr, Buffer, r1, r2, r3), "ComCom25/NREL", "HeatMaps")
    print("Plot HeatMap Done !")

def plot_cities_years_ComCom25(rewards_data, energy_data, loss_data, noService_data, cityNames, numbers, r1, r2, r3, Thr, Buffer, DATA_TYPE):

    # ---------- Plotting energy_data --------------
    plt.figure(figsize=(10, 6))

    if DATA_TYPE == 1 : 
        markers = ['o', 'x', '+', 's', 'd'] 
        xlab    = "Year"
        folder1 = "ComCom25/NSRDB"
        folder2 = "Cities_Years"

    if DATA_TYPE == 2 : 
        markers = ['o', 'x', '+', 's', 'd', '*'] 
        xlab = "Mounth"
        folder1 = "ComCom25/NREL"
        folder2 = "Cities_Mounths"
        numbers = [int(m[1:]) for m in numbers]

    for i, city in enumerate(cityNames):
        plt.plot(numbers, energy_data[city], label=city, marker=markers[i], linestyle='-') 

    #plt.ylim([0.1, 0.9])
    plt.xticks(fontsize=10, rotation=45)
    plt.xlabel(xlab)
    plt.ylabel("Mean Energy stored in batteries (Wh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---------- Plotting noService_data --------------
    plt.figure(figsize=(10, 6))

    for i, city in enumerate(cityNames):
        plt.plot(numbers, noService_data[city], label=city, marker=markers[i], linestyle='-') 

    #plt.ylim([0.055, 0.08])
    plt.xticks(fontsize=10, rotation=45)
    plt.xlabel(xlab)
    plt.ylabel("Data Packets latency probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---------- Plotting lost_data --------------
    plt.figure(figsize=(10, 6))

    for i, city in enumerate(cityNames):
        plt.plot(numbers, loss_data[city]  , label=city, marker=markers[i], linestyle='-') 

    plt.xticks(fontsize=10, rotation=45)
    plt.xlabel(xlab)
    plt.ylabel("Mean Energy lost (Wh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---------- Plotting rewards_data --------------
    plt.figure(figsize=(10, 6))

    for i, city in enumerate(cityNames):
        plt.plot(numbers, rewards_data[city], label=city, marker=markers[i], linestyle='-') 

    plt.xticks(fontsize=10, rotation=45)
    plt.xlabel(xlab)
    plt.ylabel("Combined weighted rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_multi_image("Cities_Months_Thr_{}_B_{}_r1_{}_r2_{}_r3_{}.pdf".format(Thr,Buffer,r1,r2,r3), folder1, folder2)
    print("Plots Done !")

def plot_optimal_rewards_3d_ComCom25(data, city, r1, month, title, name):
    # Extraction des données
    r2 = np.array([row[0] for row in data])
    r3 = np.array([row[1] for row in data])
    measure = np.array([row[2] for row in data])

    # Créer une grille régulière pour r2 et r3
    r2_min, r2_max = min(r2), max(r2)
    r3_min, r3_max = min(r3), max(r3)
    r2_grid, r3_grid = np.meshgrid(np.linspace(r2_min, r2_max, 100), np.linspace(r3_min, r3_max, 100))

    # Interpoler les valeurs z sur cette grille
    z_grid = griddata((r2, r3), measure, (r2_grid, r3_grid), method='cubic')  # 'cubic', 'linear', or 'nearest'

    # Créer la figure et l'axe
    fig, ax = plt.subplots()

    # Choisir le thème
    if 'energy' in name or 'Combined' in name:
        theme = 'inferno'
    else:
        theme = 'inferno_r'

    # Créer la carte colorée en utilisant pcolormesh
    color_plot = ax.pcolormesh(r2_grid, r3_grid, z_grid, cmap=theme, shading='auto')

    # Ajouter la barre de couleur pour les valeurs interpolées
    color_bar = fig.colorbar(color_plot, ax=ax)
    color_bar.set_label(name)


    """# Création du graphique 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Nuage de points avec la couleur représentant la valeur 'moyenne'
    scatter = ax.scatter(r1, r2, r3, c=measure, cmap=green_Cmap(10))

    # Ajout de la barre de couleur
    color_bar = plt.colorbar(scatter)
    color_bar.set_label(name)
    """

    # Étiquettes des axes
    ax.set_xlabel('r2-')
    ax.set_ylabel('r3-')

    # Titre du graphique
    plt.title(title)
    plt.tight_layout()  # Optimize layout

    # Save the plot as a PDF
    save_multi_image("{}_{}_r1_{}_r2Min_{}_r3Min{}.pdf".format(city,month,r1, min(r2),min(r3)), "ComCom25/NREL", "Rewards_Detailed")

