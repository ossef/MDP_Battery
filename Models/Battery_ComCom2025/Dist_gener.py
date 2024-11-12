import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DATA_TYPE   = 1  # 1 for NSRDB Data : 1991 - 2010, by city
                 # 2 for NREL  Data : Average type year (based on 30 years)

GRAPHS = 1      # 0 for not displaying Graphs
                # 1 for     displaying Graphs

PACKET_SIZE = 200  #Size of a chunk of Energy packet (Wh)

# NREL paquete_size = 20 ++>   Thr = 400, Buffer = 900 
# NREL paquete_size = 50 ++>   Thr = 200, Buffer = 450 
# NREL paquete_size = 100 ++>  Thr = 150, Buffer = 220 
# NREL paquete_size = 200 ++>  Thr = 40, Buffer = 100
# NREL paquete_size = 300 ++>  Thr = 25, Buffer = 65
# NREL paquete_size = 500 ++>  Thr = 10, Buffer = 20 


def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def save_filtered_data_to_txt(distribution_empirique_filtre, Interup_prob_par_heure_filtre, output_filename):
    with open(output_filename, 'w') as file:
        # Calculer le nombre d'heures et de paquets
        num_heures = len(distribution_empirique_filtre.index)
        num_paquets = distribution_empirique_filtre.columns[-1]+1
        print("num_paquets = ",num_paquets)

        # Écrire la matrice de distribution empirique (heure x paquets)
        file.write("Matrice des probabilités (heure x paquets):\n")
        heures = distribution_empirique_filtre.index
        paquets = distribution_empirique_filtre.columns

        # Écrire les dimensions au début du fichier : heure_Debut, heure_Fin, nombre de paquets
        file.write(f"{heures[0]} {heures[-1]} {num_paquets} {PACKET_SIZE}\n")

        # Écrire les en-têtes des paquets (les colonnes)
        file.write("Heure\t" + "\t".join(map(str,map(int, paquets))) + "\n")
        
        # Écrire chaque ligne de la matrice pour chaque heure
        for heure in heures:
            ligne_prob = "\t".join(f"{distribution_empirique_filtre.loc[heure, paquet]:.25f}" for paquet in paquets)
            file.write(f"{heure}\t{ligne_prob}\n")
        
        """file.write("\n")  # Ajouter une ligne vide pour séparer les sections

        # Écrire le tableau des probabilités d'interruption
        file.write("Tableau des probabilités d'interruption:\n")
        file.write("Heure\tProbabilité d'interruption\n")
        for heure, prob in Interup_prob_par_heure_filtre.items():
            file.write(f"{heure}\t{prob:.6f}\n")"""

    print(f"2) Write dists = {output_filename}")

def calculate_prob_interruption(row, alpha1=0.05, alpha2=0.8, alpha3=0.05, alpha4=0.05, alpha5=0.05):
    """
    Calcule la probabilité d'interruption d'énergie solaire à partir des conditions météorologiques.
    
    Parameters:
    row: une ligne du dataframe contenant les données météorologiques pour une heure donnée.
    alpha1, alpha2, alpha3, alpha4, alpha5: poids pour chaque facteur dans la formule.
    
    Returns:
    Probabilité d'interruption entre 0 et 1.
    """

    # Variables météorologiques

    #(Total Cloud Cover)
    TotCC = row['TotCC (10ths)'] #Cette colonne représente la couverture nuageuse totale en dixièmes (valeur comprise entre 0 et 10).
    
    #Zenith (deg) 
    Zenith = row['Zenith (deg)'] #C'est l'angle zénithal du soleil, exprimé en degrés, mesurant l'angle entre la verticale (directement au-dessus) et le soleil.
    
    #(Aerosol Optical Depth) 
    AOD = row['AOD (unitless)']  #L'épaisseur optique des aérosols (AOD) est une mesure de la concentration des particules en suspension dans l'atmosphère (poussière, pollution, fumée, etc.)
    
    #Liq Precip Depth (mm)
    LiqPrecip = row['Liq Precip Depth (mm)'] #Il s'agit de la profondeur des précipitations liquides en millimètres, mesurant la quantité d'eau tombée sous forme de pluie pendant une certaine période
    
    #Hor Vis (m)
    HorVis = row['Hor Vis (m)']  #La visibilité horizontale est mesurée en mètres et indique la distance maximale à laquelle un objet peut être vu. Une faible visibilité est généralement causée par des phénomènes météorologiques comme le brouillard, la fumée, ou une forte tempête de poussière
    
    # Normalisation des variables
    Zenith_norm = Zenith / 90  # Norme l'angle zénithal (90 degrés = au niveau de l'horizon)
    AOD_norm = AOD / max(1, row['AOD (unitless)'])  # Normaliser AOD par rapport à une valeur maximale potentielle
    LiqPrecip_norm = LiqPrecip / max(1, row['Liq Precip Depth (mm)'])  # Normaliser les précipitations
    HorVis_norm = 1 / max(1, HorVis)  # Inverser la visibilité pour réduire la probabilité avec une bonne visibilité

    # Calculer la probabilité d'interruption
    prob_interruption = (alpha1 * (TotCC / 10) +
                         alpha2 * Zenith_norm +
                         alpha3 * AOD_norm +
                         alpha4 * LiqPrecip_norm +
                         alpha5 * HorVis_norm)
    
    # Limiter la probabilité entre 0 et 1
    return min(max(prob_interruption, 0), 1)

def NSRDB_read_generate_stats(city, year):
    # Charger le fichier CSV
    file_path = './NSRDB_Data/NSRDB_' + city + '_19910101_20101231/NSRDB_StationData_' + year + '0101_' + year + '1231.csv'
    print("1) Read file = ", file_path)
    data = pd.read_csv(file_path)

    # Remplacer les valeurs '24:00' par '00:00' pour traiter correctement l'heure minuit
    data['HH:MM (LST)'] = data['HH:MM (LST)'].replace('24:00', '00:00')

    # Discrétiser les données en paquets d'énergie
    data['Paquets'] = data['Glo Mod (Wh/m^2)'] // PACKET_SIZE

    # Ajouter une colonne 'Heure' pour regrouper les données par heure de la journée
    data['Heure'] = pd.to_datetime(data['HH:MM (LST)'], format='%H:%M').dt.hour

    # 1) Regrouper par heure et calculer la moyenne des paquets d'énergie par heure, sur toute l'année
    Moy_paquets_par_heure = data.groupby('Heure')['Paquets'].mean()

    # 2)  Calculer la distribution empirique des paquets d'énergie par heure
    distribution_empirique = data.groupby(['Heure', 'Paquets']).size().unstack(fill_value=0)

    # Normaliser les distributions pour obtenir des probabilités (facultatif)
    distribution_empirique = distribution_empirique.div(distribution_empirique.sum(axis=1), axis=0)

    # Calculer la probabilité d'interruption pour chaque ligne du dataframe
    data['Probabilité d\'interruption'] = data.apply(calculate_prob_interruption, axis=1)

    # 3) Calculer la moyenne des probabilités d'interruption par heure
    Interup_prob_par_heure = data.groupby('Heure')['Probabilité d\'interruption'].mean()
    
    # Filtrer les heures où la moyenne des paquets par heure est > 0
    heures_filtrees = Moy_paquets_par_heure[Moy_paquets_par_heure > 0].index

    # 2') Filtrer la distribution empirique pour ne conserver que les heures avec paquets > 0
    distribution_empirique_filtre = distribution_empirique.loc[heures_filtrees]

    # 3') Filtrer les probabilités d'interruption pour ne garder que les heures où paquets > 0
    Interup_prob_par_heure_filtre = Interup_prob_par_heure.loc[heures_filtrees]

    """
    print(distribution_empirique)
    print(Interup_prob_par_heure)
    print("----------------- Aprés filtre ---------------------")
    print(distribution_empirique_filtre)
    print(Interup_prob_par_heure_filtre)"""

    save_filtered_data_to_txt(distribution_empirique_filtre, Interup_prob_par_heure_filtre, "./NSRDB_Extracts/"+city+"/"+city+"_"+year+"_filtred_Dists.data")

    #------------------------------------ Plot moyennes, histogramme ------------------------------------
    if GRAPHS == 1 :
        # Créer un fichier PDF pour enregistrer toutes les figures
        with PdfPages("./NSRDB_Extracts/"+city+"/"+city+"_"+year+".pdf") as pdf:
            # 1) Afficher la distribution des paquets d'énergie le long de la journée

            # Normalize the values (used to control the intensity)
            norm = plt.Normalize(min(Moy_paquets_par_heure), max(Moy_paquets_par_heure))
            
            # Choose a single base color from the 'inferno' colormap (e.g., the middle of the colormap)
            base_color = plt.cm.inferno(0.6)

            # Define minimum and maximum alpha values to control transparency
            min_alpha = 0.2  
            max_alpha = 1.0

            # Create the bar plot with the intensity controlled by the normalized values
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, value in enumerate(Moy_paquets_par_heure):
                # Normalize the intensity and apply a scaled alpha between min_alpha and max_alpha
                intensity = norm(value)
                alpha = min_alpha + (intensity * (max_alpha - min_alpha))  # Scale the alpha between min_alpha and max_alpha
                ax.bar(i, value, color=base_color, alpha=alpha)  # Apply the adjusted alpha

            plt.xlabel('Hour')
            plt.ylabel('Average Energy Packets produced')
            plt.xticks(range(0, 24))
            plt.grid(True)
            pdf.savefig()  # Sauvegarder cette figure dans le PDF
            plt.close()
            
            # 2) Afficher la distribution des probabilités d'interruption tout au long de la journée
            plt.figure(figsize=(10, 6))
            plt.bar(Interup_prob_par_heure.index, Interup_prob_par_heure.values, color='orange')
            plt.xlabel('Heure de la journée')
            plt.ylabel('Probabilité d\'interruption moyenne')
            plt.title('Probabilité d\'interruption d\'énergie solaire au cours de la journée')
            plt.xticks(range(0, 24))
            plt.grid(True)
            pdf.savefig()  # Sauvegarder cette figure dans le PDF
            plt.close()

            # 3) ------------------------------------ Plot distributions --------------------------------
            # Définir le nombre de sous-graphiques par figure
            plots_per_figure = 6

            # Nombre total d'heures
            total_hours = len(distribution_empirique.index)

            # Nombre total de figures
            num_figures = (total_hours + plots_per_figure - 1) // plots_per_figure  # Ajuster pour le nombre de figures

            # Créer les figures et sous-graphiques
            for fig_num in range(num_figures):
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))  # 2 lignes, 3 colonnes pour 6 sous-graphiques
                axes = axes.flatten()

                # Sélectionner les heures pour cette figure
                start_index = fig_num * plots_per_figure
                end_index = min(start_index + plots_per_figure, total_hours)

                for i, heure in enumerate(distribution_empirique.index[start_index:end_index]):
                    ax = axes[i]
                    # Plot de l'histogramme
                    ax.bar(distribution_empirique.columns, distribution_empirique.loc[heure], color='skyblue')

                    # Calcul de la moyenne pour chaque distribution empirique
                    moyenne = Moy_paquets_par_heure[heure]

                    # Ajout de la légende avec la moyenne
                    ax.set_title(f'Heure {heure}:00 - Moyenne: {moyenne:.2f}')
                    ax.set_xlabel('Nombre de paquets d\'énergie')
                    ax.set_ylabel('Probabilité')

                # Ajuster la mise en page pour ne pas avoir de chevauchement
                plt.tight_layout()

                # Sauvegarder cette figure dans le fichier PDF
                pdf.savefig(fig)
                plt.close(fig)  # Fermer la figure pour libérer la mémoire
            print("3) Plots saved in 'NSRDB_Extracts' folder")

def NREL_read_generate_stats(city):
    # Charger le fichier CSV
    file_path = './NREL_Data/' + city + '_pvwatts_hourly.csv'
    print("1) Read file = ", file_path)

    # Lire les métadonnées (lignes 1 à 31)
    with open(file_path, 'r') as f:
        metadata = [next(f).strip() for _ in range(32)]

    # Charger les données à partir de la ligne 33 (en sautant les 32 premières lignes)
    column_names = [
        "Month", "Day", "Hour", "Beam Irradiance (W/m2)", "Diffuse Irradiance (W/m2)", 
        "Ambient Temperature (C)", "Wind Speed (m/s)", "Albedo", 
        "Plane of Array Irradiance (W/m2)", "Cell Temperature (C)", 
        "DC Array Output (W)", "AC System Output (W)"
    ]

    # Charger les données avec skiprows pour sauter les lignes descriptives
    data = pd.read_csv(file_path, skiprows=32, names=column_names)

    # Discrétiser les données en paquets d'énergie à partir de "AC System Output (W)"
    data['Paquets'] = data['AC System Output (W)'] // PACKET_SIZE

    # Ajouter une colonne 'Heure' à partir des colonnes existantes (Month, Day, Hour)
    data['Heure'] = data['Hour']  # Extraire directement l'heure


    # 1) Pour chaque mois, regrouper par heure et calculer la moyenne des paquets d'énergie par heure
    for mois in range(1, 13):
        Moy_paquets_par_heure = 0
        distribution_empirique = None 
        heures_filtrees = None
        distribution_empirique_filtre = None 


        # Filtrer les données pour le mois en cours
        data_mois = data[data['Month'] == mois]

        # Calcul de la moyenne des paquets par heure pour le mois
        Moy_paquets_par_heure = data_mois.groupby('Heure')['Paquets'].mean()

        # 2) Calculer la distribution empirique des paquets d'énergie par heure pour ce mois
        distribution_empirique = data_mois.groupby(['Heure', 'Paquets']).size().unstack(fill_value=0)

        # Normaliser les distributions pour obtenir des probabilités (facultatif)
        distribution_empirique = distribution_empirique.div(distribution_empirique.sum(axis=1), axis=0)

        # Identifier le paquet maximal observé dans ce mois
        max_paquet = int(distribution_empirique.columns.max())

        # 3) Ajouter les colonnes manquantes pour les paquets non observés
        all_paquets = range(0, max_paquet + 1)  # Paquets attendus de 0 à max_paquet
        distribution_empirique = distribution_empirique.reindex(columns=all_paquets, fill_value=0)

        # 4) Filtrer les heures où la moyenne des paquets par heure est > 0
        heures_filtrees = Moy_paquets_par_heure[Moy_paquets_par_heure > 0].index

        # 5) Filtrer la distribution empirique pour ne conserver que les heures avec paquets > 0
        distribution_empirique_filtre = distribution_empirique.loc[heures_filtrees]

        # Sauvegarder la distribution filtrée dans un fichier
        save_filtered_data_to_txt(distribution_empirique_filtre, None, "./NREL_Extracts/"+city+"/"+city+"_M"+str(mois)+"_filtred_Dists.data")
        
        # Afficher les résultats pour validation
        #print("Mois M{}".format(str(mois)))
        #print(distribution_empirique_filtre)

        #------------------------------------ Plot moyennes, histogramme ------------------------------------
        if GRAPHS == 1 :
            # Créer un fichier PDF pour enregistrer toutes les figures
            with PdfPages("./NREL_Extracts/"+city+"/"+city+"_M"+str(mois)+".pdf") as pdf:
                # 1) Afficher la distribution des paquets d'énergie le long de la journée

                # Normalize the values (used to control the intensity)
                norm = plt.Normalize(min(Moy_paquets_par_heure), max(Moy_paquets_par_heure))
                
                # Choose a single base color from the 'inferno' colormap (e.g., the middle of the colormap)
                base_color = plt.cm.inferno(0.6)

                # Define minimum and maximum alpha values to control transparency
                min_alpha = 0.2  
                max_alpha = 1.0

                # Create the bar plot with the intensity controlled by the normalized values
                fig, ax = plt.subplots(figsize=(10, 6))
                for i, value in enumerate(Moy_paquets_par_heure):
                    # Normalize the intensity and apply a scaled alpha between min_alpha and max_alpha
                    intensity = norm(value)
                    alpha = min_alpha + (intensity * (max_alpha - min_alpha))  # Scale the alpha between min_alpha and max_alpha
                    ax.bar(i, value, color=base_color, alpha=alpha)  # Apply the adjusted alpha

                plt.xlabel('Hour')
                plt.ylabel('Average Energy Packets produced')
                plt.xticks(range(0, 24))
                plt.grid(True)
                pdf.savefig()  # Sauvegarder cette figure dans le PDF
                plt.close()

                # 2) ------------------------------------ Plot distributions --------------------------------
                # Définir le nombre de sous-graphiques par figure
                plots_per_figure = 6

                # Nombre total d'heures
                total_hours = len(distribution_empirique.index)

                # Nombre total de figures
                num_figures = (total_hours + plots_per_figure - 1) // plots_per_figure  # Ajuster pour le nombre de figures

                # Créer les figures et sous-graphiques
                for fig_num in range(num_figures):
                    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))  # 2 lignes, 3 colonnes pour 6 sous-graphiques
                    axes = axes.flatten()

                    # Sélectionner les heures pour cette figure
                    start_index = fig_num * plots_per_figure
                    end_index = min(start_index + plots_per_figure, total_hours)

                    for i, heure in enumerate(distribution_empirique.index[start_index:end_index]):
                        ax = axes[i]
                        # Plot de l'histogramme
                        ax.bar(distribution_empirique.columns, distribution_empirique.loc[heure], color='skyblue')

                        # Calcul de la moyenne pour chaque distribution empirique
                        moyenne = Moy_paquets_par_heure[heure]

                        # Ajout de la légende avec la moyenne
                        ax.set_title(f'Hour {heure}:00 - Average: {moyenne:.2f}')
                        ax.set_xlabel('Number of Energy Packets')
                        ax.set_ylabel('Probability')

                    # Ajuster la mise en page pour ne pas avoir de chevauchement
                    plt.tight_layout()

                    # Sauvegarder cette figure dans le fichier PDF
                    pdf.savefig(fig)
                    plt.close(fig)  # Fermer la figure pour libérer la mémoire
                print("3) Plots saved in 'NREL_Extracts' folder")

def Service_Demand():

    # Data inspired from cisco, table 1 : https://www.cisco.com/c/en/us/td/docs/ios/solutions_docs/voip_solutions/TA_ISD.html
    data = {
        "Hour": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  
        "Total_Erlangs": [30 ,40, 45, 54.6, 60.6, 58.3, 45.2, 51.0, 59.2, 55.7, 52.7, 45, 40, 35, 30]
    }

    # Créer le DataFrame
    df = pd.DataFrame(data)

    # Recalculer la somme totale des Erlangs
    total_erlangs = df["Total_Erlangs"].sum()

    # Calculer la probabilité relative d'arrivée d'une tâche
    df["P_relative_arrival"] = df["Total_Erlangs"] / total_erlangs

    first_hour = df["Hour"].min()
    last_hour = df["Hour"].max()

    # Writing to the file
    with open("./NREL_Extracts/Service_Demand.data", "w") as file:
        # Write the first line with first and last hour
        file.write(f"{first_hour} {last_hour}\n")
        
        # Write each hour and its corresponding probability
        for i, row in df.iterrows():
            file.write(f"{int(row['Hour'])} {row['P_relative_arrival']:.5f}\n")

    if GRAPHS == 1 :
        # Normalize the probabilities for controlling transparency (alpha)
        norm = plt.Normalize(df["P_relative_arrival"].min(), df["P_relative_arrival"].max())

        # Define a base color from the 'inferno' colormap
        base_color = plt.cm.inferno(0.6)

        # Set alpha range for transparency
        min_alpha = 0.2
        max_alpha = 1.0

        # Create the bar plot with intensity controlled by the normalized values
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, value in enumerate(df["P_relative_arrival"]):
            # Normalize the intensity and apply scaled alpha between min_alpha and max_alpha
            intensity = norm(value)
            alpha = min_alpha + (intensity * (max_alpha - min_alpha))  # Scale alpha between min_alpha and max_alpha
            ax.bar(df["Hour"][i], value, color=base_color, alpha=alpha)  # Apply the adjusted alpha

        # Labels and settings
        plt.xlabel('Hour')
        plt.ylabel('Service demand probability')
        plt.grid(True)
        plt.xticks(df["Hour"])
        plt.savefig("./NREL_Extracts/Service_Demand.pdf")


if DATA_TYPE == 1:
    cities = ["Chicago_Ohare", "Fairbanks", "Los_Angeles", "New_York_JFK", "Reno"]
    years = [str(year) for year in range(1991, 2010 + 1, 1)]

    Service_Demand()
    #NSRDB_read_generate_stats("Chicago_Ohare", "1992")

    for city in cities:
        for year in years :
            NSRDB_read_generate_stats(city, year)

    print("NSRDB Done !")

if DATA_TYPE == 2:
    #            2,613      3,3        4,05       5,4                                                
    cities = ["Unalaska", "Moscow", "Paris", "Barcelona", "Rabat" ]

    Service_Demand()
    for city in cities:
        NREL_read_generate_stats(city)
    print("NREL Done !")


