import streamlit as st
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import sys  # Importez le module sys

# Titre et description de l'application
st.title("Validation des Données de Forage Minier")
st.write("Application développée par Didier Ouedraogo, P.Geo")

# Options de configuration dans la barre latérale
st.sidebar.header("Options de Configuration")
depth_tolerance = st.sidebar.number_input("Tolérance de profondeur (mètres)", value=0.1)
distance_threshold = st.sidebar.number_input("Distance maximale pour les doublons composites (mètres)", value=1.0)

# Uploaders de fichiers
st.header("Téléchargement des Fichiers")
collars_file = st.file_uploader("Collar (CSV, Excel)", type=["csv", "xlsx"])
survey_file = st.file_uploader("Survey (CSV, Excel)", type=["csv", "xlsx"])
assays_file = st.file_uploader("Essais (CSV, Excel)", type=["csv", "xlsx"])
litho_file = st.file_uploader("Lithologie (CSV, Excel)", type=["csv", "xlsx"])
density_file = st.file_uploader("Densité (CSV, Excel)", type=["csv", "xlsx"])
oxidation_file = st.file_uploader("Oxydation (CSV, Excel)", type=["csv", "xlsx"])
geometallurgy_file = st.file_uploader("Géométallurgie (CSV, Excel)", type=["csv", "xlsx"])
composites_file = st.file_uploader("Composites (CSV, Excel)", type=["csv", "xlsx"])

# Chargement des données avec gestion des erreurs
@st.cache_data
def load_data(file, file_name):
    if file is None:
        return None
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file, encoding='latin1')
        except UnicodeDecodeError:
            st.error(f"Erreur : Impossible de décoder le fichier {file_name}. Essayez un autre encodage ou vérifiez le format du fichier.")
            return None
    except Exception as e:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier {file_name}: {e}")
            return None
    if df is not None:
        st.success(f"Fichier {file_name} chargé avec succès.")
    return df

# Chargement des DataFrames
collars = load_data(collars_file, "Collar")
survey = load_data(survey_file, "Survey")
assays = load_data(assays_file, "Essais")
litho = load_data(litho_file, "Lithologie")
density = load_data(density_file, "Densité")
oxidation = load_data(oxidation_file, "Oxydation")
geometallurgy = load_data(geometallurgy_file, "Géométallurgie")
composites = load_data(composites_file, "Composites")

# Affichage des aperçus
if collars is not None:
    st.subheader("Aperçu des Collars")
    st.dataframe(collars.head())

if survey is not None:
    st.subheader("Aperçu des Surveys")
    st.dataframe(survey.head())

if assays is not None:
    st.subheader("Aperçu des Essais")
    st.dataframe(assays.head())

# ... (Aperçus des autres fichiers)

# Validations
st.header("Validations")

# 1. Forages manquants
if collars is not None:
    collars_ids = set(collars['BHID'].unique())

    def check_missing(data, data_name):
        if data is not None:
            data_ids = set(data['BHID'].unique())
            missing = collars_ids - data_ids
            if missing:
                st.warning(f"Forages manquants dans {data_name}: {missing}")
            else:
                st.success(f"Tous les forages de collars sont présents dans {data_name}")
        else:
            st.info(f"Fichier {data_name} non chargé.")

    check_missing(survey, "Survey")
    check_missing(assays, "Essais")
    check_missing(litho, "Lithologie")
    check_missing(density, "Densité")
    check_missing(oxidation, "Oxydation")
    check_missing(geometallurgy, "Géométallurgie")
    check_missing(composites, "Composites")

    # 2. Forages présents dans d'autres fichiers mais absents de collars
    def check_extra(data, data_name):
        if data is not None:
            data_ids = set(data['BHID'].unique())
            extra = data_ids - collars_ids
            if extra:
                st.warning(f"Forages présents dans {data_name} mais absents de collars: {extra}")
            else:
                st.success(f"Tous les forages de {data_name} sont présents dans collars.")
        else:
            st.info(f"Fichier {data_name} non chargé.")

    check_extra(survey, "Survey")
    check_extra(assays, "Essais")
    check_extra(litho, "Lithologie")
    check_extra(density, "Densité")
    check_extra(oxidation, "Oxydation")
    check_extra(geometallurgy, "Géométallurgie")
    check_extra(composites, "Composites")

    # 3. Intervalles dépassant la profondeur maximale
    if assays is not None:
        max_depths = collars.set_index('BHID')['DEPTH'].to_dict()

        def check_depths(data, data_name):
            if data is not None:
                errors = []
                for index, row in data.iterrows():
                    bhid = row['BHID']
                    to_depth = row['TO']
                    if bhid in max_depths and to_depth > max_depths[bhid] + depth_tolerance:
                        errors.append(f"Intervalle invalide pour {bhid}: {to_depth} > {max_depths[bhid]}")
                if errors:
                    st.error(f"Erreurs de profondeur dans {data_name}:")
                    for error in errors:
                        st.write(error)
                else:
                    st.success(f"Aucune erreur de profondeur détectée dans {data_name}.")
            else:
                st.info(f"Fichier {data_name} non chargé.")

        check_depths(assays, "Essais")
        check_depths(litho, "Lithologie")
        check_depths(density, "Densité")
        check_depths(oxidation, "Oxydation")
        check_depths(geometallurgy, "Géométallurgie")
        check_depths(composites, "Composites")

    # 4. Doublons de forages
    def check_duplicate_bhids(data, data_name):
        if data is not None:
            if data['BHID'].duplicated().any():
                st.warning(f"Doublons de forages détectés dans {data_name}")
                st.dataframe(data[data['BHID'].duplicated(keep=False)])
            else:
                st.success(f"Aucun doublon de forage détecté dans {data_name}")
        else:
            st.info(f"Fichier {data_name} non chargé.")

    check_duplicate_bhids(collars, "Collar")
    check_duplicate_bhids(survey, "Survey")
    check_duplicate_bhids(assays, "Essais")
    check_duplicate_bhids(litho, "Lithologie")
    check_duplicate_bhids(density, "Densité")
    check_duplicate_bhids(oxidation, "Oxydation")
    check_duplicate_bhids(geometallurgy, "Géométallurgie")
    check_duplicate_bhids(composites, "Composites")

    # 5. Doublons d'échantillons
    def check_duplicate_samples(data, data_name):
        if data is not None:
            if 'SAMPLE_ID' in data.columns:
                if data['SAMPLE_ID'].duplicated().any():
                    st.warning(f"Doublons d'échantillons détectés dans {data_name}")
                    st.dataframe(data[data['SAMPLE_ID'].duplicated(keep=False)])
                else:
                    st.success(f"Aucun doublon d'échantillon détecté dans {data_name}")
            else:
                st.info(f"Colonne 'SAMPLE_ID' non trouvée dans {data_name}, impossible de vérifier les doublons d'échantillons.")
        else:
            st.info(f"Fichier {data_name} non chargé.")

    check_duplicate_samples(assays, "Essais")
    check_duplicate_samples(litho, "Lithologie")
    check_duplicate_samples(density, "Densité")
    check_duplicate_samples(oxidation, "Oxydation")
    check_duplicate_samples(geometallurgy, "Géométallurgie")
    check_duplicate_samples(composites, "Composites")

    # 6. Doublons d'échantillons composites (coordonnées proches)
    if composites is not None:
        st.subheader("Vérification des doublons d'échantillons composites (coordonnées proches)")

        def check_composite_duplicates(data, distance_threshold):
            duplicates = []
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    try:
                        coord1 = (data['LATITUDE'].iloc[i], data['LONGITUDE'].iloc[i])
                        coord2 = (data['LATITUDE'].iloc[j], data['LONGITUDE'].iloc[j])
                        distance = geodesic(coord1, coord2).meters
                        if distance < distance_threshold:
                            duplicates.append((i, j, distance))
                    except (ValueError, KeyError) as e:
                        st.error(f"Erreur lors du calcul de la distance : {e}. Assurez-vous que les colonnes 'LATITUDE' et 'LONGITUDE' existent et contiennent des valeurs numériques.")
                        return

            if duplicates:
                st.warning("Doublons d'échantillons composites détectés (coordonnées proches) :")
                for i, j, distance in duplicates:
                    st.write(f"Échantillon {i} et échantillon {j} sont à {distance:.2f} mètres l'un de l'autre.")
            else:
                st.success("Aucun doublon d'échantillon composite détecté (coordonnées proches).")

        check_composite_duplicates(composites, distance_threshold)

# Visualisations (Exemple avec Histogramme des Profondeurs)
if collars is not None:
    st.subheader("Distribution des Profondeurs Maximales (Collar)")
    fig, ax = plt.subplots()
    ax.hist(collars['DEPTH'], bins=30)
    ax.set_xlabel("Profondeur Maximale")
    ax.set_ylabel("Nombre de Forages")
    st.pyplot(fig)

# Génération du rapport de validation (exemple)
def generate_report(results):
    report = "Rapport de Validation\n\n"
    for key, value in results.items():
        report += f"{key}: {value}\n"
    return report

# Exemple d'utilisation (à adapter en fonction de vos validations)
results = {
    "Nombre de forages dans Collar": len(collars) if collars is not None else "N/A",
    "Nombre de forages dans Assays": len(assays) if assays is not None else "N/A",
    # ... (Ajoutez d'autres résultats)
}

report = generate_report(results)

# Téléchargement du rapport
def download_button(object_to_download, download_filename, button_text):
    try:
        if isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
        elif not isinstance(object_to_download, str):
            object_to_download = str(object_to_download)

        b64 = base64.b64encode(object_to_download.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{button_text}</a>'
        return href
    except Exception as e:
        return f"Erreur lors de la préparation du téléchargement : {str(e)}"

if collars is not None:
    download_link = download_button(report, "validation_report.txt", "Télécharger le Rapport de Validation")
    st.markdown(download_link, unsafe_allow_html=True)