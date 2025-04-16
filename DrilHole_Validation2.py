import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from geopy.distance import geodesic
from scipy.spatial import KDTree
import datetime
import os
import sys
import traceback
from functools import partial

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Validation des Données de Forage Minier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Définition des styles CSS personnalisés
def apply_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stButton>button {
            border-radius: 5px;
            background-color: #4b7bec;
            color: white;
            font-weight: 500;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #3867d6;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        /* Stylisation des tableaux de données */
        .dataframe {
            border-collapse: collapse;
            margin: 25px 0;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .dataframe thead tr {
            background-color: #4b7bec;
            color: white;
            text-align: left;
        }
        .dataframe th, .dataframe td {
            padding: 12px 15px;
        }
        .dataframe tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .dataframe tbody tr:last-of-type {
            border-bottom: 2px solid #4b7bec;
        }
        /* Badges de statut */
        .success-badge {
            background-color: #5cb85c;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        .warning-badge {
            background-color: #f0ad4e;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        .error-badge {
            background-color: #d9534f;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        /* Section info */
        .info-box {
            background-color: #e8f4f8;
            border-left: 5px solid #4b7bec;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0 4px 4px 0;
        }
        /* Footer */
        .footer {
            position: relative;
            padding-top: 2rem;
            margin-top: 2rem;
            text-align: center;
            color: #7f8c8d;
            border-top: 1px solid #eee;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Initialisation des variables de session
if 'files' not in st.session_state:
    st.session_state.files = {}
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = {}
if 'report_data' not in st.session_state:
    st.session_state.report_data = {}

# Titre et description de l'application
st.title("🔍 Validation des Données de Forage Minier")
st.markdown("""
<div class="info-box">
Cette application permet de valider les données de forage d'exploration minière en vérifiant la cohérence entre les fichiers,
en détectant les doublons et en s'assurant que les données respectent les contraintes définies.
</div>
""", unsafe_allow_html=True)

st.markdown("**Développé par:** Didier Ouedraogo, P.Geo")

# Fonctions utilitaires
def standardize_column_names(df):
    """Standardise les noms de colonnes"""
    # Dictionnaire de mapping entre différentes variations de noms de colonnes et noms standardisés
    column_mapping = {
        # Identifiants de forage
        'bhid': 'holeid', 'hole_id': 'holeid', 'hole': 'holeid', 'id': 'holeid', 'dhid': 'holeid',
        'drill_hole_id': 'holeid', 'sondage': 'holeid', 'forage': 'holeid',
        
        # Profondeurs
        'depth': 'depth', 'max_depth': 'depth', 'final_depth': 'depth', 'eoh': 'depth',
        'end_of_hole': 'depth', 'length': 'depth', 'longueur': 'depth', 'prof': 'depth',
        'profondeur': 'depth',
        
        # Coordonnées
        'xcollar': 'x', 'east': 'x', 'easting': 'x', 'coordx': 'x', 'coord_x': 'x', 'x_utm': 'x',
        'ycollar': 'y', 'north': 'y', 'northing': 'y', 'coordy': 'y', 'coord_y': 'y', 'y_utm': 'y',
        'zcollar': 'z', 'elev': 'z', 'elevation': 'z', 'rl': 'z', 'msl': 'z', 'coordz': 'z',
        'coord_z': 'z', 'z_utm': 'z', 'altitude': 'z',
        
        # Intervalle de début
        'from': 'from', 'start': 'from', 'top': 'from', 'debut': 'from', 'de': 'from',
        'from_m': 'from', 'fr': 'from',
        
        # Intervalle de fin
        'to': 'to', 'end': 'to', 'bottom': 'to', 'fin': 'to', 'a': 'to', 'à': 'to',
        'to_m': 'to', 'jusqua': 'to',
        
        # Azimuth et Dip
        'azimuth': 'azimuth', 'azi': 'azimuth', 'azim': 'azimuth', 'azm': 'azimuth', 'az': 'azimuth',
        'bearing': 'azimuth', 'dir': 'azimuth', 'direction': 'azimuth',
        
        'dip': 'dip', 'plunge': 'dip', 'inclination': 'dip', 'incl': 'dip',
        'pend': 'dip', 'pendage': 'dip',
        
        # Identifiants d'échantillons
        'sample_id': 'sampleid', 'sampleno': 'sampleid', 'sample_no': 'sampleid', 'sample': 'sampleid',
        'echantillon': 'sampleid', 'lab_id': 'sampleid',
        
        # Autres
        'latitude': 'latitude', 'lat': 'latitude', 
        'longitude': 'longitude', 'long': 'longitude', 'lng': 'longitude',
        'lithology': 'lithology', 'litho': 'lithology', 'rocktype': 'lithology',
    }
    
    # Convertir tous les noms de colonnes en minuscules
    df.columns = [col.lower() for col in df.columns]
    
    # Appliquer le mapping
    rename_dict = {}
    for col in df.columns:
        if col in column_mapping:
            rename_dict[col] = column_mapping[col]
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    return df

def load_data(file, file_type):
    """Charge et prépare les données à partir d'un fichier"""
    if file is None:
        return None
    
    try:
        # Détection du type de fichier et chargement
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file, encoding='latin1')
                except Exception as e:
                    st.error(f"Erreur de décodage du fichier {file.name}. Essayez d'autres encodages.")
                    return None
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error(f"Format de fichier non pris en charge: {file.name}. Utilisez CSV ou Excel.")
            return None
        
        # Standardisation des noms de colonnes
        df = standardize_column_names(df)
        
        # Vérification des colonnes essentielles
        essential_columns = {
            'collars': ['holeid'],
            'survey': ['holeid'],
            'assays': ['holeid', 'from', 'to'],
            'litho': ['holeid', 'from', 'to'],
            'density': ['holeid', 'from', 'to'],
            'oxidation': ['holeid', 'from', 'to'],
            'geometallurgy': ['holeid', 'from', 'to'],
            'composites': ['holeid'],
        }
        
        if file_type in essential_columns:
            missing_cols = [col for col in essential_columns[file_type] if col not in df.columns]
            if missing_cols:
                st.error(f"Colonnes manquantes dans {file_type}: {', '.join(missing_cols)}")
                st.write("Colonnes disponibles:", ", ".join(df.columns))
                return None
        
        # Vérifier les valeurs manquantes dans les colonnes essentielles
        if file_type in essential_columns:
            for col in essential_columns[file_type]:
                if df[col].isnull().sum() > 0:
                    st.warning(f"{df[col].isnull().sum()} valeurs manquantes détectées dans la colonne '{col}' du fichier {file_type}")
        
        # Pour les fichiers avec des coordonnées spatiales
        if file_type in ['collars', 'composites']:
            coord_columns = ['x', 'y', 'z']
            for col in coord_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        st.info(f"La colonne '{col}' a été convertie en type numérique.")
                    except Exception as e:
                        st.warning(f"Impossible de convertir la colonne '{col}' en type numérique: {str(e)}")
        
        # Pour les fichiers avec des intervalles de profondeur
        if file_type in ['assays', 'litho', 'density', 'oxidation', 'geometallurgy', 'composites']:
            depth_columns = ['from', 'to']
            for col in depth_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        st.info(f"La colonne '{col}' a été convertie en type numérique.")
                    except Exception as e:
                        st.warning(f"Impossible de convertir la colonne '{col}' en type numérique: {str(e)}")
        
        st.success(f"Fichier {file_type} chargé avec succès: {file.name}")
        return df
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {file.name}: {str(e)}")
        st.error(traceback.format_exc())
        return None

def download_button(object_to_download, download_filename, button_text):
    """Crée un bouton de téléchargement pour tout type d'objet"""
    try:
        if isinstance(object_to_download, pd.DataFrame):
            if download_filename.endswith('.csv'):
                object_to_download = object_to_download.to_csv(index=False)
            elif download_filename.endswith(('.xlsx', '.xls')):
                towrite = io.BytesIO()
                object_to_download.to_excel(towrite, index=False)
                towrite.seek(0)
                b64 = base64.b64encode(towrite.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{download_filename}">{button_text}</a>'
                return href
            else:
                object_to_download = object_to_download.to_csv(index=False)
        elif not isinstance(object_to_download, str):
            object_to_download = str(object_to_download)

        b64 = base64.b64encode(object_to_download.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{button_text}</a>'
        return href
    except Exception as e:
        return f"Erreur lors de la préparation du téléchargement : {str(e)}"

def get_status_badge(status):
    """Renvoie le badge HTML correspondant au statut"""
    if status == "success":
        return '<span class="success-badge">✓ Succès</span>'
    elif status == "warning":
        return '<span class="warning-badge">⚠ Attention</span>'
    elif status == "error":
        return '<span class="error-badge">✗ Erreur</span>'
    else:
        return status

def display_status_line(message, status):
    """Affiche une ligne avec un badge de statut"""
    st.markdown(f"{get_status_badge(status)} {message}", unsafe_allow_html=True)

# Fonction pour vérifier les doublons d'échantillons composites (coordonnées proches)
def check_composite_duplicates(data, distance_threshold):
    """Vérifie les doublons d'échantillons composites par proximité de coordonnées"""
    if data is None or len(data) < 2:
        return {'status': 'info', 'message': "Pas assez de données pour l'analyse.", 'data': None}
    
    # Vérifier que les colonnes de coordonnées existent
    coord_cols = ['x', 'y', 'z']
    if not all(col in data.columns for col in coord_cols):
        return {
            'status': 'error', 
            'message': f"Colonnes de coordonnées manquantes. Colonnes nécessaires: {', '.join(coord_cols)}",
            'data': None
        }
    
    # Extraire les coordonnées
    try:
        # S'assurer que les coordonnées sont numériques
        coords = data[coord_cols].copy()
        for col in coord_cols:
            if not pd.api.types.is_numeric_dtype(coords[col]):
                coords[col] = pd.to_numeric(coords[col], errors='coerce')
        
        # Supprimer les lignes avec des valeurs manquantes
        coords.dropna(inplace=True)
        if len(coords) < 2:
            return {'status': 'warning', 'message': "Pas assez de coordonnées valides pour l'analyse.", 'data': None}
        
        # Utiliser KDTree pour trouver les paires proches
        tree = KDTree(coords.values)
        pairs = tree.query_pairs(distance_threshold, output_type='ndarray')
        
        if len(pairs) == 0:
            return {'status': 'success', 'message': f"Aucun doublon trouvé à moins de {distance_threshold} mètres", 'data': None}
        
        # Créer un DataFrame avec les détails des paires trouvées
        result_rows = []
        for i, j in pairs:
            i, j = int(i), int(j)
            point1 = coords.iloc[i].values
            point2 = coords.iloc[j].values
            distance = np.linalg.norm(point1 - point2)
            
            # Créer une entrée pour chaque paire
            result_rows.append({
                'index1': coords.index[i],
                'index2': coords.index[j],
                'holeid1': data.iloc[i]['holeid'] if 'holeid' in data.columns else f"ID_{i}",
                'holeid2': data.iloc[j]['holeid'] if 'holeid' in data.columns else f"ID_{j}",
                'x1': point1[0],
                'y1': point1[1],
                'z1': point1[2],
                'x2': point2[0],
                'y2': point2[1],
                'z2': point2[2],
                'distance': distance
            })
        
        result_df = pd.DataFrame(result_rows)
        
        # Trier par distance
        result_df.sort_values('distance', inplace=True)
        
        return {
            'status': 'warning',
            'message': f"Détection de {len(result_df)} paires d'échantillons composites proches (< {distance_threshold} m)",
            'data': result_df
        }
    
    except Exception as e:
        return {'status': 'error', 'message': f"Erreur lors de l'analyse: {str(e)}", 'data': None}

# Fonction pour vérifier les forages manquants
def check_missing_holes(collar_data, other_data, file_type):
    """Vérifie les forages manquants entre datasets"""
    if collar_data is None or other_data is None:
        return {'status': 'info', 'message': "Données manquantes pour l'analyse", 'data': None}
    
    collar_ids = set(collar_data['holeid'].unique())
    other_ids = set(other_data['holeid'].unique())
    
    # Forages dans collar mais absents dans l'autre fichier
    missing_in_other = collar_ids - other_ids
    
    # Forages dans l'autre fichier mais absents dans collar
    extra_in_other = other_ids - collar_ids
    
    result = {
        'missing_in_other': list(missing_in_other),
        'extra_in_other': list(extra_in_other)
    }
    
    # Déterminer le statut
    if missing_in_other and extra_in_other:
        status = 'error'
    elif missing_in_other or extra_in_other:
        status = 'warning'
    else:
        status = 'success'
    
    return {
        'status': status,
        'message': f"Analyse des forages entre collars et {file_type}",
        'data': result
    }

# Fonction pour vérifier les doublons
def check_duplicates(data, file_type):
    """Vérifie les doublons dans un dataset"""
    if data is None:
        return {'status': 'info', 'message': "Données manquantes pour l'analyse", 'data': None}
    
    result = None
    
    if file_type == 'collars':
        # Vérifier les doublons de forages dans collars
        duplicates = data[data.duplicated(subset=['holeid'], keep=False)]
        if not duplicates.empty:
            status = 'error'
            message = f"Doublons de forages détectés dans collars: {len(duplicates)} enregistrements"
        else:
            status = 'success'
            message = "Aucun doublon de forage détecté dans collars"
        result = duplicates
    
    elif file_type in ['assays', 'litho', 'density', 'oxidation', 'geometallurgy']:
        # Vérifier les doublons d'intervalles
        if all(col in data.columns for col in ['holeid', 'from', 'to']):
            duplicates = data[data.duplicated(subset=['holeid', 'from', 'to'], keep=False)]
            if not duplicates.empty:
                status = 'error'
                message = f"Doublons d'intervalles détectés dans {file_type}: {len(duplicates)} enregistrements"
            else:
                status = 'success'
                message = f"Aucun doublon d'intervalle détecté dans {file_type}"
            result = duplicates
        else:
            status = 'warning'
            message = f"Impossible de vérifier les doublons d'intervalles dans {file_type}: colonnes requises manquantes"
    
    elif file_type == 'survey':
        # Vérifier les doublons dans les mesures de survey
        if all(col in data.columns for col in ['holeid', 'depth']):
            duplicates = data[data.duplicated(subset=['holeid', 'depth'], keep=False)]
            if not duplicates.empty:
                status = 'error'
                message = f"Doublons de mesures détectés dans {file_type}: {len(duplicates)} enregistrements"
            else:
                status = 'success'
                message = f"Aucun doublon de mesure détecté dans {file_type}"
            result = duplicates
        else:
            status = 'warning'
            message = f"Impossible de vérifier les doublons dans {file_type}: colonnes requises manquantes"
    
    # Vérifier les doublons d'échantillons si la colonne sampleid existe
    if 'sampleid' in data.columns:
        sample_duplicates = data[data.duplicated(subset=['sampleid'], keep=False)]
        if not sample_duplicates.empty:
            status = 'error'
            message += f" | Doublons d'identifiants d'échantillons détectés: {len(sample_duplicates)} enregistrements"
            if result is None:
                result = sample_duplicates
            else:
                result = pd.concat([result, sample_duplicates]).drop_duplicates()
    
    return {
        'status': status,
        'message': message,
        'data': result
    }

# Fonction pour vérifier les intervalles de profondeur
def check_interval_depths(collar_data, interval_data, file_type, tolerance=0.1):
    """Vérifie les intervalles de profondeur par rapport aux profondeurs maximales des forages"""
    if collar_data is None or interval_data is None:
        return {'status': 'info', 'message': "Données manquantes pour l'analyse", 'data': None}
    
    if 'depth' not in collar_data.columns:
        return {'status': 'error', 'message': "Colonne 'depth' manquante dans les données de collars", 'data': None}
    
    if not all(col in interval_data.columns for col in ['holeid', 'from', 'to']):
        return {'status': 'error', 'message': "Colonnes requises ('holeid', 'from', 'to') manquantes dans les données d'intervalle", 'data': None}
    
    # Créer un dictionnaire des profondeurs maximales pour chaque forage
    max_depths = dict(zip(collar_data['holeid'], collar_data['depth']))
    
    # Vérifier chaque intervalle
    issues = []
    for idx, row in interval_data.iterrows():
        hole_id = row['holeid']
        from_depth = row['from']
        to_depth = row['to']
        
        # Vérifier si le forage existe dans collar
        if hole_id in max_depths:
            max_depth = max_depths[hole_id]
            
            # Vérifier si les profondeurs sont valides
            if to_depth > max_depth + tolerance:
                issues.append({
                    'index': idx,
                    'holeid': hole_id,
                    'from': from_depth,
                    'to': to_depth,
                    'max_depth': max_depth,
                    'issue': f"La profondeur 'to' ({to_depth}) dépasse la profondeur maximale du forage ({max_depth})"
                })
            
            if from_depth > to_depth:
                issues.append({
                    'index': idx,
                    'holeid': hole_id,
                    'from': from_depth,
                    'to': to_depth,
                    'max_depth': max_depth,
                    'issue': f"La profondeur 'from' ({from_depth}) est supérieure à 'to' ({to_depth})"
                })
    
    results_df = pd.DataFrame(issues) if issues else pd.DataFrame()
    
    if len(issues) > 0:
        status = 'error'
        message = f"Problèmes d'intervalles détectés dans {file_type}: {len(issues)} intervalles problématiques"
    else:
        status = 'success'
        message = f"Aucun problème d'intervalle détecté dans {file_type}"
    
    return {
        'status': status,
        'message': message,
        'data': results_df
    }

# Interface utilisateur avec onglets
import_tab, validate_tab, report_tab = st.tabs(["Import de Données", "Validation", "Rapport"])

with import_tab:
    st.header("Import des fichiers de données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fichier des collars (obligatoire)")
        collar_file = st.file_uploader("Sélectionner le fichier collars (CSV, Excel)", type=["csv", "xlsx", "xls"], key="collar_uploader")
        if collar_file is not None:
            collar_data = load_data(collar_file, 'collars')
            if collar_data is not None:
                st.session_state.files['collars'] = collar_data
                st.dataframe(collar_data.head())
                st.info(f"Total des forages: {len(collar_data)}")
    
    with col2:
        st.subheader("Fichier des surveys")
        survey_file = st.file_uploader("Sélectionner le fichier survey (CSV, Excel)", type=["csv", "xlsx", "xls"], key="survey_uploader")
        if survey_file is not None:
            survey_data = load_data(survey_file, 'survey')
            if survey_data is not None:
                st.session_state.files['survey'] = survey_data
                st.dataframe(survey_data.head())
                st.info(f"Total des enregistrements survey: {len(survey_data)}")
    
    st.subheader("Données d'intervalles et d'échantillons")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        assay_file = st.file_uploader("Fichier d'analyses (assays)", type=["csv", "xlsx", "xls"], key="assay_uploader")
        if assay_file is not None:
            assay_data = load_data(assay_file, 'assays')
            if assay_data is not None:
                st.session_state.files['assays'] = assay_data
                st.dataframe(assay_data.head())
                st.info(f"Total des échantillons d'analyses: {len(assay_data)}")
    
    with col2:
        litho_file = st.file_uploader("Fichier de lithologie", type=["csv", "xlsx", "xls"], key="litho_uploader")
        if litho_file is not None:
            litho_data = load_data(litho_file, 'litho')
            if litho_data is not None:
                st.session_state.files['litho'] = litho_data
                st.dataframe(litho_data.head())
                st.info(f"Total des intervalles de lithologie: {len(litho_data)}")
    
    with col3:
        density_file = st.file_uploader("Fichier de densité", type=["csv", "xlsx", "xls"], key="density_uploader")
        if density_file is not None:
            density_data = load_data(density_file, 'density')
            if density_data is not None:
                st.session_state.files['density'] = density_data
                st.dataframe(density_data.head())
                st.info(f"Total des mesures de densité: {len(density_data)}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        oxidation_file = st.file_uploader("Fichier d'oxydation", type=["csv", "xlsx", "xls"], key="oxidation_uploader")
        if oxidation_file is not None:
            oxidation_data = load_data(oxidation_file, 'oxidation')
            if oxidation_data is not None:
                st.session_state.files['oxidation'] = oxidation_data
                st.dataframe(oxidation_data.head())
                st.info(f"Total des intervalles d'oxydation: {len(oxidation_data)}")
    
    with col2:
        geomet_file = st.file_uploader("Fichier de géométallurgie", type=["csv", "xlsx", "xls"], key="geomet_uploader")
        if geomet_file is not None:
            geomet_data = load_data(geomet_file, 'geometallurgy')
            if geomet_data is not None:
                st.session_state.files['geometallurgy'] = geomet_data
                st.dataframe(geomet_data.head())
                st.info(f"Total des intervalles géométallurgiques: {len(geomet_data)}")
    
    with col3:
        composite_file = st.file_uploader("Fichier de composites", type=["csv", "xlsx", "xls"], key="composite_uploader")
        if composite_file is not None:
            composite_data = load_data(composite_file, 'composites')
            if composite_data is not None:
                st.session_state.files['composites'] = composite_data
                st.dataframe(composite_data.head())
                st.info(f"Total des composites: {len(composite_data)}")

with validate_tab:
    st.header("Validation des données")
    
    # Paramètres de validation dans la barre latérale
    st.sidebar.header("Paramètres de validation")
    depth_tolerance = st.sidebar.number_input("Tolérance de profondeur (mètres)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    distance_threshold = st.sidebar.number_input("Seuil de distance pour composites proches (mètres)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    
    # Vérifier si le fichier collars est présent
    if 'collars' not in st.session_state.files:
        st.warning("Veuillez d'abord importer le fichier des collars dans l'onglet 'Import de données'.")
    else:
        if st.button("🔍 Lancer la validation complète", use_container_width=True):
            st.session_state.validation_results = {}
            st.session_state.report_data = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'files_analyzed': list(st.session_state.files.keys()),
                'validation_results': {},
                'summary': {}
            }
            
            with st.spinner("Validation en cours..."):
                collar_data = st.session_state.files['collars']
                
                # Analyse 1: Doublons dans les fichiers
                st.subheader("Vérification des doublons")
                
                for file_type, file_data in st.session_state.files.items():
                    result = check_duplicates(file_data, file_type)
                    st.session_state.validation_results[f'duplicates_{file_type}'] = result
                    display_status_line(result['message'], result['status'])
                    
                    # Ajouter au rapport
                    st.session_state.report_data['validation_results'][f'duplicates_{file_type}'] = {
                        'status': result['status'],
                        'message': result['message'],
                        'count': len(result['data']) if result['data'] is not None else 0
                    }
                
                # Analyse 2: Vérification des forages manquants
                st.subheader("Vérification des forages manquants")
                
                for file_type, file_data in st.session_state.files.items():
                    if file_type != 'collars':
                        result = check_missing_holes(collar_data, file_data, file_type)
                        st.session_state.validation_results[f'missing_holes_{file_type}'] = result
                        
                        if result['status'] == 'success':
                            st.success(f"Tous les forages dans {file_type} correspondent aux collars")
                        else:
                            if result['data']['missing_in_other']:
                                st.warning(f"{len(result['data']['missing_in_other'])} forages dans collars sont absents de {file_type}")
                            
                            if result['data']['extra_in_other']:
                                st.error(f"{len(result['data']['extra_in_other'])} forages dans {file_type} sont absents de collars")
                        
                        # Ajouter au rapport
                        st.session_state.report_data['validation_results'][f'missing_holes_{file_type}'] = {
                            'status': result['status'],
                            'message': result['message'],
                            'missing_count': len(result['data']['missing_in_other']) if result['data'] else 0,
                            'extra_count': len(result['data']['extra_in_other']) if result['data'] else 0
                        }
                
                # Analyse 3: Vérification des intervalles de profondeur
                st.subheader("Vérification des intervalles de profondeur")
                
                interval_types = ['assays', 'litho', 'density', 'oxidation', 'geometallurgy', 'composites']
                
                for file_type in interval_types:
                    if file_type in st.session_state.files:
                        result = check_interval_depths(collar_data, st.session_state.files[file_type], file_type, depth_tolerance)
                        st.session_state.validation_results[f'interval_depths_{file_type}'] = result
                        
                        display_status_line(result['message'], result['status'])
                        
                        if result['status'] == 'error':
                            with st.expander("Voir les détails des problèmes d'intervalles"):
                                st.dataframe(result['data'])
                        
                        # Ajouter au rapport
                        st.session_state.report_data['validation_results'][f'interval_depths_{file_type}'] = {
                            'status': result['status'],
                            'message': result['message'],
                            'count': len(result['data']) if result['data'] is not None else 0
                        }
                
                # Analyse 4: Vérification des composites proches
                if 'composites' in st.session_state.files:
                    st.subheader("Vérification des composites proches")
                    
                    result = check_composite_duplicates(st.session_state.files['composites'], distance_threshold)
                    st.session_state.validation_results['composite_duplicates'] = result
                    
                    display_status_line(result['message'], result['status'])
                    
                    if result['status'] == 'warning' and result['data'] is not None:
                        with st.expander("Voir les détails des composites proches"):
                            st.dataframe(result['data'])
                    
                    # Ajouter au rapport
                    st.session_state.report_data['validation_results']['composite_duplicates'] = {
                        'status': result['status'],
                        'message': result['message'],
                        'count': len(result['data']) if result['data'] is not None else 0
                    }
                
                # Préparation du résumé de validation
                success_count = sum(1 for result in st.session_state.validation_results.values() if result['status'] == 'success')
                warning_count = sum(1 for result in st.session_state.validation_results.values() if result['status'] == 'warning')
                error_count = sum(1 for result in st.session_state.validation_results.values() if result['status'] == 'error')
                
                st.session_state.report_data['summary'] = {
                    'total_validations': len(st.session_state.validation_results),
                    'success_count': success_count,
                    'warning_count': warning_count,
                    'error_count': error_count
                }
                
                st.success(f"Validation terminée: {success_count} succès, {warning_count} avertissements, {error_count} erreurs")
        
        # Affichage des résultats de validation précédents
        if st.session_state.validation_results:
            st.subheader("Résultats de validation existants")
            
            # Option pour afficher tous les détails
            show_details = st.checkbox("Afficher tous les détails des résultats")
            
            for validation_type, result in st.session_state.validation_results.items():
                display_status_line(result['message'], result['status'])
                
                if show_details and result['data'] is not None:
                    if isinstance(result['data'], pd.DataFrame) and not result['data'].empty:
                        st.dataframe(result['data'])
                    elif isinstance(result['data'], dict):
                        st.write(result['data'])
            
            # Bouton pour exporter les résultats
            st.subheader("Exportation des résultats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Exporter le rapport de validation (CSV)"):
                    # Créer un DataFrame pour le rapport
                    report_rows = []
                    
                    for validation_type, result in st.session_state.validation_results.items():
                        row = {
                            'validation_type': validation_type,
                            'status': result['status'],
                            'message': result['message'],
                            'details': str(result['data']) if result['data'] is not None else 'Aucun détail'
                        }
                        report_rows.append(row)
                    
                    report_df = pd.DataFrame(report_rows)
                    
                    # Générer le lien de téléchargement
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"rapport_validation_{timestamp}.csv"
                    
                    csv_href = download_button(report_df, csv_filename, "Télécharger le rapport CSV")
                    st.markdown(csv_href, unsafe_allow_html=True)
            
            with col2:
                if st.button("Exporter les résultats détaillés (Excel)"):
                    # Créer un fichier Excel avec une feuille pour chaque validation
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # D'abord, écrire un résumé
                        summary_data = {
                            'Type de validation': [],
                            'Statut': [],
                            'Message': [],
                            'Nombre de problèmes': []
                        }
                        
                        for validation_type, result in st.session_state.validation_results.items():
                            summary_data['Type de validation'].append(validation_type)
                            summary_data['Statut'].append(result['status'])
                            summary_data['Message'].append(result['message'])
                            
                            if result['data'] is None:
                                count = 0
                            elif isinstance(result['data'], pd.DataFrame):
                                count = len(result['data'])
                            elif isinstance(result['data'], dict) and 'missing_in_other' in result['data']:
                                count = len(result['data']['missing_in_other']) + len(result['data']['extra_in_other'])
                            else:
                                count = 0
                            
                            summary_data['Nombre de problèmes'].append(count)
                        
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Résumé', index=False)
                        
                        # Ensuite, écrire les détails de chaque validation
                        for validation_type, result in st.session_state.validation_results.items():
                            if result['data'] is not None:
                                if isinstance(result['data'], pd.DataFrame) and not result['data'].empty:
                                    # Limiter le nom de la feuille à 31 caractères (limite Excel)
                                    sheet_name = validation_type[:31]
                                    result['data'].to_excel(writer, sheet_name=sheet_name, index=False)
                                elif isinstance(result['data'], dict) and ('missing_in_other' in result['data'] or 'extra_in_other' in result['data']):
                                    missing = pd.DataFrame({'HOLEID': result['data'].get('missing_in_other', []), 'Status': 'Missing'})
                                    extra = pd.DataFrame({'HOLEID': result['data'].get('extra_in_other', []), 'Status': 'Extra'})
                                    pd.concat([missing, extra]).to_excel(writer, sheet_name=validation_type[:31], index=False)
                    
                    # Générer le lien de téléchargement
                    output.seek(0)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    excel_filename = f"resultats_validation_detailles_{timestamp}.xlsx"
                    
                    b64 = base64.b64encode(output.getvalue()).decode()
                    excel_href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{excel_filename}">Télécharger les résultats Excel</a>'
                    st.markdown(excel_href, unsafe_allow_html=True)

with report_tab:
    st.header("Rapport de validation")
    
    if not st.session_state.report_data:
        st.info("Veuillez d'abord lancer la validation dans l'onglet 'Validation' pour générer un rapport.")
    else:
        # Récupérer les données du rapport
        report = st.session_state.report_data
        
        # En-tête du rapport
        st.subheader("Résumé de la validation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Validations totales", report['summary']['total_validations'])
        
        with col2:
            st.metric("Succès", report['summary']['success_count'], delta=f"{report['summary']['success_count']/report['summary']['total_validations']*100:.0f}%")
        
        with col3:
            st.metric("Avertissements", report['summary']['warning_count'], delta=f"{report['summary']['warning_count']/report['summary']['total_validations']*100:.0f}%")
        
        with col4:
            st.metric("Erreurs", report['summary']['error_count'], delta=f"{report['summary']['error_count']/report['summary']['total_validations']*100:.0f}%")
        
        # Statut global
        if report['summary']['error_count'] > 0:
            st.error("⚠️ Le jeu de données présente des erreurs importantes qui doivent être corrigées.")
        elif report['summary']['warning_count'] > 0:
            st.warning("⚠️ Le jeu de données présente quelques problèmes mineurs qui pourraient nécessiter votre attention.")
        else:
            st.success("✅ Le jeu de données a passé toutes les validations avec succès!")
        
        # Résultats détaillés sous forme de tableau
        st.subheader("Détails des validations")
        
        # Créer un DataFrame pour le rapport
        report_rows = []
        
        for validation_type, result in report['validation_results'].items():
            row = {
                'Type de validation': validation_type.replace('_', ' ').title(),
                'Statut': result['status'].upper(),
                'Message': result['message'],
                'Problèmes détectés': result.get('count', 0) + result.get('missing_count', 0) + result.get('extra_count', 0)
            }
            report_rows.append(row)
        
        report_df = pd.DataFrame(report_rows)
        st.dataframe(report_df, use_container_width=True)
        
        # Visualisation des statistiques
        st.subheader("Visualisation des résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique des statuts
            status_counts = {
                'Succès': report['summary']['success_count'],
                'Avertissements': report['summary']['warning_count'],
                'Erreurs': report['summary']['error_count']
            }
            
            status_df = pd.DataFrame({
                'Statut': list(status_counts.keys()),
                'Nombre': list(status_counts.values())
            })
            
            fig = px.pie(status_df, values='Nombre', names='Statut', 
                         title='Répartition des statuts de validation',
                         color='Statut',
                         color_discrete_map={
                             'Succès': '#5cb85c',
                             'Avertissements': '#f0ad4e',
                             'Erreurs': '#d9534f'
                         })
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graphique des problèmes par type de validation
            problems_df = report_df.sort_values('Problèmes détectés', ascending=False)
            
            fig = px.bar(problems_df, x='Type de validation', y='Problèmes détectés',
                         title='Nombre de problèmes par type de validation',
                         color='Statut',
                         color_discrete_map={
                             'SUCCESS': '#5cb85c',
                             'WARNING': '#f0ad4e',
                             'ERROR': '#d9534f'
                         })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des profondeurs de forage si disponible
        if 'collars' in st.session_state.files and 'depth' in st.session_state.files['collars'].columns:
            st.subheader("Distribution des profondeurs de forage")
            
            collar_depths = st.session_state.files['collars']['depth'].dropna()
            
            fig = px.histogram(collar_depths, nbins=30,
                              labels={'value': 'Profondeur (m)', 'count': 'Nombre de forages'},
                              title='Distribution des profondeurs de forage')
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        # Afficher la distribution spatiale des forages si les coordonnées sont disponibles
        if 'collars' in st.session_state.files and all(col in st.session_state.files['collars'].columns for col in ['x', 'y']):
            st.subheader("Distribution spatiale des forages")
            
            collars = st.session_state.files['collars']
            
            fig = px.scatter(collars, x='x', y='y', 
                            hover_name='holeid' if 'holeid' in collars.columns else None,
                            hover_data=['depth'] if 'depth' in collars.columns else None,
                            color='depth' if 'depth' in collars.columns else None,
                            title='Carte des emplacements de forage')
            
            fig.update_layout(
                autosize=True,
                height=600,
                margin=dict(l=50, r=50, b=100, t=100, pad=4),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Exportation du rapport complet
        st.subheader("Exporter le rapport complet")
        
        if st.button("Générer un rapport complet (HTML)"):
            # Créer le contenu HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Rapport de Validation des Données de Forage Minier</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 30px; }}
                    h1, h2, h3 {{ margin-top: 30px; }}
                    .summary {{ display: flex; justify-content: space-between; margin: 20px 0; }}
                    .summary-card {{ padding: 15px; border-radius: 5px; width: 22%; text-align: center; }}
                    .success {{ background-color: #dff0d8; color: #3c763d; }}
                    .warning {{ background-color: #fcf8e3; color: #8a6d3b; }}
                    .error {{ background-color: #f2dede; color: #a94442; }}
                    .info {{ background-color: #d9edf7; color: #31708f; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .status-badge {{ padding: 5px 10px; border-radius: 3px; font-weight: bold; }}
                    .footer {{ margin-top: 50px; text-align: center; color: #777; font-size: 14px; }}
                </style>
            </head>
            <body>
                <header>
                    <h1>Rapport de Validation des Données de Forage Minier</h1>
                    <p>Date de génération: {report['timestamp']}</p>
                </header>
                
                <div class="container">
                    <h2>Résumé de la validation</h2>
                    
                    <div class="summary">
                        <div class="summary-card info">
                            <h3>{report['summary']['total_validations']}</h3>
                            <p>Validations totales</p>
                        </div>
                        <div class="summary-card success">
                            <h3>{report['summary']['success_count']}</h3>
                            <p>Succès</p>
                        </div>
                        <div class="summary-card warning">
                            <h3>{report['summary']['warning_count']}</h3>
                            <p>Avertissements</p>
                        </div>
                        <div class="summary-card error">
                            <h3>{report['summary']['error_count']}</h3>
                            <p>Erreurs</p>
                        </div>
                    </div>
                    
                    <h2>Fichiers analysés</h2>
                    <ul>
            """
            
            for file in report['files_analyzed']:
                html_content += f"<li>{file}</li>"
            
            html_content += """
                    </ul>
                    
                    <h2>Détails des validations</h2>
                    <table>
                        <tr>
                            <th>Type de validation</th>
                            <th>Statut</th>
                            <th>Message</th>
                            <th>Problèmes détectés</th>
                        </tr>
            """
            
            for validation_type, result in report['validation_results'].items():
                status_class = result['status']
                problem_count = result.get('count', 0) + result.get('missing_count', 0) + result.get('extra_count', 0)
                
                html_content += f"""
                        <tr>
                            <td>{validation_type.replace('_', ' ').title()}</td>
                            <td><span class="status-badge {status_class}">{result['status'].upper()}</span></td>
                            <td>{result['message']}</td>
                            <td>{problem_count}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                    
                    <h2>Conclusion</h2>
            """
            
            if report['summary']['error_count'] > 0:
                html_content += """
                    <div class="error" style="padding: 15px; border-radius: 5px;">
                        <p>⚠️ Le jeu de données présente des erreurs importantes qui doivent être corrigées.</p>
                    </div>
                """
            elif report['summary']['warning_count'] > 0:
                html_content += """
                    <div class="warning" style="padding: 15px; border-radius: 5px;">
                        <p>⚠️ Le jeu de données présente quelques problèmes mineurs qui pourraient nécessiter votre attention.</p>
                    </div>
                """
            else:
                html_content += """
                    <div class="success" style="padding: 15px; border-radius: 5px;">
                        <p>✅ Le jeu de données a passé toutes les validations avec succès!</p>
                    </div>
                """
            
            html_content += """
                    <div class="footer">
                        <p>Généré par l'application de Validation des Données de Forage Minier</p>
                        <p>Développé par Didier Ouedraogo, P.Geo</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Générer le lien de téléchargement
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"rapport_validation_complet_{timestamp}.html"
            
            b64 = base64.b64encode(html_content.encode()).decode()
            html_href = f'<a href="data:text/html;base64,{b64}" download="{html_filename}">Télécharger le rapport HTML complet</a>'
            st.markdown(html_href, unsafe_allow_html=True)

# Pied de page
st.markdown("""
<div class="footer">
    <p>© 2025 Didier Ouedraogo, P.Geo | Application de validation de données de forage minier</p>
</div>
""", unsafe_allow_html=True)