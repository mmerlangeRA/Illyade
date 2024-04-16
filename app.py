#./perso/nocode/unsupervised/2024_04_12/venv/bin/python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_ANO_per_Installation(installation,df_aggregated):
    filtered_df = df_aggregated[df_aggregated['Installation']==installation & df_aggregated['Num_ANO'].notna()]
    grouped = filtered_df.groupby('month')['Num_ANO'].agg(list).reset_index()
    # Convert grouped data to dictionary {month: list of Num_ANO}
    result_dict = dict(zip(grouped['month'], grouped['Num_ANO']))
    return result_dict

def plot_maxPA_maxPM_per_date_for_installation(df, installation,saveName=""):
    df_filtered = df[df['Installation'] == installation]
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['month'], df_filtered['maxPA'], marker='o', linestyle='-', label='maxPA')
    plt.plot(df_filtered['month'], df_filtered['maxPS'], marker='x', linestyle='--', label='maxPS')

    # Formatting the plot
    plt.title(f'maxPA and maxPS per Month for Installation {installation}')
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

def plot_centroid(id,months,centroids_original_scale):
    values = centroids_original_scale[id]
    plt.figure(figsize=(10, 6))
    plt.plot(months, values, label='Centroid ', marker='o', linestyle='-', color='blue')

    # Formatting the plot
    plt.title(f'Centroid Values Over Time for centroid {id}')
    plt.ylim(bottom=0) 
    plt.xlabel('Month')
    plt.ylabel('Centroid Value')
    plt.xticks(months, labels=months, rotation=45)  # Set x-ticks labels explicitly if needed
    plt.legend()

    plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels
    st.pyplot(plt)

def plot_PA_per_date_for_installation(df, installation,saveName=""):
    filtered_df = df[df['Installation'] == installation]
    # Get unique 'Cadran' values
    cadran_values = filtered_df['Numero_Cadran_PA'].unique()
    plt.figure(figsize=(12, 8))  # Adjust figure size as needed
    # Loop through each 'Cadran' value and plot
    for cadran in cadran_values:
        subset_df = filtered_df[filtered_df['Numero_Cadran_PA'] == cadran]
        plt.plot(subset_df['month'], subset_df['PA'], marker='o', label=f'Cadran {cadran}')

    plt.title(f'PA per Date for Installation: {installation}')
    plt.xlabel('Month')
    plt.ylabel('PA')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def top_5_common_NAF_and_shares(series, pd_naf):
    # Calculate the top 5 common values as before
    value_counts = series.value_counts(normalize=True)
    top_5 = value_counts.head(5)
    top_5_percentage = top_5.apply(lambda x: "{:.2%}".format(x))
    top_5_df = pd.DataFrame({
        'NAF': top_5.index,
        'Share': top_5_percentage
    })
    top_5_df = top_5_df.merge(pd_naf, left_on='NAF', right_on='code_naf', how='left')
    top_5_df.drop(columns='code_naf', inplace=True)
    return top_5_df.reset_index(drop=True)

def top_5_common_values_and_shares(series):
    value_counts = series.value_counts(normalize=True)
    top_5 = value_counts.head(5)
    top_5_percentage = top_5.apply(lambda x: "{:.2%}".format(x))
    result_df = pd.DataFrame({
        'Value': top_5.index,
        'Share': top_5_percentage
    }).reset_index(drop=True)
    
    return result_df

def get_installations_based_on_cluster_selection(cluster_selection, df_cluster):
    if cluster_selection[0] == 'All':
        # If 'All' is selected, include all installations
        selected_installations = df_cluster['Installation']
    else:
        # Otherwise, filter installations by the selected cluster
        selected_installations = df_cluster[df_cluster['cluster'] == cluster_selection[0]]['Installation']
    return selected_installations

def get_information_from_cluster(pa_df, cluster_id,pd_naf):
    filter = df_cluster[df_cluster['cluster'] == cluster_id]
    installations = filter['Installation'].unique()
    df = pa_df[pa_df['Installation'].isin(installations)]
    df['Reduced_NAF'] = df['Code_NAF'].str.slice(0, 5)

    # Display most common NAF and regions
    code_naf_common = top_5_common_NAF_and_shares(df['Reduced_NAF'],pd_naf)
    region_common = top_5_common_values_and_shares(df['REGION'])

    st.write(f"Most common NAF codes in cluster {cluster_id}:")
    st.dataframe(code_naf_common)
    st.write(f"Most common regions in cluster {cluster_id}:")
    st.dataframe(region_common)
    plot_centroid(cluster_id, filtered_columns, centroids_original_scale)


# Load data
@st.cache_data
def load_data():
    df_aggregated = pd.read_csv('Installation_PA_PS_ANO.csv')
    df_cluster = pd.read_csv('installation_cluster.csv')
    df_pa_ps = pd.read_csv('PA_PS_Installation.csv')
    pivot_outliers = pd.read_csv('pivot_ouliers.csv',index_col="Installation")
    pivot_outliers = pivot_outliers.sort_values(by='MaxDistance', ascending=False)
    centroids_original_scale = np.load('centroids_original_scale.npy')
    pd_naf = pd.read_csv('code_naf.csv')
    print("all loaded")
    return df_aggregated, df_cluster, df_pa_ps, pivot_outliers,centroids_original_scale,pd_naf

df_aggregated, df_cluster, df_pa_ps, pivot_outliers,centroids_original_scale, pd_naf = load_data()

exclude_columns = ['cluster', 'MaxDistColumn', 'SignedMaxDistance', 'MaxDistance','Installation']
filtered_columns = [col for col in pivot_outliers.columns if col not in exclude_columns]


# Header and Summary
st.title('Installation Data Analysis')
unique_installations = len(df_aggregated['Installation'].unique())
unique_clusters = len(df_cluster['cluster'].unique())
st.header(f'Summary')
st.write(f'Number of unique installations: {unique_installations}')
st.write(f'Number of unique clusters: {unique_clusters}')

# Number of installations per cluster
cluster_counts = df_cluster['cluster'].value_counts().sort_values()
cluster_options = [(index, count) for index, count in cluster_counts.items()]
cluster_options.insert(0, ('All', 'All'))  # Insert 'All' option at the beginning
cluster_selection = st.selectbox(
    'Select a cluster',
    options=cluster_options,
    format_func=lambda x: f'Cluster {x[0]}: {x[1]} installations' if x[0] != 'All' else 'All clusters'
)
# Filter data based on cluster selection
selected_cluster_installations = get_installations_based_on_cluster_selection(cluster_selection, df_cluster)


# Call the function based on user input
if st.button('Show Cluster Info'):
    get_information_from_cluster(df_aggregated, cluster_selection[0],pd_naf)

distance_filter = st.radio(
    "Filter installations by SignedMaxDistance:",
    ('Positive', 'Negative', 'Both')
)

def filter_installations_by_distance(filter_choice, cluster_installations, pivot_data):
    # Filter based on SignedMaxDistance
    if filter_choice == 'Positive':
        condition = pivot_data['SignedMaxDistance'] > 0
    elif filter_choice == 'Negative':
        condition = pivot_data['SignedMaxDistance'] < 0
    else:
        condition = pd.Series(True, index=pivot_data.index)  # All rows are valid if "Both" is selected

    # Apply filter condition
    filtered_installations = cluster_installations[cluster_installations.isin(pivot_data[condition].index)]

    # Sort these filtered installations by 'MaxDistance' descending
    if not filtered_installations.empty:
        # Get the 'MaxDistance' for these installations from pivot_data
        max_distances = pivot_data.loc[filtered_installations, 'MaxDistance']
        # Sort by 'MaxDistance' in descending order
        filtered_installations = max_distances.sort_values(ascending=False).index

    return filtered_installations



# Assuming selected_cluster_installations is already filtered to the selected cluster
selected_cluster_installations = filter_installations_by_distance(
    distance_filter, selected_cluster_installations, pivot_outliers
)
# Dropdown for installations within the selected cluster
dropdown_options = {}
for installation in selected_cluster_installations.unique():
    if installation in pivot_outliers.index:
        max_distance = pivot_outliers.loc[installation, 'SignedMaxDistance']
        # Format the string as "Installation - SignedMaxDistance"
        option_label = f"{installation} - {max_distance}"
        dropdown_options[option_label] = installation

# Create the dropdown in Streamlit
selected_label = st.selectbox('Select an Installation', options=list(dropdown_options.keys()))
# Retrieve the actual installation from the dictionary using the selected label

# Check if an installation is selected and a button is pressed
if selected_label and st.button('Show Installation Details'):
    installation = dropdown_options[selected_label]
    # Display basic installation info
    print(f'installation {installation}')
    details = df_aggregated[df_aggregated["Installation"] == installation].iloc[0]
    libelle_row = pd_naf[pd_naf['code_naf']==details['Code_NAF']]
    details["Libelle"]=pd_naf[pd_naf['code_naf']==details['Code_NAF']]['libelle_naf'].values[0]
    details['cluster']=df_cluster[df_cluster["Installation"]==installation]['cluster'].values[0]
    st.dataframe(details.to_frame().T) 
    installation_cluster = pivot_outliers.loc[installation, 'cluster']
    installation_month = pivot_outliers.loc[installation, 'MaxDistColumn']
    st.write(f'Displaying Installation {installation} for month {installation_month} in cluster {installation_cluster}')
    st.write(f"Found anomalies  {get_ANO_per_Installation(installation,df_aggregated)}")
    # Plot functions
    plot_maxPA_maxPM_per_date_for_installation(df_aggregated, installation)
    plot_centroid(installation_cluster, filtered_columns, centroids_original_scale)
    plot_PA_per_date_for_installation(df_pa_ps, installation)
