import pandas as pd
import pickle
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
from PIL import Image


# Load the CSV file into a DataFrame
OG_predictive_df = pd.read_csv('data\predictive_df.csv')
OG_filtered_df = pd.read_csv('data\\filtered_df.csv')
OG_target_column = OG_filtered_df['ICU']

# Loading the correlations dictionary from the file
with open('data//correlations.pkl', 'rb') as file:
    correlations = pickle.load(file)

# Step 2: Open the CSV file back into a DataFrame
feature_info = pd.read_csv("data/feature_info.csv")

# Defining the categories for data

# patient_general = ["Codice identificativo:", 'Sesso', 'Nazionalità', 'Età', 'Domicilio','Altezza','Peso', 'Sub-Saharan Africa', 'Pregressa malaria', 'Profilassi']
# comorbidities = ['Comorbilità', 'Diabete', 'HIV', 'Cirrosi', 'IRC']
# patient_state_arrival = ['GCS', 'Seizures', 'Prostrazione', 'Shock', 'Bleeding', 'ARDS', 'Anemia', 'Creatinina', 'Glicemia', 'Acidosi', 'Bilirubina', 'Hyperparasitaemia', 'Numero criteri']
# diagnosis_type = ['RDT', 'Emoscopia', 'NAAT']
# diagnosis_result = ['Falciparum', 'Ovale', 'Parassitemia valore assoluto', 'Percentuale parassitemia']
# baseline_other = ['PA sistolica', 'PA diastolica', 'FC', 'FR', 'Temperatura', 'Ritardo terapeutico']
# QTc = ['QTc ingresso', 'QTc dopo ACT', 'QTC dopo ultima somministrazione artesunato *']
# T0 =  ["T0. [GB (in cell/ul):]", "T0. [GR (in cell/ul):]", "T0. [Hb (in g/dl):]", "T0. [PLT (in cell/ul):]", "T0. [Glicemia (in mg/dl):]", "T0. [Azotemia (in mg/dl):]", "T0. [Creatinina (in mg/dl):]", "T0. [LDH (in U/L):]", "T0. [AST (in U/L):]", "T0. [ALT (in U/L):]", "T0. [Bilirubina tot (in mg/dl):]", "T0. [Bilirubina diretta (in mg/dl):]", "T0. [Sodio (in mEq/l):]", "T0. [Potassio (in mEq/l):]", "T0. [Ca (in mg/dl):]", "T0. [INR:]", "T0. [fibrinogeno (in mg/dl):]", "T0. [pH:]", "T0. [bicarbonati (in mmol/l):]", "T0. [Lattati # (in mmol/l):"]
# T1 = ["T1. [Goccia spessa e striscio periferico:]", "T1. [TC (temperatura corporea) in °C:]"]
# T2 = ["T2. [Goccia spessa e striscio periferico:]", "T2. [TC (temperatura corporea) in °C:]"]
# T3 = ["T3. [Goccia spessa e striscio periferico:]", "T3. [TC (temperatura corporea) in °C:]"]
# T7 = [ "T7. [Goccia spessa e striscio periferico:]","T7. [TC (temperatura corporea) in °C:]"]
# treatment = ["Artesunato ev: Somministrazione [1][Data inizio e ora]", "Artemether/ Lumefantrina: Somministrazione [1][Data inizio e ora]", "Diidroartemisinina/Piperachina: Somministrazione [1][Data inizio e ora]", "ACT", "Artesunato + ACT", "Atovaquone/Proguanile", "Doxiciclina per os", "Clindamicina", "Antibiotici", "Chinino", "Primachina", "Durata Artesunato"]
# outcome = ["Durata ricovero", "Decesso.", "ICU", "Eventuali sequele:", "PADH, post-artesunate delayed haemoly1s", "Insorgenza PADH", "Permanenza in Terapia Intensiva (giorni):", "Trasferimento in Rianimazione (anche in altro centro)?", "Guarigione"]
# PADH_info = ["Insorgenza PADH", "Si prega di fornire tutti i parameri vitali. [GB (in cell/ul):]", "Si prega di fornire tutti i parameri vitali. [Hb (in g/dl):]", "Si prega di fornire tutti i parameri vitali. [PLT (in cell/ul):]", "Si prega di fornire tutti i parameri vitali. [reticoliti (in cell/ul):]", "Si prega di fornire tutti i parameri vitali. [LDH (in U/l)lcio:]", "Si prega di fornire tutti i parameri vitali. [AST (U/L):]", "Si prega di fornire tutti i parameri vitali. [ALT (U/L):]", "Si prega di fornire tutti i parameri vitali. [Bilirubina tot (mg/dl):]", "Si prega di fornire tutti i parameri vitali. [Bilirubina diretta (mg/dl):]", "Si prega di fornire tutti i parameri vitali. [aptoglobina (in mg/dl):]", "Si prega di fornire tutti i parameri vitali. [test di Coombs diretto:]", "Si prega di fornire tutti i parameri vitali. [test di coombs indiretto:]", "Vuole riportare ulteriori informazioni relative ai prelievi effettuati durante le visite intermedie fino alla risoluzione dell’emolisi?", "Nadir Hb", "Trasfusione:", "Unità trasfuse"]
# follow_up = ["Altri eventi avversi", "Diarrea e disidratazione", "Ipertransaminasemia tardiva", "Polmonite", "IVU nosocomiale", "Esofagite", "Dispepsia", "Tachiaritmia sopraventricolare"]


# # Define the category lists
# categories = [patient_general, comorbidities, patient_state_arrival, diagnosis_type, diagnosis_result, baseline_other, QTc, T0, T1, T2, T3, T7, treatment, outcome, PADH_info, follow_up]
# category_names = ['Patient General', 'Comorbidities', 'Patient State Arrival', 'Diagnosis Type', 'Diagnosis Result', 'Baseline Other', 'QTc', 'T0', 'T1', 'T2', 'T3', 'T7', 'Treatment', 'Outcome', 'PADH Info', 'Follow-up']


# # From now on we will drop the "Codice identificativo" because it's string data with no info.
# # So we make an updated version of Patient_general for prediciton
# patient_general_noID = ['Sesso', 'Nazionalità', 'Età', 'Domicilio','Altezza','Peso', 'Sub-Saharan Africa', 'Pregressa malaria', 'Profilassi']

# # And we will only keep the features that do not introduce "bias" in our graphe
# predictive_categories = [patient_general_noID, comorbidities, patient_state_arrival, diagnosis_result, baseline_other, T0, treatment]
# ////

# # Create a new dataframe to store the results
# feature_info = pd.DataFrame(columns=['feature', 'type', 'mean', 'variance'])

# # Iterate over the columns in the filtered dataframe
# for column in filtered_df.columns:
#     col_data = filtered_df[column]
#     col_type = ''
#     col_mean = ''
#     col_var_entropy = ''
    
#     # Check if the column has string values
#     if col_data.dtype == object:
#         col_type = 'string'
#     elif set(col_data.dropna().unique()) == {0, 1}:
#         col_type = 'categorical'
#         col_mean = col_data.mean()
#         # col_var_entropy = np.nans
#     elif col_data.dtype == np.int64 or all(pd.isnull(val) or val.is_integer() for val in col_data.dropna().unique()):
#         col_type = 'int'
#         col_mean = col_data.mean()
#         col_var_entropy = col_data.var()
#     elif col_data.dtype == np.float64 or any('.' in str(val) for val in col_data.dropna().unique()):
#         col_type = 'float'
#         col_mean = col_data.mean()
#         col_var_entropy = col_data.var()
#     else:
#         col_type = 'unknown'

#     # Add the results to the new dataframe
#     feature_info = feature_info.append({'feature': column, 'type': col_type, 'mean': col_mean, 'variance': col_var_entropy},
#                                     ignore_index=True)


##########################

# # Iterate over each category
# for category, category_name in zip(categories, category_names):
#     # Filter the features based on the current category
#     features = [feature for feature in filtered_df.columns if feature in category]

#     # Calculate the correlation between each feature and ICU
#     correlations = []
#     for feature in features:
#         if filtered_df[feature].dtype != np.object:  # Check if the feature is not a string
#             correlation = filtered_df[feature].corr(target_column)
#             correlations.append((feature, correlation))

#     # Sort the correlation values in ascending order
#     correlations.sort(key=lambda x: x[1])


# Loading the correlations dictionary from the file
with open('data//correlations.pkl', 'rb') as file:
    correlations = pickle.load(file)

# Function to remove special characters from a string
def remove_special_characters(string):
    return re.sub(r'\W+', '', string)

def preprocess(raw_df) :

    
    # Load the CSV file into a DataFrame
    # OG_predictive_df = pd.read_csv('../data\predictive_df.csv')
    # OG_filtered_df = pd.read_csv('../data\\filtered_df.csv')
    # OG_target_column = OG_filtered_df['ICU']

    # Loading the correlations dictionary from the file
    # with open('../data//correlations.pkl', 'rb') as file:
    #     correlations = pickle.load(file)

    # Step 2: Open the CSV file back into a DataFrame
    feature_info = pd.read_csv("data/feature_info.csv")

    # Defining the categories for data

    patient_general = ["Codice identificativo:", 'Sesso', 'Nazionalità', 'Età', 'Domicilio','Altezza','Peso', 'Sub-Saharan Africa', 'Pregressa malaria', 'Profilassi']
    comorbidities = ['Comorbilità', 'Diabete', 'HIV', 'Cirrosi', 'IRC']
    patient_state_arrival = ['GCS', 'Seizures', 'Prostrazione', 'Shock', 'Bleeding', 'ARDS', 'Anemia', 'Creatinina', 'Glicemia', 'Acidosi', 'Bilirubina', 'Hyperparasitaemia', 'Numero criteri']
    diagnosis_type = ['RDT', 'Emoscopia', 'NAAT']
    diagnosis_result = ['Falciparum', 'Ovale', 'Parassitemia valore assoluto', 'Percentuale parassitemia']
    baseline_other = ['PA sistolica', 'PA diastolica', 'FC', 'FR', 'Temperatura', 'Ritardo terapeutico']
    QTc = ['QTc ingresso', 'QTc dopo ACT', 'QTC dopo ultima somministrazione artesunato *']
    T0 =  ["T0. [GB (in cell/ul):]", "T0. [GR (in cell/ul):]", "T0. [Hb (in g/dl):]", "T0. [PLT (in cell/ul):]", "T0. [Glicemia (in mg/dl):]", "T0. [Azotemia (in mg/dl):]", "T0. [Creatinina (in mg/dl):]", "T0. [LDH (in U/L):]", "T0. [AST (in U/L):]", "T0. [ALT (in U/L):]", "T0. [Bilirubina tot (in mg/dl):]", "T0. [Bilirubina diretta (in mg/dl):]", "T0. [Sodio (in mEq/l):]", "T0. [Potassio (in mEq/l):]", "T0. [Ca (in mg/dl):]", "T0. [INR:]", "T0. [fibrinogeno (in mg/dl):]", "T0. [pH:]", "T0. [bicarbonati (in mmol/l):]", "T0. [Lattati # (in mmol/l):"]
    T1 = ["T1. [Goccia spessa e striscio periferico:]", "T1. [TC (temperatura corporea) in °C:]"]
    T2 = ["T2. [Goccia spessa e striscio periferico:]", "T2. [TC (temperatura corporea) in °C:]"]
    T3 = ["T3. [Goccia spessa e striscio periferico:]", "T3. [TC (temperatura corporea) in °C:]"]
    T7 = [ "T7. [Goccia spessa e striscio periferico:]","T7. [TC (temperatura corporea) in °C:]"]
    treatment = ["Artesunato ev: Somministrazione [1][Data inizio e ora]", "Artemether/ Lumefantrina: Somministrazione [1][Data inizio e ora]", "Diidroartemisinina/Piperachina: Somministrazione [1][Data inizio e ora]", "ACT", "Artesunato + ACT", "Atovaquone/Proguanile", "Doxiciclina per os", "Clindamicina", "Antibiotici", "Chinino", "Primachina", "Durata Artesunato"]
    outcome = ["Durata ricovero", "Decesso.", "ICU", "Eventuali sequele:", "PADH, post-artesunate delayed haemoly1s", "Insorgenza PADH", "Permanenza in Terapia Intensiva (giorni):", "Trasferimento in Rianimazione (anche in altro centro)?", "Guarigione"]
    PADH_info = ["Insorgenza PADH", "Si prega di fornire tutti i parameri vitali. [GB (in cell/ul):]", "Si prega di fornire tutti i parameri vitali. [Hb (in g/dl):]", "Si prega di fornire tutti i parameri vitali. [PLT (in cell/ul):]", "Si prega di fornire tutti i parameri vitali. [reticoliti (in cell/ul):]", "Si prega di fornire tutti i parameri vitali. [LDH (in U/l)lcio:]", "Si prega di fornire tutti i parameri vitali. [AST (U/L):]", "Si prega di fornire tutti i parameri vitali. [ALT (U/L):]", "Si prega di fornire tutti i parameri vitali. [Bilirubina tot (mg/dl):]", "Si prega di fornire tutti i parameri vitali. [Bilirubina diretta (mg/dl):]", "Si prega di fornire tutti i parameri vitali. [aptoglobina (in mg/dl):]", "Si prega di fornire tutti i parameri vitali. [test di Coombs diretto:]", "Si prega di fornire tutti i parameri vitali. [test di coombs indiretto:]", "Vuole riportare ulteriori informazioni relative ai prelievi effettuati durante le visite intermedie fino alla risoluzione dell’emolisi?", "Nadir Hb", "Trasfusione:", "Unità trasfuse"]
    follow_up = ["Altri eventi avversi", "Diarrea e disidratazione", "Ipertransaminasemia tardiva", "Polmonite", "IVU nosocomiale", "Esofagite", "Dispepsia", "Tachiaritmia sopraventricolare"]


    # Define the category lists
    categories = [patient_general, comorbidities, patient_state_arrival, diagnosis_type, diagnosis_result, baseline_other, QTc, T0, T1, T2, T3, T7, treatment, outcome, PADH_info, follow_up]
    category_names = ['Patient General', 'Comorbidities', 'Patient State Arrival', 'Diagnosis Type', 'Diagnosis Result', 'Baseline Other', 'QTc', 'T0', 'T1', 'T2', 'T3', 'T7', 'Treatment', 'Outcome', 'PADH Info', 'Follow-up']
    
    

    # From now on we will drop the "Codice identificativo" because it's string data with no info.
    # So we make an updated version of Patient_general for prediciton
    patient_general_noID = ['Sesso', 'Nazionalità', 'Età', 'Domicilio','Altezza','Peso', 'Sub-Saharan Africa', 'Pregressa malaria', 'Profilassi']

    # Update the list of features for each category
    patient_general = [remove_special_characters(feature) for feature in patient_general]
    comorbidities = [remove_special_characters(feature) for feature in comorbidities]
    patient_state_arrival = [remove_special_characters(feature) for feature in patient_state_arrival]
    diagnosis_type = [remove_special_characters(feature) for feature in diagnosis_type]
    diagnosis_result = [remove_special_characters(feature) for feature in diagnosis_result]
    baseline_other = [remove_special_characters(feature) for feature in baseline_other]
    QTc = [remove_special_characters(feature) for feature in QTc]
    T0 = [remove_special_characters(feature) for feature in T0]
    T1 = [remove_special_characters(feature) for feature in T1]
    T2 = [remove_special_characters(feature) for feature in T2]
    T3 = [remove_special_characters(feature) for feature in T3]
    T7 = [remove_special_characters(feature) for feature in T7]
    treatment = [remove_special_characters(feature) for feature in treatment]
    outcome = [remove_special_characters(feature) for feature in outcome]
    PADH_info = [remove_special_characters(feature) for feature in PADH_info]
    follow_up = [remove_special_characters(feature) for feature in follow_up]

    # And we will only keep the features that do not introduce "bias" in our graphe
    predictive_categories = [patient_general_noID, comorbidities, patient_state_arrival, diagnosis_result, baseline_other, T0, treatment]

    # /!\ Problematic for testing /!\
    # # Remove features with constant values
    # noInfo_columns = raw_df.columns[raw_df.nunique() <= 1] # columns with no values or with only the same value

    # filtered_df = raw_df.loc[:, raw_df.nunique() > 1] # this operation also removes the empty columns
    filtered_df = raw_df
    noInfo_columns = []

    

    # Rename the features in filtered_df
    filtered_df.rename(columns=lambda x: remove_special_characters(x), inplace=True)





    # Removing special caracters from the category description




    

    # Define the categories
    categories = [patient_general, comorbidities, patient_state_arrival, diagnosis_type, diagnosis_result, baseline_other, QTc, T0, T1, T2, T3, T7, treatment, outcome, PADH_info, follow_up]

    # Check if features belong to a category (partial match)
    missing_features = []
    for column in filtered_df.columns:
        matched = False
        for category in categories:
            for feature in category:
                if re.search(re.escape(feature), column, re.IGNORECASE):
                    matched = True
                    break
            if matched:
                break
        if not matched:
            missing_features.append(column)




    # # Get the 'ICU' column from filtered_df
    # target_column = filtered_df['ICU']



    # Select int and float features to normalize
    numeric_features = feature_info[feature_info['type'].isin(['int', 'float'])]['feature'].tolist()

    # Select categorical features to transform 0/1 to -1/1
    categorical_features = feature_info[feature_info['type'] == 'categorical']['feature'].tolist()

    # Normalize numeric features
    scaler = StandardScaler()  # or scaler = MinMaxScaler() for min-max normalization
    filtered_df[numeric_features] = scaler.fit_transform(filtered_df[numeric_features])

    # Loop through each feature in categorical_features
    for feature in categorical_features:   
        if feature in filtered_df.columns:
            # Replace zeros with ones in the selected feature
            filtered_df.loc[filtered_df[feature] == 0, feature] = -1

    # Create an empty list to store selected feature names
    selected_features = []

    # Iterate over predictive_categories
    for category in predictive_categories:
        # Check if any column in filtered_df is present in the current category
        selected_features.extend([feature for feature in filtered_df.columns if feature in category])

    # Create predictive_df DataFrame with selected features
    predictive_df = filtered_df[selected_features]

    return predictive_df



def corrIndex2(path_data) : 
    raw_df = pd.read_excel(path_data)
    input_data_ICU = raw_df['ICU']
    input_data_ICU = np.nan_to_num(input_data_ICU, nan=-1)
    predictive_df = preprocess(raw_df)
    input_data_len = predictive_df.shape[0]

    OG_predictive_df = pd.read_csv('data/corrIndex2Basedf.csv')


    

    # Step 4: Rename the DataFrame columns using the function
    new_columns = {col: remove_special_characters(col) for col in predictive_df.columns}
    predictive_df.rename(columns=new_columns, inplace=True)

    # Step 4: Rename the DataFrame columns using the function
    new_columns = {col: remove_special_characters(col) for col in OG_predictive_df.columns}
    OG_predictive_df.rename(columns=new_columns, inplace=True)

    merged_df = pd.concat([OG_predictive_df, predictive_df], axis=0)
    merged_target = np.concatenate((OG_target_column, input_data_ICU))

    # Create the corrIndex_df DataFrame
    merged_corrIndex_df = pd.DataFrame(columns=["rectified_corr_index", "corr_index", "reachable_max", "uncertainty_value"])
    corrIndex_df = pd.DataFrame(columns=["rectified_corr_index", "corr_index", "reachable_max", "uncertainty_value"])
    OG_corrIndex_df = pd.DataFrame(columns=["rectified_corr_index", "corr_index", "reachable_max", "uncertainty_value"])
    

    abs_max = 0
    for _, correlation in correlations:
        abs_max += abs(correlation)

    # Iterate over each sample (row) in predictive_df
    for _, sample in predictive_df.iterrows():
        corr_index = 0
        reachable_max = 0
        uncertinty_value = 0
        
        # Iterate over each feature and its correlation value
        for feature, correlation in correlations:
            # Check if the value of the sample for this feature is not NaN
            if not pd.isna(sample[feature]):
                corr_index += correlation * sample[feature]
                reachable_max += abs(correlation)
            else :
                uncertinty_value += abs(correlation)
                

        # We are equilibrating with 
        # corrIndex           >>> maxReachable 
        # rectified_corrindex >>> absolutMax
        # So rectified_corrindex = corr_index * abs_max / reachable_max

        # Add a row to corrIndex_df
        corrIndex_df = corrIndex_df.append(
            {"rectified_corr_index": corr_index * abs_max / reachable_max, "corr_index": corr_index, "reachable_max": reachable_max, "uncertainty_value": uncertinty_value},
            ignore_index=True
        )

    # Iterate over each sample (row) in predictive_df
    for _, sample in OG_predictive_df.iterrows():
        corr_index = 0
        reachable_max = 0
        uncertinty_value = 0
        
        # Iterate over each feature and its correlation value
        for feature, correlation in correlations:
            # Check if the value of the sample for this feature is not NaN
            if not pd.isna(sample[feature]):
                corr_index += correlation * sample[feature]
                reachable_max += abs(correlation)
            else :
                uncertinty_value += abs(correlation)
                

        # We are equilibrating with 
        # corrIndex           >>> maxReachable 
        # rectified_corrindex >>> absolutMax
        # So rectified_corrindex = corr_index * abs_max / reachable_max

        # Add a row to corrIndex_df
        OG_corrIndex_df = OG_corrIndex_df.append(
            {"rectified_corr_index": corr_index * abs_max / reachable_max, "corr_index": corr_index, "reachable_max": reachable_max, "uncertainty_value": uncertinty_value},
            ignore_index=True
        )

    # Merged df

    abs_max = 0
    for _, correlation in correlations:
        abs_max += abs(correlation)

    # Iterate over each sample (row) in predictive_df
    for _, sample in merged_df.iterrows():
        corr_index = 0
        reachable_max = 0
        uncertinty_value = 0
        
        # Iterate over each feature and its correlation value
        for feature, correlation in correlations:
            # Check if the value of the sample for this feature is not NaN
            if not pd.isna(sample[feature]):
                corr_index += correlation * sample[feature]
                reachable_max += abs(correlation)
            else :
                uncertinty_value += abs(correlation)
                

        # We are equilibrating with 
        # corrIndex           >>> maxReachable 
        # rectified_corrindex >>> absolutMax
        # So rectified_corrindex = corr_index * abs_max / reachable_max

        # Add a row to corrIndex_df
        merged_corrIndex_df = merged_corrIndex_df.append(
            {"rectified_corr_index": corr_index * abs_max / reachable_max, "corr_index": corr_index, "reachable_max": reachable_max, "uncertainty_value": uncertinty_value},
            ignore_index=True
        )



    # Assuming you have the corrIndex_df and filtered_df DataFrames
    # You can replace 'corrIndex_df' and 'filtered_df' with the actual names of your DataFrames

    # Create a larger figure
    # plt.figure(figsize=(10, 8))


    fig, ax = plt.subplots(figsize=(5,4))
    
    plt.scatter(merged_corrIndex_df["corr_index"], merged_corrIndex_df["uncertainty_value"],
            c=["blue" if icu == -1 else "red" if icu == 1 else "green" for icu in merged_target])

    
    # plt.scatter(OG_corrIndex_df["corr_index"], OG_corrIndex_df["uncertainty_value"],
    #             c=["red" if icu == 1 else "blue" for icu in OG_filtered_df["ICU"]],
    #             label="ICU = 1")

    # Add labels and title to the plot
    plt.xlabel("Rectified Corr_Index")
    plt.ylabel("Uncertainty Value")
    plt.title("Scatter Plot of Corr_Index vs. Uncertainty")

    # Get the index of the last n samples to annotate
    last_n_idx = merged_corrIndex_df.index[-input_data_len:]

    # Annotate the last n samples
    for i, idx in enumerate(last_n_idx):
        plt.annotate(i, (merged_corrIndex_df.loc[idx, "corr_index"], merged_corrIndex_df.loc[idx, "uncertainty_value"]), fontsize=6)

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buffer)
    buffer.seek(0)
    plot_data = buffer.getvalue()

    # Close the plot to release resources
    plt.close(fig)

    # Convert the plot data to a PIL Image object
    img = Image.open(BytesIO(plot_data))

    # Resize the image to fit your desired resolution
    img = img.resize((800, 600), Image.ANTIALIAS)

    # Save the resized image to a BytesIO object
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    resized_plot_data = buffer.getvalue()

    # Return the plot data as base64
    return base64.b64encode(resized_plot_data).decode('utf-8')


    

