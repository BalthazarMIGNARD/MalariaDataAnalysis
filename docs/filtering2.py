
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import re
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle




def filtering(path_data) : 
    raw_df = pd.read_excel(path_data)



    # Remove features with constant values
    noInfo_columns = raw_df.columns[raw_df.nunique() <= 1] # columns with no values or with only the same value

    filtered_df = raw_df.loc[:, raw_df.nunique() > 1] # this operation also removes the empty columns




    # Function to remove special characters from a string
    def remove_special_characters(string):
        return re.sub(r'\W+', '', string)

    # Rename the features in filtered_df
    filtered_df.rename(columns=lambda x: remove_special_characters(x), inplace=True)



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



    # Removing special caracters from the category description


    # Function to remove special characters from a string
    def remove_special_characters(string):
        return re.sub(r'\W+', '', string)

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




    # Check if noInfo_columns are present in categories and remove them
    removed_features = []
    for column in noInfo_columns:
        for i, category in enumerate(categories):
            if column in category:
                categories[i].remove(column)
                removed_features.append(column)
                break





    # Define the category lists
    categories = [patient_general, comorbidities, patient_state_arrival, diagnosis_type, diagnosis_result, baseline_other, QTc, T0, T1, T2, T3, T7, treatment, outcome, PADH_info, follow_up]
    category_names = ['Patient General', 'Comorbidities', 'Patient State Arrival', 'Diagnosis Type', 'Diagnosis Result', 'Baseline Other', 'QTc', 'T0', 'T1', 'T2', 'T3', 'T7', 'Treatment', 'Outcome', 'PADH Info', 'Follow-up']

    # Get the 'ICU' column from filtered_df
    target_column = filtered_df['ICU']

    # Iterate over each category
    for category, category_name in zip(categories, category_names):
        # Filter the features based on the current category
        features = [feature for feature in filtered_df.columns if feature in category]

        # Calculate the correlation between each feature and ICU
        correlations = []
        for feature in features:
            if filtered_df[feature].dtype != np.object:  # Check if the feature is not a string
                correlation = filtered_df[feature].corr(target_column)
                correlations.append((feature, correlation))

        # Sort the correlation values in ascending order
        correlations.sort(key=lambda x: x[1])





    # Create a new dataframe to store the results
    feature_info = pd.DataFrame(columns=['feature', 'type', 'mean', 'variance'])

    # Iterate over the columns in the filtered dataframe
    for column in filtered_df.columns:
        col_data = filtered_df[column]
        col_type = ''
        col_mean = ''
        col_var_entropy = ''
        
        # Check if the column has string values
        if col_data.dtype == object:
            col_type = 'string'
        elif set(col_data.dropna().unique()) == {0, 1}:
            col_type = 'categorical'
            col_mean = col_data.mean()
            # col_var_entropy = np.nans
        elif col_data.dtype == np.int64 or all(pd.isnull(val) or val.is_integer() for val in col_data.dropna().unique()):
            col_type = 'int'
            col_mean = col_data.mean()
            col_var_entropy = col_data.var()
        elif col_data.dtype == np.float64 or any('.' in str(val) for val in col_data.dropna().unique()):
            col_type = 'float'
            col_mean = col_data.mean()
            col_var_entropy = col_data.var()
        else:
            col_type = 'unknown'

        # Add the results to the new dataframe
        feature_info = feature_info.append({'feature': column, 'type': col_type, 'mean': col_mean, 'variance': col_var_entropy},
                                        ignore_index=True)





    # Define the data types for imputation
    data_types = {
        'categorical': np.int64,  # Categorical data type
        'int': np.int64,          # Integer data type
        'float': np.float64       # Float data type
    }

    # Identify the missing values in filtered_df
    missing_values = filtered_df.isnull().sum()

    # Separate features based on their data types
    categorical_features = feature_info[feature_info['type'] == 'categorical']['feature'].tolist()
    int_features = feature_info[feature_info['type'] == 'int']['feature'].tolist()
    float_features = feature_info[feature_info['type'] == 'float']['feature'].tolist()

    # Impute missing values for each data type
    for data_type, features in [('categorical', categorical_features), ('int', int_features), ('float', float_features)]:
        # Filter features based on data type
        features_to_impute = [feature for feature in features if feature in missing_values.index and missing_values[feature] > 0]

        if len(features_to_impute) > 0:
            # Prepare the imputation array
            impute_array = filtered_df[features_to_impute].values

            if data_type in ['categorical', 'int']:
                # Perform imputation for 'categorical' and 'int' features using median strategy
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                imputed_values = imputer.fit_transform(impute_array)
                filtered_df.loc[:, features_to_impute] = np.round(imputed_values).astype(data_types[data_type])
            elif data_type == 'float':
                # Perform imputation for 'float' features using mean strategy
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                imputed_values = imputer.fit_transform(impute_array)
                filtered_df.loc[:, features_to_impute] = imputed_values

    # Verify if any missing values remain in the DataFrame
    missing_values_after_imputation = filtered_df.isnull().sum()
    missing_values_to_print = missing_values_after_imputation[missing_values_after_imputation != 0]
    if not missing_values_to_print.empty:
        print(f"There are still missing values in the DataFrame after imputation:\n{missing_values_to_print}")
    else:
        print("All missing values have been imputed successfully.")




    # Select int and float features to normalize
    numeric_features = feature_info[feature_info['type'].isin(['int', 'float'])]['feature'].tolist()

    # Normalize numeric features
    scaler = StandardScaler()  # or scaler = MinMaxScaler() for min-max normalization
    filtered_df[numeric_features] = scaler.fit_transform(filtered_df[numeric_features])



    # From now on we will drop the "Codice identificativo" because it's string data with no info.
    # So we make an updated version of Patient_general for prediciton
    patient_general_noID = ['Sesso', 'Nazionalità', 'Età', 'Domicilio','Altezza','Peso', 'Sub-Saharan Africa', 'Pregressa malaria', 'Profilassi']

    # And we will only keep the features that do not introduce "bias" in our graphe
    predictive_categories = [patient_general_noID, comorbidities, patient_state_arrival, diagnosis_result, baseline_other, T0, treatment]



    # Create an empty list to store selected feature names
    selected_features = []

    # Iterate over predictive_categories
    for category in predictive_categories:
        # Check if any column in filtered_df is present in the current category
        selected_features.extend([feature for feature in filtered_df.columns if feature in category])

    # Create predictive_df DataFrame with selected features
    predictive_df = filtered_df[selected_features]



    correlations = []

    # Iterate over selected features
    for feature in selected_features:
        # Calculate correlation between the selected feature and "ICU"
        correlation = filtered_df[feature].corr(filtered_df["ICU"])
        correlations.append((feature, correlation))

    # Sort the correlations in descending order based on absolute values
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print the feature/correlation pairs
    for feature, correlation in correlations:
        print(f"Feature: {feature}, Correlation with ICU: {correlation}")




    # Saving the correlations dictionary to a file
    with open('../docsTest/data/correlations.pkl', 'wb') as file:
        pickle.dump(correlations, file)



    filtered_df.to_csv('../docsTest/data/filtered_df.csv', index=False)
    predictive_df.to_csv('../docsTest/data/predictive_df.csv', index=False)

