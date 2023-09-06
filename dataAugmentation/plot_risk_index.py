import numpy as np
import pandas as pd


def corr_risk_uncer(train_df, train_target, test_df, test_target) : 
    correlations = []
    features = train_df.columns

    for feature in features:
        if train_df[feature].dtype != np.object:  # Check if the feature is not a string
            
            # Print the types of feature and train_target
            print("Type of feature:", type(train_df[feature]))
            print("Type of train_target:", type(train_target))
            
            print(train_df[feature])

            print(train_target)

            correlation = train_df[feature].corr(train_target)
            print("correlation = "+str(correlation))


            # Check if the correlation is NaN, and append 0 if it is
            if pd.notna(correlation):
                correlations.append((feature, correlation))
            else:
                correlations.append((feature, 0))

            

    


    corrIndex_df = pd.DataFrame(columns=["corr_index"])

    for _, sample in test_df.iterrows():
        corr_index = 0
        
        # Iterate over each feature and its correlation value
        for feature, correlation in correlations:
            # Check if the value of the sample for this feature is not NaN
            if not pd.isna(sample[feature]):
                corr_index += correlation * sample[feature]

        # Add a row to corrIndex_df
        corrIndex_df = corrIndex_df.append({"corr_index": corr_index}, ignore_index=True)

    correlation = corrIndex_df['corr_index'].corr(test_target)

    return correlation