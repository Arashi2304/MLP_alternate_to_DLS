import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pandas as pd
import random

MARIGIN_OF_ERROR = 10  # Adjust as needed

'''
Section 1: We prep the dataset from data obtained over 1999-2011 ODI games
'''

def import_train_data(str):
    '''
    Extract data from the csv file and split into training and validation sets.
    
    Parameters
    ----------
    str : string
        The path to the csv file containing the data.
    
    Returns
    -------
    X_train : numpy array
        The training data. (run_rate, overs_left, wickets_in_hand)
    y_train : numpy array
        The training labels. (innings_total)
    X_val : numpy array
        The validation data. (run_rate, overs_left, wickets_in_hand)
    y_val : numpy array
        The validation labels. (innings_total)
    
    '''
    df = pd.read_csv(str)
    sub_df = df[df['Innings'] == 1]
    run_rate = sub_df['Run.Rate'].values
    overs_left = 50 - sub_df['Over'].values
    wickets_in_hand = sub_df['Wickets.in.Hand'].values
    innings_total = sub_df['Innings.Total.Runs'].values
    
    #Split the data into a training and validation dataset, with a 90/10 split
    split_index = int(0.9 * len(run_rate))
    
    #Train data
    X_train = np.array(list(zip(run_rate[:split_index], overs_left[:split_index], wickets_in_hand[:split_index])))
    y_train = np.array(innings_total[:split_index])
    
    #Validation data
    X_val = np.array(list(zip(run_rate[split_index:], overs_left[split_index:], wickets_in_hand[split_index:])))
    y_val = np.array(innings_total[split_index:])
    
    return X_train, y_train, X_val, y_val

'''
Section 2: Initialising both prediction models and training them on the dataset
'''

def DLS_params():
    '''
    Imports the parameters for the DLS model as obtained by Non-linear regression as implemented in DLS.py.
    
    Returns
    -------
    Z : numpy array
        The parameters for the DLS model. (Wickets-in-hand dependant parameter)
    L : float
        The parameters for the DLS model. (Constant parameter)
    '''
    from DLS import result
    
    Z = result[:-1]
    L = result[-1]
    
    return Z,L

def DLS_predict(overs_rem, wickets_in_hand):
    '''
    Predicts the total runs at the end of the innings using the DLS model, extrapolated from the current situation.
    
    Parameters
    ----------
    overs_rem : float
        The number of overs remaining in the innings.
    wickets_in_hand : int
        The number of wickets in hand.
    
    Returns
    -------
    total : float
        The predicted total runs at the end of the innings.
    '''
    wickets_in_hand = max(1, min(10, wickets_in_hand))
    
    return Z[wickets_in_hand-1] * (1 - np.exp(-L * overs_rem / Z[wickets_in_hand-1]))

def MLP_init(X_train, y_train, X_val, y_val):
    '''
    Create simple MLP model for regression and train it on the data.
    
    Parameters
    ----------
    X_train : numpy array
        The training data. (run_rate, overs_left, wickets_in_hand)
    y_train : numpy array
        The training labels. (innings_total)
    X_val : numpy array
        The validation data. (run_rate, overs_left, wickets_in_hand)
    y_val : numpy array
        The validation labels. (innings_total)
    
    Returns
    -------
    model : keras model
        The trained model.
    history : keras history
        The history of the training.
    '''
    #Initialize the MLP architecture
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(3,)), #Input layer with three nodes
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])  
    
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    #Generate history of training to study growth and visualise accuracy
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
    
    return model, history

'''
Section 3.1: Visualizing the results of the training of the MLP model
'''

def MLP_progress(model, history, X_train, y_train, save_path=None):
    '''
    Visualize the training results of the model
    
    Parameters
    ----------
    model : keras model
        The trained model.
    history : keras history
        The history of the training.
    X_train : numpy array
        The training data. (run_rate, overs_left, wickets_in_hand)
    y_train : numpy array
        The training labels. (innings_total)
    save_path : string, optional
        The path to save the plot to.
    '''
    plt.figure(figsize=(10, 5))
    
    # Scatter plot of actual vs. predicted innings total
    plt.subplot(1,2,1)
    plt.scatter(y_train, model.predict(X_train), color='blue', label='Actual vs. Predicted')
    plt.xlabel('Actual Innings Total')
    plt.ylabel('Predicted Innings Total')
    
    # Regression line
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Perfect Prediction')
    plt.legend()
    plt.title('Actual vs. Predicted Innings Total')
    
    # Extract MAE from the training history
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(train_mae) + 1)
    
    # Plot MAE vs. epochs
    plt.subplot(1,2,2)
    plt.plot(epochs, train_mae, 'bo-', label='Training MAE')
    plt.plot(epochs, val_mae, 'ro-', label='Validation MAE')
    plt.title('MAE vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.show()
    
    if save_path:
        plt.savefig(save_path)
    
    return None

'''
Section 3.2: Comparing the prediction accuracy of the MLP model vs. the DLS model
'''

def comparison(str, model, save_path=None):
    '''
    Compare the prediction accuracy of the MLP model vs. the DLS model
    
    Parameters
    ----------
    str : string
        The path to the csv file containing the data.
    model : keras model
        The trained model.
    save_path : string, optional
        The path to save the plot to.
    '''
    random.seed(42) #For reproducibility
    df = pd.read_csv(str)
    sub_df = df[df['Innings'] == 1]
    selected_rows = random.sample(sub_df.index.tolist(), 100)
    
    # Initialize lists to store the data for plotting
    case_numbers = []
    neural_network_predictions = []
    dls_predictions = []
    actual_runs = []

    # Loop through the selected rows
    for i, row_idx in enumerate(selected_rows):
        row = sub_df.loc[row_idx]
        run_rate = row['Run.Rate']
        overs_left = 50 - row['Over']
        wickets_in_hand = row['Wickets.in.Hand']
        innings_total = row['Innings.Total.Runs']


        neural_network_run_prediction = model.predict([[run_rate, overs_left, wickets_in_hand]])[0][0]


        # Predict runs using the DLS method
        dls_run_prediction = DLS_predict(wickets_in_hand, overs_left) + (50 - overs_left) * run_rate

        # Store the data
        case_numbers.append(i + 1)
        neural_network_predictions.append(neural_network_run_prediction)
        dls_predictions.append(dls_run_prediction)
        actual_runs.append(innings_total)
        
    # Calculate percentage accuracy within a margin of error
    accuracy_neural_network = np.mean(np.abs(np.array(actual_runs) - np.array(neural_network_predictions)) <= MARIGIN_OF_ERROR) * 100
    accuracy_dls = np.mean(np.abs(np.array(actual_runs) - np.array(dls_predictions)) <= MARIGIN_OF_ERROR) * 100
    
    # Print the percentage accuracy
    print("Neural Network Accuracy within {} runs: {:.2f}%".format(MARIGIN_OF_ERROR, accuracy_neural_network))
    print("DLS Accuracy within {} runs: {:.2f}%".format(MARIGIN_OF_ERROR, accuracy_dls))

    # Create a plot with three lines
    plt.plot(case_numbers, neural_network_predictions, label='Neural Network Predictions', marker='o')
    plt.plot(case_numbers, dls_predictions, label='DLS Predictions', marker='x')
    plt.plot(case_numbers, actual_runs, label='Actual Runs', linestyle='-')
    plt.xlabel('Case Number')
    plt.ylabel('Runs')
    plt.legend()
    plt.title('Comparison of Predicted Runs vs. Actual Runs')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return None
    
if __name__ == '__main__':
    path = 'Data/04_cricket_1999to2011.csv' #Name of datafile
    
    X_train, y_train, X_val, y_val = import_train_data(path)
    
    Z, L = DLS_params()
    MLP_model, history = MLP_init(X_train, y_train, X_val, y_val)
    
    MLP_progress(MLP_model, history, X_train, y_train, save_path='Figures/MLP_progress.png')
    
    comparison(path,MLP_model, save_path='Figures/Comparison.png')