import matplotlib.pyplot as plt

def plot_predicted_vs_actual_close_prices(test_data, company, predicted_closeprices, actual_closeprices, latency, method):

    plt.figure(figsize=(40, 10))
    plt.plot(test_data[company].index[:-1], actual_closeprices, label='Actual Close Prices', color='green', alpha=0.5, marker='o', markersize=5)
    plt.plot(test_data[company].index[:-1], predicted_closeprices, label=f'Predicted Close Prices ({method})', color='blue', alpha=0.5, marker='o', markersize=5)
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD$)')
    plt.title(f'Predicted vs Actual Close Prices for {company} with {latency}-day latency ({method})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{company}_Predicted_vs_Actual_Close_Prices_{latency}_days_{method}.png")
    plt.close()

def plot_mape_errors(latencies, mape_errors, mape_error_stds, company, method):

    plt.figure(figsize=(40, 10))
    plt.errorbar(latencies, mape_errors, yerr=mape_error_stds, fmt='-o', label=f'MAPE Errors ({method})')
    plt.xlabel('Latency (days)')
    plt.ylabel('MAPE Error (%)')
    plt.title(f'MAPE Errors for {company} with different latencies ({method})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{company}_MAPE_Errors_{method}.png")
    plt.close()

def visualize_states(n_states, train_data, train_data_wo_date, company, model, method, latency):
    
    train_data = train_data.copy()
    Z = model.predict(train_data_wo_date[company][["fracChange", "fracHigh", "fracLow"]])
    train_data.loc[:, "state"] = Z 
    plt.figure(figsize=(40, 10))
    for i in range(n_states):
        want = (Z == i)
        x = train_data[company].index[want]
        y = train_data[company]["Open"].iloc[want]
        plt.plot(x, y, '.', label=f"State {i+1}")
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Close Price (USD$)', fontsize=14, fontweight='bold')
    plt.title(f'States for {company} with {latency}-day latency ({method})', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=14)
    plt.grid(True)
    plt.savefig(f"{company}_States_HMM_{method}_{latency}latency.png")
    plt.close()