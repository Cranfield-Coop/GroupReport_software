import pandas as pd
import numpy as np


def prepare_prediction_csv(Re_tau, y_plus_min, y_plus_max, y_plus_delta):
    """
    Prepare a prediction dataset CSV file for a given Re_tau value.

    Args:
        Re_tau (int): The Re_tau value for the prediction dataset.
        y_plus_min (float): The minimum y_plus value.
        y_plus_max (float): The maximum y_plus value.
        y_plus_delta (float): The delta between each y_plus value.
    """


    # Define the initial data dictionary
    data_dict = {
        5200: {"Re_tau": 5185.897, "u_tau": 4.14872e-02, "nu": 8.00000e-06},
        2000: {"Re_tau": 1994.756, "u_tau": 4.58794e-02, "nu": 2.30000e-05},
        1000: {"Re_tau": 1000.512, "u_tau": 5.00256e-02, "nu": 5.00000e-05},
        550: {"Re_tau": 543.496, "u_tau": 5.43496e-02, "nu": 1.00000e-04},
        180: {"Re_tau": 182.088, "u_tau": 6.37309e-02, "nu": 3.50000e-04},
    }

    # Create a list of y_plus values
    y_plus_values = np.arange(y_plus_min, y_plus_max, y_plus_delta)

    if Re_tau not in data_dict.keys():
        raise ValueError("Re_tau not found in the given data dictionary.")

    results = []

    for y_plus in y_plus_values:
        Re_tau_float = data_dict[Re_tau]["Re_tau"]
        u_tau = data_dict[Re_tau]["u_tau"]
        nu = data_dict[Re_tau]["nu"]
        y_delta = (y_plus * nu) / u_tau

        # Verify if 0 <= y_delta < 1
        if 0 <= y_delta < 1:
            result = {
                "y^+": y_plus,
                "y/delta": y_delta,
                "u_tau": u_tau,
                "nu": nu,
                "Re_tau": Re_tau_float,
            }
            results.append(result)

    # Convert the results to a pandas DataFrame
    prediction_df = pd.DataFrame(results)

    # Sort the DataFrame by y^+ value
    prediction_df = prediction_df.sort_values("y^+")

    # Save the DataFrame as a CSV file
    filename = f"prediction_{Re_tau}.csv"
    prediction_df.to_csv(filename, index=False)

    return filename


if __name__ == "__main__":
    prepare_prediction_csv(
        Re_tau=5200,
        y_plus_min=7.372666080054942,  # Min = 0
        y_plus_max=305.6078192807962,  # Max = Re_tau (should be smaller)
        y_plus_delta=1,  # delta between each coordinates
    )
