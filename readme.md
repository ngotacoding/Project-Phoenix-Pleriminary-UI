### README for Streamlit App

---

## Overview

This Streamlit application provides a user interface to predict settlement values for personal injury cases. The application is organized into multiple tabs, each serving a specific purpose. 

### Structure of the App

- **app.py**: The main script that runs the app and handles tab navigation.
- **home_tab.py**: Contains the content and layout for the Home tab.
- **prediction_tab.py**: Contains the content and layout for the Prediction tab.
- **analysis_tab.py**: Contains the content and layout for the Analysis tab.
- **team_tab.py**: Contains the content and layout for the Team tab.

## How to Run the App

### Prerequisites

Ensure you have the required libraries installed. The required libraries are listed in the `requirements.txt` file.

### Installation Steps

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

### Conda Environment Note

If you prefer using a conda environment, note that `streamlit_option_menu` is not available in the conda repository and should be installed using pip. Here's how you can set up a conda environment and install the required libraries:

1. **Create a conda environment**:
    ```bash
    conda create --name myenv python=3.10
    conda activate myenv
    ```

2. **Install the required libraries using pip**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

After setting up the environment and installing the required libraries, you can run the app with the following command:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser. You can navigate through the various tabs to explore the different functionalities of the app.
