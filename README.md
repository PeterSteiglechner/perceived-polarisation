# Actual and Perceived Opinion Polarisation from survey data and opinion dynamics models.

### Introduction
This project analyses perceived and actual polarisation in political debates using survey data. The primary focus is on the subjective perception of the opinion space that is shaped dynamically by the opinion distribution in the observer's social identity. 

We provide an example of this analysis for the German climate debate using two waves of the European Social Survey (ESS). The analysis aims to understand polarisation contributions from opinion divergence between individuals and from perception changes between the identity groups.

### Requirements
- Python 3.11 or later
- Required Python packages: pandas, numpy, (for the analysis scipy, matplotlib, seaborn)

### Installation
1. Clone the repository: `git clone https://github.com/PeterSteiglechner/perceived-polarisation`
2. Install the required packages: `conda create --name <env> --file requirements.txt`

### Usage

```python
python main.py
```

### Functions

#### `prepareData(data, wave_t0_prtcl, wave_t1_prtcl, ESSparty_dict_t0, ESSparty_dict_t1, variables, variables_na={})`
Prepare survey data by defining respondents' identities and opinions at times t0 and t1.

- **Parameters:**
  - `data` (pd.DataFrame): Pandas dataframe containing survey data.
  - `wave_t0_prtcl` (str): Column name indicating the party in the first wave.
  - `wave_t1_prtcl` (str): Column name indicating the party in the second wave.
  - `ESSparty_dict_t0` (dict): Dictionary with keys corresponding to values in wave_t0_prtcl.
  - `ESSparty_dict_t1` (dict): Dictionary with keys corresponding to values in wave_t1_prtcl.
  - `variables` (list): List of opinion dimensions.
  - `variables_na` (dict): Dictionary coding values of each variable for which the respondent should be removed.

- **Returns:**
  - `pd.DataFrame`: Processed dataframe with necessary columns for analysis.

#### `subjDist(opinions_observer, opinions_observed, ids_observer, currTrafo_set)`
Calculate the subjective pairwise distance matrix.

- **Parameters:**
  - `opinions_observer` (np.ndarray): N x M opinion vector of N observers.
  - `opinions_observed` (np.ndarray): K x M opinion vector of K observed individuals.
  - `ids_observer` (list): Length N vector containing the identities of the observers.
  - `currTrafo_set` (dict): Dictionary of transformation matrices (M x M) used by each identity group.

- **Returns:**
  - `np.ndarray`: Subjective pairwise squared distance matrix.

#### `avg_distances(dist_matrix, w_obs=None, w_obd=None, all_observe_all=False)`
Average pairwise distance matrix with weights.

- **Parameters:**
  - `dist_matrix` (np.ndarray): Pairwise distance matrix of shape N x K.
  - `w_obs` (np.ndarray): N x 1 array containing the weights of the observers.
  - `w_obd` (np.ndarray): K x 1 array containing the weights of the observed individuals.
  - `all_observe_all` (bool): If True, the denominator is 1/(sum(w)-1); if False, 1/sum(w).

- **Returns:**
  - `float`: Averaged distance/disagreement index.

#### `get_Trafo(df, parties, waves, variables)`
Extract the transformation matrix and coordinate system axes for each partisan identity-group.

- **Parameters:**
  - `df` (pd.DataFrame): Dataframe containing the required columns.
  - `parties` (list): List containing the parties.
  - `waves` (list): List of waves.
  - `variables` (list): List of the opinion columns.

- **Returns:**
  - `dict`: Transformation matrices for each wave and party.

#### `calc_polarisation(df, waves, variables, Trafo)`
Calculate polarisation contributions from the opinion data between two waves for minimum and maximum bias.

- **Parameters:**
  - `df` (pd.DataFrame): Dataframe containing the required columns.
  - `waves` (list): List of waves.
  - `variables` (list): List of opinion columns.
  - `Trafo` (dict): Transformation matrices for each wave and party.

- **Returns:**
  - `dict`: Perceived mean distances for each wave and for each set of transformation matrices.

#### `calc_polarisation_PxPs(df, waves, parties, variables, Trafo)`
Calculate polarisation contributions from opinion data between waves for different parties for minimum and maximum bias.

- **Parameters:**
  - `df` (pd.DataFrame): Dataframe containing the required columns.
  - `waves` (list): List of waves.
  - `parties` (list): List of parties.
  - `variables` (list): List of opinion columns.
  - `Trafo` (dict): Transformation matrices for each wave and party.

- **Returns:**
  - `dict`: Perceived mean distances for each wave and each set of transformation matrices by party.

### Main Run

The main run loads survey data, defines parties, and executes the analysis functions. The results are printed for both "All by All" and individual parties.

### Example

```python
if __name__ == "__main__":
    # ... (code for loading data and setting up variables)
    data = prepareData(data, wave_t0_prtcl, wave_t1_prtcl, ESSparty_dict_t0, ESSparty_dict_t1, variables=variables, variables_na=variables_na)
    filtered_data = data.dropna(subset=["identity"]+variables, how="any", axis="index")
    CSS_dict, Trafo = get_Trafo(filtered_data, parties=parties, waves=waves, variables=variables)
    meand_w0, meand_w1 = calc_polarisation(filtered_data, waves=waves, variables=variables, Trafo=Trafo)
    PxA_0_md, PxA_1_md, PxP_0_md, PxP_1_md = calc_polarisation_PxPs(filtered_data, waves=waves, parties=parties, variables=variables, Trafo=Trafo)
    # ... (code for printing and further analysis)
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

