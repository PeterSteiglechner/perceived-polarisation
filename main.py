import pandas as pd
import numpy as np
import time 


def prepareData(data, wave_t0_prtcl, wave_t1_prtcl, ESSparty_dict_t0, ESSparty_dict_t1, variables, variables_na={}):
    """
    Prepare survey data by defining respondents' identities and opinions at times t0 and t1.
    
    Parameters:
        data (pd.DataFrame): Pandas dataframe containing survey data with columns wave_t0_prtcl, wave_t1_prtcl, "prtdgcl", and those listed in variables
        wave_t0_prtcl (str): Column name indicating the party that respondents felt closest to in the first wave.
        wave_t1_prtcl (str): Column name indicating the party that respondents felt closest to in the second wave.
        ESSparty_dict_t0 (dict): Dictionary with keys corresponding to values in wave_t0_prtcl.
        ESSparty_dict_t1 (dict): Dictionary with keys corresponding to values in wave_t1_prtcl.
        variables (list): List of opinion dimensions corresponding to the columns in the ESS.
        variables_na (dict): Dictionary coding values of each variable for which the respondent should be removed.
    
    Returns:
        pd.DataFrame: Processed dataframe with necessary columns for analysis.
    """
    # Ensure required columns are present in the dataframe
    for column in [wave_t0_prtcl, wave_t1_prtcl, "prtdgcl"] + variables:
        assert column in data.columns
        
    # Data preparation
    for p in [77, 99]:  # coded "Refusal" as np.nan
        ESSparty_dict_t0[p] = -1
        ESSparty_dict_t1[p] = -1
    data[wave_t0_prtcl].replace(to_replace=ESSparty_dict_t0, inplace=True)
    data[wave_t1_prtcl].replace(to_replace=ESSparty_dict_t1, inplace=True)
    for prtcl in [wave_t0_prtcl, wave_t1_prtcl]:
        data.loc[(data.prtdgcl > 3) & (data[prtcl] != -1), prtcl] = "None"
    data["identity"] = data[wave_t1_prtcl].combine_first(data[wave_t0_prtcl])
    data["identity"].replace(to_replace=[-1], value=np.nan, inplace=True)

    for var in variables:
        data[var].replace(to_replace=variables_na.get(var, []), value=np.nan, inplace=True)
    
    # Remove unnecessary columns
    data = data.loc[:, variables + ['identity', 'essround', 'anweight']]

    return data


def subjDist(opinions_observer, opinions_observed, ids_observer, currTrafo_set):
    """
    Calculate the subjective pairwise distance matrix.
    
    Parameters:
        opinions_observer (np.ndarray): N x M opinion vector of N observers.
        opinions_observed (np.ndarray): K x M opinion vector of K observed individuals.
        ids_observer (list): Length N vector containing the identities of the observers.
        currTrafo_set (dict): Dictionary of transformation matrices (M x M) used by each identity group.
    
    Returns:
        np.ndarray: Subjective pairwise squared distance matrix.
    """
    subj_squared_dist_matrix = np.empty((len(ids_observer), len(opinions_observed)))
    for ni, (id_i, op_i) in enumerate(zip(ids_observer, opinions_observer)):
        D = currTrafo_set[id_i]
        d = opinions_observed - op_i  # distance vector from observer i
        subj_squared_dist_matrix[ni, :] = np.sum(np.dot(d, D) * d, axis=1) # sum over the M topics
    
    return subj_squared_dist_matrix


def avg_distances(dist_matrix, w_obs=None, w_obd=None, all_observe_all=False):
    """
    Average pairwise distance matrix with weights.
    
    Parameters:
        dist_matrix (np.ndarray): Pairwise distance matrix of shape N x K.
        w_obs (np.ndarray): N x 1 array containing the weights of the observers.
        w_obd (np.ndarray): K x 1 array containing the weights of the observed individuals.
        all_observe_all (bool): If True, the denominator is 1/(sum(w)-1); if False, 1/sum(w).
    
    Returns:
        float: Averaged distance/disagreement index.
    """
    if dist_matrix.shape[1] == 1:  
        # single observer --> no averaging.
        return dist_matrix
    
    # Define weights if w_obs=w_obd=None
    if w_obs is None and w_obd is None:
        w_obs = np.ones(dist_matrix.shape[0])
        w_obd = np.ones(dist_matrix.shape[1])
    
    # Normalise weights  
    # ... for observer w_i / sum_k(w_k)
    w_obs_norm = w_obs / w_obs.sum()
    # ... for observed w_i / (sum_k(w_k)-w_i)  if observer i is included in the observed
    w_obd_norm = w_obd / (w_obd.sum() - w_obd) if all_observe_all else w_obd / w_obd.sum()
    
    # d_mean = w^T * Distance_Matrix * w 
    avg = np.dot(w_obs_norm.T, np.dot(dist_matrix, w_obd_norm))
    # dimensions: [[1 x N] * [N x K] * [K x 1] = [1,1]]
    
    return avg


def get_Trafo(df, parties, waves, variables):
    """
    Extract the transformation matrix and coordinate system axes for each partisan identity-group.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing the columns "essround", "identity", and those listed in variables.
        parties (list): List containing the parties.
        waves (list): List of the waves (values in df.essround).
        variables (list): List of the opinion columns.
    
    Returns:
        dict: Transformation matrices for each wave and party.
    """
    Trafo = {}
    CSS_dict = {}
    
    for r in waves:
        # (1) T^0: no bias / identity group "None"
        X = df.loc[df.essround == r, variables].dropna(how="any", axis="index").to_numpy()
        eigval, evec = np.linalg.eig(np.cov(X.T))   # returns w, v: column v[:,i] is the eigenvector corresponding to the  eigenvalue w[i]
        T0 = np.array([eigval[ni]**0.5 * evec[:, ni] for ni, i in enumerate(range(np.cov(X.T).shape[0]))]).T
        CSS_dict[r] = {"None": T0}
        T0inv = np.linalg.inv(T0)
        Trafo[r] = {"None": np.dot(T0inv.T, T0inv)}
        
        # (2) T^1_p: full bias; for each party
        for p in parties:
            if not (p == "None"):
                X_p = df.loc[(df.essround == r) & (df.identity == p), variables].dropna(how="any", axis="index").to_numpy()
                if not len(X_p) == 0:
                    eigval, evec = np.linalg.eig(np.cov(X_p.T))
                    T1_p = np.array([eigval[ni]**0.5 * evec[:, ni] for ni, i in enumerate(range(np.cov(X_p.T).shape[0]))]).T
                    if len(variables) > 1:
                        assert np.dot(T1_p[:, 0], T1_p[:, 1]) < 1e-13
                    CSS_dict[r][p] = T1_p
                    T1inv_p = np.linalg.inv(T1_p)
                    Trafo[r][p] = np.dot(T1inv_p.T, T1inv_p)
    
    return CSS_dict, Trafo


def calc_polarisation(df, waves, variables, Trafo):   
    """
    Calculate polarisation contributions from the opinion data between two waves for minimum and maximum bias.
    Parameters:
        df (pd.DataFrame): Dataframe that contains the columns "essround", "identity", and those listed in variables
        waves (list): List of waves (values in df.essround)
        variables (list): List of opinion columns.
        Trafo (dict): Transformation matrices for each wave and party.
    
    Returns:
        dict: Perceived mean distances for each wave and for each set of transformation matrices/representations of the opinion space used.
    """
    meand_w1 = dict(zip(waves, [{} for _ in waves]))
    meand_w0 = dict(zip(waves, [{} for _ in waves]))
    
    s0 = time.time()
    for n, r in enumerate(waves):
        df_wave = df.loc[df.essround == r,:]
        for C in waves[:n+1]:
            obs = df_wave
            obd = df_wave
            ids = df_wave["identity"]
            pairwiseSquaredDistances_w1 = subjDist(obs[variables].to_numpy(), obd[variables].to_numpy(), ids, Trafo[C])
            meand_w1[r][C] = avg_distances(pairwiseSquaredDistances_w1**(1/2), obs.anweight, obd.anweight, True)
            ids = ["None"] * len(obs)
            pairwiseSquaredDistances_w0 = subjDist(obs[variables].to_numpy(), obd[variables].to_numpy(), ids, Trafo[C])
            meand_w0[r][C] = avg_distances(pairwiseSquaredDistances_w0**(1/2), obs.anweight, obd.anweight, True)
            print(f"Opinions from wave {r} (# {obs.anweight.shape}), with transformation matrices from wave {C}. Time elapsed: {int(time.time() - s0)}s")

    return meand_w0, meand_w1


def calc_polarisation_PxPs(df, waves, parties, variables, Trafo):
    """
    Calculate polarisation contributions from opinion data between waves for different parties for minimum and maximum bias.
    
    Parameters:
        df (pd.DataFrame): Dataframe that contains the columns "essround", "identity", and those listed in variables
        waves (list): List of waves.
        parties (list): List of parties.
        variables (list): List of opinion columns.
        Trafo (dict): Transformation matrices for each wave and party.
    
    Returns:
        dict: Perceived mean distances for each wave and each set of transformation matrices by party.
    """
    PxA_1_md = dict(zip(waves, [dict(zip(waves, [{} for _ in waves])) for _ in waves]))
    PxA_0_md = dict(zip(waves, [dict(zip(waves, [{} for _ in waves])) for _ in waves]))
    PxP_0_md = dict(zip(waves, [dict(zip(waves, [{} for _ in waves])) for _ in waves]))
    PxP_1_md = dict(zip(waves, [dict(zip(waves, [{} for _ in waves])) for _ in waves]))
    
    for n, party_obs in enumerate(parties):
        print(f"{party_obs}")
        p_obs_k = party_obs[0].lower()
        for n, r in enumerate(waves):
            s0 = time.time()
            print(f"wave {r}", end="  ")
            df_wave = df.loc[df.essround == r, :]
            obs = df_wave.loc[df_wave.identity == party_obs]
            for C in waves[:n+1]:
                for party_obd in parties: 
                    p_obd_k = party_obd[0].lower()
                    AoA = True if party_obd == party_obs else False
                    obd = df_wave.loc[df_wave.identity == party_obd]
                    
                    # PxP: party-by-party
                    PxP_1_d2 = subjDist(obs[variables].to_numpy(), obd[variables].to_numpy(), obs.identity, Trafo[C])
                    PxP_0_d2 = subjDist(obs[variables].to_numpy(), obd[variables].to_numpy(), ["None"] * len(obs), Trafo[C])
                    PxP_1_md[r][C][p_obs_k + p_obd_k] = avg_distances(PxP_1_d2**(1/2), w_obs=obs.anweight, w_obd=obd.anweight, all_observe_all=AoA)
                    PxP_0_md[r][C][p_obs_k + p_obd_k] = avg_distances(PxP_0_d2**(1/2), w_obs=obs.anweight, w_obd=obd.anweight, all_observe_all=AoA)

                # PxA: party-to-all
                obd = df_wave
                PxA_1_d2 = subjDist(obs[variables].to_numpy(), obd[variables].to_numpy(), obs.identity, Trafo[C])
                PxA_0_d2 = subjDist(obs[variables].to_numpy(), obd[variables].to_numpy(), ["None"] * len(obs), Trafo[C])
                PxA_1_md[r][C][p_obs_k] = avg_distances(PxA_1_d2**(1/2), w_obs=obs.anweight, w_obd=obd.anweight, all_observe_all=True)
                PxA_0_md[r][C][p_obs_k] = avg_distances(PxA_0_d2**(1/2), w_obs=obs.anweight, w_obd=obd.anweight, all_observe_all=True)

            print(f"--> done ({(time.time() - s0):.0f} seconds, n={len(obs)})")
    
    return PxA_0_md, PxA_1_md, PxP_0_md, PxP_1_md


if __name__ == "__main__":
    
    # Load data
    cntry = "DE"
    variables = ["ccnthum", "wrclmch"]
    variables_na = {"ccnthum": [55, 66, 77, 88, 99], "wrclmch": [6, 7, 8, 9]} 
    waves = [8, 10]
    
    # Define German parties
    wave_t0_prtcl = "prtclede"
    wave_t1_prtcl = "prtclfde"
    
    ESSparty_dict_t0 = {
        1: "Union",
        2: "SPD",
        3: "Left Party",
        4: "Greens", 
        5: "FDP",
        6: "AfD"
    }
    for p in [7,8,9]: # coded "other Party" as np.nan
        ESSparty_dict_t0[p] = -1
    ESSparty_dict_t1 = ESSparty_dict_t0
    for p in [7,8,9]: # coded "other Party" as np.nan
        ESSparty_dict_t1[p] = -1
    for p in [66,88]: # coded as separate "None" fraction 
        ESSparty_dict_t0[p] = "None"
        ESSparty_dict_t1[p] = "None"

    parties = ["Left Party", "Greens", "SPD", "None", "FDP", "Union", "AfD"]
    assert all([((p in ESSparty_dict_t0.values()) and (p in ESSparty_dict_t1.values())) for p in parties])

    
    cols = ["essround", "anweight", "cntry", "prtdgcl"] + variables
    rawdataC = pd.concat(
            [
                pd.read_csv(f"/home/peter.steiglechner/labspaces/cognitive-biases-in-opinion-formation/data/ms3-subjOpSpace/ess/{essfile}.csv", 
                            usecols=cols + [prtclcol]) 
                for essfile, prtclcol in zip(["ESS8e02_3", "ESS10SC"], [wave_t0_prtcl, wave_t1_prtcl])
            ], 
            axis=0
        )
    data = rawdataC.loc[rawdataC.cntry == cntry]
    data = data.reset_index()
    
    
    # Run analysis
    
    data = prepareData(data, wave_t0_prtcl, wave_t1_prtcl, ESSparty_dict_t0, ESSparty_dict_t1, variables=variables, variables_na=variables_na)
    filtered_data = data.dropna(subset=["identity"]+variables, how="any", axis="index")
    CSS_dict, Trafo = get_Trafo(filtered_data, parties=parties, waves=waves, variables=variables)
    meand_w0, meand_w1 = calc_polarisation(filtered_data, waves=waves, variables=variables, Trafo=Trafo)
    PxA_0_md, PxA_1_md, PxP_0_md, PxP_1_md = calc_polarisation_PxPs(filtered_data, waves=waves, parties=parties, variables=variables, Trafo=Trafo)


    # Print 

    # P_perc = meand_w1[10][10] - meand_w1[8][8]
    # P_X = meand_w1[10][10] - meand_w1[10][8]
    # P_actual = meand_w1[10][8] - meand_w1[8][8]

    def print_table(DM, t0,t1, w):
        P_perc = DM[t1][t1] - DM[t0][t0]
        tableVal = r"   & T_t0 & T_t1 & P_X"+ "\n \t"+\
            rf"X_t0  &  {DM[t0][t0]:.3f} &  . &  . "+"\n \t"+\
            rf"X_t1 &  {DM[t1][t0]:.3f} &  {DM[t1][t1]:.3f} & {(DM[t1][t1]-DM[t1][t0]):.3f}   "+"\n \t"+\
            rf"P_actual  &  {(DM[t1][t0]-DM[t0][t0]):.3f}  & .. &  P_perc = {P_perc:.3f}  "
        print(f"---\n w={w}:\n \t {tableVal}")
        return
    
    print();print("All by All")
    print_table(meand_w0, waves[0], waves[1], 0)
    print(); print("All by All")
    print_table(meand_w1, waves[0], waves[1], 1)
    print("")
    for p in parties:
        print(p)
        pl = p[0].lower()
        print_table(
            dict(zip(waves, [dict(zip(waves, [PxA_1_md[r][C][pl] for C in waves[:n+1]])) for n, r in enumerate(waves)])),
            8, 10, 1)
        print("")