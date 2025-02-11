import os
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch

from data_utils import load_raw_data, load_codes_data, load_clean_data, load_phekg_emb, get_patient_profile, save_data


def parse_args():
    parser = argparse.ArgumentParser(description="Process raw EHR data.")

    # Task
    parser.add_argument("--raw", type=bool, default=False, help="Process raw data")
    parser.add_argument("--split", type=str, default=None, choices=["patient", "time"], help="Split data")

    # Input/Output
    parser.add_argument("--data_dir", type=str, default="../data/", help="Directory to raw EHR data")
    parser.add_argument('--save_prefix', type=str, default='../data/', help='Prefix of all saved files')

    args = parser.parse_args()
    return args


def calc_data_stats(meta_df, pt_df_merged, data_source):

    if data_source == "MIMIC":

        print("Categories:", meta_df.category.unique())
        print("Number of Blood Gas:", len(meta_df[meta_df.category == "Blood Gas"]))
        print("Number of Chemistry:", len(meta_df[meta_df.category == "Chemistry"]))
        print("Number of Hematology:", len(meta_df[meta_df.category == "Hematology"]))
        
        print("Unique itemid:", len(meta_df.itemid.unique()))
        print("Unique label:", len(meta_df.label.unique()))

        print("Number of entries:", len(pt_df_merged))
        print("Number of patients:", len(pt_df_merged.subject_id.unique()))
        print("Number of lab events:", len(pt_df_merged.labevent_id.unique()))
        print("Number of lab items:", len(pt_df_merged.itemid.unique()))
        print("Number of lab items labels:", len(pt_df_merged.label.unique()))
    
    elif data_source == "eICU":
        
        # From documentation: the type of lab that is represented in the values,
        #   1 for chemistry, 2 for drug level, 3 for hemo, 4 for misc, 5 for non-mapped, 6 for sensitive, 7 for ABG lab
        print("Categories:", pt_df_merged.labtypeid.unique())
        print("Number of Blood Gas:", len(pt_df_merged[pt_df_merged.labtypeid == 7]))
        print("Blood Gas:", pt_df_merged[pt_df_merged.labtypeid == 7]["labname"].unique())
        print("Number of Chemistry:", len(pt_df_merged[pt_df_merged.labtypeid == 1]))
        print("Chemistry:", pt_df_merged[pt_df_merged.labtypeid == 1]["labname"].unique())
        print("Number of Hematology:", len(pt_df_merged[pt_df_merged.labtypeid == 3]))
        print("Hematology:", pt_df_merged[pt_df_merged.labtypeid == 3]["labname"].unique())
        print("Number of Drug-level labs:", len(pt_df_merged[pt_df_merged.labtypeid == 2]))
        print("Drug-level labs:", pt_df_merged[pt_df_merged.labtypeid == 2]["labname"].unique())
        print("Number of Misc:", len(pt_df_merged[pt_df_merged.labtypeid == 4]))
        print("Misc:", pt_df_merged[pt_df_merged.labtypeid == 4]["labname"].unique())
        print("Number of Non-mapped:", len(pt_df_merged[pt_df_merged.labtypeid == 5]))
        print("Non-mapped:", pt_df_merged[pt_df_merged.labtypeid == 5]["labname"].unique())
        print("Number of Sensitive:", len(pt_df_merged[pt_df_merged.labtypeid == 6]))
        print("Sensitive:", pt_df_merged[pt_df_merged.labtypeid == 6]["labname"].unique())
        
        print("Number of entries:", len(pt_df_merged))
        print("Number of patients:", len(pt_df_merged.subject_id.unique()))
        print("Unique itemid:", len(pt_df_merged.labid.unique()))
        print("Unique label:", len(pt_df_merged.labname.unique()))


def process_raw_data(pt_df_merged, data_source):

    ##############################################################################
    # Step 0: Define helper functions
    ##############################################################################

    def print_labdf_stats(df, label_col):
        """
        Print basic statistics about the dataframe (useful to understand
        the data after each processing step
        """
        print(df.head())
        print(df.shape)
        print("Number of entries:", len(df))
        print("Number of patients:", len(df.subject_id.unique()))
        print("Number of lab items labels:", len(df[label_col].unique()))
    
    def get_top_itemids(top_pt_lab_freq):
        """
        Identify the most frequently observed labs
        """
        print(top_pt_lab_freq)
        if data_source == "MIMIC": top_itemids = [i for i, l in top_pt_lab_freq.index]
        elif data_source == "eICU": top_itemids = list(top_pt_lab_freq.index)
        else: raise NotImplementedError
        print(top_itemids)
        return top_itemids
    
    def check_units(filter_df):
        """
        Print out the number of units per lab test. Ideally, there should NOT
        be multiple units per lab test
        """
        print("Check if there are multiple units per lab test...")
        if data_source == "MIMIC":
            lab_units = filter_df[[labid_col, labunit_col, label_col]].drop_duplicates()
            for l in lab_units[labid_col].unique():
                l_df = lab_units[lab_units[labid_col] == l]
                print("Number of units for lab test %s: %d" % (l, len(l_df)))
        elif data_source == "eICU":
            lab_units = filter_df[[labunit_col, label_col]].drop_duplicates()
            for l in lab_units[label_col].unique():
                l_df = lab_units[lab_units[label_col] == l]
                print("Number of units for lab test %s: %d" % (l, len(l_df)))
        else: raise NotImplementedError

    ##############################################################################
    # Step 1: Initial filtering of relevant labs, set data parameters
    ##############################################################################
    if data_source == "MIMIC":
        process_type = "ROUTINE"
        labid_col = "itemid"
        label_col = "label"
        labval_col = "valuenum"
        labunit_col = "valueuom"
        
        df = pt_df_merged[pt_df_merged.priority == process_type]
        print("Extract %s data..." % process_type)
        print(df.head())
        print(df.shape)

        print("Number of entries:", len(df))
        print("Number of patients:", len(df.subject_id.unique()))
        print("Number of lab events:", len(df.labevent_id.unique()))
        print("Number of lab items:", len(df.itemid.unique()))
        print("Number of lab items labels:", len(df.label.unique()))

    elif data_source == "eICU":
        # Note: eICU's lab IDs and lab names are not a 1-to-1 match. Using lab name instead.

        keep_categories = [] #[1, 3]
        labid_col = "labid"
        label_col = "labname"
        labval_col = "labresult"
        labunit_col = "labmeasurenamesystem"

        if len(keep_categories) > 0:
            df = pt_df_merged[pt_df_merged["labtypeid"].isin(keep_categories)]
            print("Extract data under categories:", keep_categories)
        else:
            df = pt_df_merged
        print(df.head())
        print(df.shape)
        print("Number of entries:", len(df))
        print("Number of patients:", len(df.subject_id.unique()))
        print("Number of lab items:", len(df.labid.unique()))
        print("Number of lab items labels:", len(df.labname.unique()))

    ##############################################################################
    # Step 2: Extract the most commonly ordered lab tests
    ##############################################################################

    # Compute (and sort) frequency of labs
    if data_source == "MIMIC":
        lab_freq = df[["subject_id", labid_col, label_col]].drop_duplicates()
        pt_lab_freq = lab_freq.groupby([labid_col, label_col]).count().sort_values(by="subject_id", ascending=False)
    elif data_source == "eICU":
        lab_freq = df[["subject_id", label_col]].drop_duplicates()
        pt_lab_freq = lab_freq.groupby(label_col).count().sort_values(by="subject_id", ascending=False)
    else: raise NotImplementedError
    print(lab_freq)
    print(pt_lab_freq)

    # Filter labs that appear in at least 80% of patients
    print("Extract most common lab codes...")
    top_pt_lab_freq = pt_lab_freq[pt_lab_freq["subject_id"] >= 0.8*len(df.subject_id.unique())]
    top_itemids = get_top_itemids(top_pt_lab_freq)
    if data_source == "MIMIC": filter_df = df[df[labid_col].isin(top_itemids)]
    elif data_source == "eICU": filter_df = df[df[label_col].isin(top_itemids)]
    else: raise NotImplementedError
    print_labdf_stats(filter_df, label_col)

    # Filter patients that have all the lab tests (i.e., number of lab tests per patient equals to number of selected labs)
    print("Keep patients with all lab tests...")
    if data_source == "MIMIC":
        pt_count_unique_labs = filter_df[["subject_id", labid_col]].drop_duplicates().groupby("subject_id").count()
        print(filter_df[["subject_id", labid_col]].drop_duplicates())
        print(pt_count_unique_labs)
        pt_count_unique_labs = pt_count_unique_labs[pt_count_unique_labs[labid_col] == len(top_itemids)]
        print(pt_count_unique_labs)
    elif data_source == "eICU":
        pt_count_unique_labs = filter_df[["subject_id", label_col]].drop_duplicates().groupby("subject_id").count()
        print(filter_df[["subject_id", label_col]].drop_duplicates())
        print(pt_count_unique_labs)
        pt_count_unique_labs = pt_count_unique_labs[pt_count_unique_labs[label_col] == len(top_itemids)]
        print(pt_count_unique_labs)
    else: raise NotImplementedError
    filter_df = filter_df[filter_df.subject_id.isin(pt_count_unique_labs.index.tolist())]
    print_labdf_stats(filter_df, label_col)

    ##############################################################################
    # Step 3: Tidy up patient data
    ##############################################################################

    # Extract data of selected patients and labs
    print("Extract relevant columns from lab data...")
    if data_source == "MIMIC":
        filter_df = filter_df[["subject_id", labid_col, "charttime", labval_col, labunit_col, label_col]]
    elif data_source == "eICU":
        filter_df = filter_df[["subject_id", label_col, "charttime", labval_col, labunit_col]]
    else: raise NotImplementedError
    print_labdf_stats(filter_df, label_col)

    # Are there multiple units per test?
    check_units(filter_df)
    
    # Take mean of duplicate lab tests per time
    print("Take mean of duplicate lab tests per time...")
    if data_source == "MIMIC":
        print(filter_df[["subject_id", labid_col, "charttime", labunit_col, label_col]].drop_duplicates())
        print(filter_df.groupby(["subject_id", labid_col, "charttime", labunit_col, label_col]).mean())
        filter_df = filter_df.groupby(["subject_id", labid_col, "charttime", labunit_col, label_col]).mean().reset_index()
    elif data_source == "eICU":
        print(filter_df[["subject_id", label_col, "charttime", labunit_col]].drop_duplicates())
        print(filter_df.groupby(["subject_id", label_col, "charttime", labunit_col]).mean())
        filter_df = filter_df.groupby(["subject_id", label_col, "charttime", labunit_col]).mean().reset_index()
    else: raise NotImplementedError
    print_labdf_stats(filter_df, label_col)

    # Extract patients with multiple visits
    print("Keep patients with more than one visit...")
    pt_count_visits = filter_df[["subject_id", "charttime"]].drop_duplicates().groupby("subject_id").count()
    print(pt_count_visits)
    print(pt_count_visits.sort_values(by = "charttime", ascending = True))
    pt_count_visits = pt_count_visits[pt_count_visits["charttime"] > 1]
    print(pt_count_visits.sort_values(by = "charttime", ascending = True))
    filter_df = filter_df[filter_df.subject_id.isin(pt_count_visits.index.tolist())]
    print_labdf_stats(filter_df, label_col)

    # Rename eICU to match MIMIC-IV
    if data_source == "eICU":
        filter_df = filter_df.rename(columns = {"labname": "itemid", "labmeasurenamesystem": "valueuom", "labresult": "valuenum", "subject_id": "subject_id_str"})
        subject_id_mapper = {v: i for i, v in enumerate(filter_df["subject_id_str"].unique())}
        filter_df["subject_id"] = filter_df["subject_id_str"].map(subject_id_mapper)
        print(filter_df)

        # Sanity check
        check1 = filter_df.groupby(["subject_id", "subject_id_str", "charttime", "itemid", "valueuom"]).mean().reset_index()
        check2 = filter_df[["subject_id_str", "charttime", "itemid", "valueuom", "valuenum"]].groupby(["subject_id_str", "charttime", "itemid", "valueuom"]).mean().reset_index()
        check3 = filter_df[["subject_id", "charttime", "itemid", "valueuom", "valuenum"]].groupby(["subject_id", "charttime", "itemid", "valueuom"]).mean().reset_index()
        assert len(check1) == len(check2)
        assert len(check1) == len(check3)

    return filter_df


def split_patients(data, split_type):

    if split_type == "patient":

        total_patients = len(data.subject_id.unique())

        # Train patients
        train_pts = random.sample(list(data.subject_id.unique()), int(total_patients * 0.6))

        # Validation patients
        remain_pts = list(set(data.subject_id.unique().tolist()).difference(set(train_pts)))
        val_pts = random.sample(remain_pts, int(total_patients * 0.2))

        # Test patients
        test_pts = list(set(data.subject_id.unique().tolist()).difference(set(train_pts + val_pts)))

        # Sanity check
        assert len(set(train_pts).intersection(set(val_pts))) == 0
        assert len(set(test_pts).intersection(set(val_pts))) == 0
        assert len(set(test_pts).intersection(set(train_pts))) == 0

        # Create data loaders
        train_loader = [str(p) for p in train_pts]
        val_loader = [str(p) for p in val_pts]
        test_loader = [str(p) for p in test_pts]
        split_data = {"train": ";".join(train_loader), "val": ";".join(val_loader), "test": ";".join(test_loader)}

    elif split_type == "time":
        train_timepoints = 20
        val_timepoints = 10

        # Get all patients' visit counts
        patient_visit_counts = data[["subject_id", "charttime"]].drop_duplicates().groupby("subject_id").count()
        min_timepoints = patient_visit_counts["charttime"].min()
        median_timepoints = patient_visit_counts["charttime"].median()
        mean_timepoints = patient_visit_counts["charttime"].mean()
        max_timepoints = patient_visit_counts["charttime"].max()
        print("Minimum number of time points:", min_timepoints)
        print("Median number of time points:", median_timepoints)
        print("Mean number of time points:", mean_timepoints)
        print("Maximum number of time points:", max_timepoints)
        patient_visit_counts = patient_visit_counts.to_dict()["charttime"]

        # Split and create data loaders
        train_loader = dict()
        val_loader = dict()
        test_loader = dict()
        for p_id, p_count in patient_visit_counts.items():
            p_total_idx = np.arange(0, p_count, 1, dtype = int)
            assert len(p_total_idx) == p_count
            assert len(np.unique(p_total_idx)) == p_count

            # Split time points
            pt_train_timepoints = min(train_timepoints, p_count)
            pt_val_timepoints = min(val_timepoints, p_count - pt_train_timepoints)
            pt_test_timepoints = p_count - pt_train_timepoints - pt_val_timepoints

            # Save train data
            train_data = p_total_idx[0:pt_train_timepoints]
            train_loader[p_id] = ";".join([str(i) for i in train_data.tolist()])

            # Save val/test data
            if pt_val_timepoints > 0:
                val_data = p_total_idx[train_timepoints:train_timepoints + val_timepoints]
                val_loader[p_id] = ";".join([str(i) for i in val_data.tolist()])
            if pt_test_timepoints > 0: # Enough data for val and test
                test_data = p_total_idx[train_timepoints + val_timepoints:]
                test_loader[p_id] = ";".join([str(i) for i in test_data.tolist()])

            # Sanity check
            assert pt_train_timepoints + pt_val_timepoints + pt_test_timepoints == p_count

        split_data = {"train": train_loader, "val": val_loader, "test": test_loader}

    else:
        raise NotImplementedError

    print("Train:", len(train_loader))
    print("Val:", len(val_loader))
    print("Test:", len(test_loader))
    return split_data


def merge_multiple_actions(curr_codes, action_embs_dict):

    def combine_actions(actions, existing_actions = []):
        combined_a = [] # List of unique actions (parse semicolon-delimited strings)
        for a in actions:
            combined_a.extend(a.split(";"))
        if len(existing_actions) > 0: # Retrieve embeddings of actions
            combined_a = [a for a in set(combined_a) if a in existing_actions]
        else: # No need to retrieve embeddings of actions
            combined_a = [a for a in set(combined_a)]
        if len(combined_a) == 0: # No relevant actions = replace with None
            combined_a = "None"
        else: # Combine actions into a semicolon-delimited string
            combined_a = ";".join(combined_a)
        return {a: combined_a for a in actions}

    # Iterate through each code time
    new_t_df = []
    for t_idx in curr_codes["code_time"].unique():
        t_df = curr_codes[curr_codes["code_time"] == t_idx]
        actions = t_df["action"].tolist()
        action_cats = t_df["action_cat"].tolist()
        
        if len(actions) == 1: # Only one action -> If there is an embedding, keep action; otherwise, replace with None.
            if actions[0] in action_embs_dict: t_df["action"] = actions[0]
            else: t_df["action"] = "None"
            t_df["action_cat"] = action_cats[0]
        
        else: # Update actions to be a combined string
            t_df["action"] = t_df["action"].map(combine_actions(actions, action_embs_dict))
            t_df["action_cat"] = t_df["action_cat"].map(combine_actions(action_cats))

        new_t_df.append(t_df)
    curr_codes = pd.concat(new_t_df)
    curr_codes = curr_codes.drop_duplicates()
    return curr_codes


def retrieve_action_embs(pt_codes_df, phekg_nodemap, phekg_emb, save_f):

    # Parse codes
    all_codes = []
    for c in pt_codes_df["action"].unique():
        all_codes.extend(c.split(";"))
    
    # Retrieve actions from KG
    traj_actions = phekg_nodemap[phekg_nodemap["node_id"].isin(all_codes)]
    missing_actions = [a for a in all_codes if a not in traj_actions["node_id"].tolist()]
    print("Missing actions?", missing_actions)
    
    # Create action embedding dictionary
    action_embs = phekg_emb[traj_actions["global_graph_index"].tolist(), :]
    action_embs_dict = {k: action_embs[i, :] for i, k in enumerate(traj_actions["node_id"].tolist()) if len(k) > 0}
    
    # Save embedding (all zeros) for no action
    no_action_emb = torch.zeros(len(list(action_embs_dict.values())[0]))
    if "None" not in action_embs_dict: action_embs_dict["None"] = no_action_emb # No action embedding for None action
    #action_embs_dict.update({k: no_action_emb for k in missing_actions}) # No action embedding for missing PheKG embeddings

    # Save and return
    save_data(save_f, action_embs_dict)
    return action_embs_dict


def merge_all_patients_actions(data_source, pt_codes_df, labs_df, action_embs_dict):
    total_pts_with_action = len(pt_codes_df["subject_id"].unique())

    # Parameters
    if data_source == "MIMIC":
        tolerance = pd.Timedelta("2d")
        direction = "nearest"
    elif data_source == "eICU":
        tolerance = pd.Timedelta("12h")
        direction = "forward"
    else: raise NotImplementedError

    # Get patients with actions
    all_merged = []
    with_action = []
    unique_actions = set()
    for p_idx, pt in enumerate(pt_codes_df["subject_id"].unique()):
        print(p_idx, "out of", total_pts_with_action)

        # Get labs and codes for current patient
        curr_labs = labs_df[labs_df["subject_id"] == pt]
        curr_codes = pt_codes_df[pt_codes_df["subject_id"] == pt].drop_duplicates()
        if len(curr_labs) == 0 or len(curr_codes) == 0: continue
        with_action.append(pt)

        # Concatenate multiple actions per time
        if len(curr_codes) > len(curr_codes["code_time"].unique()):
            curr_codes = merge_multiple_actions(curr_codes, action_embs_dict)
        assert len(curr_codes["code_time"].unique()) == len(curr_codes), (curr_codes)
        
        # Merge labs and codes
        curr_lab_codes_df = pd.merge_asof(curr_labs, curr_codes, by = "subject_id", left_on = "lab_time", right_on = "code_time", direction = direction, tolerance = tolerance)
        curr_lab_codes_df = curr_lab_codes_df.drop(columns = ["code_time"]).fillna("None")
        all_merged.append(curr_lab_codes_df)
        unique_actions = unique_actions.union(curr_lab_codes_df["action"].unique().tolist())
        if len(all_merged) % 100 == 0: print("%d patients with action; %d actions so far" % (len(with_action), len(unique_actions)))
    
    # Get patients with no actions
    labs_df = labs_df[~labs_df["subject_id"].isin(with_action)]
    labs_df["action"] = "None"
    all_merged.append(labs_df)
    
    # Concatenate all dataframes
    clean_data = pd.concat(all_merged).rename(columns = {"lab_time": "charttime"})
    print(clean_data)
    return clean_data


def main():
    args = parse_args()

    # Parse data source
    if "MIMIC" in args.data_dir: data_source = "MIMIC"
    elif "eICU" in args.data_dir: data_source = "eICU"
    else: raise NotImplementedError
    
    if args.raw: # Perform processing steps
        
        #######################################
        # Step 1: Filter lab data from raw
        #######################################
        if not os.path.exists(args.save_prefix + "filtered_lab_data.csv"):
            """
            Load in the raw EHR/lab data, calculate basic data statistics, clean data, and
            save the processed data -> Output: Intermediate file for initially processed data
            """
            meta_df, pt_df, pt_df_merged = load_raw_data(args.data_dir, data_source)
            print(pt_df_merged)
            calc_data_stats(meta_df, pt_df_merged, data_source)
            clean_data = process_raw_data(pt_df_merged, data_source)
            save_data(args.save_prefix + "filtered_lab_data.csv", clean_data)
        else:
            """
            Read in already-processed data
            """
            if data_source == "eICU": # Processing for eICU requires additional meta data
                meta_df, pt_df, pt_df_merged = load_raw_data(args.data_dir, data_source)
                print(pt_df_merged)
            else: meta_df = []
            clean_data = load_clean_data(args.save_prefix + "filtered_lab_data.csv")

        #######################################
        # Step 2: Extract actions for labs
        #######################################
        if not os.path.exists(args.save_prefix + "filtered_code_data.csv"):
            """
            Retrieve actions (i.e., diagnostic codes) and their embeddings (i.e., from a clinical
            knowledge graph), merge actions with the dates of the lab records, and save the 
            processed action data -> Output: Intermediate file for action data
            """

            # Load PheKG embeddings
            phekg_nodemap, phekg_emb = load_phekg_emb("data/PheKG/")

            # Process actions and retrieve their embeddings
            print("Reading in codes...")
            pt_codes_df, clean_data = load_codes_data(args.data_dir, data_source, meta_df, clean_data)
            action_embs_dict = retrieve_action_embs(pt_codes_df, phekg_nodemap, phekg_emb, args.save_prefix + "action_embs_dict.pth")
            print(pt_codes_df)

            # Merge actions
            labs_df = clean_data[["subject_id", "lab_time"]].drop_duplicates() # De-duplicate patient visits (to make the merge faster)
            print(labs_df)
            clean_data = merge_all_patients_actions(data_source, pt_codes_df, labs_df, action_embs_dict)
            if "subject_id_str" in clean_data.columns: clean_data = clean_data.drop(columns = ["subject_id_str"])
            save_data(args.save_prefix + "filtered_code_data.csv", clean_data)

        #######################################
        # Step 3: Merge labs and actions data
        #######################################
        if not os.path.exists(args.save_prefix + "filtered_lab_code_data.csv"):
            """
            Merge lab data with the actions to create a single file with both lab information
            and actions (i.e., diagnostic codes) -> Output: Final patient dataset
            """
            
            # Read data
            code_data = load_clean_data(args.save_prefix + "filtered_code_data.csv")
            print(code_data)
            clean_data = load_clean_data(args.save_prefix + "filtered_lab_data.csv")
            print(clean_data)

            # Clean up columns
            clean_data["subject_id"] = clean_data["subject_id"].astype(str)
            code_data["subject_id"] = code_data["subject_id"].astype(str)
            clean_data["charttime"] = clean_data["charttime"].astype(str)
            code_data["charttime"] = code_data["charttime"].astype(str)
            clean_data = clean_data.merge(code_data, on = ["subject_id", "charttime"])
            clean_data = clean_data.sort_values(by = ["subject_id", "charttime", "itemid"]).reset_index(drop = True)

            print("Before merge")
            patients_with_actions = code_data[["subject_id", "action"]].dropna().drop_duplicates()
            print("Number of unique patients with actions:", len(patients_with_actions["subject_id"].unique()))
            print("Number of unique actions:", len(patients_with_actions["action"].unique()))

            print("After merge (should be the same as before merging)")
            patients_with_actions = clean_data[["subject_id", "action"]].dropna().drop_duplicates()
            print("Number of unique patients with actions:", len(patients_with_actions["subject_id"].unique()))
            print("Number of unique actions:", len(patients_with_actions["action"].unique()))

            save_data(args.save_prefix + "filtered_lab_code_data.csv", clean_data)

    else: # No processing necessary
        print("Reading in cleaned data...")
        clean_data = load_clean_data(args.save_prefix + "filtered_lab_code_data.csv") # filtered_lab_data.csv
        print(clean_data)

    #######################################
    # Step 4: Split data
    #######################################
    if args.split is not None:
        """
        Generate data splits for training the model -> Output: Final dictionary of data splits
        """
        split_data = split_patients(clean_data, args.split)
        save_data(args.save_prefix + ("filtered_lab_data_split=%s.json" % args.split), split_data)


if __name__ == "__main__":
    main()
