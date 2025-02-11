import torch
import pandas as pd
import datetime
import json
import pickle
from glob import glob

from patient_similarity import read_icd_maps


def load_raw_data(data_dir, data_source = "MIMIC"):

  if data_source == "MIMIC":

    # Lab metadata
    print("Load raw metadata data at %s" %  (data_dir + "hosp/d_labitems.csv"))
    meta_df = pd.read_csv(data_dir + "hosp/d_labitems.csv")
    print(meta_df.head())
    print(meta_df.shape)
    print("Unique itemids:", len(meta_df["itemid"].unique()))
    print("Unique labels:", len(meta_df["label"].unique()))

    # Patient labs
    print("Load raw patient data at %s" % (data_dir + "hosp/labevents.csv"))
    pt_df = pd.read_csv(data_dir + "hosp/labevents.csv")
    print(pt_df.head())
    print(pt_df.shape)
    print(pt_df.columns)
    print("Unique patients:", len(pt_df["subject_id"].unique()))
    print("Unique labs (itemid):", len(pt_df["itemid"].unique()))

    # Combined patient labs with lab metadata
    pt_df_merged = pt_df.merge(meta_df, on="itemid", how="left")
    print("Merge raw meta and patient data")
    print(pt_df_merged.head())
    print(pt_df_merged.shape)

  elif data_source == "eICU":

    # Patient meta data (retrieve unique IDs)
    meta_df = pd.read_csv(data_dir + "patient.csv")
    print(meta_df)
    print(meta_df.columns)
    meta_df = meta_df[["uniquepid", "patientunitstayid", "unitadmittime24"]].dropna().drop_duplicates().rename(columns = {"uniquepid": "subject_id"})
    print(meta_df)
    print("Unique number of patients in meta_df:", len(meta_df["subject_id"].unique()))
    
    # Patient labs
    print("Load raw patient data at %s" % (data_dir + "lab.csv"))
    pt_df = pd.read_csv(data_dir + "lab.csv")
    print(pt_df.head())
    print(pt_df.shape)
    print(pt_df.columns)

    # Use unitadmittime24 + labresultoffset to get time of lab (set a fixed starting YYYY-MM-DD, same for everyone)
    # Merge patient meta data and labs
    pt_df = pt_df.merge(meta_df, on = "patientunitstayid")
    pt_df["charttime"] = pd.to_datetime(pt_df["unitadmittime24"], format='%H:%M:%S') + pd.to_timedelta(pt_df["labresultrevisedoffset"], unit='m')
    print(pt_df)
    print("Unique patients:", len(pt_df["subject_id"].unique()))
    print("Unique labs (labid):", len(pt_df["labid"].unique()))
    pt_df_merged = pt_df

  elif data_source == "WaddingtonOT":

    keep_cols = ["timestep", "group", "cell id", "action", "actions", "trajectory", "observations"]

    def process_WOT(f):
      import ast
      print(f)
      states_df = pd.read_csv(f)
      print(states_df)
      states_df = states_df[keep_cols]
      print(states_df)
      return states_df
    
    states_data = dict()
    if "counterfactual" in data_dir:
      total_unique_orig_trajs = 0
      total_unique_cf_trajs = 0
      states_data = {"original": [], "counterfactual": []}
      for idx, f in enumerate(glob(data_dir.replace("WOT_counterfactual", "") + "WOT_diverge*_start*.csv")):
        f_data = process_WOT(f)
        
        # Split into original and counterfactual
        originals = f_data[f_data["trajectory"] % 2 == 0].reset_index(drop = True)
        counterfactuals = f_data[f_data["trajectory"] % 2 == 1].reset_index(drop = True)

        # Rename to maintain identity of this specific dataset
        originals["trajectory"] = originals["trajectory"] + ((idx + 1) * 1000)
        counterfactuals["trajectory"] = counterfactuals["trajectory"] + ((idx + 1) * 1000)

        # Sanity check calculations
        num_orig_traj = len(originals["trajectory"].unique())
        num_counterfactual_traj = len(counterfactuals["trajectory"].unique())
        assert num_orig_traj == num_counterfactual_traj, (num_orig_traj, num_counterfactual_traj)
        total_unique_orig_trajs += num_orig_traj
        total_unique_cf_trajs += num_counterfactual_traj

        # Save data
        states_data["original"].append(originals)
        states_data["counterfactual"].append(counterfactuals)
        print("Number of unique original trajectories:", num_orig_traj)
        print("Number of unique counterfactual trajectories:", num_counterfactual_traj)
      
      # Concatenate and sanity check
      states_data = {k: pd.concat(v) for k, v in states_data.items()}
      for k, v in states_data.items():
        print(k, v.shape)
        if k == "original": assert len(v["trajectory"].unique()) == total_unique_orig_trajs
        if k == "counterfactual": assert len(v["trajectory"].unique()) == total_unique_cf_trajs
    
    else:
      for start in ["start1", "start2", "start3"]:
        if data_dir.endswith("WOT_traj_all"):
          states_data[start] = process_WOT(data_dir.replace("WOT_traj_all", "") + ("WOT_%s.csv" % start))
        else:
          raise NotImplementedError

    return states_data

  else:
    raise NotImplementedError

  return meta_df, pt_df, pt_df_merged


def load_codes_data(data_dir, data_source, meta_df, clean_data):

  _, cm_rev_mapping, _ = read_icd_maps("data/icd_mappings/")
  
  def map_icd_codes(emar, icd_col):
    code_mapping = dict()
    code_mapping_trunc = dict()
    for code in emar[icd_col].unique():
        
        # Get codes
        codes = [i.strip().replace(".", "") for i in code.split(",")]
        
        # Map ICD10 to ICD9 (or map to self)
        codes = ";".join([cm_rev_mapping[i] if i in cm_rev_mapping else i for i in codes])
        
        # Save truncated and full codes
        code_mapping[code] = split_trunc(codes, -1) # No trunc
        code_mapping_trunc[code] = split_trunc(codes, 3)
    return code_mapping, code_mapping_trunc

  def split_trunc(pt, n):
    return ";".join(list(set([i[0:n] for i in pt.split(";")])))

  if data_source == "MIMIC":

    emar_meta = pd.read_csv(data_dir + "hosp/hcpcsevents.csv")
    emar_meta = emar_meta[["subject_id", "hadm_id", "chartdate"]].drop_duplicates()
    print(emar_meta)
    
    emar = pd.read_csv(data_dir + "hosp/diagnoses_icd.csv")
    emar = emar[["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"]]
    emar = emar[emar["seq_num"] == 1]
    emar = emar.merge(emar_meta, on = ["hadm_id", "subject_id"])
    print(emar)

    icd10 = emar[emar["icd_version"] == 10]
    icd9 = emar[emar["icd_version"] == 9]
    for i in icd9["icd_code"].tolist():
      if i not in cm_rev_mapping: cm_rev_mapping[i] = i
    for i in icd10["icd_code"].tolist():
      cm_rev_mapping[i] = i

    code_mapping, code_mapping_trunc = map_icd_codes(emar, "icd_code")
    emar["action_cat"] = emar["icd_code"].map(code_mapping_trunc)
    emar["icd_code"] = emar["icd_code"].map(code_mapping)
    
    emar = emar.rename(columns = {"icd_code": "action", "chartdate": "charttime"}).drop(columns = ["seq_num", "icd_version", "hadm_id"])
    emar["charttime"] = pd.to_datetime(emar["charttime"]) +  pd.to_timedelta(1, unit='s')
    emar = emar.groupby(by = ["subject_id", "action", "action_cat"]).min().reset_index()
    print(emar)


  elif data_source == "eICU":
    
    # Load and clean diagnosis data
    emar = pd.read_csv(data_dir + "diagnosis.csv")
    emar = emar[["patientunitstayid", "icd9code", "diagnosisoffset", "diagnosispriority"]].dropna().drop_duplicates()
    emar = emar[emar["diagnosispriority"] == "Primary"].drop(columns = ["diagnosispriority"])
    emar = emar.rename(columns = {"icd9code": "action", "diagnosisoffset": "timeoffset"})

    # Map ICD codes
    code_mapping, code_mapping_trunc = map_icd_codes(emar, "action")
    emar["action_cat"] = emar["action"].map(code_mapping_trunc)
    emar["action"] = emar["action"].map(code_mapping)
    emar = emar.groupby(by = ["patientunitstayid", "action", "action_cat"]).min().reset_index()

    # Merge on meta data and clean data
    emar = emar.merge(meta_df, on = "patientunitstayid").drop(columns = ["patientunitstayid"])
    emar["charttime"] = pd.to_datetime(emar["unitadmittime24"], format='%H:%M:%S') + pd.to_timedelta(emar["timeoffset"], unit='m')
    emar = emar.drop(columns = ["unitadmittime24", "timeoffset"])
    emar = emar[emar["subject_id"].isin(clean_data["subject_id_str"].unique())]
    print(emar)

  else:
    raise NotImplementedError
  
  # Clean up: Convert charttime from string to datetime
  clean_data["charttime"] = pd.to_datetime(clean_data["charttime"])
  emar["charttime"] = pd.to_datetime(emar["charttime"])

  # Clean up: Convert subject ID to string
  clean_data["subject_id"] = clean_data["subject_id"].astype(str)
  emar["subject_id"] = emar["subject_id"].astype(str)
  
  # Clean up: Rename and sort table
  clean_data = clean_data.rename(columns = {"charttime": "lab_time"})
  clean_data = clean_data.sort_values(by = ["subject_id", "lab_time"])
  emar = emar.rename(columns = {"charttime": "code_time"})
  emar = emar.sort_values(by = ["subject_id", "code_time"])

  # Clean up: Map the proper subject IDs (relevant only for eICU)
  if "subject_id_str" in clean_data:
    subject_id_mapper = {k: v for k, v in zip(clean_data["subject_id_str"].tolist(), clean_data["subject_id"].tolist())}
    emar["subject_id_str"] = emar["subject_id"].tolist()
    emar["subject_id"] = emar["subject_id_str"].map(subject_id_mapper)
  
  return emar, clean_data


def load_clean_data(f):
  df = pd.read_csv(f, sep = "\t")
  return df


def load_phekg_emb(kg_dir):

  # Load node map of KG
  phekg_nodemap = pd.read_csv(kg_dir + "new_node_map_df.csv")
  phekg_nodemap = phekg_nodemap[phekg_nodemap["ntype"] == "ICD9CM"]
  phekg_nodemap["node_id"] = phekg_nodemap["node_id"].str.replace(".", "")

  # Load embeddings of nodes in KG
  phekg_emb = pickle.load(open(kg_dir + "full_h_embed_hms.pkl", "rb"))
  return phekg_nodemap, phekg_emb


def get_time(s):
  time = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
  assert type(time) == datetime
  return time


def get_patient_profile(data, filter_list, pt):
  patient_df = data[data.subject_id == pt]
  patient_df = patient_df[["subject_id", "charttime", "itemid", "valuenum"]]
  pt_ct = patient_df.pivot(index="charttime", columns="itemid", values="valuenum")
  return pt_ct


# Calculate maximum number of timepoints
def calc_max_timepoints(data):
  all_patients = [get_patient_profile(p) for p in data.subject_id.unique()]
  max_timepoints = 0
  for pt in all_patients:
      num_t = pt.shape[0]
  if num_t > max_timepoints:
      max_timepoints = num_t
  print(max_timepoints)
  return max_timepoints


def save_data(f, data):
    print("Saving data to %s..." % f)
    if f.endswith(".csv"):
      data.to_csv(f, sep = "\t", index = False)
    elif f.endswith(".json"):
        with open(f, "w") as outfile:
          json.dump(data, outfile)
    elif f.endswith(".pth"):
        torch.save(data, f)
    else:
      raise NotImplementedError