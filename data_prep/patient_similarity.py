# Compute patient similarity
#   1. Extract patient codes (e.g., diagnosis, medication)
#   2. Define metrics for patient similarity
#   3. Optional: Compute data statistics of patients' similarities


import random
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def resolve_gender_conflicts(data, pid_col):
    patient_gender = data[[pid_col, "gender"]].drop_duplicates().groupby([pid_col]).count()
    conflict_gender_entries = patient_gender[patient_gender["gender"] > 1].index.tolist()
    no_conflict_data = data[~data[pid_col].isin(conflict_gender_entries)]
    conflict_data = data[data[pid_col].isin(conflict_gender_entries)]
    
    resolve_conflict = dict()
    for p in conflict_data[pid_col].unique():
        p_df = conflict_data[conflict_data[pid_col] == p]

        if "Other" in p_df["gender"].unique():
            resolve_conflict[p] = "Other"
        else:
            resolve_conflict[p] = "Unknown"
    resolve_conflict = pd.DataFrame({pid_col: list(resolve_conflict.keys()), "gender": list(resolve_conflict.values())})
    print(resolve_conflict)

    new_patient_data = pd.concat([no_conflict_data, resolve_conflict])
    assert len(data[pid_col].unique()) == len(new_patient_data[pid_col].unique())
    return new_patient_data


def load_patient_data(data_dir, data_source):
    if data_source == "MIMIC":
        patient_data = pd.read_csv(data_dir + "hosp/patients.csv")
        # Use gender, age
        patient_data = patient_data[["subject_id", "gender", "anchor_age"]]
        print("Unique number of patients in patient_data:", len(patient_data["subject_id"].unique()))
        print("Distribution of ages:", Counter(patient_data["anchor_age"].tolist()))
        print("Distribution of gender:", Counter(patient_data["gender"].tolist()))
        
        diagnoses = pd.read_csv(data_dir + "hosp/diagnoses_icd.csv")
        # Use ICD code, seq_num
        diagnoses = diagnoses[["subject_id", "icd_code", "icd_version", "seq_num"]]
        print("Unique number of patients in diagnoses:", len(diagnoses["subject_id"].unique()))
        print("Distribution of diagnoses ICD codes:", Counter(diagnoses["icd_code"].tolist()).most_common(10))
        print("Distribution of diagnoses ICD versions:", Counter(diagnoses["icd_version"].tolist()))
        print("Distribution of unique diagnoses per patient:", Counter(diagnoses[["subject_id", "icd_code"]].drop_duplicates()["subject_id"].tolist()).most_common(10))

        drgcodes = pd.read_csv(data_dir + "hosp/drgcodes.csv")
        # Use drg_code
        drgcodes = drgcodes[["subject_id", "drg_code"]]
        print("Unique number of patients in drgcodes:", len(drgcodes["subject_id"].unique()))
        print("Distribution of drug codes:", Counter(drgcodes["drg_code"].tolist()).most_common(10))
        print("Distribution of unique drug codes per patient:", Counter(drgcodes[["subject_id", "drg_code"]].drop_duplicates()["subject_id"].tolist()).most_common(10))
        
        # Sanity check that ICD-9 and ICD-10 have no overlapping codes
        dx_icd9 = diagnoses[diagnoses["icd_version"] == "9"]["icd_code"].unique()
        dx_icd10 = diagnoses[diagnoses["icd_version"] == "10"]["icd_code"].unique()
        print("Overlapping codes in diagnosis:", set(list(dx_icd9)).intersection(set(list(dx_icd10))))
        assert len(set(list(dx_icd9)).intersection(set(list(dx_icd10)))) == 0

    elif data_source == "eICU":
        patient_data = pd.read_csv(data_dir + "patient.csv")
        # Use gender, age (convert to age group?)
        patient_data = patient_data[["uniquepid", "patientunitstayid", "gender", "age"]].dropna()
        patient_data["anchor_age"] = [int(a) if a != "> 89" else 89 for a in patient_data["age"]]
        print(patient_data)
        print("Unique number of patients in patient_data:", len(patient_data["uniquepid"].unique()))
        print("Distribution of ages:", Counter(patient_data["anchor_age"].tolist()))
        print("Distribution of gender:", Counter(patient_data["gender"].tolist()))
        visit2pid_map = dict()
        for visit_id, pid in zip(patient_data["patientunitstayid"].tolist(), patient_data["uniquepid"].tolist()):
            assert visit_id not in visit2pid_map
            visit2pid_map[visit_id] = pid
        print("Number of total visits:", len(visit2pid_map))
        patient_data = patient_data[["uniquepid", "gender", "anchor_age"]].drop_duplicates()
        # Resolve conflicts in gender and age
        patient_gender = resolve_gender_conflicts(patient_data[["uniquepid", "gender"]], "uniquepid")
        patient_age = patient_data[["uniquepid", "anchor_age"]].groupby(["uniquepid"]).max().reset_index() # Take max age
        assert len(patient_age["uniquepid"].unique()) == len(patient_gender["uniquepid"].unique())
        assert len(patient_data["uniquepid"].unique()) == len(patient_gender["uniquepid"].unique())
        patient_data = patient_gender.merge(patient_age, on = "uniquepid")
        print(patient_data)

        diagnoses = pd.read_csv(data_dir + "diagnosis.csv")
        # Use ICD code, seq_num
        diagnoses = diagnoses[["patientunitstayid", "icd9code"]].dropna()
        diagnoses["uniquepid"] = diagnoses["patientunitstayid"].map(visit2pid_map)
        diagnoses = diagnoses[["uniquepid", "icd9code"]].drop_duplicates()
        print(diagnoses)
        print("Unique number of patients in diagnoses:", len(diagnoses["uniquepid"].unique()))
        print("Distribution of diagnoses ICD codes:", Counter(diagnoses["icd9code"].tolist()).most_common(10))
        print("Distribution of unique diagnoses per patient:", Counter(diagnoses[["uniquepid", "icd9code"]].drop_duplicates()["uniquepid"].tolist()).most_common(10))

        drgcodes = pd.read_csv(data_dir + "medication.csv") # Note: drughiclseqno has many NaNs
        # Use drg_code
        drgcodes = drgcodes[["patientunitstayid", "drugname", "gtc"]].drop_duplicates().dropna()
        print(drgcodes)
        drugname2gtc = dict()
        for drugname, gtc in zip(drgcodes["drugname"].tolist(), drgcodes["gtc"].tolist()):
            if drugname in drugname2gtc:
                #print(drugname, drugname2gtc[drugname], gtc)
                if gtc not in drugname2gtc[drugname]: drugname2gtc[drugname].append(gtc)
            else: drugname2gtc[drugname] = [gtc]
        print("Number of drugs:", len(drugname2gtc))
        print("Number of drugs with multiple GTCs:", len([d for d, i in drugname2gtc.items() if len(i) > 1]))
        drgcodes["uniquepid"] = drgcodes["patientunitstayid"].map(visit2pid_map)
        drgcodes = drgcodes[["uniquepid", "gtc"]].drop_duplicates().rename(columns = {"gtc": "drg_code"})
        print(drgcodes)
        print("Unique number of patients in drgcodes:", len(drgcodes["uniquepid"].unique()))
        print("Distribution of drug codes:", Counter(drgcodes["drg_code"].tolist()).most_common(10))
        print("Distribution of unique drug codes per patient:", Counter(drgcodes[["uniquepid", "drg_code"]].drop_duplicates()["uniquepid"].tolist()).most_common(10))
        
    else:
        raise NotImplementedError

    data_dict = {
                 "data_source": data_source,
                 "patient_data": patient_data,
                 "diagnoses": diagnoses,
                 "drgcodes": drgcodes,
                }
    
    return data_dict


def convert_age(data_dict):

    def trunc_age(x):
        return np.trunc(x / 10) * 10
    
    patient_data = data_dict["patient_data"]
    print(patient_data)
    patient_data["age_trunc"] = trunc_age(patient_data["anchor_age"].to_numpy())
    patient_data = patient_data.drop(columns = ["anchor_age"]).drop_duplicates()
    print(patient_data)
    print("Updating patient_data in data_dict...")
    data_dict["patient_data"] = patient_data
    return data_dict


# Clinical Modification (CM) https://data.nber.org/gem/icd9toicd10cmgem.csv
def read_icd_maps(f):

    def get_icd_mapping(df, reverse = False):
        if not reverse:
            icd_a_col = "icd9cm"
            icd_b_col = "icd10cm"
        else:
            icd_a_col = "icd10cm"
            icd_b_col = "icd9cm"
        mapping = dict()
        for icd_a, icd_b in zip(df[icd_a_col].tolist(), df[icd_b_col].tolist()):
            if str(icd_a) in mapping:
                mapping[str(icd_a)].append(str(icd_b))
            else: mapping[str(icd_a)] = [str(icd_b)]
        mapping = {k: ";".join(list(set(v))) for k, v in mapping.items()}
        return mapping

    cm_maps = pd.read_csv(f + "icd9toicd10cmgem.csv")
    cm_rev_maps = pd.read_csv(f + "icd10toicd9gem.csv")
    
    cm_mapping = get_icd_mapping(cm_maps)
    cm_rev_mapping = get_icd_mapping(cm_rev_maps, reverse = True)

    assert len(cm_mapping) == len(cm_maps["icd9cm"].unique()), (len(cm_mapping), len(cm_maps["icd9cm"].unique()))
    assert len(cm_rev_mapping) == len(cm_rev_maps["icd10cm"].unique()), (len(cm_rev_mapping), len(cm_rev_maps["icd10cm"].unique()))

    return cm_mapping, cm_rev_mapping


def extract_icd_category(data_dict, cm_mapping):

    def split_trunc(pt, n):
        return ";".join(list(set([i[0:n] for i in pt.split(";")])))

    for data_type in ["diagnoses"]:
        pt_data = data_dict[data_type]
        print(pt_data)
        
        if data_dict["data_source"] == "MIMIC":

            icd10 = pt_data[pt_data["icd_version"] == 10]
            print(icd10)
            
            icd9 = pt_data[pt_data["icd_version"] == 9]
        
            if data_type == "diagnoses":
                
                # Unmappable ICD9 to ICD10 (i.e., map to itself)
                for i in icd9["icd_code"].tolist():
                    if i not in cm_mapping: cm_mapping[i] = i

                icd9["icd9_code"] = icd9["icd_code"]
                icd9["icd_code"] = icd9["icd_code"].map(cm_mapping)

                assert len(icd9[icd9["icd_code"].isna()]) == 0
            
            new_pt_data = pd.concat([icd10, icd9])
            assert len(new_pt_data[new_pt_data["icd_code"].isna()]) == 0
            new_pt_data["icd_trunc"] = [split_trunc(i, 3) for i in new_pt_data["icd_code"].astype(str)]
            print(new_pt_data)
            print(new_pt_data[new_pt_data["icd_trunc"].str.contains(";")])
            print("Updating %s in data_dict..." % data_type)
            data_dict[data_type] = new_pt_data
    
        elif data_dict["data_source"] == "eICU":
            
            code_mapping = dict()
            for code in pt_data["icd9code"].unique():
                codes = [i.strip().replace(".", "") for i in code.split(",")]

                # Map ICD9 to ICD10 (or map to self)
                codes = ";".join([cm_mapping[i] if i in cm_mapping else i for i in codes])
                code_mapping[code] = split_trunc(codes, 3)
            
            pt_data["icd_trunc"] = pt_data["icd9code"].map(code_mapping)
            print(pt_data)
            print(pt_data[pt_data["icd_trunc"].str.contains(";")])
            print("Updating %s in data_dict..." % data_type)
            data_dict[data_type] = pt_data

        else:
            raise NotImplementedError
        
    return data_dict


def jaccard_similarity(p1, p2):
    return len(p1.intersection(p2)) / len(p1.union(p2))


def parse_codes(pt):
    codes = []
    for i in pt:
        codes.extend(i.split(";"))
    return set(codes)


def calc_similarity(data_dict, data_type):

    if data_dict["data_source"] == "MIMIC":
        pid_col = "subject_id"
    elif data_dict["data_source"] == "eICU":
        pid_col = "uniquepid"
    else:
        raise NotImplementedError

    patient_demog = data_dict["patient_data"]
    patient_codes = []
    for d in data_type:
        patient_d = data_dict[d]
        if d == "diagnoses":
            patient_d = patient_d[[pid_col, "icd_trunc"]].rename(columns={"icd_trunc": "code"})
        elif d == "drgcodes":
            patient_d = patient_d[[pid_col, "drg_code"]].rename(columns={"drg_code": "code"})
            patient_d["code"] = "drg:" + patient_d["code"].astype(str)
        else: raise NotImplementedError
        patient_codes.append(patient_d)
    patient_codes = pd.concat(patient_codes)
    print(patient_codes)

    sample_patients = random.sample(list(patient_codes[pid_col].unique()), 1000)
    patient_codes_subset = patient_codes[patient_codes[pid_col].isin(sample_patients)]
    print(patient_codes_subset)

    all_sims = []
    all_pairs = []
    for pt1 in patient_codes_subset[pid_col].unique():
        pt1_sims = []
        for pt2 in patient_codes_subset[pid_col].unique():
            if pt1 == pt2: continue
            if pt2 in all_pairs: continue
            
            pt1_data = patient_codes_subset[patient_codes_subset[pid_col] == pt1]
            pt2_data = patient_codes_subset[patient_codes_subset[pid_col] == pt2]
            
            pt1_codes = parse_codes(pt1_data["code"].unique())
            pt2_codes = parse_codes(pt2_data["code"].unique())

            sim = jaccard_similarity(set(pt1_codes), set(pt2_codes))
            pt1_sims.append(sim)

        all_sims.extend(pt1_sims)
        all_pairs.append(pt1) # List of patients that did all pairwise comparisons
        if len(pt1_sims) > 0:
            print("Average %s sim (%s):" % ("-".join(data_type), pt1), np.mean(pt1_sims))
            print("Median %s sim (%s):" % ("-".join(data_type), pt1), np.median(pt1_sims))
            print("Min %s sim (%s):" % ("-".join(data_type), pt1), min(pt1_sims))
            print("Max %s sim (%s):" % ("-".join(data_type), pt1), max(pt1_sims))

    ax = sns.histplot(all_sims)
    plt.ylabel("Frequency")
    plt.xlabel("Similarity")

    # Format plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(bottom = False)
    plt.tight_layout()
    plt.savefig("subset_%s_sim.pdf" % ("-".join(data_type)))
    plt.close()


def save_patient_json(data_dict, data_type, f):

    if data_dict["data_source"] == "MIMIC":
        pid_col = "subject_id"
    elif data_dict["data_source"] == "eICU":
        pid_col = "uniquepid"
    else:
        raise NotImplementedError

    patient_demog = data_dict["patient_data"]
    patient_codes = []
    for d in data_type:
        patient_d = data_dict[d]
        if d == "diagnoses":
            patient_d = patient_d[[pid_col, "icd_trunc"]].rename(columns={"icd_trunc": "code"})
        elif d == "drgcodes":
            patient_d = patient_d[[pid_col, "drg_code"]].rename(columns={"drg_code": "code"})
            patient_d["code"] = "drg:" + patient_d["code"].astype(str)
        else: raise NotImplementedError
        patient_codes.append(patient_d)
    patient_codes = pd.concat(patient_codes)
    print(patient_codes)

    num_pts = len(patient_demog[pid_col].unique())
    code_counter = dict(Counter(remove_delim(patient_codes["code"].tolist())))
    filter_out_codes = []
    for k, v in code_counter.items():
        if v / num_pts > 0.2:
            print(k, v, v / num_pts)
            if v / num_pts > 0.5: filter_out_codes.append(k)
    print("Maybe should filter out (>0.5 of patients):", filter_out_codes)

    save_dict = dict()
    for idx, pt in enumerate(patient_demog[pid_col].unique()):
        print("Running %d out of %d..." % (idx, num_pts))
        pt_demog = patient_demog[patient_demog[pid_col] == pt]
        assert len(pt_demog) == 1, pt_demog
        pt_codes = patient_codes[patient_codes[pid_col] == pt]
        if len(pt_codes) == 0: continue
        pt = str(pt)
        save_dict[pt] = dict()
        save_dict[pt]["gender"] = pt_demog["gender"].tolist()[0]
        save_dict[pt]["age_trunc"] = str(pt_demog["age_trunc"].tolist()[0])
        save_dict[pt]["codes"] = pt_codes["code"].tolist()
    
    with open(f, "w") as outfile:
        json.dump(save_dict, outfile)

    print("Finished saving to %s" % f)
    
    return save_dict


def remove_delim(data, delim = ";"):
    new_data = []
    for i in data:
        if delim in i: i = i.split(delim)
        else: i = [i]
        new_data.extend(i)
    return new_data


def main():
    cm_mapping, cm_rev_mapping = read_icd_maps("data/icd_mappings/")

    data_source = "eICU" # "MIMIC_IV"
    data_dir = "data/raw/%s/" % data_source
    save_dir = "data/%s/" % data_source
    
    # Make sure the correct variables are uncommented 
    assert data_source in data_dir
    assert data_source in save_dir

    data_dict = load_patient_data(data_dir, data_source)
    print("Data available:", data_dict.keys())    

    # Processing steps
    data_dict = convert_age(data_dict)
    data_dict = extract_icd_category(data_dict, cm_mapping) # First three numbers in code

    # Save JSON data
    save_patient_json(data_dict, ["diagnoses", "drgcodes"], save_dir + "patient_data.json")

    # Compute similarity
    #   Warning: Computationally intensive
    #dx_sims = calc_similarity(data_dict, ["diagnoses"])
    #drg_sims = calc_similarity(data_dict, ["drgcodes"])
    #dxdrg_sims = calc_similarity(data_dict, ["diagnoses", "drgcodes"])
    

if __name__ == "__main__":
    main()




