import pandas as pd
import torch
from collections import Counter
import json


def parse_codes(codes):
    parsed = []
    for c in codes:
        parsed.extend(c.split(";"))
    return list(set(parsed))


def create_cohort(patient_data, disease_code, exclude_codes_in_healthy, threshold):

    def is_not_healthy(pt_codes):
        overlap_codes = set(pt_codes).intersection(set(exclude_codes_in_healthy))
        if len(overlap_codes) == 0: return True
        else: return False

    patients_A = [] # Presence ("disease")
    codes_in_A = []
    patients_B = [] # Absence ("healthy")
    codes_in_B = []
    for pt, val in patient_data.items():
        pt_codes = parse_codes(val["codes"])
        if is_not_healthy(pt_codes):
            if disease_code in pt_codes:
                patients_A.append(pt)
                codes_in_A.extend(pt_codes)
            else:
                patients_B.append(pt)
                codes_in_B.extend(pt_codes)
    print("Number of patients in A (disease):", len(patients_A))
    print("Number of patients in B (healthy):", len(patients_B))

    # Co-occurring codes?
    co_occurring_codes = set(codes_in_A).intersection(set(codes_in_B))
    print("Shared codes:", len(co_occurring_codes))
    codes_in_A = Counter(codes_in_A)
    codes_in_B = Counter(codes_in_B)
    remove_codes = set()
    for c in co_occurring_codes:
        if codes_in_A[c] > threshold and codes_in_B[c] > threshold: remove_codes.add(c)
    print("Remove codes:", len(remove_codes))
    print(remove_codes)

    return patients_A, patients_B, remove_codes


def find_matches(patient_data, icd10to9, disease_code, patients_A, patients_B, remove_codes):

    # How many patients have a match with the other cohort?
    matches = dict()
    for pt_A in patients_A:
        pt_A_val = patient_data[pt_A]
        A_codes = set(parse_codes(pt_A_val["codes"])).difference(remove_codes)
        for pt_B in patients_B:
            pt_B_val = patient_data[pt_B]

            # Compare
            if pt_A_val["gender"] == pt_B_val["gender"] and pt_A_val["age_trunc"] == pt_B_val["age_trunc"]:
                B_codes = set(parse_codes(pt_B_val["codes"])).difference(remove_codes)

                AB_intersect = A_codes.intersection(B_codes)
                AB_union = A_codes.union(B_codes)
                
                if len(AB_intersect) > 0.5 * len(AB_union):
                    print("Match:", pt_A, pt_B)
                    assert disease_code in A_codes and disease_code not in B_codes
                    if pt_A not in matches: matches[pt_A] = []
                    if pt_B not in matches: matches[pt_B] = []
                    matches[pt_A].append(pt_B)
                    matches[pt_B].append(pt_A)

    for pt_A in patients_A:
        if pt_A not in matches: continue
        assert len(matches[pt_A]) > 0
        matches[pt_A] = {"cohort": icd10to9[disease_code], "match": matches[pt_A]}
    for pt_B in patients_B:
        if pt_B not in matches: continue
        assert len(matches[pt_B]) > 0, len(matches[pt_B])
        matches[pt_B] = {"cohort": ("No %s" % disease_code), "match": matches[pt_B]}

    print("Number of total matches:", len(matches) / 2)

    return matches


def read_patient_data(patient_dir):
    patient_data = json.load(open(patient_dir + "/patient_data.json", "r"))
    print("Number of patients:", len(patient_data))

    # Compute most common codes
    codes_freq = []
    for pt, val in patient_data.items():
        codes_freq.extend(val["codes"])
    codes_freq = Counter(codes_freq)
    for c, f in codes_freq.most_common(100):
        if c.startswith("drg:"): continue
        if f <= len(patient_data) / 2: print(c, f)

    return patient_data


def main():

    
    ######################################################################
    # 1. Read patient data
    ######################################################################
    
    patient_dataset = "MIMIC-IV"
    #patient_dataset = "eICU"

    patient_dir = "data/" + patient_dataset
    patient_data = read_patient_data(patient_dir)
    if patient_dataset == "eICU":
        lab_data = pd.read_csv(patient_dir + "/filtered_lab_code_data.csv", sep="\t")
        patient_ids = lab_data[["subject_id_str", "subject_id"]].drop_duplicates()
        patient_id_mapper = {k: str(v) for k, v in zip(patient_ids["subject_id_str"].tolist(), patient_ids["subject_id"].tolist())}
        patient_data = {patient_id_mapper[p]: v for p, v in patient_data.items() if p in patient_id_mapper}

    ######################################################################
    # 2. Select code to get disease versus no disease)
    ######################################################################
    
    disease_code = "E10"
    exclude_codes_in_healthy = ["E11", "E13", "E12", "E08", "E09", "R73", "O24"] # "E16", "P70"
    threshold = 20
    icd10to9 = {"E10": "250"}

    ######################################################################
    # 3. Create disease cohort (A) and healthy cohort (B)
    ######################################################################

    patients_A, patients_B, remove_codes = create_cohort(patient_data, disease_code, exclude_codes_in_healthy, threshold)

    ######################################################################
    # 4. Find and save matches
    ######################################################################

    matches = find_matches(patient_data, icd10to9, disease_code, patients_A, patients_B, remove_codes)

    with open(patient_dir + "/matched_patients.json", "w") as outfile:
        json.dump(matches, outfile)


if __name__ == "__main__":
    main()