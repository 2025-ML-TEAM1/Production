import torch

def encode_meta(meta: dict):
    vec = []

    vec.extend([0])  # rc_unknown
    vec.extend([0])  # rc_rash
    vec.extend([0])  # condition_symptoms_itching
    vec.extend([0])  # other_symptoms_no_relevant_symptoms
    vec.extend([0])  # textures_raised_or_bumpy
    vec.extend([0])  # fitzpatrick_skin_type_encoded
    vec.extend([0])  # sex_other_or_unspecified
    vec.extend([float(meta["age"])])  # age_group_ordinal
    vec.extend([0])  # textures_rough_or_flaky
    vec.extend([0])  # derm_fitz_2
    vec.extend([0])  # derm_fitz_3
    vec.extend([1])  # race_ethnicity_white (기본값)
    vec.extend([0])  # condition_symptoms_bothersome_appearance
    vec.extend([1 if meta["gender"] == "sex_female" else 0])  # sex_female
    vec.extend([1 if meta["body_part"] == "body_parts_arm" else 0])
    vec.extend([1 if meta["body_part"] == "body_parts_leg" else 0])
    vec.extend([0])  # condition_symptoms_increasing_size
    vec.extend([0])  # condition_symptoms_darkening
    vec.extend([0])  # condition_symptoms_burning
    vec.extend([0])  # textures_flat
    vec.extend([0])  # condition_symptoms_bleeding
    vec.extend([1 if meta["body_part"] == "body_parts_torso_front" else 0])
    vec.extend([1 if meta["gender"] == "sex_male" else 0])  # sex_male
    vec.extend([0])  # other_symptoms_fatigue
    vec.extend([0])  # condition_symptoms_pain
    vec.extend([0])  # derm_fitz_1
    vec.extend([0])  # other_symptoms_joint_pain
    vec.extend([1 if meta["body_part"] == "body_parts_torso_back" else 0])
    vec.extend([1 if meta["body_part"] == "body_parts_back_of_hand" else 0])
    vec.extend([1 if meta["body_part"] == "body_parts_head_or_neck" else 0])

    return torch.tensor(vec).float()
