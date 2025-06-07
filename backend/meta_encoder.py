import torch

def encode_meta(meta: dict):
    """
    meta = {
        'gender': 'sex_female' or 'sex_male',
        'age': '3',   # 정수 문자열 (0~10)
        'body_part': one of SCIN part keys
    }
    """
    vec = []

    # 0. textures_raised_or_bumpy → default 0
    vec.append(0)

    # 1. fitzpatrick_skin_type_encoded → default 0
    vec.append(0)

    # 2. sex_female
    vec.append(1 if meta['gender'] == 'sex_female' else 0)

    # 3. condition_symptoms_burning → default 0
    vec.append(0)

    # 4. body_parts_head_or_neck
    vec.append(1 if meta['body_part'] == 'body_parts_head_or_neck' else 0)

    # 5. body_parts_leg
    vec.append(1 if meta['body_part'] == 'body_parts_leg' else 0)

    # 6. textures_rough_or_flaky → default 0
    vec.append(0)

    # 7. condition_symptoms_darkening → default 0
    vec.append(0)

    # 8. condition_symptoms_bothersome_appearance → default 0
    vec.append(0)

    # 9. age_group_ordinal
    vec.append(float(meta['age']))

    # 10. condition_symptoms_increasing_size → default 0
    vec.append(0)

    # 11. body_parts_arm
    vec.append(1 if meta['body_part'] == 'body_parts_arm' else 0)

    # 12. other_symptoms_no_relevant_symptoms → default 0
    vec.append(0)

    # 13. sex_other_or_unspecified → default 0
    vec.append(0)

    # 14. body_parts_foot_top_or_side
    vec.append(1 if meta['body_part'] == 'body_parts_foot_top_or_side' else 0)

    # 15. condition_symptoms_itching → default 0
    vec.append(0)

    # 16. race_ethnicity_white → default 1 (기본값)
    vec.append(1)

    # 17. body_parts_buttocks
    vec.append(1 if meta['body_part'] == 'body_parts_buttocks' else 0)

    return torch.tensor(vec).float()
