# ─────────────────────────────────────────────────────────────────────────
# 7) 모델 정의 (Dropout 추가된 MultiTaskResNet)
# ─────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

# 학습용 데이터 df_train에서 실제 라벨 개수를 구함
meta_cols = [
    "textures_raised_or_bumpy",
    "fitzpatrick_skin_type_encoded",
    "sex_female",
    "condition_symptoms_burning",
    "body_parts_head_or_neck",
    "body_parts_leg",
    "textures_rough_or_flaky",
    "condition_symptoms_darkening",
    "condition_symptoms_bothersome_appearance",
    "age_group_ordinal",
    "condition_symptoms_increasing_size",
    "body_parts_arm",
    "other_symptoms_no_relevant_symptoms",
    "sex_other_or_unspecified",
    "body_parts_foot_top_or_side",
    "condition_symptoms_itching",
    "race_ethnicity_white",
    "body_parts_buttocks",
]
onehot_cols_train = ["abrasion,_scrape,_or_scab","abscess","acne","actinic_keratosis","acute_and_chronic_dermatitis","acute_dermatitis,_nos","allergic_contact_dermatitis","basal_cell_carcinoma","bullous_pemphigoid","burn_of_skin","cd_-_contact_dermatitis","candida_intertrigo","cellulitis","chronic_dermatitis,_nos","condyloma_acuminatum","cutaneous_t_cell_lymphoma","cutaneous_larva_migrans","cutaneous_lupus","cutaneous_sarcoidosis","cyst","dermatofibroma","drug_rash","ecthyma","eczema","erythema_ab_igne","erythema_migrans","erythema_multiforme","folliculitis","geographic_tongue","granuloma_annulare","hemangioma","herpes_simplex","herpes_zoster","hidradenitis","hypersensitivity","impetigo","infected_eczema","inflicted_skin_lesions","insect_bite","intertrigo","irritant_contact_dermatitis","kaposi's_sarcoma_of_skin","keratosis_pilaris","leukocytoclastic_vasculitis","lichen_simplex_chronicus","lichen_nitidus","lichen_planus_lichenoid_eruption","lichen_spinulosus","livedo_reticularis","melanocytic_nevus","miliaria","molluscum_contagiosum","o_e_-_ecchymoses_present","onychomycosis","perioral_dermatitis","photodermatitis","pigmented_purpuric_eruption","pityriasis_lichenoides","pityriasis_rosea","porokeratosis","post-inflammatory_hyperpigmentation","prurigo_nodularis","psoriasis","purpura","rosacea","scc_sccis","sk_isk","scabies","scar_condition","seborrheic_dermatitis","skin_and_soft_tissue_atypical_mycobacterial_infection","skin_infection","stasis_dermatitis","superficial_wound_of_body_region","syphilis","tinea","tinea_versicolor","urticaria","vasculitis_of_the_skin","verruca_vulgaris","viral_exanthem","vitiligo","xerosis"]
num_labels_train = 83
print(f"Number of disease labels: {len(onehot_cols_train)}")  # e.g. 83
# meta_cols를 위에서 이미 정의했으므로 그 길이를 사용
num_meta_features = len(meta_cols)  # e.g. 31

class MultiTaskResNetWithMeta(nn.Module):
    def __init__(self,
                 num_duration_classes=10,
                 num_disease_labels=num_labels_train,
                 num_meta_features=num_meta_features,
                 dropout_p=0.5):
        super().__init__()
        # ─── 1) Backbone (ResNet50) ───────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features    # 보통 2048
        backbone.fc = nn.Identity()           # 마지막 FC를 삭제하여 특징 벡터만 추출
        self.backbone = backbone

        # ─── 2) Meta 데이터용 FC (31 → mid_meta_dim) ──────────────────────
        mid_meta_dim = 64
        self.meta_net = nn.Sequential(
            nn.Linear(num_meta_features, mid_meta_dim),
            nn.BatchNorm1d(mid_meta_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        # ─── 3) 이미지 + 메타 특징을 합한 뒤 배치정규화 + 드롭아웃 ────────────
        merged_dim = feat_dim + mid_meta_dim   # 2048 + 64 = 2112
        self.post_merge_bn = nn.BatchNorm1d(merged_dim)
        self.post_merge_dropout = nn.Dropout(p=dropout_p)

        # ─── 4) Duration 분기 ─────────────────────────────────────────────
        dur_branch_dim = 256
        self.dur_branch = nn.Sequential(
            nn.Linear(merged_dim, dur_branch_dim),
            nn.BatchNorm1d(dur_branch_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(dur_branch_dim, num_duration_classes)
        )

        # ─── 5) Disease 분기 ──────────────────────────────────────────────
        dis_branch_dim = 256
        self.dis_branch = nn.Sequential(
            nn.Linear(merged_dim, dis_branch_dim),
            nn.BatchNorm1d(dis_branch_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(dis_branch_dim, num_disease_labels)
        )

    def forward(self, x_img, x_meta):
        # 1) 이미지 특징 추출
        feat_img = self.backbone(x_img)         # (batch_size, feat_dim)

        # 2) 메타 특징 추출
        feat_meta = self.meta_net(x_meta)       # (batch_size, mid_meta_dim)

        # 3) 두 벡터 합치기
        merged_feat = torch.cat([feat_img, feat_meta], dim=1)  # (batch_size, merged_dim)

        # 4) 병합 후 정규화 + 드롭아웃
        merged_feat = self.post_merge_bn(merged_feat)
        merged_feat = self.post_merge_dropout(merged_feat)

        # 5) Duration, Disease 분기
        out_duration = self.dur_branch(merged_feat)  # (batch_size, num_duration_classes)
        out_disease  = self.dis_branch(merged_feat)  # (batch_size, num_disease_labels)
        return out_duration, out_disease