import torch
from PIL import Image
import io
from torchvision import transforms
from model import MultiTaskResNetWithMeta
from meta_encoder import encode_meta
from google import genai


# 클래스 라벨 정의 (질환 클래스는 너 프로젝트 기준에 맞게 작성해야 함)
idx_to_label = ["abrasion,_scrape,_or_scab","abscess","acne","actinic_keratosis","acute_and_chronic_dermatitis","acute_dermatitis,_nos","allergic_contact_dermatitis","basal_cell_carcinoma","bullous_pemphigoid","burn_of_skin","cd_-_contact_dermatitis","candida_intertrigo","cellulitis","chronic_dermatitis,_nos","condyloma_acuminatum","cutaneous_t_cell_lymphoma","cutaneous_larva_migrans","cutaneous_lupus","cutaneous_sarcoidosis","cyst","dermatofibroma","drug_rash","ecthyma","eczema","erythema_ab_igne","erythema_migrans","erythema_multiforme","folliculitis","geographic_tongue","granuloma_annulare","hemangioma","herpes_simplex","herpes_zoster","hidradenitis","hypersensitivity","impetigo","infected_eczema","inflicted_skin_lesions","insect_bite","intertrigo","irritant_contact_dermatitis","kaposi's_sarcoma_of_skin","keratosis_pilaris","leukocytoclastic_vasculitis","lichen_simplex_chronicus","lichen_nitidus","lichen_planus_lichenoid_eruption","lichen_spinulosus","livedo_reticularis","melanocytic_nevus","miliaria","molluscum_contagiosum","o_e_-_ecchymoses_present","onychomycosis","perioral_dermatitis","photodermatitis","pigmented_purpuric_eruption","pityriasis_lichenoides","pityriasis_rosea","porokeratosis","post-inflammatory_hyperpigmentation","prurigo_nodularis","psoriasis","purpura","rosacea","scc_sccis","sk_isk","scabies","scar_condition","seborrheic_dermatitis","skin_and_soft_tissue_atypical_mycobacterial_infection","skin_infection","stasis_dermatitis","superficial_wound_of_body_region","syphilis","tinea","tinea_versicolor","urticaria","vasculitis_of_the_skin","verruca_vulgaris","viral_exanthem","vitiligo","xerosis"]

duration_label = [
    'one_day',
    'less_than_one_week',
    'one_to_four_weeks',
    'one_to_three_months',
    'three_to_twelve_months',
    'more_than_one_year',
    'more_than_five_years',
    'since_childhood',
    'unknown',
    'unspecified'
]

# 모델 초기화 및 weight 불러오기
model = MultiTaskResNetWithMeta()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_disease(image_bytes, meta):
    # 1. 이미지 로딩 및 전처리
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)        # [1, 3, 224, 224]

    # 2. 메타 인코딩
    print(f"[LOG] meta: {meta}")  # 디버깅용 로그
    meta_tensor = encode_meta(meta).unsqueeze(0)        # [1, 18] ← 정확하게 맞춘 구조

    # 3. 모델 예측
    with torch.no_grad():
        out_duration, out_disease = model(image_tensor, meta_tensor)

    # 4. 결과 처리
    dur_idx = torch.argmax(out_duration, dim=1).item()
    dis_idx = torch.argmax(out_disease, dim=1).item()

    # 5. 방어 코드 + 로그
    print(f"[LOG] duration idx: {dur_idx}, disease idx: {dis_idx}")

    duration = duration_label[dur_idx] if 0 <= dur_idx < len(duration_label) else "unknown"
    disease  = idx_to_label[dis_idx]  if 0 <= dis_idx  < len(idx_to_label)  else "unknown"

    client = genai.Client(api_key="AIzaSyCSfdrLd6jzEIWQFNOLa0Owgi-4NhcCMns")

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=f"""너는 피부 질병 전문가야. 사용자의 증상과 경과 기간을 알려줄테니, 
        증상의 일반적인 예상 회복 기간을 확인하고 사용자의 경과 시간과 비교하여, 
        회복까지 남은 예상 회복 기간이 임계값을 초과할 경우 의료기관 방문을 권유해.
        임계값을 초과하지 않았을 경우 증상의 경과 기간과 회복 예측 기간을 바탕으로 적절한 자가 치료 방법 또는 생활 습관 개선 팁을 제공해.

        답변 룰
        - 한국어로 답변
        - 답변만 작성하고, 다른 말은 하지 않아
        - 사용자는 전문가가 아니므로, 이해하기 쉬운 언어로 답변
        - 치료법은 약물, 생활습관, 식이요법 등 다양한 방법을 포함
        - 치료법은 증상과 지속기간에 따라 다르게 제시
        - 치료법은 구체적이고 실용적인 방법으로 제시
        - 출처 표시
        - 답변 마지막에는 꼭 의사와 상담을 권장하는 문구를 포함


        입력
        - 증상: {disease}
        - 경과과 기간: {duration}
        """
    )
    print(response.text)



    return {
        "disease": disease,
        "duration": duration,
        "treatment": response.text
    }
