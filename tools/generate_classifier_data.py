"""Generate longer, more realistic synthetic training documents (no PHI).

The original classifier_data/ files are one-sentence templates, so the model was
unsure on real multi-section documents. These add 12 realistic documents per
class on topics that do NOT overlap with the demo samples (no acne, no knee),
so the demo documents remain a fair unseen test.

Run: python tools/generate_classifier_data.py
"""
import random
from pathlib import Path

random.seed(7)
OUT = Path("classifier_data")

CLINICS = ["Riverside Family Clinic", "Northgate Internal Medicine", "Lakeview Medical Group",
           "Hillcrest Primary Care", "Westside Health Center", "Oak Park Specialty Clinic"]
DOCTORS = ["Dr. A. Rivera", "Dr. B. Chen", "Dr. C. Okafor", "Dr. D. Patel", "Dr. E. Novak", "Dr. F. Haddad"]

NOTES = [
    ("hypertension follow-up", "blood pressure readings at home around 150/95", "BP 152/94, HR 78, heart regular, no edema",
     "Essential hypertension, not at goal", "Increase lisinopril to 20 mg daily. Recheck BP in 4 weeks."),
    ("type 2 diabetes visit", "fatigue and increased thirst for 2 months", "A1c 8.9%, BMI 33, monofilament normal",
     "Type 2 diabetes mellitus, uncontrolled", "Start metformin 500 mg twice daily. Diabetes education referral."),
    ("asthma exacerbation", "wheezing and shortness of breath for 4 days", "Expiratory wheezes bilaterally, O2 sat 95%",
     "Moderate persistent asthma with acute exacerbation", "Prednisone burst 40 mg x5 days, start inhaled steroid."),
    ("migraine", "recurrent throbbing headaches with nausea 6 times per month", "Neuro exam normal, no papilledema",
     "Migraine without aura", "Start topiramate 25 mg nightly. Headache diary."),
    ("low back pain", "lower back pain after lifting boxes 3 weeks ago", "Paraspinal tenderness, straight leg raise negative",
     "Acute mechanical low back pain", "NSAIDs, activity as tolerated, return if numbness or weakness."),
    ("shoulder pain", "right shoulder pain with overhead reaching for 2 months", "Painful arc, positive Hawkins test",
     "Rotator cuff tendinopathy", "Refer to physical therapy. Ibuprofen as needed."),
    ("GERD", "burning chest discomfort after meals for 6 weeks", "Abdomen soft, mild epigastric tenderness",
     "Gastroesophageal reflux disease", "Start omeprazole 20 mg daily. Avoid late meals."),
    ("depression follow-up", "low mood and poor sleep, improved since last visit", "PHQ-9 score 11, down from 17",
     "Major depressive disorder, improving", "Continue sertraline 100 mg. Follow up in 6 weeks."),
    ("hypothyroidism", "weight gain, cold intolerance and dry skin", "TSH 9.8, free T4 low, thyroid not enlarged",
     "Primary hypothyroidism", "Start levothyroxine 50 mcg daily. Repeat TSH in 6 weeks."),
    ("urinary tract infection", "burning with urination and frequency for 2 days", "Urinalysis positive for nitrites and leukocytes",
     "Acute uncomplicated cystitis", "Nitrofurantoin 100 mg twice daily for 5 days."),
    ("COPD", "chronic cough and breathlessness walking one block", "Prolonged expiration, FEV1/FVC 0.62",
     "Chronic obstructive pulmonary disease", "Start tiotropium inhaler. Pulmonary rehab referral."),
    ("ankle sprain", "left ankle pain after twisting it playing soccer yesterday", "Lateral swelling, can bear weight, Ottawa rules negative",
     "Lateral ankle sprain, grade 1", "RICE, ankle brace, follow up in 2 weeks."),
]

REQUESTS = [
    ("CT abdomen and pelvis with contrast", "74177", "R10.31"), ("Polysomnography sleep study", "95810", "G47.33"),
    ("CPAP machine and supplies", "E0601", "G47.33"), ("Adalimumab 40 mg injection", "J0135", "M05.79"),
    ("Semaglutide injection", "J3490", "E11.65"), ("Physical therapy, 12 visits", "97110", "M54.50"),
    ("Screening colonoscopy", "45378", "Z12.11"), ("Lumbar epidural steroid injection", "62323", "M54.16"),
    ("Power wheelchair", "K0823", "G35"), ("MRI brain with and without contrast", "70553", "G43.909"),
    ("Insulin pump", "E0784", "E10.65"), ("Home oxygen concentrator", "E1390", "J44.9"),
]

THERAPY = [
    ("Physical Therapy", "right shoulder", "flexion 120 deg, abduction 100 deg", "Therapeutic exercise, manual therapy, scapular stabilization"),
    ("Physical Therapy", "lumbar spine", "lumbar flexion limited 50%", "Core strengthening, McKenzie extension, gait training"),
    ("Occupational Therapy", "left wrist after fracture", "grip strength 18 lb vs 45 lb", "Fine motor activities, splint check, ROM exercises"),
    ("Physical Therapy", "right hip after replacement", "hip flexion 85 deg", "Bed mobility, stair training, hip abductor strengthening"),
    ("Speech Therapy", "swallowing after stroke", "tolerating soft solids with chin tuck", "Swallow strategies, oral motor exercises"),
    ("Physical Therapy", "cervical spine", "rotation 50 deg each side", "Posture training, cervical retraction, soft tissue mobilization"),
    ("Occupational Therapy", "right hand after tendon repair", "composite fist 70%", "Tendon glides, edema control, scar massage"),
    ("Physical Therapy", "left ankle", "dorsiflexion 8 deg", "Balance board, calf strengthening, proprioception drills"),
    ("Physical Therapy", "post-stroke left side weakness", "gait speed 0.6 m/s with cane", "Gait training, balance, functional transfers"),
    ("Occupational Therapy", "upper body after cardiac surgery", "sternal precautions maintained", "ADL retraining, energy conservation"),
    ("Physical Therapy", "thoracic spine", "thoracic rotation limited", "Mobility exercises, breathing exercises, posture"),
    ("Speech Therapy", "aphasia after stroke", "naming accuracy 60%", "Naming drills, word retrieval strategies"),
]

POLICIES = [
    ("Polysomnography for Obstructive Sleep Apnea", "MED-201"), ("CPAP Devices", "DME-114"),
    ("Bariatric Surgery", "SURG-310"), ("GLP-1 Receptor Agonists for Weight Management", "RX-422"),
    ("Home Oxygen Therapy", "DME-130"), ("Genetic Testing for Hereditary Cancer", "LAB-509"),
    ("Spinal Cord Stimulators", "SURG-355"), ("Hearing Aids", "DME-170"),
    ("Cataract Surgery", "SURG-220"), ("Treatment of Varicose Veins", "SURG-240"),
    ("Outpatient Physical Therapy Visit Limits", "REHAB-101"), ("TNF Inhibitors for Rheumatoid Arthritis", "RX-305"),
]


def note(i, t):
    topic, hpi, exam, assess, plan = t
    return f"""{random.choice(CLINICS).upper()} - OFFICE VISIT NOTE (SYNTHETIC, NO PHI)
Patient: Test Patient {100+i}    Age: {random.randint(22, 78)}    Date of service: 0{random.randint(1,9)}/1{random.randint(0,9)}/2026
Provider: {random.choice(DOCTORS)}

Chief complaint:
Visit for {topic}.

History of present illness:
Patient reports {hpi}. Denies fever or weight loss. Medications reviewed.

Physical examination:
{exam}.

Assessment:
{assess}.

Plan:
{plan}
Patient counseled and agrees with plan.
"""


def request(i, t):
    service, code, icd = t
    return f"""PRIOR AUTHORIZATION REQUEST FORM (SYNTHETIC, NO PHI)
Health plan: Sample Health Plan    Fax: (512) 555-01{10+i}

SECTION A - MEMBER INFORMATION
Member name: Test Member {200+i}    Member ID: DEMO-{5000+i}    DOB: 01/01/19{60+i}

SECTION B - PROVIDER INFORMATION
Ordering provider: {random.choice(DOCTORS)}    NPI: 1{random.randint(100000000,999999999)}
Facility: {random.choice(CLINICS)}    Phone: (512) 555-02{10+i}

SECTION C - SERVICE REQUESTED
Service: {service}
CPT/HCPCS code: {code}    Diagnosis code (ICD-10): {icd}
Units: 1    Requested start date: 11/01/2026
Request type: [{'X' if i % 3 else ' '}] Standard  [{' ' if i % 3 else 'X'}] Urgent

SECTION D - ATTESTATION
I attest that the information provided is accurate. Clinical notes attached.
Provider signature: ____________________    Date: 10/{10+i}/2026
"""


def therapy(i, t):
    kind, area, measure, interventions = t
    return f"""{kind.upper()} DAILY TREATMENT NOTE (SYNTHETIC, NO PHI)
Patient: Test Patient {300+i}    Visit {random.randint(3, 14)} of {random.choice([12, 16, 20])}    Therapist: J. Lee, {'PT' if kind.startswith('Physical') else 'OT' if kind.startswith('Occ') else 'SLP'}

Diagnosis / focus: {area}

Subjective:
Patient reports pain {random.randint(2,6)}/10 today and is doing the home exercise program most days.

Objective:
{measure}. Tolerated session well.

Interventions ({random.choice([38, 45, 53])} minutes):
{interventions}.

Assessment:
Progressing toward short-term goals. Improved function compared with initial evaluation.

Plan:
Continue plan of care 2x per week. Update home exercise program. Re-assess goals at visit {random.choice([12, 16])}.
"""


def policy(i, t):
    title, pid = t
    return f"""SAMPLE HEALTH PLAN - MEDICAL POLICY (SYNTHETIC)
Policy number: {pid}
Title: {title}
Effective date: 0{random.randint(1,9)}/01/2026    Last reviewed: 06/01/2026

Policy statement:
{title} is considered medically necessary when the coverage criteria below are met.

Coverage criteria:
1. The diagnosis is documented by a qualified provider.
2. Conservative or first-line treatment has been tried and was not effective, when applicable.
3. Required tests or measurements are documented in the medical record.

Not medically necessary:
Requests that do not meet the criteria above, or for convenience, are not covered.

Documentation requirements:
Office notes, test results and treatment history must be submitted with the request.

Applicable codes and references:
See coding appendix. This policy is reviewed annually by the medical policy committee.
"""


def main():
    for label, items, fn in [("clinical_note", NOTES, note), ("request_form", REQUESTS, request),
                             ("therapy_note", THERAPY, therapy), ("policy_document", POLICIES, policy)]:
        for i, item in enumerate(items, start=13):
            (OUT / label / f"{label}_{i:02d}.txt").write_text(fn(i, item), encoding="utf-8")
    print("Wrote 12 realistic documents per class (files _13 to _24).")


if __name__ == "__main__":
    main()
