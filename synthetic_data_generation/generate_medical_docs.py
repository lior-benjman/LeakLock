import os
import json
import uuid
import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "dataset", "synthetic_medical_docs")

CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")
HIDDEN_DIR = os.path.join(OUTPUT_DIR, "hidden")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

NUM_IMAGES = 10

fake = Faker("en_US")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(HIDDEN_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


def load_font(size):
    font_options = [
        "arial.ttf",
        "calibri.ttf",
        "times.ttf",
        "cour.ttf",
        "DejaVuSans.ttf",
        "DejaVuSerif.ttf",
    ]

    for font_name in font_options:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    return ImageFont.load_default()


def generate_patient():
    return {
        "name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=1, maximum_age=95).strftime("%m/%d/%Y"),
        "patient_id": f"PID-{random.randint(10000000, 99999999)}",
        "mrn": f"MRN-{random.randint(1000000, 9999999)}",
        "synthetic_id": f"SYN-{uuid.uuid4().hex[:10].upper()}",
        "phone": fake.phone_number(),
        "email": fake.email(),
        "address": fake.address().replace("\n", ", "),
        "doctor": f"Dr. {fake.name()}, MD",
        "clinic": random.choice([
            "City Medical Center",
            "HealthPlus Laboratory",
            "General Hospital",
            "Wellness Clinic",
            "Northside Medical Group",
            "Radiology Associates",
            "Family Care Clinic",
            "Advanced Diagnostics Center"
        ])
    }


def draw_header(draw, width, title, clinic, color):
    title_font = load_font(38)
    subtitle_font = load_font(24)

    draw.rectangle([0, 0, width, 120], fill=color)
    draw.text((40, 30), clinic, font=title_font, fill=(25, 45, 80))
    draw.text((width - 360, 45), title, font=subtitle_font, fill="black")


def draw_patient_block(draw, patient, y):
    font = load_font(21)

    draw.rectangle([45, y, 855, y + 145], outline=(120, 120, 120), width=2)

    left_lines = [
        f"Patient Name: {patient['name']}",
        f"Date of Birth: {patient['dob']}",
        f"Patient ID: {patient['patient_id']}",
        f"MRN: {patient['mrn']}",
    ]

    right_lines = [
        f"Ordering Doctor: {patient['doctor']}",
        f"Clinic/Hospital: {patient['clinic']}",
        f"Phone: {patient['phone']}",
        f"Email: {patient['email']}",
    ]

    for i, line in enumerate(left_lines):
        draw.text((65, y + 18 + i * 30), line, font=font, fill="black")

    for i, line in enumerate(right_lines):
        draw.text((470, y + 18 + i * 30), line, font=font, fill="black")


def create_lab_report(index):
    width, height = 900, 1200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    patient = generate_patient()

    draw_header(draw, width, "LABORATORY REPORT", patient["clinic"], (232, 240, 250))
    draw_patient_block(draw, patient, 155)

    header_font = load_font(27)
    table_font = load_font(18)

    y = 340
    draw.text((50, y), "HEMATOLOGY", font=header_font, fill=(25, 55, 100))
    y += 42

    columns = ["TEST", "RESULT", "UNIT", "REFERENCE RANGE", "FLAG"]
    col_x = [55, 360, 515, 650, 820]

    tests = [
        ("WBC", round(random.uniform(4.0, 11.0), 1), "x10^3/uL", "4.0 - 11.0"),
        ("RBC", round(random.uniform(4.2, 5.9), 2), "x10^6/uL", "4.20 - 5.90"),
        ("Hemoglobin", round(random.uniform(12.0, 17.5), 1), "g/dL", "12.0 - 17.5"),
        ("Hematocrit", round(random.uniform(36.0, 53.0), 1), "%", "36 - 53"),
        ("Platelet Count", random.randint(150, 450), "x10^3/uL", "150 - 450"),
        ("Glucose", random.randint(70, 135), "mg/dL", "70 - 99"),
        ("Creatinine", round(random.uniform(0.55, 1.35), 2), "mg/dL", "0.60 - 1.20"),
        ("Sodium", random.randint(132, 147), "mmol/L", "135 - 145"),
        ("Potassium", round(random.uniform(3.2, 5.5), 1), "mmol/L", "3.5 - 5.1"),
        ("ALT", random.randint(7, 65), "U/L", "7 - 55"),
        ("AST", random.randint(8, 58), "U/L", "8 - 48"),
    ]

    draw.rectangle([45, y, 860, y + 35], fill=(30, 65, 110))
    for i, col in enumerate(columns):
        draw.text((col_x[i], y + 8), col, font=table_font, fill="white")

    y += 35

    for test, result, unit, ref_range in tests:
        flag = random.choice(["-", "-", "-", "H", "L"])
        draw.rectangle([45, y, 860, y + 35], outline=(180, 180, 180))
        row = [test, str(result), unit, ref_range, flag]

        for i, value in enumerate(row):
            draw.text((col_x[i], y + 8), value, font=table_font, fill="black")

        y += 35

    note_font = load_font(21)
    draw.text((50, 1040), "Notes: Please correlate clinically with patient history.", font=note_font, fill="black")
    draw.text((50, 1090), f"Signed: {patient['doctor']}", font=note_font, fill="black")
    draw.text((50, 1135), "Synthetic document - generated for research use only.", font=load_font(16), fill="gray")

    keywords = [
        "laboratory report", "patient", "doctor", "blood test", "hematology",
        "clinical chemistry", "hemoglobin", "wbc", "rbc", "glucose",
        "creatinine", "reference range"
    ]

    return img, "lab_report", keywords


def create_prescription(index):
    width, height = 900, 1200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    patient = generate_patient()

    draw_header(draw, width, "PRESCRIPTION", patient["clinic"], (236, 246, 236))
    draw_patient_block(draw, patient, 155)

    big_font = load_font(72)
    text_font = load_font(25)
    small_font = load_font(20)

    medications = [
        "Amoxicillin 500 mg",
        "Ibuprofen 400 mg",
        "Azithromycin 250 mg",
        "Metformin 500 mg",
        "Loratadine 10 mg",
        "Omeprazole 20 mg",
        "Acetaminophen 500 mg",
    ]

    instructions = [
        "Take 1 tablet by mouth twice daily for 7 days",
        "Take 1 capsule every morning before food",
        "Take 1 tablet every 8 hours as needed",
        "Take 2 tablets daily with water",
        "Apply as directed by physician",
    ]

    y = 350
    draw.text((70, y), "Rx", font=big_font, fill="black")

    for i in range(random.randint(2, 4)):
        med = random.choice(medications)
        inst = random.choice(instructions)

        draw.text((180, y + 25 + i * 115), f"{i + 1}. {med}", font=text_font, fill="black")
        draw.text((210, y + 62 + i * 115), inst, font=small_font, fill="black")

    draw.text((60, 950), f"Refills: {random.randint(0, 3)}", font=text_font, fill="black")
    draw.text((470, 950), "Signature: __________________", font=text_font, fill="black")
    draw.text((60, 1060), f"Prescribing Physician: {patient['doctor']}", font=small_font, fill="black")
    draw.text((60, 1120), "Synthetic document - generated for research use only.", font=load_font(16), fill="gray")

    keywords = [
        "prescription", "rx", "patient", "doctor", "medication",
        "dosage", "refills", "pharmacy", "physician"
    ]

    return img, "prescription", keywords


def create_discharge_summary(index):
    width, height = 900, 1200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    patient = generate_patient()

    draw_header(draw, width, "DISCHARGE SUMMARY", patient["clinic"], (248, 235, 235))
    draw_patient_block(draw, patient, 155)

    text_font = load_font(23)

    diagnoses = [
        "Acute bronchitis with mild dehydration.",
        "Viral upper respiratory infection.",
        "Abdominal pain, improved after observation.",
        "Mild pneumonia, clinically stable.",
        "Post-operative follow-up, stable condition.",
    ]

    treatments = [
        "IV fluids, observation, and supportive care.",
        "Antibiotic therapy and clinical monitoring.",
        "Pain control and routine blood work.",
        "Respiratory treatment and follow-up imaging.",
    ]

    y = 345
    lines = [
        "Final Diagnosis:",
        random.choice(diagnoses),
        "",
        "Hospital Course:",
        random.choice(treatments),
        "The patient improved during hospitalization and was discharged in stable condition.",
        "",
        "Discharge Medications:",
        f"1. {random.choice(['Acetaminophen 500 mg', 'Ibuprofen 400 mg', 'Azithromycin 250 mg'])}",
        f"2. {random.choice(['Omeprazole 20 mg', 'Loratadine 10 mg', 'Amoxicillin 500 mg'])}",
        "",
        "Follow-up:",
        "Please follow up with primary care physician within 7 days.",
        "",
        "Return to emergency care if symptoms worsen."
    ]

    for line in lines:
        draw.text((60, y), line, font=text_font, fill="black")
        y += 42

    draw.text((60, 1070), f"Attending Physician: {patient['doctor']}", font=text_font, fill="black")
    draw.text((60, 1125), "Synthetic document - generated for research use only.", font=load_font(16), fill="gray")

    keywords = [
        "discharge summary", "final diagnosis", "hospital course",
        "patient", "doctor", "medications", "follow-up", "emergency care"
    ]

    return img, "discharge_summary", keywords


def add_document_effects(img):
    img = img.convert("RGB")

    angle = random.uniform(-4, 4)
    img = img.rotate(angle, expand=True, fillcolor=(235, 235, 235))

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.82, 1.18))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.25))

    if random.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))

    return img


def create_hidden_scene(doc_img, index, doc_type, keywords):
    canvas_w, canvas_h = 1200, 900

    background_color = random.choice([
        (185, 160, 120),
        (220, 215, 200),
        (165, 165, 165),
        (130, 105, 85),
        (235, 230, 215),
        (200, 190, 175),
    ])

    canvas = Image.new("RGB", (canvas_w, canvas_h), background_color)
    draw = ImageDraw.Draw(canvas)

    doc = doc_img.copy()
    doc.thumbnail((520, 720))

    x = random.randint(120, 550)
    y = random.randint(70, 210)

    canvas.paste(doc, (x, y))
    bbox = [x, y, x + doc.width, y + doc.height]

    for _ in range(random.randint(1, 3)):
        object_type = random.choice(["folder", "book", "sticky_note", "dark_object"])

        ox = x + random.randint(-120, 300)
        oy = y + random.randint(-90, 420)

        if object_type == "folder":
            draw.rectangle(
                [ox, oy, ox + random.randint(360, 620), oy + random.randint(160, 300)],
                fill=random.choice([(155, 110, 60), (180, 135, 75), (200, 160, 95)])
            )

        elif object_type == "book":
            draw.rectangle(
                [ox, oy, ox + random.randint(330, 520), oy + random.randint(180, 300)],
                fill=random.choice([(45, 45, 45), (70, 55, 45), (35, 60, 80)])
            )

        elif object_type == "sticky_note":
            draw.rectangle(
                [ox, oy, ox + 160, oy + 130],
                fill=random.choice([(245, 230, 120), (240, 210, 130), (230, 240, 150)])
            )

        elif object_type == "dark_object":
            draw.rectangle(
                [ox, oy, ox + random.randint(350, 550), oy + random.randint(160, 270)],
                fill=random.choice([(30, 30, 30), (55, 55, 60), (80, 70, 65)])
            )

    canvas = add_document_effects(canvas)

    label = {
        "image_id": f"hidden_{index:05d}",
        "document_type": doc_type,
        "is_medical_document": True,
        "keywords": keywords,
        "bbox_document_before_effects": bbox,
        "visibility": "partially_hidden"
    }

    return canvas, label


def save_image_and_label(img, label, image_dir):
    image_path = os.path.join(image_dir, f"{label['image_id']}.png")
    label_path = os.path.join(LABELS_DIR, f"{label['image_id']}.json")

    img.save(image_path)

    with open(label_path, "w", encoding="utf-8") as file:
        json.dump(label, file, indent=4)


def generate_dataset(num_images=NUM_IMAGES):
    generators = [
        create_lab_report,
        create_prescription,
        create_discharge_summary,
    ]

    for i in range(num_images):
        generator = random.choice(generators)
        doc_img, doc_type, keywords = generator(i)

        doc_img = add_document_effects(doc_img)

        clean_label = {
            "image_id": f"clean_{i:05d}",
            "document_type": doc_type,
            "is_medical_document": True,
            "keywords": keywords,
            "visibility": "fully_visible"
        }

        save_image_and_label(doc_img, clean_label, CLEAN_DIR)

        hidden_img, hidden_label = create_hidden_scene(doc_img, i, doc_type, keywords)
        save_image_and_label(hidden_img, hidden_label, HIDDEN_DIR)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_images} examples")

    print("Done.")
    print(f"Clean images: {CLEAN_DIR}")
    print(f"Hidden images: {HIDDEN_DIR}")
    print(f"Labels: {LABELS_DIR}")


if __name__ == "__main__":
    generate_dataset(num_images=3000)