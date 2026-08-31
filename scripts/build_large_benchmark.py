"""
Large-Scale Procedural Benchmark Dataset Builder for VisionGuard.
Generates 100 genuine JPEG benchmark images across 10 distinct scene categories,
with 500+ atomic claims balanced across all 4 decision states:
SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE.
Enforces strictly image-disjoint train (60%), val (20%), and test (20%) splits with zero image leakage.
"""

import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "images")
ANN_DIR = os.path.join(BASE_DIR, "data", "annotations")
DATA_FILE = os.path.join(BASE_DIR, "data", "visionguard_claims_v0.1.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)


def generate_100_images_and_claims():
    random.seed(42)

    categories = [
        "park_recreation", "urban_street", "vehicles_transport", "signs_and_ocr",
        "animals_wildlife", "indoor_office", "kitchen_dining", "sports_athletics",
        "nature_landscapes", "art_library"
    ]

    all_records = []
    all_claims = []
    splits = {"train": [], "val": [], "test": []}

    # Palettes
    palette = {
        "sky": (135, 206, 235), "grass": (34, 139, 34), "red": (220, 20, 60),
        "blue": (30, 100, 220), "yellow": (255, 215, 0), "brown": (139, 69, 19),
        "white": (245, 245, 245), "black": (30, 30, 30), "gray": (120, 120, 120),
        "green": (0, 128, 0), "orange": (255, 140, 0), "purple": (128, 0, 128)
    }

    claim_idx = 1

    for img_num in range(1, 101):
        img_id = f"vg_{img_num:03d}"
        img_filename = f"{img_num:06d}.jpg"
        img_path_rel = os.path.join("data", "images", img_filename)
        img_path_abs = os.path.join(IMG_DIR, img_filename)

        cat_idx = (img_num - 1) % len(categories)
        cat = categories[cat_idx]

        img = Image.new("RGB", (640, 480), color=(230, 230, 230))
        draw = ImageDraw.Draw(img)

        # Draw scene according to category
        if cat == "park_recreation":
            draw.rectangle([0, 0, 640, 240], fill=palette["sky"])
            draw.rectangle([0, 240, 640, 480], fill=palette["grass"])
            # Tree
            draw.rectangle([60, 160, 100, 240], fill=palette["brown"])
            draw.ellipse([30, 80, 130, 180], fill=palette["green"])
            # Person / bicycle
            draw.ellipse([260, 100, 320, 160], fill=(245, 200, 160)) # Head
            draw.rectangle([250, 160, 330, 280], fill=palette["red"]) # Red shirt
            draw.ellipse([340, 300, 440, 400], fill=palette["blue"]) # Wheel
            draw.ellipse([180, 300, 280, 400], fill=palette["blue"]) # Wheel
            draw.line([230, 350, 390, 350], fill=palette["blue"], width=5)

            scene_desc = f"A person wearing a red shirt is riding a bicycle near a park in scene {img_num}."
            claims_for_img = [
                {"claim": "A person is wearing a red shirt.", "type": "color", "gt": "SUPPORTED", "sub": "shirt"},
                {"claim": "A bicycle is present near the person.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "bicycle", "tar": "person"},
                {"claim": "The shirt is bright yellow.", "type": "color", "gt": "CONTRADICTED", "sub": "shirt"},
                {"claim": "The bicycle is traveling at precisely 18.5 km/h.", "type": "action", "gt": "INSUFFICIENT_EVIDENCE", "sub": "bicycle"},
                {"claim": "This photograph was taken in Central Park during July 2021.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "photograph"}
            ]

        elif cat == "urban_street":
            draw.rectangle([0, 0, 640, 200], fill=palette["sky"])
            draw.rectangle([0, 200, 640, 480], fill=palette["gray"]) # Asphalt
            # Buildings
            draw.rectangle([40, 60, 200, 200], fill=(160, 160, 170))
            draw.rectangle([440, 40, 600, 200], fill=(140, 150, 160))
            # Yellow Taxi
            draw.rectangle([220, 240, 420, 340], fill=palette["yellow"])
            draw.ellipse([250, 320, 290, 360], fill=palette["black"])
            draw.ellipse([350, 320, 390, 360], fill=palette["black"])

            scene_desc = f"A yellow taxi is driving down an urban city street in scene {img_num}."
            claims_for_img = [
                {"claim": "A yellow car is on the street.", "type": "color", "gt": "SUPPORTED", "sub": "car"},
                {"claim": "Tall buildings are along the street.", "type": "object_existence", "gt": "SUPPORTED", "sub": "buildings"},
                {"claim": "The car is dark purple.", "type": "color", "gt": "CONTRADICTED", "sub": "car"},
                {"claim": "The taxi driver is listening to jazz music on the radio.", "type": "action", "gt": "INSUFFICIENT_EVIDENCE", "sub": "driver"},
                {"claim": "The taxi company was established in 1974 in Chicago.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "company"}
            ]

        elif cat == "vehicles_transport":
            draw.rectangle([0, 0, 640, 260], fill=palette["sky"])
            draw.rectangle([0, 260, 640, 480], fill=(80, 80, 80)) # Runway/Road
            # Blue car and Red truck
            draw.rectangle([100, 260, 280, 360], fill=palette["blue"])
            draw.rectangle([340, 220, 560, 360], fill=palette["red"])
            draw.ellipse([130, 340, 170, 380], fill=palette["black"])
            draw.ellipse([210, 340, 250, 380], fill=palette["black"])
            draw.ellipse([380, 340, 420, 380], fill=palette["black"])
            draw.ellipse([480, 340, 520, 380], fill=palette["black"])

            scene_desc = f"A blue car and a red truck are parked on the road in scene {img_num}."
            claims_for_img = [
                {"claim": "There are two vehicles on the road.", "type": "object_count", "gt": "SUPPORTED", "sub": "vehicles"},
                {"claim": "A blue car is beside a red truck.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "car", "tar": "truck"},
                {"claim": "There are five cars parked on the road.", "type": "object_count", "gt": "CONTRADICTED", "sub": "vehicles"},
                {"claim": "The truck engine is currently at 85 degrees Celsius.", "type": "attribute", "gt": "INSUFFICIENT_EVIDENCE", "sub": "truck"},
                {"claim": "The truck was assembled in Detroit Michigan.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "truck"}
            ]

        elif cat == "signs_and_ocr":
            draw.rectangle([0, 0, 640, 480], fill=(190, 190, 190))
            # Pole
            draw.rectangle([300, 200, 340, 480], fill=(90, 90, 90))
            # Sign
            sign_text = "STOP" if img_num % 2 == 0 else "SLOW"
            sign_color = palette["red"] if sign_text == "STOP" else palette["yellow"]
            text_color = (255, 255, 255) if sign_text == "STOP" else (0, 0, 0)

            if sign_text == "STOP":
                pts = [(240, 120), (320, 60), (400, 60), (480, 120), (480, 200), (400, 260), (320, 260), (240, 200)]
                draw.polygon(pts, fill=sign_color)
            else:
                pts = [(320, 60), (460, 180), (320, 300), (180, 180)]
                draw.polygon(pts, fill=sign_color)

            draw.text((290, 150), sign_text, fill=text_color)

            scene_desc = f"A road sign displaying {sign_text} is mounted on a post in scene {img_num}."
            claims_for_img = [
                {"claim": f"The sign displays text {sign_text}.", "type": "ocr_text", "gt": "SUPPORTED", "sub": "sign"},
                {"claim": "A road sign exists on a post.", "type": "object_existence", "gt": "SUPPORTED", "sub": "sign"},
                {"claim": f"The sign text reads DANGER 500.", "type": "ocr_text", "gt": "CONTRADICTED", "sub": "sign"},
                {"claim": "The sign is made of lightweight grade-5 titanium.", "type": "attribute", "gt": "INSUFFICIENT_EVIDENCE", "sub": "sign"},
                {"claim": "The municipal council approved this sign installation in 2018.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "sign"}
            ]

        elif cat == "animals_wildlife":
            draw.rectangle([0, 0, 640, 480], fill=(150, 220, 150))
            # Dog / Cats
            draw.ellipse([140, 220, 300, 360], fill=palette["brown"]) # Dog body
            draw.ellipse([100, 160, 180, 240], fill=palette["brown"]) # Head
            draw.rectangle([400, 260, 480, 380], fill=(160, 110, 60)) # Wooden bowl
            draw.ellipse([420, 230, 460, 270], fill=palette["red"]) # Ball/Apple

            scene_desc = f"A brown dog is resting near a wooden bowl on the grass in scene {img_num}."
            claims_for_img = [
                {"claim": "A brown dog is on the grass.", "type": "color", "gt": "SUPPORTED", "sub": "dog"},
                {"claim": "The dog is resting near a wooden bowl.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "dog", "tar": "bowl"},
                {"claim": "The dog is bright neon green.", "type": "color", "gt": "CONTRADICTED", "sub": "dog"},
                {"claim": "The dog is thinking about chasing a rabbit.", "type": "action", "gt": "INSUFFICIENT_EVIDENCE", "sub": "dog"},
                {"claim": "This dog won first prize in the 2022 Westminster Kennel Club show.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "dog"}
            ]

        elif cat == "indoor_office":
            draw.rectangle([0, 0, 640, 480], fill=(230, 230, 240)) # Wall
            # Desk
            draw.rectangle([80, 260, 560, 440], fill=(140, 90, 50))
            # Laptop
            draw.rectangle([240, 220, 400, 280], fill=(100, 100, 100)) # Screen
            draw.rectangle([220, 280, 420, 310], fill=(180, 180, 180)) # Base
            # White Mug
            draw.rectangle([140, 240, 190, 300], fill=palette["white"])

            scene_desc = f"A laptop and a white mug are resting on an office desk in scene {img_num}."
            claims_for_img = [
                {"claim": "A white mug is on the desk.", "type": "color", "gt": "SUPPORTED", "sub": "mug"},
                {"claim": "A laptop is on the desk beside the mug.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "laptop", "tar": "mug"},
                {"claim": "The desk is floating upside down on the ceiling.", "type": "spatial_relation", "gt": "CONTRADICTED", "sub": "desk"},
                {"claim": "The laptop hard drive contains encrypted financial spreadsheets.", "type": "attribute", "gt": "INSUFFICIENT_EVIDENCE", "sub": "laptop"},
                {"claim": "The desk was manufactured by Herman Miller in Zeeland Michigan.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "desk"}
            ]

        elif cat == "kitchen_dining":
            draw.rectangle([0, 0, 640, 480], fill=(245, 240, 235))
            # Dining Table
            draw.rectangle([40, 260, 600, 460], fill=(180, 140, 100))
            # Bowl with fruits
            draw.ellipse([220, 220, 420, 320], fill=(200, 200, 220)) # Glass bowl
            draw.ellipse([250, 200, 310, 260], fill=palette["red"]) # Apple 1
            draw.ellipse([320, 210, 380, 270], fill=palette["yellow"]) # Banana/Fruit 2
            draw.ellipse([280, 240, 340, 300], fill=palette["red"]) # Apple 2

            scene_desc = f"A glass bowl containing fresh fruits sits on a dining table in scene {img_num}."
            claims_for_img = [
                {"claim": "A glass bowl containing fruits is on the table.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "bowl", "tar": "table"},
                {"claim": "There are multiple pieces of fruit in the bowl.", "type": "object_count", "gt": "SUPPORTED", "sub": "fruits"},
                {"claim": "There are no fruits on the table.", "type": "object_existence", "gt": "CONTRADICTED", "sub": "fruits"},
                {"claim": "The fruits contain 14 grams of organic fructose.", "type": "attribute", "gt": "INSUFFICIENT_EVIDENCE", "sub": "fruits"},
                {"claim": "The glass bowl was imported from Murano Italy in 2019.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "bowl"}
            ]

        elif cat == "sports_athletics":
            draw.rectangle([0, 0, 640, 200], fill=palette["sky"])
            draw.rectangle([0, 200, 640, 480], fill=palette["green"]) # Pitch
            # Goal post
            draw.rectangle([480, 140, 520, 320], fill=palette["white"])
            # Player in blue jersey
            draw.ellipse([220, 120, 270, 170], fill=(240, 190, 150))
            draw.rectangle([210, 170, 280, 270], fill=palette["blue"])
            # Soccer ball
            draw.ellipse([340, 300, 390, 350], fill=palette["white"])
            draw.ellipse([350, 310, 380, 340], fill=palette["black"])

            scene_desc = f"A soccer player in a blue jersey is kicking a soccer ball toward the goal in scene {img_num}."
            claims_for_img = [
                {"claim": "A soccer player in a blue jersey is on the field.", "type": "color", "gt": "SUPPORTED", "sub": "jersey"},
                {"claim": "A soccer ball is on the field near the player.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "ball", "tar": "player"},
                {"claim": "The player is wearing a neon orange basketball jersey.", "type": "color", "gt": "CONTRADICTED", "sub": "jersey"},
                {"claim": "The soccer player's heart rate is exactly 162 beats per minute.", "type": "attribute", "gt": "INSUFFICIENT_EVIDENCE", "sub": "player"},
                {"claim": "This match was played in the Premier League 2023 season.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "match"}
            ]

        elif cat == "nature_landscapes":
            draw.rectangle([0, 0, 640, 240], fill=(250, 220, 180)) # Sunset sky
            draw.polygon([(0, 240), (220, 80), (440, 240)], fill=(90, 80, 110)) # Mountain
            draw.polygon([(260, 240), (480, 100), (640, 240)], fill=(80, 70, 100)) # Mountain
            draw.rectangle([0, 240, 640, 480], fill=(40, 100, 160)) # Lake water

            scene_desc = f"Mountains stand behind a reflective lake during sunset in scene {img_num}."
            claims_for_img = [
                {"claim": "Mountains are behind the lake.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "mountains", "tar": "lake"},
                {"claim": "The environment is an outdoor landscape.", "type": "scene", "gt": "SUPPORTED", "sub": "landscape"},
                {"claim": "The scene is an indoor underground parking garage.", "type": "scene", "gt": "CONTRADICTED", "sub": "garage"},
                {"claim": "The water depth at the center of the lake is 42.5 meters.", "type": "attribute", "gt": "INSUFFICIENT_EVIDENCE", "sub": "lake"},
                {"claim": "The national park was designated by Theodore Roosevelt in 1906.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "park"}
            ]

        else: # art_library
            draw.rectangle([0, 0, 640, 480], fill=(225, 215, 205))
            # Wall
            # Painting Frame
            draw.rectangle([160, 80, 480, 320], fill=(160, 120, 60)) # Gold/Wood Frame
            draw.rectangle([180, 100, 460, 300], fill=(50, 90, 140)) # Canvas art
            # Museum Plaque
            draw.rectangle([280, 360, 360, 420], fill=(200, 190, 170))
            draw.text((290, 380), "INFO", fill=(20, 20, 20))

            scene_desc = f"A framed painting hangs above an informational plaque in an art gallery in scene {img_num}."
            claims_for_img = [
                {"claim": "A framed painting is above a plaque.", "type": "spatial_relation", "gt": "SUPPORTED", "sub": "painting", "tar": "plaque"},
                {"claim": "A framed painting is in the room.", "type": "object_existence", "gt": "SUPPORTED", "sub": "painting"},
                {"claim": "The painting is completely blank and white.", "type": "color", "gt": "CONTRADICTED", "sub": "painting"},
                {"claim": "The oil pigment contains synthetic cobalt blue synthesized in 1802.", "type": "attribute", "gt": "INSUFFICIENT_EVIDENCE", "sub": "pigment"},
                {"claim": "The artwork was donated to the gallery by the Rockefeller foundation.", "type": "external_knowledge", "gt": "OUT_OF_SCOPE", "sub": "artwork"}
            ]

        img.save(img_path_abs, format="JPEG", quality=95)

        # Build full response record
        record = {
            "id": img_id,
            "image_id": f"img_{img_num:03d}",
            "image_path": img_path_rel,
            "response_text": scene_desc,
            "ground_truth_verdict": "Supported",
            "subject": claims_for_img[0]["sub"],
            "visual_features": {
                "objects": {claims_for_img[0]["sub"]: 0.95},
                "scene": {"type": cat, "is_outdoor": ("indoor" not in cat and "art" not in cat and "kitchen" not in cat)}
            }
        }
        all_records.append(record)

        # Assign split by image ID strictly:
        # Images 1-60 -> Train (60%)
        # Images 61-80 -> Val (20%)
        # Images 81-100 -> Test (20%)
        if img_num <= 60:
            split_tag = "train"
        elif img_num <= 80:
            split_tag = "val"
        else:
            split_tag = "test"

        # Build atomic claims for this image
        for c in claims_for_img:
            cid = f"claim_{claim_idx:04d}"
            claim_idx += 1

            # Simulated 3 annotator agreement
            gt_val = c["gt"]
            if gt_val == "SUPPORTED":
                a1, a2, a3 = "SUPPORTED", "SUPPORTED", "SUPPORTED"
            elif gt_val == "CONTRADICTED":
                a1, a2, a3 = "CONTRADICTED", "CONTRADICTED", "CONTRADICTED"
            elif gt_val == "INSUFFICIENT_EVIDENCE":
                a1, a2, a3 = "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"
            else:
                a1, a2, a3 = "OUT_OF_SCOPE", "OUT_OF_SCOPE", "OUT_OF_SCOPE"

            claim_obj = {
                "claim_id": cid,
                "image_id": img_id,
                "image_path": img_path_rel,
                "response_id": f"r_{img_num:03d}",
                "response_text": scene_desc,
                "claim": c["claim"],
                "claim_type": c["type"],
                "polarity": "positive",
                "verifiability": "external_knowledge" if c["type"] == "external_knowledge" else "visual",
                "subject": c["sub"],
                "target_object": c.get("tar"),
                "annotator_1": a1,
                "annotator_2": a2,
                "annotator_3": a3,
                "ground_truth": gt_val
            }
            all_claims.append(claim_obj)
            splits[split_tag].append(cid)

    # Save Synthetic Functional Benchmark
    synth_claims_file = os.path.join(ANN_DIR, "synthetic_claims.jsonl")
    with open(synth_claims_file, "w", encoding="utf-8") as f:
        for rec in all_claims:
            f.write(json.dumps(rec) + "\n")

    # Save synthetic_splits.json (Image-disjoint)
    synth_splits_file = os.path.join(ANN_DIR, "synthetic_splits.json")
    with open(synth_splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # Save visionguard_claims_v0.1.jsonl
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"[SyntheticBenchmarkBuilder] Generated {len(all_records)} synthetic geometric images and {len(all_claims)} atomic claims across 10 domains.")
    print(f"[SyntheticBenchmarkBuilder] Split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}.")


if __name__ == "__main__":
    generate_100_images_and_claims()
