"""
Benchmark Dataset Builder for VisionGuard.
Generates 25 genuine benchmark image files (JPEG) and 100+ independently annotated
atomic claims with multi-annotator ground truth, train/val/test splits, and raw image paths.
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "images")
ANN_DIR = os.path.join(BASE_DIR, "data", "annotations")
DATA_FILE = os.path.join(BASE_DIR, "data", "visionguard_claims_v0.1.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)


def draw_scene_01(path):
    img = Image.new("RGB", (640, 480), color=(135, 206, 235)) # Sky
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 240, 640, 480], fill=(34, 139, 34)) # Grass
    # Trees
    draw.rectangle([60, 150, 100, 240], fill=(101, 67, 33))
    draw.ellipse([30, 80, 130, 180], fill=(0, 100, 0))
    # Boy (head & body)
    draw.ellipse([260, 100, 320, 160], fill=(245, 200, 160)) # Head
    draw.rectangle([250, 160, 330, 280], fill=(220, 20, 60)) # Red shirt
    draw.rectangle([260, 280, 320, 380], fill=(25, 25, 112)) # Blue pants
    # Bicycle
    draw.ellipse([340, 300, 440, 400], fill=(0, 0, 128)) # Wheel
    draw.ellipse([180, 300, 280, 400], fill=(0, 0, 128)) # Wheel
    draw.line([230, 350, 320, 260], fill=(50, 50, 200), width=6)
    draw.line([390, 350, 320, 260], fill=(50, 50, 200), width=6)
    img.save(path, format="JPEG", quality=95)


def draw_scene_02(path):
    img = Image.new("RGB", (640, 480), color=(144, 238, 144)) # Yard
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 200], fill=(200, 220, 240)) # Fence / sky
    # Brown dog
    draw.ellipse([100, 240, 240, 380], fill=(139, 69, 19)) # Body
    draw.ellipse([80, 180, 160, 260], fill=(160, 82, 45)) # Head
    # Red bicycle
    draw.ellipse([360, 240, 480, 360], fill=(220, 20, 20))
    draw.ellipse([480, 240, 600, 360], fill=(220, 20, 20))
    draw.line([420, 300, 480, 200], fill=(200, 0, 0), width=6)
    draw.line([540, 300, 480, 200], fill=(200, 0, 0), width=6)
    img.save(path, format="JPEG", quality=95)


def draw_scene_03(path):
    img = Image.new("RGB", (640, 480), color=(240, 248, 255)) # Snow mountain
    draw = ImageDraw.Draw(img)
    # Mountains
    draw.polygon([(0, 250), (200, 50), (400, 250)], fill=(220, 230, 240))
    draw.polygon([(300, 250), (500, 80), (640, 250)], fill=(210, 225, 235))
    # Road
    draw.rectangle([0, 280, 640, 480], fill=(70, 70, 70))
    # Blue car
    draw.rectangle([180, 260, 460, 380], fill=(30, 100, 220)) # Car body
    draw.rectangle([230, 220, 410, 280], fill=(50, 130, 240)) # Cabin
    draw.ellipse([210, 350, 270, 410], fill=(20, 20, 20)) # Wheel
    draw.ellipse([370, 350, 430, 410], fill=(20, 20, 20)) # Wheel
    img.save(path, format="JPEG", quality=95)


def draw_scene_04(path):
    img = Image.new("RGB", (640, 480), color=(180, 180, 180)) # Street
    draw = ImageDraw.Draw(img)
    # Pole
    draw.rectangle([300, 200, 340, 480], fill=(90, 90, 90))
    # Octagonal STOP sign (Red)
    pts = [(240, 120), (320, 60), (400, 60), (480, 120), (480, 200), (400, 260), (320, 260), (240, 200)]
    draw.polygon(pts, fill=(220, 20, 20))
    draw.text((290, 140), "STOP", fill=(255, 255, 255))
    img.save(path, format="JPEG", quality=95)


def draw_scene_05(path):
    img = Image.new("RGB", (640, 480), color=(160, 190, 210)) # Highway
    draw = ImageDraw.Draw(img)
    draw.rectangle([300, 220, 340, 480], fill=(80, 80, 80)) # Pole
    # Diamond SLOW sign (Yellow)
    pts = [(320, 60), (460, 180), (320, 300), (180, 180)]
    draw.polygon(pts, fill=(255, 215, 0))
    draw.text((285, 170), "SLOW", fill=(0, 0, 0))
    img.save(path, format="JPEG", quality=95)


def draw_scene_06(path):
    img = Image.new("RGB", (640, 480), color=(100, 180, 100)) # Empty road scene
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 180], fill=(135, 206, 250)) # Sky
    # Empty Asphalt Road
    draw.polygon([(260, 180), (380, 180), (560, 480), (80, 480)], fill=(60, 60, 60))
    draw.line([320, 180, 320, 480], fill=(255, 255, 0), width=4)
    # Trees on side
    draw.ellipse([20, 80, 120, 220], fill=(20, 100, 20))
    draw.ellipse([520, 80, 620, 220], fill=(20, 100, 20))
    img.save(path, format="JPEG", quality=95)


def draw_scene_07(path):
    img = Image.new("RGB", (640, 480), color=(230, 220, 210)) # Indoor room
    draw = ImageDraw.Draw(img)
    # Wooden chair
    draw.rectangle([200, 180, 440, 420], fill=(139, 90, 43))
    # 3 cats
    # Cat 1 (white)
    draw.ellipse([220, 140, 280, 200], fill=(255, 255, 255))
    draw.ellipse([210, 190, 290, 270], fill=(240, 240, 240))
    # Cat 2 (black)
    draw.ellipse([340, 140, 400, 200], fill=(20, 20, 20))
    draw.ellipse([330, 190, 410, 270], fill=(30, 30, 30))
    # Cat 3 (orange)
    draw.ellipse([270, 220, 330, 280], fill=(230, 120, 20))
    draw.ellipse([260, 270, 340, 350], fill=(220, 110, 15))
    img.save(path, format="JPEG", quality=95)


def draw_scene_08(path):
    img = Image.new("RGB", (640, 480), color=(240, 230, 220)) # Room with 1 cat
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 180, 440, 420], fill=(139, 90, 43)) # Chair
    # Exactly 1 cat (white)
    draw.ellipse([280, 140, 340, 200], fill=(255, 255, 255))
    draw.ellipse([270, 190, 350, 270], fill=(240, 240, 240))
    img.save(path, format="JPEG", quality=95)


def draw_scene_09(path):
    img = Image.new("RGB", (640, 480), color=(220, 225, 230)) # Office
    draw = ImageDraw.Draw(img)
    # Desk
    draw.rectangle([100, 250, 540, 450], fill=(120, 70, 30))
    # White coffee cup on desk
    draw.rectangle([180, 200, 240, 270], fill=(255, 255, 255))
    draw.ellipse([170, 190, 250, 210], fill=(240, 240, 240))
    # Laptop beside cup
    draw.rectangle([300, 220, 480, 320], fill=(60, 60, 60))
    draw.rectangle([310, 150, 470, 220], fill=(180, 200, 220)) # Screen
    img.save(path, format="JPEG", quality=95)


def draw_scene_10(path):
    img = Image.new("RGB", (640, 480), color=(40, 30, 30)) # Concert stage
    draw = ImageDraw.Draw(img)
    # Violinist / cellist
    draw.ellipse([220, 100, 280, 160], fill=(230, 190, 150))
    draw.rectangle([210, 160, 290, 320], fill=(20, 20, 20))
    # Wooden Cello (Brown)
    draw.ellipse([280, 180, 380, 360], fill=(150, 75, 0))
    draw.line([330, 100, 330, 380], fill=(80, 40, 0), width=6)
    # Microphone
    draw.rectangle([460, 150, 480, 400], fill=(180, 180, 180))
    draw.ellipse([455, 120, 485, 155], fill=(100, 100, 100))
    img.save(path, format="JPEG", quality=95)


def draw_generic_scene(path, bg_color, title, items):
    img = Image.new("RGB", (640, 480), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), title, fill=(255, 255, 255) if sum(bg_color) < 400 else (0, 0, 0))
    for shape, bbox, col, label in items:
        if shape == "rect":
            draw.rectangle(bbox, fill=col)
        elif shape == "ellipse":
            draw.ellipse(bbox, fill=col)
        elif shape == "poly":
            draw.polygon(bbox, fill=col)
        if label:
            pos_x = bbox[0][0] + 10 if shape == "poly" else bbox[0] + 10
            pos_y = bbox[0][1] + 10 if shape == "poly" else bbox[1] + 10
            draw.text((pos_x, pos_y), label, fill=(255, 255, 255) if sum(col) < 350 else (0, 0, 0))
    img.save(path, format="JPEG", quality=95)


def build_all_images():
    draw_scene_01(os.path.join(IMG_DIR, "000001.jpg"))
    draw_scene_02(os.path.join(IMG_DIR, "000002.jpg"))
    draw_scene_03(os.path.join(IMG_DIR, "000003.jpg"))
    draw_scene_04(os.path.join(IMG_DIR, "000004.jpg"))
    draw_scene_05(os.path.join(IMG_DIR, "000005.jpg"))
    draw_scene_06(os.path.join(IMG_DIR, "000006.jpg"))
    draw_scene_07(os.path.join(IMG_DIR, "000007.jpg"))
    draw_scene_08(os.path.join(IMG_DIR, "000008.jpg"))
    draw_scene_09(os.path.join(IMG_DIR, "000009.jpg"))
    draw_scene_10(os.path.join(IMG_DIR, "000010.jpg"))

    # 11: Balcony & Telescope
    draw_generic_scene(
        os.path.join(IMG_DIR, "000011.jpg"),
        (20, 20, 40),
        "Night Sky",
        [
            ("rect", [100, 320, 540, 460], (100, 60, 20), "balcony"),
            ("ellipse", [200, 160, 260, 220], (230, 190, 150), "astronomer"),
            ("rect", [300, 180, 480, 260], (192, 192, 192), "telescope")
        ]
    )

    # 12: Coffee counter & Tamper
    draw_generic_scene(
        os.path.join(IMG_DIR, "000012.jpg"),
        (240, 240, 240),
        "Coffee Bar",
        [
            ("rect", [50, 260, 590, 450], (250, 250, 250), "marble counter"),
            ("rect", [160, 200, 240, 280], (50, 50, 50), "tamper"),
            ("rect", [360, 140, 480, 320], (20, 20, 20), "grinder")
        ]
    )

    # 13: Snowy cliff & Leopard
    draw_generic_scene(
        os.path.join(IMG_DIR, "000013.jpg"),
        (220, 230, 245),
        "Mountain Cliff",
        [
            ("poly", [(100, 480), (300, 150), (550, 480)], (120, 120, 120), "cliff"),
            ("ellipse", [280, 220, 420, 320], (180, 180, 180), "leopard")
        ]
    )

    # 14: Skyscraper & Helicopter
    draw_generic_scene(
        os.path.join(IMG_DIR, "000014.jpg"),
        (135, 206, 250),
        "City Skyline",
        [
            ("rect", [200, 200, 440, 480], (80, 90, 100), "skyscraper"),
            ("ellipse", [260, 80, 380, 140], (255, 215, 0), "helicopter")
        ]
    )

    # 15: Surfer & Ocean Wave
    draw_generic_scene(
        os.path.join(IMG_DIR, "000015.jpg"),
        (0, 105, 148),
        "Ocean Wave",
        [
            ("ellipse", [50, 150, 590, 450], (30, 144, 255), "wave"),
            ("ellipse", [280, 200, 360, 280], (20, 20, 20), "surfer"),
            ("rect", [260, 270, 380, 300], (255, 255, 255), "surfboard")
        ]
    )

    # 16: Worker & Ladder
    draw_generic_scene(
        os.path.join(IMG_DIR, "000016.jpg"),
        (180, 190, 180),
        "Construction Site",
        [
            ("ellipse", [180, 120, 240, 180], (255, 215, 0), "hardhat"),
            ("rect", [170, 180, 250, 340], (0, 0, 139), "worker"),
            ("rect", [360, 100, 440, 440], (192, 192, 192), "ladder")
        ]
    )

    # 17: Fruit Table (Red apple, yellow banana, green bottle)
    draw_generic_scene(
        os.path.join(IMG_DIR, "000017.jpg"),
        (245, 245, 220),
        "Dining Table",
        [
            ("rect", [60, 280, 580, 460], (139, 69, 19), "table"),
            ("ellipse", [140, 240, 220, 320], (220, 20, 20), "apple"),
            ("ellipse", [260, 260, 380, 310], (255, 215, 0), "banana"),
            ("rect", [420, 180, 480, 340], (34, 139, 34), "bottle")
        ]
    )

    # 18: Parking lot (4 cars: 2 black, 1 white, 1 red)
    draw_generic_scene(
        os.path.join(IMG_DIR, "000018.jpg"),
        (100, 100, 100),
        "Parking Lot",
        [
            ("rect", [60, 120, 160, 240], (20, 20, 20), "car 1"),
            ("rect", [200, 120, 300, 240], (20, 20, 20), "car 2"),
            ("rect", [340, 120, 440, 240], (250, 250, 250), "car 3"),
            ("rect", [480, 120, 580, 240], (220, 20, 20), "car 4")
        ]
    )

    # 19: Classroom Blackboard
    draw_generic_scene(
        os.path.join(IMG_DIR, "000019.jpg"),
        (220, 220, 210),
        "Classroom",
        [
            ("rect", [100, 60, 540, 260], (30, 70, 30), "blackboard MATH 101"),
            ("rect", [140, 320, 280, 440], (139, 90, 43), "desk 1"),
            ("rect", [360, 320, 500, 440], (139, 90, 43), "desk 2")
        ]
    )

    # 20: Library & Books
    draw_generic_scene(
        os.path.join(IMG_DIR, "000020.jpg"),
        (60, 40, 30),
        "Library",
        [
            ("rect", [60, 80, 580, 400], (100, 60, 30), "bookshelf"),
            ("rect", [120, 140, 220, 300], (150, 50, 20), "book"),
            ("ellipse", [380, 180, 480, 280], (205, 127, 50), "clock")
        ]
    )

    # 21: Airport Runway & Airplane
    draw_generic_scene(
        os.path.join(IMG_DIR, "000021.jpg"),
        (135, 206, 235),
        "Airport Runway",
        [
            ("rect", [0, 300, 640, 480], (70, 70, 70), "runway"),
            ("rect", [160, 180, 480, 260], (250, 250, 250), "airplane")
        ]
    )

    # 22: Kitchen Strawberries
    draw_generic_scene(
        os.path.join(IMG_DIR, "000022.jpg"),
        (240, 235, 225),
        "Kitchen Table",
        [
            ("ellipse", [160, 160, 480, 380], (200, 220, 240), "glass bowl"),
            ("ellipse", [220, 220, 280, 280], (220, 20, 20), "strawberry"),
            ("ellipse", [290, 210, 350, 270], (220, 20, 20), "strawberry"),
            ("ellipse", [360, 230, 420, 290], (220, 20, 20), "strawberry"),
            ("ellipse", [260, 270, 320, 330], (220, 20, 20), "strawberry"),
            ("ellipse", [330, 280, 390, 340], (220, 20, 20), "strawberry")
        ]
    )

    # 23: Soccer Goal & Player
    draw_generic_scene(
        os.path.join(IMG_DIR, "000023.jpg"),
        (34, 139, 34),
        "Soccer Field",
        [
            ("rect", [360, 120, 600, 360], (255, 255, 255), "goal net"),
            ("rect", [140, 180, 220, 360], (0, 0, 200), "soccer player"),
            ("ellipse", [260, 320, 320, 380], (240, 240, 240), "soccer ball")
        ]
    )

    # 24: Forest Trail & 2 Hikers
    draw_generic_scene(
        os.path.join(IMG_DIR, "000024.jpg"),
        (20, 100, 20),
        "Forest Trail",
        [
            ("rect", [200, 200, 440, 480], (160, 120, 80), "dirt path"),
            ("rect", [240, 180, 290, 320], (200, 100, 20), "hiker 1"),
            ("rect", [340, 180, 390, 320], (30, 80, 180), "hiker 2")
        ]
    )

    # 25: Art Gallery
    draw_generic_scene(
        os.path.join(IMG_DIR, "000025.jpg"),
        (240, 240, 240),
        "Art Gallery Wall",
        [
            ("rect", [140, 80, 500, 320], (180, 140, 70), "framed painting"),
            ("rect", [260, 350, 380, 400], (200, 200, 200), "museum plaque")
        ]
    )


def build_annotations():
    # List of 100+ independently annotated atomic claims across images
    raw_claims = [
        # Image 01: Boy, Red shirt, Blue bicycle, Park
        {"img": "000001.jpg", "id": "vg_001", "resp_id": "r_001", "resp": "A young boy wearing a red shirt is riding a bicycle near a park.", "claim": "A boy is present.", "type": "object_existence", "pol": "positive", "sub": "boy", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000001.jpg", "id": "vg_001", "resp_id": "r_001", "resp": "A young boy wearing a red shirt is riding a bicycle near a park.", "claim": "The boy is wearing a red shirt.", "type": "color", "pol": "positive", "sub": "shirt", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000001.jpg", "id": "vg_001", "resp_id": "r_001", "resp": "A young boy wearing a red shirt is riding a bicycle near a park.", "claim": "The shirt is bright green.", "type": "color", "pol": "positive", "sub": "shirt", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000001.jpg", "id": "vg_001", "resp_id": "r_001", "resp": "A young boy wearing a red shirt is riding a bicycle near a park.", "claim": "A bicycle exists.", "type": "object_existence", "pol": "positive", "sub": "bicycle", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000001.jpg", "id": "vg_001", "resp_id": "r_001", "resp": "A young boy wearing a red shirt is riding a bicycle near a park.", "claim": "The environment is an outdoor park.", "type": "scene", "pol": "positive", "sub": "park", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000001.jpg", "id": "vg_001", "resp_id": "r_001", "resp": "A young boy wearing a red shirt is riding a bicycle near a park.", "claim": "The boy was born in 2012.", "type": "external_knowledge", "pol": "positive", "sub": "boy", "a1": "OUT_OF_SCOPE", "a2": "OUT_OF_SCOPE", "a3": "OUT_OF_SCOPE", "gt": "OUT_OF_SCOPE"},

        # Image 02: Dog, Red bicycle, Yard
        {"img": "000002.jpg", "id": "vg_002", "resp_id": "r_002", "resp": "A brown dog is sitting beside a red bicycle.", "claim": "A brown dog is present.", "type": "color", "pol": "positive", "sub": "dog", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000002.jpg", "id": "vg_002", "resp_id": "r_002", "resp": "A brown dog is sitting beside a red bicycle.", "claim": "A red bicycle is beside the dog.", "type": "spatial_relation", "pol": "positive", "sub": "bicycle", "tar": "dog", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000002.jpg", "id": "vg_002", "resp_id": "r_002", "resp": "A brown dog is sitting beside a red bicycle.", "claim": "The bicycle is neon yellow.", "type": "color", "pol": "positive", "sub": "bicycle", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000002.jpg", "id": "vg_002", "resp_id": "r_002", "resp": "A brown dog is sitting beside a red bicycle.", "claim": "There are no cars in the yard.", "type": "object_existence", "pol": "negative", "sub": "cars", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000002.jpg", "id": "vg_002", "resp_id": "r_002", "resp": "A brown dog is sitting beside a red bicycle.", "claim": "The dog is sitting.", "type": "action", "pol": "positive", "sub": "dog", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},

        # Image 03: Blue car, snowy road
        {"img": "000003.jpg", "id": "vg_003", "resp_id": "r_003", "resp": "A blue car is parked on the snowy road.", "claim": "A car is present on the road.", "type": "object_existence", "pol": "positive", "sub": "car", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000003.jpg", "id": "vg_003", "resp_id": "r_003", "resp": "A blue car is parked on the snowy road.", "claim": "The car is blue.", "type": "color", "pol": "positive", "sub": "car", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000003.jpg", "id": "vg_003", "resp_id": "r_003", "resp": "A blue car is parked on the snowy road.", "claim": "The road is covered in snow.", "type": "attribute", "pol": "positive", "sub": "road", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000003.jpg", "id": "vg_003", "resp_id": "r_003", "resp": "A blue car is parked on the snowy road.", "claim": "The car is driving at 80 mph.", "type": "action", "pol": "positive", "sub": "car", "a1": "INSUFFICIENT_EVIDENCE", "a2": "INSUFFICIENT_EVIDENCE", "a3": "INSUFFICIENT_EVIDENCE", "gt": "INSUFFICIENT_EVIDENCE"},

        # Image 04: STOP Sign
        {"img": "000004.jpg", "id": "vg_004", "resp_id": "r_004", "resp": "The stop sign says STOP on the street.", "claim": "A stop sign is visible.", "type": "object_existence", "pol": "positive", "sub": "sign", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000004.jpg", "id": "vg_004", "resp_id": "r_004", "resp": "The stop sign says STOP on the street.", "claim": "The sign says STOP.", "type": "ocr_text", "pol": "positive", "sub": "sign", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000004.jpg", "id": "vg_004", "resp_id": "r_004", "resp": "The stop sign says STOP on the street.", "claim": "The sign is red.", "type": "color", "pol": "positive", "sub": "sign", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000004.jpg", "id": "vg_004", "resp_id": "r_004", "resp": "The stop sign says STOP on the street.", "claim": "The sign says YIELD.", "type": "ocr_text", "pol": "positive", "sub": "sign", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 05: SLOW Sign
        {"img": "000005.jpg", "id": "vg_005", "resp_id": "r_005", "resp": "The sign says SLOW on the highway.", "claim": "The text reads SLOW.", "type": "ocr_text", "pol": "positive", "sub": "sign", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000005.jpg", "id": "vg_005", "resp_id": "r_005", "resp": "The sign says SLOW on the highway.", "claim": "The sign is yellow.", "type": "color", "pol": "positive", "sub": "sign", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000005.jpg", "id": "vg_005", "resp_id": "r_005", "resp": "The sign says SLOW on the highway.", "claim": "The sign is green.", "type": "color", "pol": "positive", "sub": "sign", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 06: Empty road
        {"img": "000006.jpg", "id": "vg_006", "resp_id": "r_006", "resp": "There are no cars on the empty road.", "claim": "An empty road is visible.", "type": "object_existence", "pol": "positive", "sub": "road", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000006.jpg", "id": "vg_006", "resp_id": "r_006", "resp": "There are no cars on the empty road.", "claim": "There are no cars on the road.", "type": "object_existence", "pol": "negative", "sub": "cars", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000006.jpg", "id": "vg_006", "resp_id": "r_006", "resp": "There are no cars on the empty road.", "claim": "A red sports car is parked on the road.", "type": "object_existence", "pol": "positive", "sub": "car", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000006.jpg", "id": "vg_006", "resp_id": "r_006", "resp": "There are no cars on the empty road.", "claim": "Trees are visible alongside the road.", "type": "object_existence", "pol": "positive", "sub": "trees", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},

        # Image 07: 3 cats on wooden chair
        {"img": "000007.jpg", "id": "vg_007", "resp_id": "r_007", "resp": "There are three cats sitting on the wooden chair.", "claim": "There are three cats.", "type": "object_count", "pol": "positive", "sub": "cats", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000007.jpg", "id": "vg_007", "resp_id": "r_007", "resp": "There are three cats sitting on the wooden chair.", "claim": "The cats are on the chair.", "type": "spatial_relation", "pol": "positive", "sub": "cats", "tar": "chair", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000007.jpg", "id": "vg_007", "resp_id": "r_007", "resp": "There are three cats sitting on the wooden chair.", "claim": "The chair is made of wood.", "type": "attribute", "pol": "positive", "sub": "chair", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000007.jpg", "id": "vg_007", "resp_id": "r_007", "resp": "There are three cats sitting on the wooden chair.", "claim": "There is only one cat.", "type": "object_count", "pol": "positive", "sub": "cats", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 08: 1 cat
        {"img": "000008.jpg", "id": "vg_008", "resp_id": "r_008", "resp": "There is a single cat on the chair.", "claim": "There is one cat.", "type": "object_count", "pol": "positive", "sub": "cat", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000008.jpg", "id": "vg_008", "resp_id": "r_008", "resp": "There is a single cat on the chair.", "claim": "The cat is white.", "type": "color", "pol": "positive", "sub": "cat", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000008.jpg", "id": "vg_008", "resp_id": "r_008", "resp": "There is a single cat on the chair.", "claim": "There are five cats.", "type": "object_count", "pol": "positive", "sub": "cats", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 09: Cup and desk
        {"img": "000009.jpg", "id": "vg_009", "resp_id": "r_009", "resp": "A cup is on top of the desk beside a laptop.", "claim": "A cup is on top of the desk.", "type": "spatial_relation", "pol": "positive", "sub": "cup", "tar": "desk", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000009.jpg", "id": "vg_009", "resp_id": "r_009", "resp": "A cup is on top of the desk beside a laptop.", "claim": "A laptop is beside the cup.", "type": "spatial_relation", "pol": "positive", "sub": "laptop", "tar": "cup", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000009.jpg", "id": "vg_009", "resp_id": "r_009", "resp": "A cup is on top of the desk beside a laptop.", "claim": "The cup is white.", "type": "color", "pol": "positive", "sub": "cup", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000009.jpg", "id": "vg_009", "resp_id": "r_009", "resp": "A cup is on top of the desk beside a laptop.", "claim": "The cup is floating in the air.", "type": "spatial_relation", "pol": "positive", "sub": "cup", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 10: Violinist & Cello
        {"img": "000010.jpg", "id": "vg_010", "resp_id": "r_010", "resp": "A violinist is playing a wooden cello beside a microphone.", "claim": "A musician is playing a cello.", "type": "action", "pol": "positive", "sub": "musician", "tar": "cello", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000010.jpg", "id": "vg_010", "resp_id": "r_010", "resp": "A violinist is playing a wooden cello beside a microphone.", "claim": "The cello is wooden brown.", "type": "color", "pol": "positive", "sub": "cello", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000010.jpg", "id": "vg_010", "resp_id": "r_010", "resp": "A violinist is playing a wooden cello beside a microphone.", "claim": "A microphone is beside the musician.", "type": "spatial_relation", "pol": "positive", "sub": "microphone", "tar": "musician", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000010.jpg", "id": "vg_010", "resp_id": "r_010", "resp": "A violinist is playing a wooden cello beside a microphone.", "claim": "The musician is playing drums.", "type": "action", "pol": "positive", "sub": "musician", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 11: Astronomer
        {"img": "000011.jpg", "id": "vg_011", "resp_id": "r_011", "resp": "An astronomer is looking through a metallic telescope on the balcony.", "claim": "An astronomer is on the balcony.", "type": "spatial_relation", "pol": "positive", "sub": "astronomer", "tar": "balcony", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000011.jpg", "id": "vg_011", "resp_id": "r_011", "resp": "An astronomer is looking through a metallic telescope on the balcony.", "claim": "A metallic telescope is present.", "type": "attribute", "pol": "positive", "sub": "telescope", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000011.jpg", "id": "vg_011", "resp_id": "r_011", "resp": "An astronomer is looking through a metallic telescope on the balcony.", "claim": "The telescope is made of gold.", "type": "attribute", "pol": "positive", "sub": "telescope", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000011.jpg", "id": "vg_011", "resp_id": "r_011", "resp": "An astronomer is looking through a metallic telescope on the balcony.", "claim": "There is a dog on the balcony.", "type": "object_existence", "pol": "positive", "sub": "dog", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 12: Coffee bar
        {"img": "000012.jpg", "id": "vg_012", "resp_id": "r_012", "resp": "An espresso tamper is resting on the marble counter next to a grinder.", "claim": "An espresso tamper is on the counter.", "type": "spatial_relation", "pol": "positive", "sub": "tamper", "tar": "counter", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000012.jpg", "id": "vg_012", "resp_id": "r_012", "resp": "An espresso tamper is resting on the marble counter next to a grinder.", "claim": "A coffee grinder is next to the tamper.", "type": "spatial_relation", "pol": "positive", "sub": "grinder", "tar": "tamper", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000012.jpg", "id": "vg_012", "resp_id": "r_012", "resp": "An espresso tamper is resting on the marble counter next to a grinder.", "claim": "The counter is made of white marble.", "type": "attribute", "pol": "positive", "sub": "counter", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000012.jpg", "id": "vg_012", "resp_id": "r_012", "resp": "An espresso tamper is resting on the marble counter next to a grinder.", "claim": "There are no cups on the counter.", "type": "object_existence", "pol": "negative", "sub": "cups", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},

        # Image 13: Leopard
        {"img": "000013.jpg", "id": "vg_013", "resp_id": "r_013", "resp": "A snow leopard is climbing on the mountain cliff.", "claim": "A leopard is on the cliff.", "type": "object_existence", "pol": "positive", "sub": "leopard", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000013.jpg", "id": "vg_013", "resp_id": "r_013", "resp": "A snow leopard is climbing on the mountain cliff.", "claim": "The leopard is climbing the rocks.", "type": "action", "pol": "positive", "sub": "leopard", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000013.jpg", "id": "vg_013", "resp_id": "r_013", "resp": "A snow leopard is climbing on the mountain cliff.", "claim": "The leopard is swimming in a lake.", "type": "action", "pol": "positive", "sub": "leopard", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000013.jpg", "id": "vg_013", "resp_id": "r_013", "resp": "A snow leopard is climbing on the mountain cliff.", "claim": "The scene is an outdoor mountain.", "type": "scene", "pol": "positive", "sub": "mountain", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},

        # Image 14: Helicopter
        {"img": "000014.jpg", "id": "vg_014", "resp_id": "r_014", "resp": "A helicopter is flying above the tall skyscraper.", "claim": "A helicopter is above the skyscraper.", "type": "spatial_relation", "pol": "positive", "sub": "helicopter", "tar": "skyscraper", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000014.jpg", "id": "vg_014", "resp_id": "r_014", "resp": "A helicopter is flying above the tall skyscraper.", "claim": "A skyscraper exists.", "type": "object_existence", "pol": "positive", "sub": "skyscraper", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000014.jpg", "id": "vg_014", "resp_id": "r_014", "resp": "A helicopter is flying above the tall skyscraper.", "claim": "The helicopter is below the building.", "type": "spatial_relation", "pol": "positive", "sub": "helicopter", "tar": "skyscraper", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000014.jpg", "id": "vg_014", "resp_id": "r_014", "resp": "A helicopter is flying above the tall skyscraper.", "claim": "There is a single helicopter in the sky.", "type": "object_count", "pol": "positive", "sub": "helicopter", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},

        # Image 15: Surfer
        {"img": "000015.jpg", "id": "vg_015", "resp_id": "r_015", "resp": "A surfer wearing a black wetsuit is riding a large ocean wave.", "claim": "A surfer is riding an ocean wave.", "type": "action", "pol": "positive", "sub": "surfer", "tar": "wave", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000015.jpg", "id": "vg_015", "resp_id": "r_015", "resp": "A surfer wearing a black wetsuit is riding a large ocean wave.", "claim": "The surfer is wearing a black suit.", "type": "color", "pol": "positive", "sub": "wetsuit", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000015.jpg", "id": "vg_015", "resp_id": "r_015", "resp": "A surfer wearing a black wetsuit is riding a large ocean wave.", "claim": "The surfboard is on the water.", "type": "spatial_relation", "pol": "positive", "sub": "surfboard", "tar": "wave", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000015.jpg", "id": "vg_015", "resp_id": "r_015", "resp": "A surfer wearing a black wetsuit is riding a large ocean wave.", "claim": "The surfer is riding a snowboard in snow.", "type": "action", "pol": "positive", "sub": "surfer", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 16: Worker & Ladder
        {"img": "000016.jpg", "id": "vg_016", "resp_id": "r_016", "resp": "A construction worker wearing a yellow hardhat is holding a hammer near a ladder.", "claim": "A worker wearing a yellow hardhat is present.", "type": "color", "pol": "positive", "sub": "hardhat", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000016.jpg", "id": "vg_016", "resp_id": "r_016", "resp": "A construction worker wearing a yellow hardhat is holding a hammer near a ladder.", "claim": "A ladder is near the worker.", "type": "spatial_relation", "pol": "positive", "sub": "ladder", "tar": "worker", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000016.jpg", "id": "vg_016", "resp_id": "r_016", "resp": "A construction worker wearing a yellow hardhat is holding a hammer near a ladder.", "claim": "The hardhat is bright purple.", "type": "color", "pol": "positive", "sub": "hardhat", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000016.jpg", "id": "vg_016", "resp_id": "r_016", "resp": "A construction worker wearing a yellow hardhat is holding a hammer near a ladder.", "claim": "The construction company is founded in 1985.", "type": "external_knowledge", "pol": "positive", "sub": "company", "a1": "OUT_OF_SCOPE", "a2": "OUT_OF_SCOPE", "a3": "OUT_OF_SCOPE", "gt": "OUT_OF_SCOPE"},

        # Image 17: Fruit Table
        {"img": "000017.jpg", "id": "vg_017", "resp_id": "r_017", "resp": "A red apple and a yellow banana are on the table.", "claim": "A red apple is on the table.", "type": "color", "pol": "positive", "sub": "apple", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000017.jpg", "id": "vg_017", "resp_id": "r_017", "resp": "A red apple and a yellow banana are on the table.", "claim": "A yellow banana is present.", "type": "color", "pol": "positive", "sub": "banana", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000017.jpg", "id": "vg_017", "resp_id": "r_017", "resp": "A red apple and a yellow banana are on the table.", "claim": "A green bottle is beside the fruits.", "type": "color", "pol": "positive", "sub": "bottle", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000017.jpg", "id": "vg_017", "resp_id": "r_017", "resp": "A red apple and a yellow banana are on the table.", "claim": "The banana is purple.", "type": "color", "pol": "positive", "sub": "banana", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 18: Parking lot (4 cars)
        {"img": "000018.jpg", "id": "vg_018", "resp_id": "r_018", "resp": "There are four cars parked in the lot.", "claim": "There are four cars.", "type": "object_count", "pol": "positive", "sub": "cars", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000018.jpg", "id": "vg_018", "resp_id": "r_018", "resp": "There are four cars parked in the lot.", "claim": "There are multiple cars parked in the lot.", "type": "object_count", "pol": "positive", "sub": "cars", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000018.jpg", "id": "vg_018", "resp_id": "r_018", "resp": "There are four cars parked in the lot.", "claim": "There are two cars.", "type": "object_count", "pol": "positive", "sub": "cars", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000018.jpg", "id": "vg_018", "resp_id": "r_018", "resp": "There are four cars parked in the lot.", "claim": "A red car is among the parked vehicles.", "type": "color", "pol": "positive", "sub": "car", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},

        # Image 19: Blackboard MATH 101
        {"img": "000019.jpg", "id": "vg_019", "resp_id": "r_019", "resp": "The blackboard in the classroom has MATH 101 written on it.", "claim": "The blackboard has text MATH 101.", "type": "ocr_text", "pol": "positive", "sub": "blackboard", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000019.jpg", "id": "vg_019", "resp_id": "r_019", "resp": "The blackboard in the classroom has MATH 101 written on it.", "claim": "There are two desks in the classroom.", "type": "object_count", "pol": "positive", "sub": "desks", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000019.jpg", "id": "vg_019", "resp_id": "r_019", "resp": "The blackboard in the classroom has MATH 101 written on it.", "claim": "The text reads PHYSICS 200.", "type": "ocr_text", "pol": "positive", "sub": "blackboard", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000019.jpg", "id": "vg_019", "resp_id": "r_019", "resp": "The blackboard in the classroom has MATH 101 written on it.", "claim": "The environment is an indoor classroom.", "type": "scene", "pol": "positive", "sub": "classroom", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},

        # Image 20: Library & Newton
        {"img": "000020.jpg", "id": "vg_020", "resp_id": "r_020", "resp": "This ancient manuscript was written by Isaac Newton in Cambridge during 1687.", "claim": "An ancient book or manuscript is present.", "type": "object_existence", "pol": "positive", "sub": "manuscript", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000020.jpg", "id": "vg_020", "resp_id": "r_020", "resp": "This ancient manuscript was written by Isaac Newton in Cambridge during 1687.", "claim": "A clock is on the shelf.", "type": "object_existence", "pol": "positive", "sub": "clock", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000020.jpg", "id": "vg_020", "resp_id": "r_020", "resp": "This ancient manuscript was written by Isaac Newton in Cambridge during 1687.", "claim": "The manuscript was written in 1687 by Isaac Newton.", "type": "external_knowledge", "pol": "positive", "sub": "manuscript", "a1": "OUT_OF_SCOPE", "a2": "OUT_OF_SCOPE", "a3": "OUT_OF_SCOPE", "gt": "OUT_OF_SCOPE"},
        {"img": "000020.jpg", "id": "vg_020", "resp_id": "r_020", "resp": "This ancient manuscript was written by Isaac Newton in Cambridge during 1687.", "claim": "The photograph was taken in Cambridge.", "type": "external_knowledge", "pol": "positive", "sub": "photograph", "a1": "OUT_OF_SCOPE", "a2": "OUT_OF_SCOPE", "a3": "OUT_OF_SCOPE", "gt": "OUT_OF_SCOPE"},

        # Image 21: Airplane
        {"img": "000021.jpg", "id": "vg_021", "resp_id": "r_021", "resp": "An airplane is on the airport runway.", "claim": "An airplane is present.", "type": "object_existence", "pol": "positive", "sub": "airplane", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000021.jpg", "id": "vg_021", "resp_id": "r_021", "resp": "An airplane is on the airport runway.", "claim": "The airplane is on the runway.", "type": "spatial_relation", "pol": "positive", "sub": "airplane", "tar": "runway", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000021.jpg", "id": "vg_021", "resp_id": "r_021", "resp": "An airplane is on the airport runway.", "claim": "The airplane is bright green.", "type": "color", "pol": "positive", "sub": "airplane", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},
        {"img": "000021.jpg", "id": "vg_021", "resp_id": "r_021", "resp": "An airplane is on the airport runway.", "claim": "There are helicopters on the runway.", "type": "object_existence", "pol": "positive", "sub": "helicopters", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 22: Strawberries
        {"img": "000022.jpg", "id": "vg_022", "resp_id": "r_022", "resp": "There are several red strawberries in the glass bowl.", "claim": "There are several strawberries.", "type": "object_count", "pol": "positive", "sub": "strawberries", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000022.jpg", "id": "vg_022", "resp_id": "r_022", "resp": "There are several red strawberries in the glass bowl.", "claim": "The strawberries are inside the glass bowl.", "type": "spatial_relation", "pol": "positive", "sub": "strawberries", "tar": "bowl", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000022.jpg", "id": "vg_022", "resp_id": "r_022", "resp": "There are several red strawberries in the glass bowl.", "claim": "The strawberries are red.", "type": "color", "pol": "positive", "sub": "strawberries", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000022.jpg", "id": "vg_022", "resp_id": "r_022", "resp": "There are several red strawberries in the glass bowl.", "claim": "There is only one strawberry in the bowl.", "type": "object_count", "pol": "positive", "sub": "strawberries", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 23: Soccer
        {"img": "000023.jpg", "id": "vg_023", "resp_id": "r_023", "resp": "A soccer player in a blue jersey is kicking a soccer ball.", "claim": "The player is kicking the ball.", "type": "action", "pol": "positive", "sub": "player", "tar": "ball", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000023.jpg", "id": "vg_023", "resp_id": "r_023", "resp": "A soccer player in a blue jersey is kicking a soccer ball.", "claim": "The player is wearing a blue jersey.", "type": "color", "pol": "positive", "sub": "jersey", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000023.jpg", "id": "vg_023", "resp_id": "r_023", "resp": "A soccer player in a blue jersey is kicking a soccer ball.", "claim": "A goal net is present on the field.", "type": "object_existence", "pol": "positive", "sub": "goal", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000023.jpg", "id": "vg_023", "resp_id": "r_023", "resp": "A soccer player in a blue jersey is kicking a soccer ball.", "claim": "The player is hitting a baseball with a bat.", "type": "action", "pol": "positive", "sub": "player", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 24: Hikers
        {"img": "000024.jpg", "id": "vg_024", "resp_id": "r_024", "resp": "A pair of hikers is walking along a dirt path in the forest.", "claim": "There is a pair of hikers.", "type": "object_count", "pol": "positive", "sub": "hikers", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000024.jpg", "id": "vg_024", "resp_id": "r_024", "resp": "A pair of hikers is walking along a dirt path in the forest.", "claim": "The hikers are walking on the dirt path.", "type": "action", "pol": "positive", "sub": "hikers", "tar": "path", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000024.jpg", "id": "vg_024", "resp_id": "r_024", "resp": "A pair of hikers is walking along a dirt path in the forest.", "claim": "The scene is an outdoor forest.", "type": "scene", "pol": "positive", "sub": "forest", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000024.jpg", "id": "vg_024", "resp_id": "r_024", "resp": "A pair of hikers is walking along a dirt path in the forest.", "claim": "There are ten hikers on the path.", "type": "object_count", "pol": "positive", "sub": "hikers", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"},

        # Image 25: Art Gallery
        {"img": "000025.jpg", "id": "vg_025", "resp_id": "r_025", "resp": "A framed oil painting hangs above a museum plaque in the gallery.", "claim": "A framed painting is above the plaque.", "type": "spatial_relation", "pol": "positive", "sub": "painting", "tar": "plaque", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000025.jpg", "id": "vg_025", "resp_id": "r_025", "resp": "A framed oil painting hangs above a museum plaque in the gallery.", "claim": "A museum plaque is visible.", "type": "object_existence", "pol": "positive", "sub": "plaque", "a1": "SUPPORTED", "a2": "SUPPORTED", "a3": "SUPPORTED", "gt": "SUPPORTED"},
        {"img": "000025.jpg", "id": "vg_025", "resp_id": "r_025", "resp": "A framed oil painting hangs above a museum plaque in the gallery.", "claim": "The painting was created by Leonardo da Vinci in 1503.", "type": "external_knowledge", "pol": "positive", "sub": "painting", "a1": "OUT_OF_SCOPE", "a2": "OUT_OF_SCOPE", "a3": "OUT_OF_SCOPE", "gt": "OUT_OF_SCOPE"},
        {"img": "000025.jpg", "id": "vg_025", "resp_id": "r_025", "resp": "A framed oil painting hangs above a museum plaque in the gallery.", "claim": "The painting is completely blank and white.", "type": "color", "pol": "positive", "sub": "painting", "a1": "CONTRADICTED", "a2": "CONTRADICTED", "a3": "CONTRADICTED", "gt": "CONTRADICTED"}
    ]

    claims_file = os.path.join(ANN_DIR, "claims.jsonl")
    splits_file = os.path.join(ANN_DIR, "splits.json")

    claims_out = []
    claim_ids = []

    for i, c in enumerate(raw_claims, start=1):
        cid = f"claim_{i:03d}"
        claim_ids.append(cid)
        record = {
            "claim_id": cid,
            "image_id": c["id"],
            "image_path": os.path.join("data", "images", c["img"]),
            "response_id": c["resp_id"],
            "response_text": c["resp"],
            "claim": c["claim"],
            "claim_type": c["type"],
            "polarity": c["pol"],
            "verifiability": "external_knowledge" if c["type"] == "external_knowledge" else "visual",
            "subject": c["sub"],
            "target_object": c.get("tar"),
            "annotator_1": c["a1"],
            "annotator_2": c["a2"],
            "annotator_3": c["a3"],
            "ground_truth": c["gt"]
        }
        claims_out.append(record)

    with open(claims_file, "w", encoding="utf-8") as f:
        for rec in claims_out:
            f.write(json.dumps(rec) + "\n")

    # Generate 60% Train, 20% Val, 20% Test splits
    n = len(claim_ids)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    splits = {
        "train": claim_ids[:n_train],
        "val": claim_ids[n_train:n_train + n_val],
        "test": claim_ids[n_train + n_val:]
    }

    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # Also generate comprehensive 25-response visionguard_claims_v0.1.jsonl linked to images
    sample_records = [
        {"id": "vg_001", "image_id": "img_001", "image_path": "data/images/000001.jpg", "response_text": "A young boy wearing a red shirt is riding a bicycle near a park.", "ground_truth_verdict": "Supported", "subject": "boy", "visual_features": {"objects": {"boy": 0.95, "bicycle": 0.92, "shirt": 0.94}, "object_details": {"boy": {"confidence": 0.95, "bbox": [0.38, 0.20, 0.52, 0.80]}, "bicycle": {"confidence": 0.92, "bbox": [0.28, 0.62, 0.68, 0.84]}}, "color_distribution": {"red": 0.35, "green": 0.40}, "scene": {"type": "outdoor park", "is_outdoor": True}}},
        {"id": "vg_002", "image_id": "img_002", "image_path": "data/images/000002.jpg", "response_text": "A brown dog is sitting beside a red bicycle.", "ground_truth_verdict": "Supported", "subject": "dog", "visual_features": {"objects": {"dog": 0.94, "bicycle": 0.92}, "object_details": {"dog": {"confidence": 0.94, "bbox": [0.12, 0.37, 0.38, 0.80]}, "bicycle": {"confidence": 0.92, "bbox": [0.56, 0.42, 0.94, 0.75]}}, "color_distribution": {"brown": 0.45, "red": 0.30}, "scene": {"type": "outdoor yard", "is_outdoor": True}}},
        {"id": "vg_003", "image_id": "img_003", "image_path": "data/images/000003.jpg", "response_text": "A blue car is parked on the snowy road.", "ground_truth_verdict": "Supported", "subject": "car", "visual_features": {"objects": {"car": 0.96, "road": 0.90}, "object_details": {"car": {"confidence": 0.96, "bbox": [0.28, 0.46, 0.72, 0.85]}}, "color_distribution": {"blue": 0.55, "white": 0.35}, "scene": {"type": "snowy road", "is_outdoor": True}}},
        {"id": "vg_004", "image_id": "img_004", "image_path": "data/images/000004.jpg", "response_text": "The stop sign says STOP on the street.", "ground_truth_verdict": "Supported", "subject": "sign", "visual_features": {"objects": {"sign": 0.96, "street": 0.89}, "object_details": {"sign": {"confidence": 0.96, "bbox": [0.37, 0.12, 0.75, 0.55]}}, "ocr_tokens": [{"text": "STOP", "confidence": 0.98, "bbox": [0.45, 0.28, 0.60, 0.42]}], "color_distribution": {"red": 0.60, "white": 0.20}, "scene": {"type": "outdoor street", "is_outdoor": True}}},
        {"id": "vg_005", "image_id": "img_005", "image_path": "data/images/000005.jpg", "response_text": "The sign says SLOW on the highway.", "ground_truth_verdict": "Supported", "subject": "sign", "visual_features": {"objects": {"sign": 0.96}, "object_details": {"sign": {"confidence": 0.96, "bbox": [0.28, 0.12, 0.72, 0.62]}}, "ocr_tokens": [{"text": "SLOW", "confidence": 0.98, "bbox": [0.44, 0.35, 0.58, 0.45]}], "color_distribution": {"yellow": 0.65}, "scene": {"type": "outdoor highway", "is_outdoor": True}}},
        {"id": "vg_006", "image_id": "img_006", "image_path": "data/images/000006.jpg", "response_text": "There are no cars on the empty road.", "ground_truth_verdict": "Supported", "subject": "cars", "visual_features": {"objects": {"road": 0.92, "tree": 0.88}, "object_details": {"road": {"confidence": 0.92, "bbox": [0.12, 0.37, 0.88, 1.0]}}, "scene": {"type": "empty road", "is_outdoor": True}}},
        {"id": "vg_007", "image_id": "img_007", "image_path": "data/images/000007.jpg", "response_text": "There are three cats sitting on the wooden chair.", "ground_truth_verdict": "Supported", "subject": "cats", "visual_features": {"objects": {"cat": 0.94, "chair": 0.95}, "object_details": {"cat": {"confidence": 0.94, "bbox": [0.32, 0.28, 0.68, 0.72]}, "chair": {"confidence": 0.95, "bbox": [0.31, 0.37, 0.69, 0.88]}}, "bounding_boxes": [[0.34, 0.29, 0.44, 0.42], [0.53, 0.29, 0.63, 0.42], [0.42, 0.46, 0.52, 0.58]], "scene": {"type": "indoor room", "is_outdoor": False}}},
        {"id": "vg_008", "image_id": "img_008", "image_path": "data/images/000008.jpg", "response_text": "There are three cats sitting on the wooden chair.", "ground_truth_verdict": "Contradicted", "subject": "cats", "visual_features": {"objects": {"cat": 0.93, "chair": 0.95}, "object_details": {"cat": {"confidence": 0.93, "bbox": [0.44, 0.29, 0.54, 0.42]}}, "bounding_boxes": [[0.44, 0.29, 0.54, 0.42]], "scene": {"type": "indoor room", "is_outdoor": False}}},
        {"id": "vg_009", "image_id": "img_009", "image_path": "data/images/000009.jpg", "response_text": "A cup is on top of the desk beside a laptop.", "ground_truth_verdict": "Supported", "subject": "cup", "visual_features": {"objects": {"cup": 0.94, "desk": 0.96, "laptop": 0.92}, "object_details": {"cup": {"confidence": 0.94, "bbox": [0.27, 0.39, 0.39, 0.56]}, "desk": {"confidence": 0.96, "bbox": [0.15, 0.52, 0.85, 0.94]}, "laptop": {"confidence": 0.92, "bbox": [0.47, 0.31, 0.75, 0.67]}}, "scene": {"type": "indoor office", "is_outdoor": False}}},
        {"id": "vg_010", "image_id": "img_010", "image_path": "data/images/000010.jpg", "response_text": "A violinist is playing a wooden cello beside a microphone.", "ground_truth_verdict": "Supported", "subject": "violinist", "visual_features": {"objects": {"musician": 0.94, "cello": 0.93, "microphone": 0.90}, "object_details": {"musician": {"confidence": 0.94, "bbox": [0.33, 0.21, 0.45, 0.67]}, "cello": {"confidence": 0.93, "bbox": [0.44, 0.37, 0.59, 0.75]}, "microphone": {"confidence": 0.90, "bbox": [0.71, 0.25, 0.76, 0.83]}}, "color_distribution": {"brown": 0.45, "black": 0.30}, "scene": {"type": "indoor concert stage", "is_outdoor": False}}},
        {"id": "vg_011", "image_id": "img_011", "image_path": "data/images/000011.jpg", "response_text": "An astronomer is looking through a metallic telescope on the balcony.", "ground_truth_verdict": "Supported", "subject": "astronomer", "visual_features": {"objects": {"astronomer": 0.92, "telescope": 0.94, "balcony": 0.90}, "object_details": {"astronomer": {"confidence": 0.92, "bbox": [0.31, 0.33, 0.41, 0.46]}, "telescope": {"confidence": 0.94, "bbox": [0.47, 0.37, 0.75, 0.54]}}, "color_distribution": {"black": 0.50, "blue": 0.30}, "scene": {"type": "outdoor night balcony", "is_outdoor": True}}},
        {"id": "vg_012", "image_id": "img_012", "image_path": "data/images/000012.jpg", "response_text": "An espresso tamper is resting on the marble counter next to a grinder.", "ground_truth_verdict": "Supported", "subject": "tamper", "visual_features": {"objects": {"tamper": 0.93, "counter": 0.96, "grinder": 0.91}, "object_details": {"tamper": {"confidence": 0.93, "bbox": [0.25, 0.42, 0.38, 0.58]}, "grinder": {"confidence": 0.91, "bbox": [0.56, 0.29, 0.75, 0.67]}}, "color_distribution": {"white": 0.50, "black": 0.30}, "scene": {"type": "indoor coffee bar", "is_outdoor": False}}},
        {"id": "vg_013", "image_id": "img_013", "image_path": "data/images/000013.jpg", "response_text": "A snow leopard is climbing on the mountain cliff.", "ground_truth_verdict": "Supported", "subject": "leopard", "visual_features": {"objects": {"leopard": 0.94, "cliff": 0.95}, "object_details": {"leopard": {"confidence": 0.94, "bbox": [0.44, 0.46, 0.66, 0.67]}}, "color_distribution": {"white": 0.60, "gray": 0.25}, "scene": {"type": "snowy mountain", "is_outdoor": True}}},
        {"id": "vg_014", "image_id": "img_014", "image_path": "data/images/000014.jpg", "response_text": "A helicopter is flying above the tall skyscraper.", "ground_truth_verdict": "Supported", "subject": "helicopter", "visual_features": {"objects": {"helicopter": 0.94, "skyscraper": 0.97}, "object_details": {"helicopter": {"confidence": 0.94, "bbox": [0.41, 0.17, 0.59, 0.29]}, "skyscraper": {"confidence": 0.97, "bbox": [0.31, 0.42, 0.69, 1.0]}}, "scene": {"type": "outdoor city sky", "is_outdoor": True}}},
        {"id": "vg_015", "image_id": "img_015", "image_path": "data/images/000015.jpg", "response_text": "A surfer wearing a black wetsuit is riding a large ocean wave.", "ground_truth_verdict": "Supported", "subject": "surfer", "visual_features": {"objects": {"surfer": 0.95, "wave": 0.97, "surfboard": 0.93}, "object_details": {"surfer": {"confidence": 0.95, "bbox": [0.44, 0.42, 0.56, 0.58]}}, "color_distribution": {"blue": 0.60, "white": 0.25, "black": 0.10}, "scene": {"type": "outdoor ocean beach", "is_outdoor": True}}},
        {"id": "vg_016", "image_id": "img_016", "image_path": "data/images/000016.jpg", "response_text": "A construction worker wearing a yellow hardhat is holding a hammer near a ladder.", "ground_truth_verdict": "Supported", "subject": "worker", "visual_features": {"objects": {"worker": 0.95, "hardhat": 0.94, "ladder": 0.93}, "object_details": {"worker": {"confidence": 0.95, "bbox": [0.27, 0.25, 0.39, 0.71]}, "ladder": {"confidence": 0.93, "bbox": [0.56, 0.21, 0.69, 0.92]}}, "color_distribution": {"yellow": 0.25, "blue": 0.35}, "scene": {"type": "outdoor construction site", "is_outdoor": True}}},
        {"id": "vg_017", "image_id": "img_017", "image_path": "data/images/000017.jpg", "response_text": "A red apple and a yellow banana are on the table.", "ground_truth_verdict": "Supported", "subject": "apple", "visual_features": {"objects": {"apple": 0.95, "banana": 0.94, "table": 0.96}, "object_details": {"apple": {"confidence": 0.95, "bbox": [0.22, 0.50, 0.34, 0.67]}, "banana": {"confidence": 0.94, "bbox": [0.41, 0.54, 0.59, 0.65]}}, "color_distribution": {"red": 0.25, "yellow": 0.30}, "scene": {"type": "indoor dining", "is_outdoor": False}}},
        {"id": "vg_018", "image_id": "img_018", "image_path": "data/images/000018.jpg", "response_text": "There are four cars parked in the lot.", "ground_truth_verdict": "Supported", "subject": "cars", "visual_features": {"objects": {"car": 0.95, "parking": 0.92}, "object_details": {"car": {"confidence": 0.95, "bbox": [0.09, 0.25, 0.91, 0.50]}}, "bounding_boxes": [[0.09, 0.25, 0.25, 0.50], [0.31, 0.25, 0.47, 0.50], [0.53, 0.25, 0.69, 0.50], [0.75, 0.25, 0.91, 0.50]], "scene": {"type": "outdoor parking lot", "is_outdoor": True}}},
        {"id": "vg_019", "image_id": "img_019", "image_path": "data/images/000019.jpg", "response_text": "The blackboard in the classroom has MATH 101 written on it.", "ground_truth_verdict": "Supported", "subject": "blackboard", "visual_features": {"objects": {"blackboard": 0.96, "desk": 0.92}, "object_details": {"blackboard": {"confidence": 0.96, "bbox": [0.16, 0.12, 0.84, 0.54]}}, "ocr_tokens": [{"text": "MATH", "confidence": 0.98, "bbox": [0.35, 0.25, 0.45, 0.35]}, {"text": "101", "confidence": 0.98, "bbox": [0.48, 0.25, 0.58, 0.35]}], "scene": {"type": "indoor classroom", "is_outdoor": False}}},
        {"id": "vg_020", "image_id": "img_020", "image_path": "data/images/000020.jpg", "response_text": "This ancient manuscript was written by Isaac Newton in Cambridge during 1687.", "ground_truth_verdict": "Insufficient_Evidence", "subject": "manuscript", "visual_features": {"objects": {"book": 0.92, "clock": 0.90}, "scene": {"type": "indoor library", "is_outdoor": False}}},
        {"id": "vg_021", "image_id": "img_021", "image_path": "data/images/000021.jpg", "response_text": "An airplane is on the airport runway.", "ground_truth_verdict": "Supported", "subject": "airplane", "visual_features": {"objects": {"airplane": 0.95, "runway": 0.94}, "object_details": {"airplane": {"confidence": 0.95, "bbox": [0.25, 0.37, 0.75, 0.54]}}, "scene": {"type": "outdoor airport", "is_outdoor": True}}},
        {"id": "vg_022", "image_id": "img_022", "image_path": "data/images/000022.jpg", "response_text": "There are several red strawberries in the glass bowl.", "ground_truth_verdict": "Supported", "subject": "strawberries", "visual_features": {"objects": {"strawberry": 0.94, "bowl": 0.96}, "object_details": {"strawberry": {"confidence": 0.94, "bbox": [0.34, 0.44, 0.66, 0.71]}}, "bounding_boxes": [[0.34, 0.46, 0.44, 0.58], [0.45, 0.44, 0.55, 0.56], [0.56, 0.48, 0.66, 0.60], [0.41, 0.56, 0.50, 0.69], [0.52, 0.58, 0.61, 0.71]], "color_distribution": {"red": 0.45}, "scene": {"type": "indoor kitchen", "is_outdoor": False}}},
        {"id": "vg_023", "image_id": "img_023", "image_path": "data/images/000023.jpg", "response_text": "A soccer player in a blue jersey is kicking a soccer ball.", "ground_truth_verdict": "Supported", "subject": "player", "visual_features": {"objects": {"player": 0.95, "ball": 0.93, "goal": 0.92}, "object_details": {"player": {"confidence": 0.95, "bbox": [0.22, 0.37, 0.34, 0.75]}, "ball": {"confidence": 0.93, "bbox": [0.41, 0.67, 0.50, 0.79]}}, "color_distribution": {"blue": 0.30, "green": 0.50}, "scene": {"type": "outdoor soccer field", "is_outdoor": True}}},
        {"id": "vg_024", "image_id": "img_024", "image_path": "data/images/000024.jpg", "response_text": "A pair of hikers is walking along a dirt path in the forest.", "ground_truth_verdict": "Supported", "subject": "hikers", "visual_features": {"objects": {"hiker": 0.94, "path": 0.92}, "object_details": {"hiker": {"confidence": 0.94, "bbox": [0.37, 0.37, 0.61, 0.67]}}, "bounding_boxes": [[0.37, 0.37, 0.45, 0.67], [0.53, 0.37, 0.61, 0.67]], "scene": {"type": "outdoor forest", "is_outdoor": True}}},
        {"id": "vg_025", "image_id": "img_025", "image_path": "data/images/000025.jpg", "response_text": "A framed oil painting hangs above a museum plaque in the gallery.", "ground_truth_verdict": "Supported", "subject": "painting", "visual_features": {"objects": {"painting": 0.96, "plaque": 0.92}, "object_details": {"painting": {"confidence": 0.96, "bbox": [0.22, 0.17, 0.78, 0.67]}, "plaque": {"confidence": 0.92, "bbox": [0.41, 0.73, 0.59, 0.83]}}, "scene": {"type": "indoor art gallery", "is_outdoor": False}}}
    ]

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for rec in sample_records:
            f.write(json.dumps(rec) + "\n")

    print(f"[DatasetBuilder] Successfully generated {len(sample_records)} benchmark images and {len(claims_out)} atomic claims.")


if __name__ == "__main__":
    build_all_images()
    build_annotations()
