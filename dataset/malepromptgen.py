import random

# Define ethnicity features with probabilities
ethnicities = {
    "Caucasian": {"wrinkle_mod": 1.0, "rare_eye_color": ["blue", "green", "hazel"]},
    "African": {"wrinkle_mod": 0.8, "rare_eye_color": ["amber", "gray"]},
    "Asian": {"wrinkle_mod": 0.7, "rare_eye_color": ["amber"]},
    "Hispanic": {"wrinkle_mod": 1.0, "rare_eye_color": ["hazel"]},
    "Middle Eastern": {"wrinkle_mod": 1.2, "rare_eye_color": []},
}

weights = ["underweight", "average", "overweight", "obese"]

# Probabilities for rare features
rare_conditions = {
    "violet_eyes": 0.01,
    "freckles": 0.1,
    "wrinkles": 0.2,
}

def random_age():
    return random.randint(20, 70)

def calculate_wrinkle_probability(age, ethnicity):
    base_prob = 0.05 if age < 30 else 0.3 if age <= 50 else 0.7
    mod = ethnicities.get(ethnicity, {"wrinkle_mod": 1.0})["wrinkle_mod"]
    return random.random() < base_prob * mod

def random_hair(ethnicity):
    lengths = ["short", "medium-length", "long", "shoulder-length", "buzzed", "bald"]
    styles = {
        "short": ["crew cut", "Caesar cut", "induction cut", "buzz cut", "brush cut"],
        "medium-length": ["curly hair", "pompadour", "side part", "wavy hair"],
        "long": ["afro", "braid", "long straight hair", "ponytail", "long curly hair", "dreadlocks"],
        "shoulder-length": ["straight hair", "wavy hair", "loose curls", "shoulder-length braid"],
        "buzzed": ["buzz cut"],
        "bald": []
    }
    colors = ["black", "brown", "blonde", "red", "gray", "white", "auburn", "golden", "platinum blonde", "silver"]

    length = random.choices(lengths, weights=[20, 25, 20, 15, 10, 10], k=1)[0]
    if length == "bald":
        return "bald"
    
    style = random.choice(styles[length])
    if ethnicity != "Caucasian" and style != "buzzed":
        colors = [c for c in colors if c not in ["blonde", "platinum blonde"]]
    
    color = random.choice(colors)
    return f"{length} {style} {color} hair"

def random_skin():
    tones = ["fair", "light", "medium", "tan", "brown", "dark", "golden"]
    skin = f"{random.choice(tones)} skin"
    if random.random() < rare_conditions["freckles"]:
        skin += " with freckles"
    return skin

def random_facial_hair():
    return "with beard"

def random_eyes(ethnicity):
    base_colors = ["brown", "blue", "green", "gray", "hazel", "amber"]
    rare_colors = ethnicities[ethnicity]["rare_eye_color"]
    if random.random() < rare_conditions["violet_eyes"]:
        return "violet eyes"
    color = random.choice(base_colors + rare_colors)
    shapes = ["almond-shaped", "monolid", "round", "hooded", "deep-set"]
    return f"{random.choice(shapes)} {color} eyes"

def generate_prompt():
    gender = "male"
    age = random_age()
    ethnicity = random.choice(list(ethnicities.keys()))
    weight = random.choices(weights, weights=[15, 50, 25, 10], k=1)[0]
    
    # Wrinkles based on age and ethnicity
    has_wrinkles = calculate_wrinkle_probability(age, ethnicity)
    
    # Generate features
    hair = random_hair(ethnicity)
    skin = random_skin()
    face_structure = random.choice(["oval face", "round face", "square face", "heart-shaped face", "diamond face"])
    eyes = random_eyes(ethnicity)
    eyebrows = random.choice(["arched", "straight", "rounded", "angled"]) + " eyebrows"
    nose = random.choice(["small", "medium", "large", "button-shaped", "aquiline"]) + " nose"
    mouth = random.choice(["small lips", "full lips", "thin lips", "heart-shaped lips"])
    facial_hair = random_facial_hair()

    return (
        f"A {age}-year-old {gender} of {ethnicity} ethnicity with {hair}, {skin}, {face_structure}, "
        f"{eyes}, {eyebrows}, {nose}, {mouth}, and {facial_hair}. "
        f"He {'has wrinkles' if has_wrinkles else 'does not have wrinkles'}."
    )



def generate_prompts(number_of_prompts):
    # Generate 1000 prompts
    prompts = [generate_prompt() for _ in range(number_of_prompts)]

    # Save prompts to a file
    file_path = "dataset/dataset/prompts.txt"
    with open(file_path, "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    print(f"Prompts saved to {file_path}")
