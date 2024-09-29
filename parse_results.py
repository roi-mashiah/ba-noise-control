import json
import pandas as pd


def add_scores(results_df: pd.DataFrame, input_json: dict):
    results_df["ba_score"] = None
    results_df["gp_score"] = None
    results_df["api_score"] = None

    for i in range(results_df.shape[0]):
        current_input = get_input_object(results_df.loc[i, :], input_json)
        updated_row = get_scores(current_input, results_df.loc[i, :])
        results_df.loc[i, :] = updated_row


def get_object_tokens(input_obj):
    tokens = []
    token_array = input_obj["prompt"].split(" ")
    for obj_ref in input_obj["references"]:
        if len(obj_ref) == 1:
            token = token_array[obj_ref[0] - 1].strip(".")
        else:
            token = " ".join([token_array[t - 1].strip(".")
                              for t in obj_ref])
        tokens.append(token)
    return tokens


def parse_result_row(results_row, method):
    bad_chars = ["[", "]", "'", " "]
    detections = [s.translate({ord(x): '' for x in bad_chars})
                  for s in results_row[f"detections_{method}"].split(",")]
    if detections[0] == "":
        return None, None
    confidence = [float(s.translate({ord(x): '' for x in bad_chars}))
                  for s in results_row[f"confidence_{method}"].split(",")]
    return detections, confidence


def calc_score(object_tokens, results_row, method):
    score = 0
    detections, confidence = parse_result_row(results_row, method)
    if not detections:
        return 0

    # disregard low confidence objects
    reliable_detections = [{"token": det, "p": conf, "seen": 0}
                           for det, conf in zip(detections, confidence)
                           if conf > CONFIDENCE_TH]
    # rally the desired objects
    for token in object_tokens:
        matching_objects = [d for d in reliable_detections if d["token"] in token and d["seen"] == 0]
        if matching_objects:
            max_value = max(matching_objects, key=lambda x: x["p"])
            score += 1
            max_value["seen"] = 1

    # penalize the unwanted objects
    unwanted_objects = [d for d in reliable_detections if d["seen"] == 0]
    for _ in unwanted_objects:
        score -= PENALTY_LEVEL
    final_score = score / len(object_tokens)
    return max(final_score, 0)


def get_scores(input_obj, results_row):
    object_tokens = get_object_tokens(input_obj)
    methods = ["ba", "gp", "api"]
    for method in methods:
        method_score = calc_score(object_tokens, results_row, method)
        results_row[f"{method}_score"] = method_score
    return results_row


def get_input_object(row, input_dict):
    prompt = row["prompt"]
    matches = [value for value in input_dict.values() if value["prompt"] == prompt]
    if not matches:
        raise LookupError(f"Can't find prompt - {prompt}")
    return matches[0]


def load_input_json():
    with open("db_inputs.json", "r") as reader:
        input_json = json.load(reader)
    return input_json


if __name__ == '__main__':
    CONFIDENCE_TH = 0.3
    PENALTY_LEVEL = 0.2
    df = pd.read_excel(r"C:\Users\roima\Downloads\DLprojectRES.xlsx")
    input_db = load_input_json()
    add_scores(df, input_db)
    df.to_excel("results.xlsx")
