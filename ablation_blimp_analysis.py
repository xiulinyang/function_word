#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from collections import defaultdict
import os


EPOCH = 10



baseline_natural_path = "natural_function_blimp_results/results_GPT2_natural_function_42_epoch-10.csv"
baseline_within_boundary = "within_boundary_blimp_results/results_GPT2_within_boundary_42_epoch-10.csv"
ablation_within_boundary = "blimp_ablation_epoch10_fw_mask/results_GPT2_within_boundary_42_epoch-10.csv"
ablation_natural_function = "blimp_ablation_epoch10_fw_mask/results_GPT2_natural_function_42_epoch-10.csv"
baseline_more_function = "more_function_blimp_results/results_GPT2_more_function_42_epoch-10.csv"
ablation_more_function = "blimp_ablation_epoch10_fw_mask/results_GPT2_more_function_42_epoch-10.csv"
baseline_random_function = "random_function_blimp_results/results_GPT2_random_function_42_epoch-10.csv"
baseline_bigram_function = "bigram_function_blimp_results/results_GPT2_bigram_function_42_epoch-10.csv"
ablation_random_function = "blimp_ablation_epoch10_fw_mask/results_GPT2_random_function_42_epoch-10.csv"
ablation_bigram_function = "blimp_ablation_epoch10_fw_mask/results_GPT2_bigram_function_42_epoch-10.csv"
baseline_five_function = "five_function_blimp_results/results_GPT2_five_function_42_epoch-10.csv"
ablation_five_function = "blimp_ablation_epoch10_fw_mask/results_GPT2_five_function_42_epoch-10.csv"

PHENOMENON_GROUPS = {
    "anaphor_agreement": [
        "anaphor_gender_agreement",
        "anaphor_number_agreement",
    ],
    "argument_structure": [
        "animate_subject_passive",
        "animate_subject_trans",
        "causative",
        "drop_argument",
        "inchoative",
        "intransitive",
        "passive_1",
        "passive_2",
        "transitive",
    ],
    "binding": [
        "principle_A_c_command",
        "principle_A_case_1",
        "principle_A_case_2",
        "principle_A_domain_1",
        "principle_A_domain_2",
        "principle_A_domain_3",
        "principle_A_reconstruction",
    ],
    "control_raising": [
        "existential_there_object_raising",
        "existential_there_subject_raising",
        "expletive_it_object_raising",
        "tough_vs_raising_1",
        "tough_vs_raising_2",
    ],
    "determiner_noun_agreement": [
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adjective_1",
        "determiner_noun_agreement_with_adjective_2",
        "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2",
    ],
    "ellipsis": [
        "ellipsis_n_bar_1",
        "ellipsis_n_bar_2",
    ],
    "filler_gap": [
        "wh_questions_object_gap",
        "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance",
        "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ],
    "irregular_forms": [
        "irregular_past_participle_adjectives",
        "irregular_past_participle_verbs",
    ],
    "island_effects": [
        "adjunct_island",
        "complex_NP_island",
        "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "left_branch_island_echo_question",
        "left_branch_island_simple_question",
        "sentential_subject_island",
        "wh_island",
    ],
    "npi_licensing": [
        "matrix_question_npi_licensor_present",
        "npi_present_1",
        "npi_present_2",
        "only_npi_licensor_present",
        "only_npi_scope",
        "sentential_negation_npi_licensor_present",
        "sentential_negation_npi_scope",
    ],
    "quantifiers": [
        "existential_there_quantifiers_1",
        "existential_there_quantifiers_2",
        "superlative_quantifiers_1",
        "superlative_quantifiers_2",
    ],
    "subject_verb_agreement": [
        "distractor_agreement_relational_noun",
        "distractor_agreement_relative_clause",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
    ],
}

# ===== UID → category 映射 =====
UID2CAT = {}
for cat, uids in PHENOMENON_GROUPS.items():
    for u in uids:
        UID2CAT[u] = cat

# ===== 读单个 CSV =====
def read_epoch10(csv_path):
    out = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for task, score in reader:
            uid = task.replace("blimp_", "")
            out[uid] = float(score)
    return out

# ===== 三个条件读取 =====
all_conditions = {
    "baseline_natural": baseline_natural_path,
    "ablation_natural": ablation_natural_function,
    "baseline_within_boundary": baseline_within_boundary,
    "ablation_within_boundary": ablation_within_boundary,
    'baseline_more_function': baseline_more_function,
    'ablation_more_function': ablation_more_function,
    'baseline_random_function':baseline_random_function,
    'ablation_random_function':ablation_random_function,
    'baseline_bigram_function': baseline_bigram_function,
    'ablation_bigram_function':ablation_bigram_function,
'baseline_five_function': baseline_five_function,
    'ablation_five_function':ablation_five_function
}

rows = []

for cond_name, path in all_conditions.items():
    scores = read_epoch10(path)

    for uid, acc in scores.items():
        cat = UID2CAT.get(uid, "unknown")

        if cat == "determiner_noun_agreement":
            continue

        rows.append({
            "condition": cond_name,
            "category": cat,
            "uid": uid,
            "epoch": EPOCH,
            "accuracy": acc,
        })

# ===== 加 overall（每个 condition 单独算）=====
by_condition = defaultdict(list)
for r in rows:
    by_condition[r["condition"]].append(float(r["accuracy"]))

overall_rows = []
for cond, vals in by_condition.items():
    overall_rows.append({
        "condition": cond,
        "category": "overall",
        "uid": "overall",
        "epoch": EPOCH,
        "accuracy": sum(vals) / len(vals),
    })

rows_extended = rows + overall_rows

out_file = "blimp_baseline_vs_ablation_epoch10_42.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["condition", "category", "uid", "epoch", "accuracy"]
    )
    writer.writeheader()
    writer.writerows(rows_extended)

print(f"Wrote {len(rows_extended)} rows to {out_file}")