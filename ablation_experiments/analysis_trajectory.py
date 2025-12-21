import csv
from collections import defaultdict
import os
import math
'''Note that the eval code does not completely filter out determiner noun agreement categories, so we 
add an extra filter in this analysis script.
'''


SEEDS = [42, 53, 67]
EPOCHS = range(1,11)

# # paths for function word deletion
# PATHS = {
#     "baseline_natural": "natural_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
#     "ablation_natural": "no_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
# }


# paths for function word random
# PATHS = {
#     "baseline_natural": "natural_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
#     "ablation_natural": "random_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
# }


# paths for no function word training and test
# PATHS = {
#     "baseline_natural": "natural_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-1.csv",
#     "ablation_natural": "no_function_blimp_results/results_GPT2_no_function_{seed}_epoch-1.csv",
# }


# # paths for function word masking
# PATHS={
# "baseline_natural": "natural_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
# "ablation_natural": "blimp_ablation_epoch10_fw_mask/results_GPT2_natural_function_{seed}_epoch-10.csv",
#
# "baseline_five": "five_function_blimp_results/results_GPT2_five_function_{seed}_epoch-10.csv",
# "ablation_five": "blimp_ablation_epoch10_fw_mask/results_GPT2_five_function_{seed}_epoch-10.csv",
#
# "baseline_more": "more_function_blimp_results/results_GPT2_more_function_{seed}_epoch-10.csv",
# "ablation_more": "blimp_ablation_epoch10_fw_mask/results_GPT2_more_function_{seed}_epoch-10.csv",
#
# "baseline_random": "random_function_blimp_results/results_GPT2_random_function_{seed}_epoch-10.csv",
# "ablation_random": "blimp_ablation_epoch10_fw_mask/results_GPT2_random_function_{seed}_epoch-10.csv",
#
# "baseline_boundary": "within_boundary_blimp_results/results_GPT2_within_boundary_{seed}_epoch-10.csv",
# "ablation_boundary": "blimp_ablation_epoch10_fw_mask/results_GPT2_within_boundary_{seed}_epoch-10.csv",
#
# "baseline_bigram": "bigram_function_blimp_results/results_GPT2_bigram_function_{seed}_epoch-10.csv",
# "ablation_bigram": "blimp_ablation_epoch10_fw_mask/results_GPT2_bigram_function_{seed}_epoch-10.csv",
# }


#paths for function head masking
PATHS={
"natural_function_eval": "natural_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-{epoch}.csv",
"no_function_training": "no_function_blimp_results/results_GPT2_no_function_{seed}_epoch-{epoch}.csv",
"no_function_eval": "no_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-{epoch}.csv",
"mask_function_eval": "blimp_ablation_epoch{epoch}_fw_mask/results_GPT2_natural_function_{seed}_epoch-{epoch}.csv",
"random_function_eval": "random_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-{epoch}.csv"
}

# PATHS={
# "baseline_natural": "natural_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
# "ablation_natural": "random_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
# }


## paths for funciton word masking
# PATHS = {
#     "baseline_natural": "natural_function_blimp_results/results_GPT2_natural_function_{seed}_epoch-10.csv",
#     "ablation_natural": "blimp_ablation_epoch10_fw_mask/results_GPT2_natural_function_{seed}_epoch-10.csv",
#
#     "baseline_within_boundary": "within_boundary_blimp_results/results_GPT2_within_boundary_{seed}_epoch-10.csv",
#     "ablation_within_boundary": "blimp_ablation_epoch10_fw_mask/results_GPT2_within_boundary_{seed}_epoch-10.csv",
#
#     "baseline_more_function": "more_function_blimp_results/results_GPT2_more_function_{seed}_epoch-10.csv",
#     "ablation_more_function": "blimp_ablation_epoch10_fw_mask/results_GPT2_more_function_{seed}_epoch-10.csv",
#
#     "baseline_random_function": "random_function_blimp_results/results_GPT2_random_function_{seed}_epoch-10.csv",
#     "ablation_random_function": "blimp_ablation_epoch10_fw_mask/results_GPT2_random_function_{seed}_epoch-10.csv",
#
#     "baseline_bigram_function": "bigram_function_blimp_results/results_GPT2_bigram_function_{seed}_epoch-10.csv",
#     "ablation_bigram_function": "blimp_ablation_epoch10_fw_mask/results_GPT2_bigram_function_{seed}_epoch-10.csv",
#
#     "baseline_five_function": "five_function_blimp_results/results_GPT2_five_function_{seed}_epoch-10.csv",
#     "ablation_five_function": "blimp_ablation_epoch10_fw_mask/results_GPT2_five_function_{seed}_epoch-10.csv",
# }

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


# find the main category each subcategory belongs to
UID2CAT = {}
for cat, uids in PHENOMENON_GROUPS.items():
    for u in uids:
        UID2CAT[u] = cat


def read_epoch10(csv_path):
    out = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for task, score in reader:
            uid = task.replace("blimp_", "")
            out[uid] = float(score)
    return out


rows = []

for seed in SEEDS:
    for epoch in EPOCHS:
        for cond_name, path_tpl in PATHS.items():
            path = path_tpl.format(seed=seed, epoch=epoch)

            if not os.path.exists(path):
                print(f"[WARN] Missing: {path}")
                continue

            scores = read_epoch10(path)

            for uid, acc in scores.items():
                cat = UID2CAT.get(uid, "unknown")

                if cat == "determiner_noun_agreement":
                    continue
                if 'quantifier' in cat:
                    continue
                rows.append({
                    "seed": seed,
                    "condition": cond_name,
                    "category": cat,
                    "uid": uid,
                    "epoch": epoch,
                    "accuracy": acc,
                })


by_seed_cond = defaultdict(list)

for r in rows:
    key = (r["seed"], r["condition"], r['epoch'])
    by_seed_cond[key].append(r["accuracy"])

overall_rows = []
for (seed, cond, epoch), vals in by_seed_cond.items():
    overall_rows.append({
        "seed": seed,
        "condition": cond,
        "category": "overall",
        "uid": "overall",
        "epoch": epoch,
        "accuracy": sum(vals) / len(vals),
    })


rows_extended = rows + overall_rows

out_file = "blimp_function_word_deletion_overall.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "condition",
            "category",
            "uid",
            "epoch",
            "accuracy",
        ]
    )
    writer.writeheader()
    writer.writerows(rows_extended)

print(f"Wrote {len(rows_extended)} rows to {out_file}")


def mean(xs):
    return sum(xs) / len(xs)

def sample_sd(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

# (cond, cat, seed) -> [acc...]
by_ccs = defaultdict(list)

with open(out_file, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        seed = int(r["seed"])
        cond = r["condition"]
        cat = r["category"]
        acc = float(r["accuracy"])
        epoch = int(r['epoch'])
        by_ccs[(cond, cat, seed, epoch)].append(acc)

# (cond, cat) -> [seed_mean...]
by_cc = defaultdict(list)
for (cond, cat, seed, epoch), xs in by_ccs.items():
    by_cc[(cond, cat, epoch)].append(mean(xs))

rows = []
for (cond, cat,epoch), seed_means in sorted(by_cc.items()):
    rows.append({
        "condition": cond,
        "category": cat,
        "epoch": epoch,
        "n_seeds": len(seed_means),
        "mean_acc": mean(seed_means),
        "sd_acc": sample_sd(seed_means),
    })

print(rows)
with open('learning_trajectory.csv', "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "category","epoch", "n_seeds", "mean_acc", "sd_acc"])
    w.writeheader()
    w.writerows(rows)