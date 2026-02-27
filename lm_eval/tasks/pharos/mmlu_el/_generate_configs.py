# noqa
"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

TO_GREEK = {
    'abstract_algebra': 'αφηρημένη_άλγεβρα',
    'anatomy': 'ανατομία',
    'astronomy': 'αστρονομία',
    'business_ethics': 'επιχειρηματική_ηθική',
    'clinical_knowledge': 'κλινική_γνώση',
    'college_biology': 'πανεπιστημιακή_βιολογία',
    'college_chemistry': 'πανεπιστημιακή_χημεία',
    'college_computer_science': 'πανεπιστημιακή_επιστήμη_υπολογιστών',
    'college_mathematics': 'πανεπιστημιακά_μαθηματικά',
    'college_medicine': 'ιατρική_εκπαίδευση',
    'college_physics': 'πανεπιστημιακή_φυσική',
    'computer_security': 'ασφάλεια_υπολογιστών',
    'conceptual_physics': 'έννοιες_φυσικής',
    'econometrics': 'οικονομετρία',
    'electrical_engineering': 'ηλεκτρολογική_μηχανική',
    'elementary_mathematics': 'βασικά_μαθηματικά',
    'formal_logic': 'τυπική_λογική',
    'global_facts': 'παγκόσμια_δεδομένα',
    'high_school_biology': 'σχολική_βιολογία',
    'high_school_chemistry': 'σχολική_χημεία',
    'high_school_computer_science': 'σχολική_επιστήμη_υπολογιστών',
    'high_school_european_history': 'σχολική_ευρωπαϊκή_ιστορία',
    'high_school_geography': 'σχολική_γεωγραφία',
    'high_school_government_and_politics': 'κοινωνική_και_πολιτική_αγωγή',
    'high_school_macroeconomics': 'μακροοικονομικά_λυκείου',
    'high_school_mathematics': 'μαθηματικά_λυκείου',
    'high_school_microeconomics': 'μικροοικονομία_λυκείου',
    'high_school_physics': 'σχολική_φυσική',
    'high_school_psychology': 'ψυχολογία_λυκείου',
    'high_school_statistics': 'στατιστική_λυκείου',
    'high_school_us_history': 'σχολική_ιστορία',
    'high_school_world_history': 'σχολική_παγκόσμια_ιστορία',
    'human_aging': 'ανθρώπινη_γήρανση',
    'human_sexuality': 'ανθρώπινη_σεξουαλικότητα',
    'international_law': 'διεθνές_δίκαιο',
    'jurisprudence': 'νομολογία',
    'logical_fallacies': 'λογικές_πλάνες',
    'machine_learning': 'μηχανική_μάθηση',
    'management': 'διοίκηση_επιχειρήσεων',
    'marketing': 'μάρκετινγκ',
    'medical_genetics': 'ιατρική_γενετική',
    'miscellaneous': 'διάφορα',
    'moral_disputes': 'ηθικά_διλήμματα',
    'moral_scenarios': 'ηθικά_σενάρια',
    'nutrition': 'διατροφή',
    'philosophy': 'φιλοσοφία',
    'prehistory': 'προϊστορία',
    'professional_accounting': 'λογιστική',
    'professional_law': 'δίκαιο',
    'professional_medicine': 'ιατρική',
    'professional_psychology': 'ψυχολογία',
    'public_relations': 'δημόσιες_σχέσεις',
    'security_studies': 'διεθνής_ασφάλεια',
    'sociology': 'κοινωνιολογία',
    'us_foreign_policy': 'εξωτερική_πολιτική_των_ηπα',
    'virology': 'ιολογία',
    'world_religions': 'θρησκείες_του_κόσμου'
 }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="mmlu_el")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        if args.cot_prompt_path is not None:
            description = cot_file[subject]
        else:
            description = f"Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής (που παρουσιάζονται μαζί με τις απαντήσεις τους) έχουν να κάνουν με {' '.join(TO_GREEK[subject].split('_'))}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"mmlu_el_{category}_{args.task_prefix}"
            if args.task_prefix != ""
            else f"mmlu_el_{category}_tasks",
            "task": f"mmlu_el_{subject}_{args.task_prefix}"
            if args.task_prefix != ""
            else f"mmlu_el_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    if args.task_prefix != "":
        mmlu_subcategories = [
            f"mmlu_el_{args.task_prefix}_{category}" for category in ALL_CATEGORIES
        ]
    else:
        mmlu_subcategories = [f"mmlu_el_{category}" for category in ALL_CATEGORIES]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": f"mmlu_el_{args.task_prefix}"
                if args.task_prefix != ""
                else "mmlu_el",
                "task": mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
