"""
Script to generate YAML configuration files for Multilingual AIME (mt-aime_polymath_format)
This creates task configs for 55 languages with 30 AIME 2024 problems each.
"""

from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the absolute path to the lm-evaluation-harness directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "lm_eval" / "tasks" / "pharos" / "mt-aime"

# Dataset information
DATASET_PATH = "yongzx/mt_aime_polymath_format"
SPLIT_NAME = "top"  # The dataset has only one split called "top"

# Language codes and names (55 languages - actual configs from the dataset)
# Each language is a separate dataset configuration/subset
LANGUAGES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}

# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

def create_directory():
    """Create the output directory structure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created directory: {OUTPUT_DIR}\n")


# =============================================================================
# CREATE LANGUAGE-SPECIFIC YAML FILES
# =============================================================================

def create_language_yaml(lang_code, lang_name):
    """
    Create a YAML configuration file for a specific language.
    
    Args:
        lang_code: Language code (e.g., 'en', 'zh', 'ar')
        lang_name: Full language name (e.g., 'English', 'Chinese')
    """
    yaml_content = f'''# MT AIME - {lang_name}
# Multilingual AIME 2024 problems (30 questions)

task: mt-aime_{lang_code.replace("-", "_")}
dataset_path: {DATASET_PATH}
dataset_name: {lang_code}
output_type: generate_until
test_split: {SPLIT_NAME}
fewshot_split: null
num_fewshot: 0

doc_to_text: "Question: {{{{question}}}}\\nAnswer:"
doc_to_target: "{{{{answer}}}}"

process_results: !function utils.process_results

generation_kwargs:
  until:
    - "\\n\\n"
    - "</s>"
    - "<|im_end|>"
    - "<|eot_id|>"
  do_sample: false
  temperature: 0.0
  max_gen_toks: 256

metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true

metadata:
  version: 1.0
  language: {lang_code}
  language_name: {lang_name}
  description: "Multilingual AIME 2024 problems in {lang_name} (30 problems)"
  num_problems: 30
'''
    
    # Replace dashes with underscores for task names (e.g., zh-cn -> zh_cn)
    task_suffix = lang_code.replace("-", "_")
    yaml_path = OUTPUT_DIR / f"mt-aime_{task_suffix}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    return yaml_path.name


# =============================================================================
# CREATE MAIN GROUP YAML
# =============================================================================

def create_main_group_yaml():
    """Create the main mt-aime.yaml file that groups all languages."""
    yaml_content = '''# Multilingual AIME (MT AIME)
# Dataset: yongzx/mt-aime_polymath_format

group: mt-aime
task:
'''
    
    # Add all language tasks (replace dashes with underscores)
    for lang_code in sorted(LANGUAGES.keys()):
        task_name = lang_code.replace("-", "_")
        yaml_content += f"  - mt-aime_{task_name}\n"
    
    yaml_path = OUTPUT_DIR / "mt-aime.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✓ Created main group file: {yaml_path.name}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate all configuration files."""
    print("=" * 70)
    print("GENERATING MT AIME CONFIGURATION FILES")
    print("=" * 70)
    print()
    
    # Step 1: Create directory
    create_directory()
    
    # Step 2: Create language-specific YAML files
    print("=" * 70)
    print(f"CREATING {len(LANGUAGES)} LANGUAGE-SPECIFIC YAML FILES")
    print("=" * 70)
    
    created_files = []
    for lang_code, lang_name in sorted(LANGUAGES.items()):
        filename = create_language_yaml(lang_code, lang_name)
        created_files.append(filename)
    
    print(f"✓ Created {len(created_files)} language YAML files")
    print()
    
    # Step 4: Create main group YAML
    print("=" * 70)
    print("CREATING MAIN GROUP YAML")
    print("=" * 70)
    create_main_group_yaml()
    print()
    
    # Summary
    print("=" * 70)
    print("✅ GENERATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"📊 Summary:")
    print(f"   • Output directory: {OUTPUT_DIR}")
    print(f"   • Total languages: {len(LANGUAGES)}")
    print(f"   • Files created:")
    print(f"     - {len(LANGUAGES)} language-specific YAML files (mt-aime_*.yaml)")
    print(f"     - 1 main group file (mt-aime.yaml)")
    print(f"     - Total: {len(LANGUAGES) + 1} files")
    print()
    print("   ⚠️  Note: Make sure utils.py already exists in the output directory")
    print()
    print("🚀 Usage examples:")
    print("   # Run English version:")
    print("   lm_eval --model hf --model_args pretrained=model-name --tasks mt-aime_en --limit 5")
    print()
    print("   # Run Chinese Simplified:")
    print("   lm_eval --model hf --model_args pretrained=model-name --tasks mt-aime_zh_cn --limit 5")
    print()
    print("   # Run Arabic:")
    print("   lm_eval --model hf --model_args pretrained=model-name --tasks mt-aime_ar --limit 5")
    print()
    print("   # Run all 55 languages:")
    print("   lm_eval --model hf --model_args pretrained=model-name --tasks mt-aime")
    print()


if __name__ == "__main__":
    main()
