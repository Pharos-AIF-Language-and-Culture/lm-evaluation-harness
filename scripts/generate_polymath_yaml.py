"""
FINAL COMPLETE VERSION – PolyMath Configuration
Generates ALL files required for the lm-evaluation-harness.
"""

from pathlib import Path

# =============================================================================
# SETTINGS
# =============================================================================

SCRIPT_DIR = Path("../lm_eval/tasks/pharos/polymath")

LANGUAGES = {
    "en": ("English", "Note: Please put the final answer in the $\\boxed{}$."),
    "zh": ("Chinese", "注意：请将最终答案放在 $\\boxed{}$ 中。"),
    "ar": ("Arabic", "ملاحظة: يُرجى وضع الإجابة في $\\boxed{}$."),
    "bn": ("Bengali", "বিঃদ্রঃ: অনুগ্রহ করে চূড়ান্ত উত্তরটি $\\boxed{}$ এ রাখুন।"),
    "de": ("German", "Hinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{}$."),
    "es": ("Spanish", "Nota: Por favor, coloque la respuesta final en el $\\boxed{}$."),
    "fr": ("French", "Remarque : Veuillez mettre la réponse finale dans le $\\boxed{}$."),
    "id": ("Indonesian", "Catatan: Silakan letakkan jawaban akhir di dalam $\\boxed{}$."),
    "it": ("Italian", "Nota: Per favore, metti la risposta finale nel $\\boxed{}$."),
    "ja": ("Japanese", "注意：最終的な答えを $\\boxed{}$ に入れてください。"),
    "ko": ("Korean", "참고: 최종 답안을 $\\boxed{}$ 안에 넣어 주세요."),
    "ms": ("Malay", "Nota: Sila letakkan jawapan akhir dalam $\\boxed{}$."),
    "pt": ("Portuguese", "Nota: Por favor, coloque a resposta final no $\\boxed{}$."),
    "ru": ("Russian", "Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed{}$."),
    "sw": ("Swahili", "Kumbuka: Tafadhali weka jibu la mwisho katika $\\boxed{}$."),
    "te": ("Telugu", "గమనిక: దయచేసి తుది జవాబును $\\boxed{}$ లో ఉంచండి."),
    "th": ("Thai", "หมายเหตุ: กรุณาใส่คำตอบสุดท้ายใน $\\boxed{}$."),
    "vi": ("Vietnamese", "Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\\boxed{}$."),
}

DIFFICULTIES = ["top", "high", "medium", "low"]

# =============================================================================
# CREATE ROOT DIRECTORY
# =============================================================================

SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Directory created: {SCRIPT_DIR}\n")

# =============================================================================
# 1. CREATE utils.py
# =============================================================================

print("=" * 70)
print("CREATING utils.py")
print("=" * 70)

utils_code = '''import re


def extract_boxed_answer(text):
    """
    Extracts the content inside a \\boxed{...} pattern from the model output.
    """
    if not text:
        return ""
    
    # Look for \\boxed{...}
    match = re.search(r'\\\\boxed\\{([^}]+)\\}', text)
    
    if match:
        return match.group(1).strip()
    
    return ""


def process_results(doc, results):
    """
    Extracts the model's boxed answer and compares it with the ground truth.
    """
    prediction = results[0] if results else ""
    target = str(doc.get("answer", "")).strip()
    pred_boxed = extract_boxed_answer(prediction)
    is_correct = pred_boxed == target
    
    return {
        "exact_match": float(is_correct)
    }
'''

utils_path = SCRIPT_DIR / "utils.py"
with open(utils_path, 'w', encoding='utf-8') as f:
    f.write(utils_code)

print(f"✓ Created {utils_path.name}")
print()

# =============================================================================
# 2. CREATE LANGUAGE TEMPLATES
# =============================================================================

print("=" * 70)
print("CREATING LANGUAGE TEMPLATES")
print("=" * 70)

for lang_code, (lang_name, instruction) in LANGUAGES.items():
    template = f'''dataset_path: Qwen/PolyMath
dataset_name: {lang_code}
output_type: generate_until

doc_to_text: "{{{{question}}}}\\n{instruction}"
doc_to_target: "{{{{answer}}}}"

process_results: !function utils.process_results

generation_kwargs:
  until:
    - "\\n\\n"
  max_gen_toks: 512
  do_sample: false
  temperature: 0.0

metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true

metadata:
  version: 1.0
  language: {lang_code}
'''
    
    template_filename = f"_default_template_{lang_code}_yaml"
    template_path = SCRIPT_DIR / template_filename
    
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✓ {template_filename:40s} ({lang_name})")

print()

# =============================================================================
# 3. CREATE DIFFICULTY LEVEL CONFIGS
# =============================================================================

print("=" * 70)
print("CREATING DIFFICULTY LEVEL CONFIGS")
print("=" * 70)

count = 0
for lang_code, (lang_name, _) in LANGUAGES.items():
    for difficulty in DIFFICULTIES:
        config = f'''include: _default_template_{lang_code}_yaml
task: polymath_{lang_code}_{difficulty}
test_split: {difficulty}

metadata:
  description: "PolyMath {lang_name} - {difficulty.capitalize()}"
  difficulty: {difficulty}
'''
        
        config_filename = f"polymath_{lang_code}_{difficulty}.yaml"
        config_path = SCRIPT_DIR / config_filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config)
        
        count += 1

print(f"✓ Created {count} difficulty configs")
print()

# =============================================================================
# 4. CREATE LANGUAGE GROUP FILES
# =============================================================================

print("=" * 70)
print("CREATING LANGUAGE GROUP FILES")
print("=" * 70)

for lang_code, (lang_name, _) in LANGUAGES.items():
    group = f"group: polymath_{lang_code}\ntask:\n"
    for difficulty in DIFFICULTIES:
        group += f"  - polymath_{lang_code}_{difficulty}\n"
    
    group_filename = f"polymath_{lang_code}.yaml"
    group_path = SCRIPT_DIR / group_filename
    
    with open(group_path, 'w', encoding='utf-8') as f:
        f.write(group)
    
    print(f"✓ {group_filename:30s} ({lang_name})")

print()

# =============================================================================
# 5. CREATE MAIN polymath.yaml
# =============================================================================

print("=" * 70)
print("CREATING MAIN polymath.yaml")
print("=" * 70)

main_group = "group: polymath\ntask:\n"
for lang_code in LANGUAGES.keys():
    for difficulty in DIFFICULTIES:
        main_group += f"  - polymath_{lang_code}_{difficulty}\n"

main_path = SCRIPT_DIR / "polymath.yaml"
with open(main_path, 'w', encoding='utf-8') as f:
    f.write(main_group)

print(f"✓ Created {main_path.name}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("✅ COMPLETED!")
print("=" * 70)
print(f"📦 Files created:")
print(f"   • 1 utils.py")
print(f"   • {len(LANGUAGES)} language templates (_default_template_*_yaml)")
print(f"   • {len(LANGUAGES) * len(DIFFICULTIES)} difficulty configs (polymath_*_*)")
print(f"   • {len(LANGUAGES)} language group files (polymath_*.yaml)")
print(f"   • 1 main group file (polymath.yaml)")
print()
total = 1 + len(LANGUAGES) + len(LANGUAGES) * len(DIFFICULTIES) + len(LANGUAGES) + 1
print(f"   Total: {total} files")
print()
print(f"📍 Location: {SCRIPT_DIR}")
print("=" * 70)
print()
print("🧪 Test command:")
print("   cd lm-evaluation-harness")
print("   lm_eval --model dummy --tasks polymath_en_low --limit 2")
print()
print("=" * 70)