# Dataset Description

The Multiple Choice QA Greek ASEP dataset is a set of 1200 multiple choice questions in Greek. The questions were extracted and converted from questions available at the website of the Greek Supreme Council for Civil Personnel Selection (Ανώτατο Συμβούλιο Επιλογής Προσωπικού, ΑΣΕΠ-ASEP) (1Γ/2025). The dataset includes questions in the following domains:

- Συνταγματικό Δίκαιο (Constitutional Law): 187 questions
- Διοικητικό Δίκαιο (Administrative Law): 242 questions
- Ευρωπαϊκοί Θεσμοί και Δίκαιο (European Institutional Law): 165 questions
- Οικονομικές Επιστήμες (Economics): 203 questions
- Δημόσια Διοίκηση και Διαχείριση Ανθρώπινου Δυναμικού (Public Administration and Human Resource Management): 123 questions
- Πληροφορική και Ψηφιακή Διακυβέρνηση (Informatics and Digital Governance): 157 questions
- Σύγχρονη Ιστορία της Ελλάδος (1875-σήμερα) (History of Modern Greece (1875-today)): 123 questions

# Dataset Info

- language: el
- license: cc-by-4.0
- multilinguality: monolingual
- size_categories: 1K<n<10K
- task_categories:
  - multiple-choice
- pretty_name: Multiple Choice QA Greek ASEP
- dataset_info:
  - splits:
    - name: default
      num_examples: 1200
- Features (Columns):
  - id: Value
  - question: Value
  - choices: List
  - answer: Value
  - answer_text: Value
  - subject: ClassLabel