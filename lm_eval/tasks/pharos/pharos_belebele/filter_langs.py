from lang_mapping import ALLOWED_LANGS_ISO3_SCRIPT
import yaml
import os

delete = False

if delete:
    for file in os.listdir('.'):
        if not file.startswith('belebele') or not file.endswith('.yaml'): continue
        task = file.split('.yaml')[0]
        if (task.split('_')[1]+'_'+task.split('_')[2]) not in ALLOWED_LANGS_ISO3_SCRIPT['apertus']:
            print(f'Removing {os.path.join('.', file)}...')
            os.remove(os.path.join('.', file))
else:
    tasks = {}
    with open('_belebele.yaml', 'r') as f:
        try:
            tasks = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            exit(1)

    tasks['task'] = [task for task in tasks['task'] if (task.split('_')[1]+'_'+task.split('_')[2]) in ALLOWED_LANGS_ISO3_SCRIPT['apertus']]
    with open('./_belebele_allowed.yaml', 'w') as f:
        yaml.dump(tasks, f, default_flow_style=False, sort_keys=False)