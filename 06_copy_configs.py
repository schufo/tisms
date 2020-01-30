import json
import os
from shutil import copyfile

def create_tag_named_folder(original_path, new_path, id):

    original_experiment_dir = os.path.join(original_path, '{}'.format(id))

    config_dict = json.load(open(os.path.join(original_experiment_dir, 'config.json'.format(id))))

    tag = config_dict['tag']

    new_experiment_directory = os.path.join(new_path, tag)

    if not os.path.exists(new_experiment_directory):
        os.mkdir(new_experiment_directory)

    copyfile(os.path.join(original_experiment_dir, 'config.json'),
             os.path.join(new_experiment_directory, 'config.json'))
    copyfile(os.path.join(original_experiment_dir, 'cout.txt'),
             os.path.join(new_experiment_directory, 'cout.txt'))
    copyfile(os.path.join(original_experiment_dir, 'metrics.json'),
             os.path.join(new_experiment_directory, 'metrics.json'))
    copyfile(os.path.join(original_experiment_dir, 'run.json'),
             os.path.join(new_experiment_directory, 'run.json'))


if __name__ == '__main__':

    original_path = 'sacred_experiment_logs'
    new_path = 'configs'

    exp = sorted(os.listdir(original_path))
    exp.remove('_sources')
    exp.remove('.gitignore')

    for exp_id in exp:

        create_tag_named_folder(original_path, new_path, int(exp_id))


