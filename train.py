from clearml import Task, Dataset

Task.add_requirements("./requirements.txt")
task = Task.init(
    project_name="Project-08dbb8f9-3a62-44e9-8cd3-c75ad5517b0e",
    task_name="test_02",
    reuse_last_task_id=False,
)

task.execute_remotely(queue_name="default", exit_process=True)
if not task.running_locally():
    # PREPARE DATA
    dataset = Dataset.get(dataset_id='2fc999e15e7e4001ae6247b81a99db43')
    dataset_path = dataset.get_local_copy('data')

    import os
    import yaml

    with open(os.path.join(dataset_path, 'coco.yaml'), 'r') as f:
        content = f.read()
        content = content.replace("\n\t", "\n  ")  # to fix a yaml parsing bug
        names = yaml.load(content, Loader=yaml.FullLoader)['names']

    dataset_yaml = {
        "nc": 12,
        "path": dataset_path,
        "train": "images",
        "val": "images",
        "names": names
    }

    with open(os.path.join(dataset_path, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f)

    # Change the dataset path of the config file
    config_file = "config_seg.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['data'] = os.path.join(dataset_path, 'dataset.yaml')
        model_variant = cfg['model']

    with open(config_file, "w") as f:
        yaml.dump(cfg, f)

    task.connect(cfg)  # Allow ClearML to track the configurations

    from ultralytics.models.yolo.model import YOLO
    # PERFORM TRAINING
    model = YOLO(model=model_variant)
    results = model.train(cfg=config_file)
