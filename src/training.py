import os

from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model, save_model
import argparse

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_datasize"]
    (X_train_full, y_train_full),(X_valid, y_valid),(X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_functions"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["no_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train_full, y_train_full, epochs=EPOCHS,
                    validation_data=VALIDATION_SET)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir,model_dir)
    os.mkdir(model_dir_path,exist_ok =True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c",default="config.yml")

    parsed_args = args.parse_args()

    training(config_path = parsed_args.config)