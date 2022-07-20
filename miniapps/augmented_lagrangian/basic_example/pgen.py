from os.path import exists
import json
from maestrowf.datastructures.core import ParameterGenerator

def get_custom_generator(env, **kwargs):

    # Unpack any pargs passed in
    config_filename = kwargs.get("CONFIG", env.find("CONFIG").value)
    record_filename = kwargs.get("RECORD", env.find("RECORD").value)
    iteration = kwargs.get("ENCORE_ITERATION", env.find("ENCORE_ITERATION").value)

    if iteration > 1 and exists(record_filename):
        with open(record_filename) as record_file:
            record = json.load(record_file)
            sample_size = record["Sample size"]
    else:
        with open(config_filename) as config_file:
            config = json.load(config_file)
            sample_size = config["design_update"]["initial_sample_size"]

    p_gen = ParameterGenerator()
    params = {
        "SAMPLE": {
            "values": [i for i in range(1, sample_size+1)],
            "label": "SAMPLE.%%"
        },
    }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen