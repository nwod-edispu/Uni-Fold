import os
import jax
import jax.random as jrand
from unifold.train.train_config import train_config
from unifold.model.config import model_config as get_model_config
from unifold.train.data_system import DataSystem
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def precompute_features(rng, out_dir):
    gc = train_config.global_config
    model_config = get_model_config(gc.model_name, is_training=True)
    data_system = DataSystem(model_config, train_config.data.train)
    data_system.precompute_features(rng, out_dir)


def main():
    random_seed = 2163783
    with jax.disable_jit():
        rng = jrand.PRNGKey(random_seed)
    out_dir = "/home/hanj/workplace/unifold_dataset/training_set/features_done/"
    precompute_features(rng, out_dir)


if __name__ == "__main__":
    main()
