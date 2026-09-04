from ncwm import hyperparam_sweep, load_configuration

# load_configuration() requires this script to have a configuration path as first argument
hyperparam_sweep(load_configuration(), trials=30)