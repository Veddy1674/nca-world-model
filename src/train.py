from ncwm import train_model, load_configuration

# load_configuration() requires this script to have a configuration path as first argument
train_model(load_configuration(), print_enabled=True)