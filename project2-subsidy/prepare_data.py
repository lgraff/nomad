
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from nomad.data import network, demographics
from conf import config

def main():
    network.process_data(config)
    demographics.process_data(config)

if __name__ == "__main__":
    main()