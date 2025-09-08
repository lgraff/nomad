""" Script to process raw data into a form used by NOMAD to create the multimodal network model."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from nomad.data import network, demographics
from conf import config

def main():
    network.process_data(config)  # strictly for network data
    demographics.get_od_centroids(config)  # strictly for origin/destination data

if __name__ == "__main__":
    main()