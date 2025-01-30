import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from pathlib import Path

import pandas as pd

from nomad.data import network, demographics
from conf import config
from od_demand import get_od_demand
from make_candidates import make_bikeshare_candidates, make_microtransit_candidates

def main():
    network.process_data(config)  
    get_od_demand()
    make_bikeshare_candidates(config['optimization']['candidates']['bikeshare_csv'])
    make_microtransit_candidates(config['optimization']['candidates']['microtransit_zones'])

if __name__ == "__main__":
    main()