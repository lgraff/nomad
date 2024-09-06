'''This file processes two types of data: 
      1) Raw network data, which is converted into a form that is used to build the unimodal graphs.
            Raw network data is stored in the folder nomad/data/network/raw
            Processed network data is stored in the folder nomad/data/network/processed
      2) Raw demographic data, which is used to build the origins (population centers) and destinations (job centers) of the supernetwork.
            Raw demographic data is stored in the folder nomad/data/demographic/raw
            Processed demographic data is stored in the folder nomad/data/demographic/processed
'''

# Execute this file only once.

from nomad.data import network, demographics

def main():
      network.process_data()
      demographics.process_data()

if __name__ == "__main__":
    main()