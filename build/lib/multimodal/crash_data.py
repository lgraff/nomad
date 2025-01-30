# libraries
import ckanapi
import pandas as pd
import gc

def get_resource_data(site,resource_id,count=50):

    """
    Retrieve data records from a CKAN resource using the datastore_search API endpoint.

    Parameters:
        site (str): The URL of the CKAN site.
        resource_id (str): The ID of the CKAN resource from which to fetch data.
        count (int, optional): The number of records to retrieve. Defaults to 50.

    Returns:
        list: A list of data records retrieved from the CKAN resource.
    """

    # Use the datastore_search API endpoint to get <count> records from
    # a CKAN resource.
    ckan = ckanapi.RemoteCKAN(site)
    response = ckan.action.datastore_search(id=resource_id, limit=count)
    data = response['records']
    return data

def download_crash_data(site, resource_ids, output_path):
    """
    Download crash data from CKAN resources for the last two years and save as a CSV file.

    This function fetches crash data from specified CKAN resources, concatenates the data from two
    years, selects specific columns, and saves the resulting DataFrame as a CSV file.

    Parameters:
        site (str): The URL of the CKAN site.
        resource_ids (list): A list of two CKAN resource IDs for crash data from different years.
        output_path (str): The file path to save the concatenated crash data as a CSV file.
    """
    # two years of crash data (function can [should] be generalized to include more years) 
    crash_data_0 = get_resource_data(site,resource_id=resource_ids[0],count=999999999) 
    crash_data_1 = get_resource_data(site,resource_id=resource_ids[1],count=999999999) 

    # Convert to pandas df and concatenate
    df_crash_0 = pd.DataFrame(crash_data_0)
    df_crash_1 = pd.DataFrame(crash_data_1)
    df_crash = pd.concat([df_crash_1, df_crash_0], ignore_index=True)
    del crash_data_0  # deleting original data b/c large
    del crash_data_1
    del df_crash_0  # deleting year by year data b/c large
    del df_crash_1
    gc.collect()
    cols_keep = ['DEC_LAT', 'DEC_LONG', 'BICYCLE', 'BICYCLE_COUNT', 'PEDESTRIAN', 'PED_COUNT', 
                'SPEED_LIMIT', 'VEHICLE_COUNT', 'TOT_INJ_COUNT']
    df_crash = df_crash[cols_keep]

    # save as .csv file
    df_crash.to_csv(output_path)

 #   df_crash_2020.to_csv(os.path.join(cwd,'Input_Data','df_crash_2020.csv'))
 #   df_crash_2019.to_csv(os.path.join(cwd,'Input_Data','df_crash_2019.csv'))

# site = "https://data.wprdc.org"
# resource_ids = ["514ae074-f42e-4bfb-8869-8d8c461dd824","cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"]
# download_crash_data(site, resource_ids)