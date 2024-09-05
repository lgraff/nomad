
import geopandas as gpd

def create_study_area(neighborhoods_inpath, neighborhoods_keep, study_area_outpath):
    # get the study area
    nhoods = gpd.read_file(neighborhoods_inpath)  # all neighborhoods
    #nhoods_keep = conf.config_data['Geography']['neighborhoods']  # neighborhoods to keep
    
    # dissolve the neighborhoods together 
    nhoods_union = nhoods[nhoods['hood'].isin(neighborhoods_keep)].dissolve() 
    
    #  buffer the study area by x miles
    x = 0.2 # conf.config_data['Geography']['buffer']  # miles
    nhoods_union = nhoods_union.to_crs(crs='epsg:32128').buffer(x*1609).to_crs('EPSG:4326')  # 1609 meters/mile

    # save to output file
    nhoods_union.to_file(study_area_outpath, driver='GeoJSON')


# call function
# create_study_area(os.path.join(os.getcwd(), 'Data','Input_Data', 'Neighborhoods', 'Neighborhoods_.shp'), 
#                   os.path.join(os.getcwd(), 'Data', 'Output_Data', 'study_area.csv'))
# study_area_gdf = pgh_nhoods_union.copy()
# bbox_study_area = study_area_gdf['geometry'].bounds.T.to_dict()[0]  # bounding box of neighborhood polygon layer   

