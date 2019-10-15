#Disaster Mapping

##Dataset 
Download data of interest from DigitalGlobe [OpenDataInitiative](https://www.digitalglobe.com/ecosystem/open-data)
Download Labels from [OpenStreetMap](https://www.openstreetmap.org/export)
Use QGis to extract any polylines marked as roads,trunk, link etc (essentially extract all road polylines) and export as geojson
Use code provided from SpaceNet [apls](https://github.com/CosmiQ/apls/blob/master/apls/create_spacenet_masks.py) to generate raster files from OSM vector data

##Segmentation
Initial segmentation training code based on [TernausNet](https://github.com/ternaus/robot-surgery-segmentation) 
'python  aerial_train.py' used for training binary/multiclass segmentation model

##Disaster Mapping
'python generate_masks.py' ## generates road, building, combined segmentation masks using trained models
'extract_osm_diff.py' ## converts segmentation masks to graph, generates metrics and output files


