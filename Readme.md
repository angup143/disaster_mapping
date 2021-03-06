# Disaster Mapping


Code for:  

[**CNN-Based Semantic Change Detection in Satellite Imagery**](https://link.springer.com/chapter/10.1007/978-3-030-30493-5_61), Ananya Gupta, Elisabeth Welburn, Simon Watson & Hujun Yin, ICANN 2019

[**Post Disaster Mapping With Semantic Change Detection in Satellite Imagery**](http://openaccess.thecvf.com/content_CVPRW_2019/html/WiCV/Gupta_Post_Disaster_Mapping_With_Semantic_Change_Detection_in_Satellite_Imagery_CVPRW_2019_paper.html), Ananya Gupta, Elisabeth Welburn, Simon Watson & Hujun Yin, CVPRW 2019

## Introduction

Our work focuses on identifying road networks and buildings in post-disaster scenarios using publicly available satellite imagery and neural networks for segmentation. We use a change detection framework to identify areas impacted by the disaster and use inspiration from graph theory to update road network data available from OpenStreetMap in the aftermath of a disaster.


## Dataset 

- Download data of interest from DigitalGlobe [OpenDataInitiative](https://www.digitalglobe.com/ecosystem/open-data)

- Download Labels from [OpenStreetMap](https://www.openstreetmap.org/export)

- Use QGis to extract any polylines marked as roads,trunk, link etc (essentially extract all road polylines) and export as geojson

- Use tools provided by SpaceNet [apls](https://github.com/CosmiQ/apls/blob/master/apls/create_spacenet_masks.py) to generate raster files from OSM vector data


## Segmentation

Segmentation training code based on [TernausNet](https://github.com/ternaus/robot-surgery-segmentation). Added models, updated training scripts:

> python  aerial_train.py

This script includes the following models and backends:
- 'UNet11': UNet (VGG11),
- 'UNet16': UNet (VGG16),
- 'UNet18': UNet (ResNet18)
- 'UNet34': UNet (ResNet34)
- 'UNet11Upsample' : UNet (VGG11 with linear upsampling),
- 'UNet16Upsample' : UNet (VGG16 with linear upsampling),
- 'UNet18Upsample' : UNet (ResNet18 with linear upsampling),
- 'UNet34Upsample' : UNet (ResNet34 with linear upsampling),
- 'LinkNet18': LinkNet (ResNet18),
- 'LinkNet34': LinkNet (ResNet34)
              

## Disaster Mapping

Generate road, building, combined segmentation masks using trained models
> python generate_masks.py 

 Convert segmentation masks to graph, generates metrics and output files

> extract_osm_diff.py


## Citation

Please consider citing the following if you find this work useful:

    @article{gupta2020deep,
    title={Deep Learning-based Aerial Image Segmentation with Open Data for Disaster Impact Assessment},
    author={Gupta, Ananya and Watson, Simon and Yin, Hujun},
    journal={arXiv preprint arXiv:2006.05575},
    year={2020}
    }

    @inproceedings{gupta2019cnn,
    title={CNN-Based Semantic Change Detection in Satellite Imagery},
    author={Gupta, Ananya and Welburn, Elisabeth and Watson, Simon and Yin, Hujun},
    booktitle={International Conference on Artificial Neural Networks},
    pages={669--684},
    year={2019},
    organization={Springer}
    }

    @inproceedings{gupta2019post,
    title={Post Disaster Mapping With Semantic Change Detection in Satellite Imagery},
    author={Gupta, Ananya and Welburn, Elisabeth and Watson, Simon and Yin, Hujun},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
    pages={0--0},
    year={2019}
    }

