# Graduation project

# Towards the automation of semantically enriching point cloud datasets
#### An interactive method incorporating machine learning for fast semantic classification of building indoor scenes

by Guido Porteners

MSc Construction Management & Engineering

Eindhoven University of Technology

![Byg72](http://duraark.eu/wp-content/uploads/2016/01/CITA_Byg72_August.jpg)

# Introduction
Point clouds are a valuable datatype for the AEC industry and construction management sector, as they can be used for accurate Scan-to-BIM processes, and allow construction managers to compare the as-built and as-designed situation of construction projects. This provides practical implications such as progress monitoring, analysis of structural deviations and quality control. However before being pragmatic, raw point cloud data needs to be enriched with semantic information, for which segmentation and classification are key activities. Manual processes require a significant experience and time investment, and automation is therefore desired. This is a contemporary studied problem in the field of construction management and several methods are proposed that incorporate contextual reasoning and a hierarchical workflow. However, these methods break down as buildings are vastly different and construction elements are often ambiguous. Therefore, researchers turned to methods incorporating machine learning for object recognition, specifically deep learning and neural networks, as they appear to closely resemble human reasoning and learning contextual and semantic relationships. Unfortunately, current state-of-the-art methods incorporating deep learning are not tailored to the AEC industry, and because of their complexity, are poorly understood and impractical for construction management practitioners to work with.
This research proposes a hybrid method for segmenting and classifying point cloud datasets for the AEC industry, which combines both machine learning and user interaction to exploit their advantages and to balance out their disadvantages. This is achieved by training a deep neural network as a base model, which outputs a valuable set of parameters which can be used for a region-growing algorithm to operate as a smart and quick selection tool for selecting specific element class categories.


# Prerequisites
- Python 3.5 (https://www.python.org/downloads/release/python-354/)
- PCL 1.8 or higher (http://pointclouds.org/)
- Blender v2.79 (https://www.blender.org/download/)

#### Python Packages:
- Numpy 1.13.1
- Matplotlib 2.0.2
- H5py 2.7.1
- Pandas 0.20.3
- Pyflann3 1.8.4.1
- Scikit-learn 0.19.0

#### Blender Packages
- Numpy 1.13.1
- PLY Import/Export

# Workflow
To be updated...
