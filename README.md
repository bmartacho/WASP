# WASP
**Waterfall Atrous Spatial Pooling Architecture for Efficient Semantic Segmentation**

<p align="center">
  <img src="https://www.mdpi.com/sensors/sensors-19-05361/article_deploy/html/images/sensors-19-05361-g004.png" title="WASPnet architecture for Semantic Segmentation">
  Figure 1: WASPnet architecture for Semantic Segmentation.
</p><br />

<p align="justify">
We present a new efficient architecture for semantic segmentation, based on a “Waterfall” Atrous Spatial Pooling architecture, that achieves a considerable accuracy increase while decreasing the number of network parameters and memory footprint. The proposed Waterfall architecture leverages the efficiency of progressive filtering in the cascade architecture while maintaining multiscale fields-of-view comparable to spatial pyramid configurations. Additionally, our method does not rely on a postprocessing stage with Conditional Random Fields, which further reduces complexity and required training time. We demonstrate that the Waterfall approach with a ResNet backbone is a robust and efficient architecture for semantic segmentation obtaining state-of-the-art results with significant reduction in the number of parameters for the Pascal VOC dataset and the Cityscapes dataset.<br />
  
We propose the “Waterfall Atrous Spatial Pyramid” module, shown in Figure 2. WASP is a novel architecture with Atrous Convolutions that is able to leverage both the larger Field-of-View of the Atrous Spatial Pyramid Pooling configuration and the reduced size of the cascade approach.<br />

<p align="center">
  <img src="https://www.mdpi.com/sensors/sensors-19-05361/article_deploy/html/images/sensors-19-05361-g006.png" width=500 title="WASP module"><br />
  Figure 2: WASP Module.
</p><br />

Examples of the WASPnet architecture for segmentation are shown in Figures 3 and 4 for the PASCAL VOC and Cityscapes datasets, respectively.<br />

<p align="center">
  <img src="https://www.mdpi.com/sensors/sensors-19-05361/article_deploy/html/images/sensors-19-05361-g009.png" width=500 title="WASP module"><br />
  Figure 3: Segmentation samples for the Pascal VOC dataset.
  <br /><br />
  <img src="https://www.mdpi.com/sensors/sensors-19-05361/article_deploy/html/images/sensors-19-05361-g010.png" width=500 title="WASP module"><br />
  Figure 4: Segmentation samples for the CItyscapes dataset.
</p><br /><br />
  
Link to the published article at Sensors: https://doi.org/10.3390/s19245361
</p><br />

**Datasets:**
<p align="justify">
Both datasets used in this paper and required for training, validation, and testing can be downloaded directly from the dataset websites below:<br />
  Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/<br />
  Cityscapes: https://www.cityscapes-dataset.com<br />
</p><br />

**Contact:**

<p align="justify">
Bruno Artacho:<br />
  E-mail: bmartacho@mail.rit.edu<br />
  Website: https://people.rit.edu/bm3768<br />
  
Andreas Savakis:<br />
  E-mail: andreas.savakis@rit.edu<br />
  Website: https://www.rit.edu/directory/axseec-andreas-savakis<br /><br />
</p>

**Citation:**

<p align="justify">
Artacho, B.; Savakis, A. Waterfall Atrous Spatial Pooling Architecture for Efficient Semantic Segmentation. Sensors 19, no. 24: 5361, 2019. <br />

Latex:<br />
@article{Artacho,<br />
    author =       "Bruno Artacho and Andreas Savakis",<br />
    title =        "Waterfall Atrous Spatial Pooling Architecture for Efficient Semantic Segmentation",<br />
    journal =      "Sensors",<br />
    volume =       "19",<br />
    number =       "24",<br />
    pages =        "5361",<br />
    year =         "2019",<br />
    DOI =          "https://doi.org/10.3390/s19245361"<br />
}<br />
</p>
