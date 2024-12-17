# Exploring OpenStreetMap Data for Enriched Remote Sensing Image Captioning

## Overview

This project explores using Volunteered Geographic Information (VGI) from OpenStreetMap (OSM) to create more detailed captions for remote sensing images.

We built a custom dataset combining images, OSM geographic data, and captions. Our model merges visual information with OSM data using SoTA architectures and GPT-2 for generating text.

To handle varying amounts of OSM data, we tested methods to select the most relevant info and tried different ways to represent the data.

Results show the model can generate meaningful captions, but challenges remain due to noisy and biased VGI data, limited samples, and overfitting.

This work highlights the promise of VGI for improving image captions and points to future work needed to overcome current limitations.

---
