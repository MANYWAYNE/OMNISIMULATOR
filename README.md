# OMNISIMULATOR
[<img width="1280" height="720" alt="COVER_04" src="https://github.com/user-attachments/assets/3873069f-64a2-4c7d-9cb3-0400681bf55c" />](https://t.me/omniflexinfo)
Custom node package for ComfyUI 
## OMNISIMULATOR 
### is a specialized suite of tools for ComfyUI designed for advanced image processing, metadata management, and sophisticated stylization.
### Functional Modules Overview1. 
## Transformation and Correction Module (Detection Bypass & Texture Normalization) 
This module provides tools for fine-tuning the statistical characteristics of an image: 
* Spectral Analysis: Tools for manipulating the Fourier frequency spectrum.  
* Texture Normalization: Adjusting parameters for GLCM (Gray-Level Co-occurrence Matrix) and LBP (Local Binary Patterns) to align image statistical characteristics with specified benchmarks.  
* Optical Simulation: Applying mathematical models to simulate film grain, LUT filters, and optical distortions.
---
## Metadata Management (Metadata Synthesis) 
### This module provides capabilities for advanced EXIF data manipulation:
* Metadata Synthesis: The OmniSimulator_SaveWithAuthenticMetadata and SynthesizeAuthenticMetadata nodes allow for the programmatic setting of camera parameters (model, software version, exposure). 
* Geolocation: An integrated database for simulating GPS coordinates corresponding to real-world metropolitan areas.  
---
## Ethical Use and DisclaimerWarning: 
### Warning: This software contains tools that can be used to alter digital fingerprints and manipulate metadata. 
* The tools described in the "Transformation Module" section are intended strictly for experimental purposes, artistic stylization, and image processing research.  
* The developer is not responsible for the use of these tools to mislead third parties regarding the authenticity or origin of media content.  
* The use of metadata synthesis functions must comply with applicable legal norms and standards of digital ethics.  
---
## Dependency installation.

* After installation "CMD"

`git clone https://github.com/MANYWAYNE/OMNISIMULATOR.git`

* It is necessary to install the dependencies "CMD"

`cd ComfyUI/custom_nodes/OMNISIMULATOR 
pip install -r requirements.txt`

* When installing dependencies, use the full path. Below is an example of the path where the node is located in my build

`D:\OMNIFLEX\App\python_embeded\python.exe -m pip install -r requirements.txt`

---

[<img width="3440" height="924" alt="START" src="https://github.com/user-attachments/assets/79a549a3-788d-4a73-9f60-854ffd5ef570" />](https://t.me/omniflexinfo)