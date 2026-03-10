# Weed Detection System for Agricultural Environments

A computer vision system designed to automatically detect weeds in agricultural fields.
The system processes camera images and identifies weed locations so that they can be removed mechanically or by other automated systems.

This project is designed to run on an **edge device integrated with the MARS Rover V2 platform**, enabling real-time weed detection directly in the field.


# Overview

Weeds compete with crops for essential resources such as space, water, and nutrients. If left uncontrolled, they significantly reduce crop yield and soil efficiency.

Traditional weed control methods include:

* **Chemical control (herbicides)**
* **Biological control using insects or bacteria**

While these approaches can be effective, they often have drawbacks such as environmental impact, resistance development, or limited effectiveness.

This project explores an alternative solution: **detecting weeds individually using computer vision** so they can be removed precisely without affecting crops.


# Proposed Solution

The system detects weeds in camera images using object recognition algorithms.

The detection pipeline works as follows:

1. A camera mounted on the rover captures images of the soil.
2. Images are processed by a weed detection algorithm running on an edge device.
3. The system identifies weeds and determines their position relative to the rover.
4. The coordinates can then be used by other systems to remove the weed mechanically or by applying heat.

This enables **targeted weed removal without relying on large-scale chemical treatments.**


# Project Goals

The main goal of this project is to develop a system capable of:

* Detecting weeds in a **video or image feed**
* Running efficiently on **low-power edge devices**
* Integrating with the **MARS Rover V2**
* Providing **precise weed location coordinates**

These coordinates can be used by robotic tools to remove weeds automatically.


# Challenges

Although object recognition is a well-researched field, this project introduces several challenges.

### Limited Computing Power

Modern deep learning models require significant computational resources.
Edge devices have limited processing power, making it necessary to design **efficient algorithms** that can run in real time.

### Limited Training Data

Neural networks require large datasets for effective training.
Since the task involves specific plant types and conditions, collecting a sufficiently large dataset of labeled images is challenging.


# Key Objectives

* Develop a **weed detection algorithm optimized for edge devices**
* Build or collect a **suitable training dataset**
* Integrate the detection system with **MARS Rover V2**
* Enable **automated weed removal through precise localization**


# Potential Applications

* Precision agriculture
* Autonomous farming robots
* Reduced herbicide usage
* Environmentally friendly weed management

# Project Context

This project was developed as part of the MARS (Mobile Working Robot Systems) course at Technische Universität Berlin (TU Berlin)

# Collaborators

This project was developed by:
* Anton Elminger 
* Lena Zubik
* Felix Paulus
* Oliver Markus
