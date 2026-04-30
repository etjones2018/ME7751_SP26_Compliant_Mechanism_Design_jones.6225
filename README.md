# Compliant Mechanism Design: 3R Pseudo-Rigid-Body Optimization

**Author:** Enoch Jones
**Course:** ME 7751 Compliant Mechanism Design

## Overview
This repository contains the analysis, simulation, and evolutionary optimization of a 3-Revolute (3R) Pseudo-Rigid-Body (PRB) model for a flexible beam embedded in a highly non-linear 4-bar linkage mechanism.

The primary objective of this project is to evaluate the globally optimized 3R PRB model proposed by Su (2009) against a novel, task-specific 3R PRB model derived using Differential Evolution (via SciPy). Both models are validated against the exact continuous Euler-Bernoulli Boundary Value Problem (BVP) and rigorous finite element analysis (FEA) performed with CalculiX/FreeCAD.

By training an evolutionary algorithm exclusively on the physical geometric path constraints of the 4-bar linkage, the task-specific 3R PRB model achieves a 4,400x reduction in tracking error within its operational domain compared to the globally optimized Su model.

## Repository Contents

* `optimize_prb3r.py`: Replication of the global grid search and numerical BVP integration from Su (2009).
* `optimize_task_4bar.py`: A task-specific Differential Evolution algorithm that optimizes the PRB parameters ($\gamma_i$, $k_i$) to minimize tracking error strictly along the 4-bar linkage operational path.
* `fourbar_prb3r.py`: Core kinematic solver. Generates the interactive GUI and solves the forward kinematics for both 3R PRB models against the exact continuum beam BVP.
* `fourbar_poses.py`: Visualizer that generates and saves image frames comparing the continuous beam to the rigid 3R models at specific crank angles.
* `fea_prb_compare.py`: Compares the analytical Python models against the CalculiX/FreeCAD non-linear B32R large-deflection beam node data.
* `ME7751_Report.pdf`: The final formal report detailing the methodology and conclusions, formatted as an ASME conference paper.

## Requirements
* Python 3.8+
* `numpy`
* `scipy`
* `matplotlib`

## Methodology Highlight
Rather than training on generic independent forces, the task-specific optimizer was constrained exclusively to the geometric trajectory of the 4-bar mechanism. For any given crank angle, the exact continuum beam tip position was determined using the rigorous BVP solver. The PRB tip position was computed using forward kinematics by evaluating the minimum strain energy equilibrium state of the 3R chain subject to the 4-bar closure constraints. 

A Differential Evolution search found the mechanism-specific lengths and stiffnesses that directly minimized the Euclidean distance between the PRB tip and the exact continuum tip integrated over the full 360-degree rotation of the crank.
