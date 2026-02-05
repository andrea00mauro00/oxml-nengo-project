# Bio-Inspired Cognitive Agent for Sequence Learning

This repository contains a neural-engineering project featuring a cognitive agent developed using the **Neural Engineering Framework (NEF)** and **Semantic Pointer Architecture (SPA)**. The project was submitted as the final assignment for the **Cognitive Robotics** course at the University of Trento (January 2026).



## Project Overview

The agent is designed to autonomously navigate a grid-world environment, process noisy sensory data, and perform high-level cognitive tasks through analog neural dynamics.

### Key Capabilities:
* **Autonomous Navigation**: Real-time exploration with reactive obstacle avoidance.
* **Noisy Pattern Recognition**: Classification of color-coded landmarks using noisy RGB sensors ($\sigma = 0.1$) mapped to high-dimensional Semantic Pointers.
* **Sequence Detection**: Identifying temporal transitions (e.g., detecting when the agent moves from a **RED** square to a **GREEN** one).
* **Neural Counting**: Maintaining stable statistical counts of specific transitions using recurrent neural integrators.

## Technical Architecture

The system is implemented in **Nengo**, a biological neural simulator. Unlike traditional Deep Learning, this model uses spiking neurons and structured representations to achieve cognition.

### Core Modules:
* **Compositional Binding**: Utilizes **Circular Convolution** to bind semantic vectors (e.g., `RED * current_color`), allowing the agent to represent "what it remembers" vs "what it sees" in a single neural population.
* **Robust Classification**: An ensemble of **800 Leaky Integrate-and-Fire (LIF) neurons** handles noisy inputs, maintaining a cosine similarity threshold of 0.25 for reliable categorization.
* **Working Memory**: Five dedicated **Recurrent Neural Integrators** (300 neurons each) maintain stable activity to count transitions, simulating persistent firing observed in biological prefrontal cortexes.
* **Subsumption Logic**: A movement controller that prioritizes radar-based collision avoidance while allowing for exploration noise.



## Repository Structure

* `final_agent.py`: Core simulation script including the Nengo model, SPA vocabulary, and environment logic.
* `report_cognitive_robotics.pdf`: Full academic report detailing the theoretical background (Situated Cognition), implementation strategy, and empirical results.
* `requirements.txt`: Necessary Python dependencies.

## Installation & Usage

1.  **Install dependencies**:
    ```bash
    pip install nengo nengo-spa numpy
    ```
2.  **Run the simulation**:
    ```bash
    python final_agent.py
    ```
    *Note: It is highly recommended to run this script within the **Nengo GUI** to visualize the Semantic Pointer plots and the grid-world live.*

## Academic Context

This project explores how high-level symbolic reasoning (counting and sequence matching) can emerge from low-level neural dynamics. It stands as a bridge between **computational neuroscience** and **AI**, showcasing robust performance in uncertain, embodied environments.
