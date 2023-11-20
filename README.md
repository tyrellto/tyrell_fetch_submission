---

# Tyrell Fetch Submission - Forecasting Receipt Amounts for 2022

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model and Forecasting Strategy](#model-and-forecasting-strategy)
4. [Technology Stack](#technology-stack)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installing and Running the Application](#installing-and-running-the-application)
6. [Model Training and Visualization](#model-training-and-visualization)
   - [Key Features](#key-features-1)
   - [Prerequisites for Replication](#prerequisites-for-replication)
   - [Running the Notebook](#running-the-notebook)

## Project Overview

This 2022 project aims to forecast the monthly number of scanned receipts, leveraging a TensorFlow model that analyzes trend, seasonal, and residual data, alongside the original time series. 
The model’s additive approach to incorporating trend, seasonal, and residual elements is designed to offer a more accurate and holistic view of future receipt trends.

### Model and Forecasting Strategy

- **Approach**: Combines trend, seasonal, residual, and original time series data to enhance forecasting accuracy and model stablization.
- **Data**: Used a lag of 90 days (around 3 months) to generate features from trend, seasonal, residual, and original time series data for supervised learning.
- **Visualization**: Plots forecasted values (combined trend, seasonal, and residual additively) with the original time series for comparison and better understanding of the model's performance.
- **Monthly Aggregation**: Post-prediction, sums up the total receipts for each month and visualizes them in a histogram, offering insights into monthly trends.

The project's outcome is a comprehensive model that not only forecasts receipt counts but also visualizes data trends effectively with Streamlit, aiding in informed decision-making.

## Technology Stack

- **Python**: Primary programming language for data analysis and modeling.
- **Pandas & NumPy**: Key libraries for efficient data manipulation and numerical operations.
- **Streamlit**: Interactive web application framework to demonstrate model results.
- **TensorFlow**: For advanced machine learning model development.

## Getting Started

These instructions will guide you in getting a copy of the project running on your local machine for development and testing purposes.

### Prerequisites

What you need to install:

- **Git**
- **Docker**

#### Installing Git

Download and install Git from [Git's website](https://git-scm.com/downloads). Follow the installation instructions for your operating system.

#### Installing Docker

Download Docker Desktop from [Docker's website](https://www.docker.com/products/docker-desktop) and follow the installation instructions for your operating system.

### Installing and Running the Application

Follow these steps to set up your development environment.

#### Cloning the Repository

Open a terminal and run:

```bash
git clone https://github.com/tyrellto/tyrell_fetch_submission.git
cd tyrell_fetch_submission
```

#### Building the Docker Image

Build the Docker image with:

```bash
docker build -t mystreamlitapp .
```

This command builds a Docker image named `mystreamlitapp` from the Dockerfile in your project directory.

#### Running the Streamlit App

Run the app using:

```bash
docker run -p 8501:8501 mystreamlitapp
```

This command starts a container from the `mystreamlitapp` image and maps port 8501 from the container to port 8501 on your host machine.

### Accessing the Application

Open a web browser and navigate to `http://localhost:8501`. You should now see the Streamlit app running.


## Model Training and Visualization

The core of the model training and its visual inspection are conducted through the `train_model.ipynb` Jupyter notebook. This notebook contains detailed steps and visualizations essential for understanding and replicating the model.

### Key Features

- **train_model.ipynb**: A Jupyter notebook used for:
    - Training the forecasting model.
    - Visual inspection of the model's performance.
    - Detailed documentation of the methodology and analysis.

### Prerequisites for Replication

To replicate and further develop the model, the following are required:

- **Jupyter Notebook Environment**: An IDE or platform that can run Jupyter notebooks is essential. Options include:
    - [Google Colab](https://colab.research.google.com/): A free, cloud-based service that supports Jupyter notebooks.
    - [Visual Studio Code](https://code.visualstudio.com/): With its Jupyter extension, it allows for running and visualizing notebooks.
    - [JupyterLab](https://jupyter.org/): A web-based interactive development environment for Jupyter notebooks.
- **Python Libraries**: Ensure that all Python libraries used in `train_model.ipynb` are installed. Typically, these can be found in a `requirements.txt` file.

### Running the Notebook

1. Open `train_model.ipynb` in your preferred Jupyter notebook environment.
2. Execute the cells in sequence to train the model and view the forecasts.

---
