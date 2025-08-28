# Options Modeling

This repository contains experiments and tools for modeling market options data. The project focuses on preparing, analyzing, and modeling options chain data combined with stock price data to predict market movements and evaluate trading strategies.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The goal of this project is to explore and model options chain data to predict profitable trades. The project includes data preprocessing, feature engineering, and machine learning experiments. Notebooks and images can be used/seen to analyze and visualize options data.

Key objectives:
- Combine options chain data with stock price data.
- Predict whether an option's price will double within a specified time frame.
- Evaluate trading strategies using backtesting.

---

## Features

- **Data Preprocessing**: Tools to clean and prepare options and stock price data.
- **Feature Engineering**: Generate advanced features like implied volatility, moneyness, and time-to-expiry.
- **Modeling**: Train and evaluate machine learning models for options price prediction.
- **Backtesting**: Simulate trading strategies and evaluate their performance.
- **Visualization**: Explore data and results through plots and charts.

---

## Project Structure
```plaintext
options-modeling/
├── data/                # Contains raw and processed data files
├── exp1/                # Experiment group 1 (preprocessing, training, etc.)
│   ├── preprocessing.py # Data preprocessing scripts
│   ├── constants.py     # Constants and configuration
│   ├── experiments_*.py # Experiment scripts
│   └── utils.py         # Utility functions
├── exp2/                # Experiment group 2 (advanced modeling)
│   ├── data.py          # Data loading and handling
│   ├── model.py         # Model definitions
│   └── train_utils.py   # Training utilities
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── README.md            # Project documentation
├── setup.py             # Installation script
└── local.env            # Environment variables (e.g., API keys)
```
---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/options-modeling.git
   cd options-modeling
   ```

2. Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a local.env file and add your API keys and other configurations.

---

## Usage

In folder [exp1](exp1/):

- experiments_moving_training.py to obtain the return after a year when investing a fixed amount the month.
- experiments_moving_training_compound.py to obtain the return after a year when investing an amount from the previous month including the return (compounded return).
- We used data from 2023 since it was the cheapest available.

```bash
python experiments_moving_training_compound.py
```

In folder [exp2](exp2/):

- main.py: To obtain compounded return of a full year
- We used data from 2023 since it was the cheapest available.

In folder [exp3](exp3/):

- exploratory notebooks 
- Only stock data for now. Options data to be added later on

## License

This project is licensed under the MIT License.
