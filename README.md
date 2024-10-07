# Formula 1 Driver Position Prediction

This project aims to predict the finishing positions of Formula 1 drivers using various race-related data points. The model is built using **LightGBM Regressor** and fine-tuned with **RandomizedSearchCV** for better accuracy.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Features Used](#features-used)
- [Model](#model)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Submission](#submission)
- [License](#license)
- [Contact](#contact)

## Overview

The goal of this project is to predict the position of drivers in Formula 1 races based on a variety of race metrics. We use the **LightGBM** model to predict the position and apply feature engineering techniques to maximize the performance of the model. The final model was optimized using **RandomizedSearchCV**.

### Key Steps:
1. **Data Loading**: Load the train, validation, and test datasets.
2. **Data Preprocessing**: Convert mixed-type columns, handle missing values, and create additional features such as driver age and average speed.
3. **Feature Engineering**: Derived features like `start_grid_diff` and `avg_speed`.
4. **Model Tuning**: Performed hyperparameter tuning with `RandomizedSearchCV`.
5. **Prediction and Submission**: Predicted the positions and created a submission file for the test dataset.

## Data
    [Dataset Link](https://www.kaggle.com/competitions/f1nalyze-datathon-ieeecsmuj/data)
- **Train Data**: Contains race details and final positions for past races.
- **Validation Data**: Used to validate model performance.
- **Test Data**: Contains race details for upcoming races where we predict the final positions.

### Data Files:
- `train.csv`
- `validation.csv`
- `test.csv`

## Features Used

Below is a list of the main features used for training the model:

- **grid**: Starting grid position.
- **points**: Points scored in the race.
- **laps**: Number of laps completed.
- **timetaken_in_millisec**: Time taken to complete the race in milliseconds.
- **fastestLap**: Fastest lap during the race.
- **max_speed**: Maximum speed during the race.
- **age**: Age of the driver.
- **avg_speed**: Average speed during the race.
- **start_grid_diff**: Difference between grid position and final position.
- **rank**: Driver's rank before the race.
- **year**: Year of the race.
- **round**: Round number of the race.
- **circuitId**: Circuit ID of the race.
- **driverRef**: Encoded reference of the driver.
- **constructorRef**: Encoded reference of the constructor.

## Model

The model used is **LightGBM Regressor**. The following hyperparameters were tuned using **RandomizedSearchCV**:

- **n_estimators**: Number of boosting rounds.
- **max_depth**: Maximum depth of the tree.
- **learning_rate**: Step size for shrinkage.
- **subsample**: Fraction of the training data used for fitting the model.
- **colsample_bytree**: Fraction of features used at each iteration.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Formula1-Driver-Position-Prediction.git
cd Formula1-Driver-Position-Prediction
```

### 2. Install Dependencies

To run the code, you need to install the required Python libraries. You can install them by running:

```bash
pip install -r requirements.txt
```

### 3. Run the Model Training Script

To train the model and generate predictions, execute the following command:

```bash
python src/model_training.py
```

This script will:
- Load the data.
- Perform preprocessing (converting to numeric, filling missing values).
- Train the LightGBM model.
- Generate predictions for validation and test datasets.
- Create a submission file `submission_lgbm.csv`.

## Results

- **Validation RMSE**: X.XX (You can fill in your RMSE here once available)

The model performed well, particularly in predicting the top 10 positions.

## Submission

After running the model, a submission file named `submission_lgbm.csv` will be created in the root directory. This file contains the predicted positions for the test dataset.

The submission file will have the following format:

```csv
result_driver_standing,position
1,2
2,4
3,1
...
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or contributions, feel free to contact:

- **Your Name**: [gokulprasath8600@gmail.com](mailto:gokulprasath8600@gmail.com)
- **GitHub**: [GokulPrasathM](https://github.com/GokulPrasathM)
