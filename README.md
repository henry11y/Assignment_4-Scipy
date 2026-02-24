# NBA Data Analysis Using SciPy

## Project Purpose
The purpose of this project is to perform numeric data analysis using Python and the SciPy library on real NBA player statistics data.

## What the code does
- Filters NBA regular season data
- Identifies player with most seasons played
- Calculates three-point accuracy per season
- Performs linear regression and plots trend line
- Computes average accuracy using integration
- Estimates missing saons using interpolation
- Calculates statistical measures (mean, variance, skew, kurtosis)
- Performs t-tests

 ### Class Design
- The project uses a class called 'NBAAnalyzer' to organize functionality.

### Attributes
- dataset
- filtered regular season data
- selected player data

- ### Methods
- load_data()
- filter_regular_season()
- find_player_most_seasons()
- linear_regression_and_plot()
- integrated_average_accuracy()
- interpolate_missing_seasons()
- fgm_fga_stats_and_ttests()

 ### Python libraries used:
- pandas
- numpy
- scipy
- matplotlib
