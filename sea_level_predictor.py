import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np 

def draw_plot():
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')
    
    # Create scatter plot

    plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], color='blue')

    # Create first line of best fit

    slope_all, intercept_all, r_value, p_value, std_err = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    years_extended = np.arrange(df['Year'].min(), 2051)
    line_all = intercept_all + slope_all * years_extended
    plt.plot(years_extended, line_all, color='red', label='Fit: All Years')

    # Create second line of best fit

    df_milennia = df[df['Year'] >= 2000]
    slope_milennia, intercept_milennia, r_value, p_value, std_err = linregress(df_milennia['Year'], df_milennia['CSIRO Adjusted Sea Level'])
    years_milennia_extended = np.arrange(2000, 2051)
    line_milennia = intercept_milennia + slope_milennia + years_milennia_extended
    plt.plot(years_milennia_extended, line_milennia, color='green', label='Fit: 2000 and Beyond')

    # Add labels and title

    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')
    plt.legend()
    plt.show()

    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()
