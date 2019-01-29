# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices

DATA_PATH = './data.csv'


def read_csv_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError("The file you are trying to access does not exist")
    # header=0 -> first row contains columns name and not actual data.
    return pd.read_csv(path, header=0, )


def get_fitted_ols_model(formula, data):
    y, X = dmatrices(formula, data=data, return_type='dataframe')
    ols_model = sm.OLS(y, X)
    return ols_model.fit()


def get_residuals_plot_ols(data, fitted_model, y_name, figure_height, figure_width, save_to_file=False,
                           file_name="residuals_plot"):
    # get the predicted values from the model
    fitted_values = fitted_model.fittedvalues

    # defined some graphs styles
    resid_plot = plt.figure(1)
    resid_plot.set_figwidth(figure_height)
    resid_plot.set_figwidth(figure_width)
    sns.set_style('darkgrid')
    # tell seaborn to generate a residuals plot
    resid_plot.axes[0] = sns.residplot(fitted_values, y_name, data=data,
                                       lowess=True,
                                       scatter_kws={'alpha': 0.5},
                                       line_kws={'color': 'red', 'lw': 2, 'alpha': 0.8})

    # add labels.
    resid_plot.axes[0].set_title('Residuals Plot')
    resid_plot.axes[0].set_xlabel('Fitted values')
    resid_plot.axes[0].set_ylabel('Residuals')

    if save_to_file:
        resid_plot.savefig(file_name + ".png")

    return resid_plot


def get_partial_regression_plot(fitted_model, figure_size=(12, 8), save_to_file=False, file_name="regression_plot2"):
    reg_plot = sm.graphics.plot_partregress_grid(fitted_model, fig=plt.figure(figsize=figure_size))

    if save_to_file:
        reg_plot.savefig(file_name + ".png")

    return reg_plot


# %%

# %%
batting = read_csv_file(DATA_PATH)
print(batting.head())  # print the first 5 rows to the dataset.
# %%


# %%
# Set the first formula, fit the model and print summary.
offense_formula = 'R ~ wOBA + H + OPS  + AVG + BB  + SO + OBP + SLG + ISO + BABIP + wRC + wRCplus + Season'
offense_model = get_fitted_ols_model(offense_formula, batting)
print(offense_model.summary())
# %%

# %%
# Simplify the formula given the results form previous steps.
new_offense_formula = 'R ~ wRC + wRCplus'
new_offense_model = get_fitted_ols_model(new_offense_formula, batting)
print(new_offense_model.summary())
# %%

# %%
get_residuals_plot_ols(batting, new_offense_model, 'R', 6, 8, save_to_file=True)
reg_plot = get_partial_regression_plot(new_offense_model, save_to_file=True, figure_size=(12, 8))
plt.show()
# # %%$

