import pandas as pd
import numpy as np
from scipy import stats, integrate, interpolate
import matplotlib.pyplot as plt


class NBAAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.regular_season = None
        self.player_name = None
        self.player_data = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_file)

    def filter_regular_season(self):
        self.regular_season = self.df[
            (self.df["League"] == "NBA") & (self.df["Stage"] == "Regular_Season")
        ].copy()

    def find_player_most_seasons(self):
        seasons_per_player = self.regular_season.groupby("Player")["Season"].nunique()
        self.player_name = seasons_per_player.idxmax()
        return self.player_name, int(seasons_per_player.max())

    def prepare_player_data(self):
        self.player_data = self.regular_season[self.regular_season["Player"] == self.player_name].copy()
        self.player_data = self.player_data.sort_values("Season")

        self.player_data["3PA"] = self.player_data["3PA"].replace(0, np.nan)
        self.player_data["ThreePointAccuracy"] = self.player_data["3PM"] / self.player_data["3PA"]
        self.player_data = self.player_data.dropna(subset=["ThreePointAccuracy"]).copy()

    def print_three_point_accuracy_by_season(self):
        print("\nThree-point accuracy by season:")
        for s, acc in zip(self.player_data["Season"], self.player_data["ThreePointAccuracy"]):
            print(s, ":", round(float(acc), 4))

    def linear_regression_and_plot(self):
        x = np.arange(len(self.player_data))
        y = self.player_data["ThreePointAccuracy"].values

        reg = stats.linregress(x, y)
        slope = reg.slope
        intercept = reg.intercept
        fit_line = slope * x + intercept

        print("\nLinear regression:")
        print("slope =", slope)
        print("intercept =", intercept)
        print("r-value =", reg.rvalue)
        print("p-value =", reg.pvalue)

        plt.scatter(x, y)
        plt.plot(x, fit_line)
        plt.title("3PT Accuracy Trend")
        plt.xlabel("Season Index")
        plt.ylabel("3PT Accuracy")
        plt.show()

        return x, y, fit_line

    def integrated_average_accuracy(self, x, fit_line, y):
        area = integrate.trapezoid(fit_line, x)
        span = x[-1] - x[0]
        avg_integrated_accuracy = area / span if span != 0 else float("nan")
        actual_avg_accuracy = float(np.mean(y))
        actual_avg_3pm = float(self.player_data["3PM"].mean())

        print("\nAverage 3PT accuracy:")
        print("Integrated avg (fit line):", avg_integrated_accuracy)
        print("Actual avg (real values):", actual_avg_accuracy)
        print("Actual average 3PM:", actual_avg_3pm)

    def interpolate_missing_seasons(self):
        def season_to_year(season_str):
            return int(str(season_str)[:4])

        self.player_data["SeasonYear"] = self.player_data["Season"].apply(season_to_year)

        years = self.player_data["SeasonYear"].values
        accs = self.player_data["ThreePointAccuracy"].values

        interp_func = interpolate.interp1d(years, accs, kind="linear", fill_value="extrapolate")

        missing_2002 = float(interp_func(2002))
        missing_2015 = float(interp_func(2015))

        print("\nInterpolated missing 3PT accuracy:")
        print("Estimated 2002-2003:", missing_2002)
        print("Estimated 2015-2016:", missing_2015)

    def fgm_fga_stats_and_ttests(self):
        fgm = self.regular_season["FGM"].dropna()
        fga = self.regular_season["FGA"].dropna()

        print("\nFGM stats")
        print("Mean:", float(np.mean(fgm)))
        print("Variance:", float(np.var(fgm)))
        print("Skew:", float(stats.skew(fgm)))
        print("Kurtosis:", float(stats.kurtosis(fgm)))

        print("\nFGA stats")
        print("Mean:", float(np.mean(fga)))
        print("Variance:", float(np.var(fga)))
        print("Skew:", float(stats.skew(fga)))
        print("Kurtosis:", float(stats.kurtosis(fga)))

        paired_df = self.regular_season[["FGM", "FGA"]].dropna()
        paired_t = stats.ttest_rel(paired_df["FGM"], paired_df["FGA"])
        print("\nPaired t-test:", paired_t)

        ind_t = stats.ttest_ind(fgm, fga, equal_var=False)
        print("\nIndependent t-test:", ind_t)


def main():
    analyzer = NBAAnalyzer("nba.csv")

    analyzer.load_data()
    analyzer.filter_regular_season()

    player, seasons = analyzer.find_player_most_seasons()
    print("Player with most NBA regular seasons:", player)
    print("Number of regular seasons played:", seasons)

    analyzer.prepare_player_data()
    analyzer.print_three_point_accuracy_by_season()

    x, y, fit_line = analyzer.linear_regression_and_plot()

    # ADDED: these were the missing calls
    analyzer.integrated_average_accuracy(x, fit_line, y)
    analyzer.interpolate_missing_seasons()
    analyzer.fgm_fga_stats_and_ttests()


if __name__ == "__main__":
    main()