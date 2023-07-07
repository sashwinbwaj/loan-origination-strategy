from ast import Raise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from scipy.stats import f_oneway, chi2_contingency
from scipy.stats.mstats import pearsonr, spearmanr
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

class FeatureDescription():

    def __init__(self, input_data : pd.DataFrame, feature_list : list, target : str):
        self.data = input_data
        self.feature_list = feature_list
        self.target = target

    def show_values(self, axs, orient : str ="v", space : float =.01, required_format : str = '{:.2f}'):
        """Function to display values in bar plots"""
        def _single(ax):
            if orient == "v": # for vertical bars
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                    value = required_format.format(p.get_height())
                    ax.text(_x, _y, value, ha="center") 
            elif orient == "h": # for horizontal bars
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                    value = required_format.format(p.get_width())
                    ax.text(_x, _y, value, ha="left")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _single(ax)
        else:
            _single(axs)

    
    def plot_categorical(self, ax, feature : str):
        data = self.data.copy()
        data[self.target] = data[self.target].astype('float64')
        count_summary = (data[feature].value_counts()/len(data)).reset_index()
        target_summary = data.groupby(feature).mean()[self.target].reset_index() #data[data[self.target] == '1'].groupby(feature).count()[self.target].reset_index()/count_summary
        ax = sns.barplot(data = count_summary, x = 'index', y = feature, palette = 'rocket_r')
        ax.set_xlabel('')
        ax.set_ylabel('Proportion %')
        ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
        ax2=ax.twinx()
        ax2 = sns.scatterplot(data = target_summary, x = feature, y = self.target, color = 'red', s = 60)
        ax2.set_ylim([0,1])
        ax2.axhline(y = data[self.target].mean(), ls = '--', color = 'crimson', alpha = 0.3)
        ax2.yaxis.set_major_locator(mticker.FixedLocator(ax2.get_yticks()))
        ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()])
        ax2.set_ylabel('Target %')
        ax.set_title(feature, fontsize = 15)
    
    def plot_numerical(self, ax, feature : str, kind : str = 'density'):
        data = self.data
        if kind == 'combo':
            data['Groups'] = pd.cut(data[feature], bins = 10, labels = range(10))
            count_summary = (data['Groups'].value_counts()/len(data)).reset_index()
            target_summary = data.groupby('Groups').mean()[self.target].reset_index()
            ax = sns.barplot(data = count_summary, x = 'index', y = 'Groups', palette = 'rocket_r')
            ax.set_xlabel('')
            ax.set_ylabel('Proportion %')
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
            ax.set_title(feature, fontsize = 15)
            
            ax2=ax.twinx()
            ax2 = sns.scatterplot(data = target_summary, x = 'Groups', y = self.target, color = 'red')
            ax2.set_ylim([0,1])
            ax2.axhline(y = data[self.target].mean(), ls = '--', color = 'crimson', alpha = 0.3, label = 'Average Target')
            ax2.yaxis.set_major_locator(mticker.FixedLocator(ax2.get_yticks()))
            ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()])
            ax2.set_ylabel('Target %')
        
        elif kind == 'density':
            ax = sns.kdeplot(data = self.data, x = feature, hue = self.target, shade=True, alpha=.5, palette = 'rocket_r') 
            ax.set_xlabel('')
            ax.set_title(feature, fontsize = 15)
            ax.set_ylabel('Density %')
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
            ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])

        elif kind == 'box':
            data['target_str'] = data[self.target].astype('object')
            ax = sns.boxplot(data = data, y = feature, x = "target_str",  palette = 'rocket_r') 
            ax.set_xlabel('')
            ax.set_title(feature, fontsize = 15)
            ax.set_ylabel('Value')
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
            ax.set_yticklabels(['{:,.0f}'.format(x) for x in ax.get_yticks()])
            
        else:
            raise ValueError('Kind can only be "density" for density plot or "combo" for bar chart combination')

    def plot(self, numerical_plot_type : str = 'density'):
        nPlots = len(self.feature_list)
        nRows = int(np.ceil(nPlots/3)) 
        fig, axes = plt.subplots(nRows, 3, figsize=(24, nRows * 6))

        for i, feature in enumerate(self.feature_list):
            ax = plt.subplot(nRows, 3, i + 1)
            if self.data[feature].dtype in ['object', 'str', 'bool']:
                self.plot_categorical(ax = ax, feature = feature)
            else:
                self.plot_numerical(ax = ax, feature = feature, kind = numerical_plot_type)
        plt.tight_layout()

class FeatureInformation():

    def __init__(self, input_data : pd.DataFrame, feature_list : list, target : str, method : str = 'lgbm'):
        self.data = input_data
        self.feature_list = feature_list
        self.target = target
        self.method = method

    def calculate(self):
        if self.method == 'basic':
            return self._get_important_vars()
        elif self.method == 'lgbm':
            return self._run_lgbm()
        else:
            raise ValueError('Method can only be "lgbm" for LightGBM AUC or "basic" for Spearman/Chi-Sq/ANOVA coefficients')
            

    def plot(self, num_vars : int = 10):
        if self.method == 'basic':
            num_feature_information_dict, cat_feature_information_dict = self.calculate()
            num_feature_information_df = pd.DataFrame.from_dict(
                num_feature_information_dict, orient='index', 
                columns = ['value']).sort_values('value', ascending = False).head(num_vars)
            cat_feature_information_df = pd.DataFrame.from_dict(
                cat_feature_information_dict, orient='index', 
                columns = ['value']).sort_values('value', ascending = False).head(num_vars)

            fig, ax = plt.subplots(figsize = (14,5))
            cat_test = 'Chi-square' if self.data[self.target].dtype in ['object', 'str', 'bool'] else 'ANOVA F-'
            num_test = 'ANOVA F-' if self.data[self.target].dtype in ['object', 'str', 'bool'] else 'Spearman Corr'
            ax1 = plt.subplot(1,2,1)
            ax1 = sns.barplot(y = num_feature_information_df.index, x = num_feature_information_df.value, palette='rocket_r')
            ax1.set_title(f'Numerical Features Information : {num_test}  statistic')
            ax1.set_xlabel(f'{num_test}statistic')
            ax2 = plt.subplot(1,2,2)
            ax2 = sns.barplot(y = cat_feature_information_df.index, x = cat_feature_information_df.value, palette='rocket_r')
            ax2.set_title(f'Categorical Features Information : {cat_test} statistic')
            ax2.set_xlabel(f'{cat_test} statistic')
            plt.tight_layout()
        
        elif self.method == 'lgbm':
            feature_information_dict = self.calculate()
            feature_information_df = pd.DataFrame.from_dict(
                feature_information_dict, orient='index', 
                columns = ['value']).sort_values('value', ascending = False).head(num_vars)

            fig, ax = plt.subplots(figsize = (10,5))
            ax = sns.barplot(y = feature_information_df.index, x = feature_information_df.value, palette='rocket_r')
            ax.set_title(f'Feature Information : LightGBM AUC')
            ax.set_xlabel('AUC')
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
            ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])


    def _get_important_vars(self):
        num_feature_information_dict = {}
        cat_feature_information_dict = {}
        for feature in self.feature_list:
            if self.data[self.target].dtype not in ['object', 'str', 'bool']:
                if self.data[feature].dtype not in ['object', 'str', 'bool']:
                    num_feature_information_dict[feature] = spearmanr(self.data[~self.data.isna()][feature], self.data[~self.data.isna()][self.target])[0]
                elif (self.data[feature].dtype not in ['object', 'str', 'bool']) & (self.data[self.target].dtype in ['object', 'str', 'bool']):
                    cat_feature_information_dict[feature] = f_oneway(self.data[feature], self.data[self.target])[0]
            else:
                if self.data[feature].dtype in ['object', 'str', 'bool']:
                    self.data['dummy']= 1
                    contingent_table = pd.pivot_table(data = self.data, index = feature, values = 'dummy', columns = 'num_defaults', aggfunc = 'count').fillna(0)
                    cat_feature_information_dict[feature] = chi2_contingency(contingent_table)[0]
                else:
                    num_feature_information_dict[feature] = f_oneway(self.data[feature], self.data[self.target])[0]
        return num_feature_information_dict, cat_feature_information_dict


    def _run_lgbm(self):
        feature_information_dict = {}
        for feature in self.feature_list:
            model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=4, num_iterations = 20, random_state=42)
            if self.data[feature].dtype in ['object', 'str']:
                self.data[feature] = self.data[feature].astype('category')
            X_train = self.data[[feature]]
            y_train = self.data[self.target]
            model.fit(X_train, y_train)
            y_preds = model.predict_proba(X_train)[:,1]
            feature_information_dict[feature] = roc_auc_score(y_train, y_preds) 
        return feature_information_dict

class OutlierDetection():

    def __init__(self, input_data : pd.DataFrame, feature_list : list, method : str = 'iqr'):
        self.data = input_data
        self.feature_list = feature_list
        self.method = method
    
    def _get_iqr_thresholds(self, input_series : pd.Series, coefficient : float = 1.5):

        Q1 = np.percentile(input_series, 25, interpolation = 'midpoint') 
        Q2 = np.percentile(input_series, 50, interpolation = 'midpoint') 
        Q3 = np.percentile(input_series, 75, interpolation = 'midpoint') 

        IQR = Q3 - Q1 
        low_lim = Q1 - coefficient * IQR
        up_lim = Q3 + coefficient * IQR

        return low_lim, up_lim
        
        return self.data[self.data.index.isin(outliers.index)]

    def outlier_detection_iqr(self, input_df : pd.DataFrame, input_features : list, verbose : bool = False, cap : bool = False):
            input_df['outlier_flag'] = 0 
            for column in input_features:
                input_series = self.data[column]
                low_lim, up_lim = self._get_iqr_thresholds(input_series)
                outliers = input_series[(input_series > up_lim) | (input_series < low_lim)] 
                if verbose:
                    print(f'Number of outliers detected for {column} is {len(outliers)}')
                input_df.loc[input_df.index.isin(outliers.index), 'outlier_flag'] = 1
                if cap:
                    input_df.loc[input_df[column] > up_lim, column] = up_lim
                    input_df.loc[input_df[column] < low_lim, column] = low_lim
            if cap:
                print('\nOutliers capped')
            return input_df

    def outlier_handling_iforest(self, input_df : pd.DataFrame, input_features : list, contamination : float = 0.05):
        iforest = IsolationForest(n_estimators=100, contamination=contamination)
        output_df = input_df.copy()
        input_df['anamaly_score'] = iforest.fit_predict(pd.get_dummies(input_df[input_features]))
        output_df['outlier_flag'] = np.where(input_df['anamaly_score'] == -1, 1, 0)
        return output_df

    def identify(self, remove : bool = False, cap : bool = False, verbose : bool = False):
        numerical_features = [column for column in self.feature_list if self.data[column].dtype in ['int64', 'float64']]
        if self.method == 'iqr':
            return self.outlier_detection_iqr(input_df = self.data.copy(), 
                                            input_features = numerical_features, 
                                            verbose = verbose, cap = cap)
        elif self.method == 'iforest':
            return self.outlier_handling_iforest(input_df = self.data.copy(),
                                                input_features = self.feature_list)

    def remove(self, input_df : pd.DataFrame):
        if 'outlier_flag' not in input_df.columns:
            raise KeyError('Outlier Flag not found. Please run the calculate function to identify outliers using I-forest/IQR method')
        else:
            print(f'Number of outliers observations removed is {input_df[input_df.outlier_flag == 1].shape[0]} out of {input_df.shape[0]} observations')
        return input_df[input_df.outlier_flag == 0].drop(['outlier_flag'], axis = 1)

            
        
