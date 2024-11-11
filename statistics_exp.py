import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.weightstats as stests
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot

#read the data
data = pd.read_csv('exp_data.csv')

#describe statistics
print("===================================================\ndescribe statistics:")
gender_counts = data['gender'].value_counts()
print("Counts of each gender:")
print(gender_counts)
age_mean = data['age'].mean()
age_std = data['age'].std()
print("\nAverage age:", age_mean)
print("Standard deviation of age:", age_std)

condition_counts = data['condition'].value_counts()
print("\nCounts of participants in each condition:")
print(condition_counts)

group_means = data.groupby('condition')['credibility'].mean()
group_medians = data.groupby('condition')['credibility'].median()

print("Group Means:")
print(group_means)
print("\nGroup Medians:")
print(group_medians)

#normal destribution shapiro-wilk test
print("==================================================\nnormal destribution shapiro-wilk test:")
shapiro_test = stats.shapiro(data['credibility'])
print('Shapiro-Wilk Test:', shapiro_test)

#independent groups t test
print("==================================================\nindependent t test:")
single = data[data['condition'] == 0]['credibility']
multi = data[data['condition'] == 1]['credibility']
t_test = stests.ttest_ind(single, multi)
std_err_group1 = np.std(single, ddof=1) / np.sqrt(len(single))
std_err_group2 = np.std(multi, ddof=1) / np.sqrt(len(multi))

print('Standard Error of Group single:', std_err_group1)
print('Standard Error of Group multipule:', std_err_group2)
print('t-statistic:', t_test[0])
print('p-value:', t_test[1])
print('degrees of freedom:', t_test[2])

#t test graph
plt.figure(figsize=(12, 6))
# יצירת היסטוגרמות
plt.hist(single, bins=15, alpha=0.6, label='Single Source', color='blue', density=True)
plt.hist(multi, bins=15, alpha=0.6, label='Multiple Sources', color='orange', density=True)
# הוספת עקומות צפיפות
sns.kdeplot(single, color='blue')
sns.kdeplot(multi, color='orange')
#קווי ממוצע
ylim = plt.gca().get_ylim()
plt.axvline(np.mean(single), color='blue', linestyle='dashed', linewidth=2,)
plt.axvline(np.mean(multi), color='orange', linestyle='dashed', linewidth=2)
# הגדרות נוספות
plt.title('Distribution of Credibility Scores for Single vs. Multiple Sources')
plt.xlabel('Credibility')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#linear regression
print("==================================================\nlinear regression:")
data['gender'] = data['gender'].map({'Man': 0, 'Woman': 1, 'Other': np.random.choice(1)})
X = data[['gender', 'age', 'condition']]
y = data['credibility']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

plt.figure(figsize=(12, 8))
# גרף פיזור עם קו רגרסיה עבור גיל ורמת ביטחון
plt.subplot(2, 2, 1)
sns.regplot(x='age', y='credibility', data=data)
plt.title('Regression of Age on credibility')
plt.xlabel('Age')
plt.ylabel('credibility')

# גרף פיזור עם קו רגרסיה עבור מגדר ורמת ביטחון
plt.subplot(2, 2, 2)
sns.regplot(x='gender', y='credibility', data=data, x_jitter=0.1)
plt.title('Regression of Gender on credibility')
plt.xlabel('Gender')
plt.ylabel('credibility')

# גרף פיזור עם קו רגרסיה עבור תנאי הניסוי ורמת ביטחון
plt.subplot(2, 2, 3)
sns.regplot(x='condition', y='credibility', data=data, x_jitter=0.1)
plt.title('Regression of Condition on credibility')
plt.xlabel('Condition')
plt.ylabel('credibility')

# הצגת הגרפים
plt.tight_layout(pad=4.0)
plt.show()

#age codition credibility anova
print("==================================================\nage codition credibility anova")
model = ols('credibility ~ age + C(condition) + age:C(condition)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

#uni credibility anove
print("==================================================\nuni credibility anova")
model = ols('credibility ~ C(uni)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# יצירת גרף תיבות (Box Plot) להצגת הציונים עבור כל אוניברסיטה
plt.figure(figsize=(10, 6))
sns.boxplot(x='uni', y='credibility', data=data)
plt.title('Box Plot of Credibility by University')
plt.xlabel('University')
plt.ylabel('Credibility')
plt.show()
