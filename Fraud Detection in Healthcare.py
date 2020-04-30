#!/usr/bin/env python
# coding: utf-8

# In[108]:


#import libraries
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols 
from scipy import stats
from pylab import rcParams
import math as mt
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
from sklearn import datasets, linear_model, metrics 
from statsmodels.graphics.gofplots import ProbPlot
import math

LABELS = ["Normal", "Fraud"]


# In[179]:


#Load data
claims=pd.read_csv('c:\\temp\\insuranceclaims.csv',header=0,encoding= 'unicode_escape')


# In[180]:


claims.head()


# In[181]:


claims.describe()


# In[182]:


## Lets check the shape of the data

print('claims Shape:',claims.shape,'\n')


# In[47]:


#let load the beneficiary data
Beneficiarydata=pd.read_csv('c:\\temp\\Beneficiarydata.csv',header=0,encoding= 'unicode_escape')


# In[48]:


Beneficiarydata.head()


# In[183]:


## Lets merge claims data with beneficiary details data based on 'BeneID' as joining key for inner join
Alldata=pd.merge(claims,Beneficiarydata,left_on='BeneID',right_on='BeneID',how='inner')

Alldata=pd.merge(claims,Beneficiarydata,left_on='BeneID',right_on='BeneID',how='inner')


# In[50]:


#PLotting the frequencies of fraud and non-fraud  transactions in the data

sns.set_style('white',rc={'figure.figsize':(12,8)})
count_classes = pd.value_counts(claims['PotentialFraud'], sort = True)
print("Percent Distribution of Potential Fraud class:- \n",count_classes*100/len(claims))
LABELS = ["Non Fraud", "Fraud"]
#Drawing a barplot
count_classes.plot(kind = 'bar', rot=0,figsize=(10,6))

#Giving titles and labels to the plot
plt.title("Potential Fraud distribution in Aggregated claim transactional data")
plt.xticks(range(2), LABELS)
plt.xlabel("Potential Fraud Class ")
plt.ylabel("Number of PotentialFraud per Class ")

plt.savefig('PotentialFraudDistributionInMergedData')


# In[51]:


#PLotting the frequencies of Statewise beneficiaries
count_States = pd.value_counts(Beneficiarydata['State'], sort = True)
#print("Percent Distribution of Beneficieries per state:- \n",count_States*100/len(Train_Beneficiarydata))

#Drawing a barplot
(count_States*100/len(Beneficiarydata)).plot(kind = 'bar', rot=0,figsize=(16,8),fontsize=12,legend=True)

#Giving titles and labels to the plot

plt.annotate('Maximum Beneficiaries are from this State', xy=(0.01,8), xytext=(8, 6.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.yticks(np.arange(0,10,2), ('0 %','2 %', '4 %', '6 %', '8 %', '10%'))
plt.title("State - wise Beneficiary Distribution",fontsize=18)
plt.xlabel("State Number",fontsize=15)
plt.ylabel("Percentage of Beneficiaries "'%',fontsize=15)
plt.show()

plt.savefig('StateWiseBeneficiaryDistribution')


# In[52]:


#PLotting the frequencies of race-wise beneficiaries
count_Race = pd.value_counts(Beneficiarydata['Race'], sort = True)

#Drawing a barplot
(count_Race*100/len(Beneficiarydata)).plot(kind = 'bar', rot=0,figsize=(10,6),fontsize=12)

#Giving titles and labels to the plot
plt.yticks(np.arange(0,100,20))#, ('0 %','20 %', '40 %', '60 %', '80 %', '100%'))
plt.title("Race - wise Beneficiary Distribution",fontsize=18)
plt.xlabel("Race Code",fontsize=15)
plt.ylabel("Percentage of Beneficiaries "'%',fontsize=15)

plt.show()

plt.savefig('RacewiseBeneficiaryDistribution')


# In[53]:


## Lets plot countplot for each fraud non fraud categories

sns.set(rc={'figure.figsize':(12,8)},style='white')

ax=sns.countplot(x='ClmProcedureCode_1',hue='PotentialFraud',data=claims
              ,order=claims.ClmProcedureCode_1.value_counts().iloc[:10].index)

plt.title('Top-10 Procedures invloved in Healthcare Fraud')
    
plt.show()

plt.savefig('TopProceduresinvlovedinHealthcareFraud')


# In[54]:


## lets plot Top-10 Claim Diagnosis  invloved in Healthcare Fraud

sns.set(rc={'figure.figsize':(12,8)},style='white')

sns.countplot(x='ClmDiagnosisCode_1',hue='PotentialFraud',data=claims
              ,order=claims.ClmDiagnosisCode_1.value_counts().iloc[:10].index)

plt.title('Top-10 Diagnosis invloved in Healthcare Fraud')
plt.show()

plt.savefig('TopDiagnosisInnvlovedinHealthcareFraud')


# In[55]:


### lets plot Top-20 Attending Physicians invloved in Healthcare Fraud 

sns.set(rc={'figure.figsize':(12,8)},style='white')

ax= sns.countplot(x='AttendingPhysician',hue='PotentialFraud',data=claims
              ,order=claims.AttendingPhysician.value_counts().iloc[:20].index)

    
plt.title('Top-20 Attending physicians invloved in Healthcare Fraud')
plt.xticks(rotation=90)
plt.show()

plt.savefig('TopAttendingphysiciansinvlovedinHealthcareFraud')


# In[56]:


## Lets Plot DeductibleAmtPaid and InsClaimAmtReimbursed in both fraud and non Fraud Categoories

sns.set(rc={'figure.figsize':(12,8)},style='white')

sns.lmplot(x='DeductibleAmtPaid',y='InscClaimAmtReimbursed',hue='PotentialFraud',
           col='PotentialFraud',fit_reg=False,data=Alldata)


plt.savefig('DeductibleAmtPaidandInsClaimAmtReimbursed')


# In[57]:


## Let's See Insurance Claim Amount Reimbursed Vs Race
sns.set(rc={'figure.figsize':(12,8)},style='white')

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Insurance Claim Amount Reimbursed Vs Race')

ax1.scatter(Alldata[Alldata.PotentialFraud=='Yes'].Race, 
           Alldata[Alldata.PotentialFraud=='Yes'].InscClaimAmtReimbursed)
ax1.set_title('Fraud')
ax1.axhline(y=60000,c='r')
ax1.set_ylabel('Insurance Claim Amout Reimbursed')

ax2.scatter(Alldata[Alldata.PotentialFraud=='No'].Race, 
            Alldata[Alldata.PotentialFraud=='No'].InscClaimAmtReimbursed)
ax2.set_title('Normal')
ax2.axhline(y=60000,c='r')
ax2.set_xlabel('Race')
ax2.set_ylabel('Insurance Claim Amout Reimbursed')

plt.show()
f.savefig('RaceVsClaimAmtReimbursed')


# In[184]:


df_fradulent = Alldata[Alldata['PotentialFraud'] == 'Yes']


# In[244]:


df_fradulent["Race"]= df_fradulent["Race"].astype(str) 


# In[245]:


df_fradulent["Gender"]= df_fradulent["Gender"].astype(str) 


# In[239]:


col = ['InscClaimAmtReimbursed', 'TotalClaimCost', 'Total_Amount_of_Payment_USDollars', 'PaidAmount', 'DeductibleAmtPaid']
axes = pd.plotting.scatter_matrix(df_fradulent[col], alpha=0.5, figsize=(20,20))
plt.tight_layout()
plt.show()


# In[186]:


df_fradulent[col].corr()


# In[235]:


formula = 'InscClaimAmtReimbursed ~ TotalClaimCost'
model = ols(formula, df_fradulent).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)


# In[215]:


formula = 'InscClaimAmtReimbursed ~ TotalClaimCost + Total_Amount_of_Payment_USDollars'
model = ols(formula, df_fradulent).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)


# In[234]:


formula = 'InscClaimAmtReimbursed ~ DeductibleAmtPaid'
model = ols(formula, df_fradulent).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)


# In[217]:


formula = 'InscClaimAmtReimbursed ~TotalClaimCost * Total_Amount_of_Payment_USDollars'
model = ols(formula, df_fradulent).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)


# In[218]:


formula = 'InscClaimAmtReimbursed ~ TotalClaimCost + Total_Amount_of_Payment_USDollars'
model2 = ols(formula, df_fradulent).fit()
aov_table = sm.stats.anova_lm(model, model2)
print(aov_table) 


# In[219]:


aov_table = sm.stats.anova_lm(model2, type=2)
print(aov_table) 


# In[220]:


#building the linear model
lm = ols("Total_Amount_of_Payment_USDollars ~ DeductibleAmtPaid", data=df_fradulent) . fit() 


# In[221]:


lm.summary()


# In[222]:


print(lm. params) 


# In[196]:


## The linear regression: InscClaimAmtReimbursed = 264.42+ 8.208 * DeductibleAmtPaid. R square is .029,
## which means 3% of the observed data can be explained by the linear model.


# In[233]:


## Build the quadratic model
lm2 = ols("InscClaimAmtReimbursed ~ DeductibleAmtPaid + I(DeductibleAmtPaid**2)", data=df_fradulent).fit() 


# In[224]:


lm2.summary()


# In[225]:


print(lm2. params) 


# In[200]:


## The quadratic regression: InscClaimAmtReimbursed = 292.055 - 14.368 * DeductibleAmtPaid  + 0.132 * DeductibleAmtPaid^2. R square is .363,
## which means 36.3%% of the observed data can be explained by the quadratic model. Note: This indicated that the quadratic model is  better than the linear model.


# In[232]:


## Build the cubic model
lm3 = ols("InscClaimAmtReimbursed ~ DeductibleAmtPaid  + I(DeductibleAmtPaid **2) + I(DeductibleAmtPaid **3)", data=df_fradulent).fit() 
lm.summary()


# In[227]:


print(lm3. params) 


# In[228]:


lm = ols("InscClaimAmtReimbursed ~ Total_Amount_of_Payment_USDollars", data=df_fradulent) . fit() 
lm.summary()


# In[229]:


print(lm3. params) 


# In[230]:


## The cubic regression: InscClaimAmtReimbursed = 285.64 + 0.959 * DeductibleAmtPaid - 0.033 * DeductibleAmtPaid^2 +  0.000176 * DeductibleAmtPaid ^3. R square is 0,
## which means 77.4% of the observed data can be explained by the cubic model.Note: This indicates that model 2 is still better than the others


# In[231]:


# using anova_lm to compare the three models
print(sm.stats.anova_lm(lm, lm2, lm3, typ=1)) 


# In[207]:


#Regression Analysis 
plt.scatter(df_fradulent['InscClaimAmtReimbursed'], df_fradulent['TotalClaimCost'], marker='+')
plt.xlabel("InscClaimAmtReimbursed")
plt.ylabel("TotalClaimCost")
plt.show()


# In[208]:


aX = np.asarray(df_fradulent['InscClaimAmtReimbursed'])
aY = np.asarray(df_fradulent['TotalClaimCost'])
regr = linear_model.LinearRegression()
regr.fit(aX.reshape(-1,1), aY.reshape(-1,1))
plt.scatter(aX, aY, marker='+')
plt.plot(aX, regr.predict(aX.reshape(-1,1)))
plt.xlabel("InscClaimAmtReimbursed")
plt.ylabel("TotalClaimCost")
plt.show()


# In[209]:


aX = np.asarray(df_fradulent['InscClaimAmtReimbursed'])
aY = np.asarray(df_fradulent['TotalClaimCost'])
aX = sm.add_constant(aX)

model = sm.OLS(aY, aX)
results = model.fit()
print(results.summary())


# In[210]:


df = pd.DataFrame(results.resid)
df.describe()


# In[211]:


plt.style.use('seaborn') 
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)
# fitted values (need a constant term for intercept)
model_fitted_y = results.fittedvalues
# model residuals
model_residuals = results.resid
# normalized residuals
model_norm_residuals = results.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# leverage, from statsmodels internals
model_leverage = results.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = results.get_influence().cooks_distance[0]


# In[212]:


aX = np.asarray(df_fradulent['InscClaimAmtReimbursed'])
aY = np.asarray(df_fradulent['TotalClaimCost'])
plot_lm_1 = plt.figure(1)
#plot_lm_1.set_figheight(8)
#plot_lm_1.set_figwidth(12)
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, aY)
plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plt.show()


# In[213]:


QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
#plot_lm_2.set_figheight(8)
#plot_lm_2.set_figwidth(12)
plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
plt.show()


# In[ ]:




