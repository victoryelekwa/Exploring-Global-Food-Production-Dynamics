#!/usr/bin/env python
# coding: utf-8

# # Exploring Global Food Production Dynamics

# Author: Victory Elekwa

# ### Introduction
# 
# In an era marked by unprecedented global challenges and the need for sustainable development, understanding and optimizing food production processes have become paramount. As we navigate complexities such as population growth, climate change, and resource scarcity, the role of data-driven insights in shaping agricultural strategies cannot be overstated.
# 
# This report explores an in-depth analysis of world food production trends, leveraging advanced data analytics techniques to uncover key patterns and insights. By examining production volumes, regional disparities, and emerging trends across various food categories, we aim to provide actionable insights for policymakers, agricultural stakeholders, and researchers alike.
# 
# Through a comprehensive exploration of global food production data spanning multiple decades, this report endeavors to shed light on the dynamics shaping our food systems. By harnessing the power of data analytics, we seek to empower decision-makers with the knowledge and tools necessary to foster a more sustainable and resilient food future.
# 
# Join us as we embark on a journey through the intricate web of food production, uncovering hidden trends, identifying opportunities for improvement, and paving the way towards a more nourished and prosperous world.

# #### Import Libraries

# In[74]:


import pandas as pd
import matplotlib.pyplot as plt
import datapane as dp
import plotly
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
template_style = 'plotly_white'


# #### Load and Explore Data

# In[75]:


world_food_prod = pd.read_csv('world food production.csv')


# In[76]:


world_food_prod.head(3)


# In[77]:


world_food_prod.shape


# In[78]:


world_food_prod.describe()


# In[79]:


world_food_prod.info()


# ## Data Preprocessing

# ##### Adding a decade column

# In[80]:


convert_dict = {'Year': str}
 
world_food_prod = world_food_prod.astype(convert_dict)
print(world_food_prod.dtypes)


# In[81]:


world_food_prod['Decade'] = (world_food_prod['Year'].apply(lambda x: x[:3])+'0')
world_food_prod['Decade']


# In[82]:


convert_dict = {'Year': int,
               'Decade': int}
 
world_food_prod = world_food_prod.astype(convert_dict)
print(world_food_prod.dtypes)


# ##### Creating a continent column

# In[83]:


africa_standard = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cape Verde', 'Cameroon', 
          'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Democratic Republic of Congo', 'Djibouti', 'Egypt', 
          'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 
          "Cote d'Ivoire", 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 
          'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 
          'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe']


north_america_standard = ['Canada', 'United States', 'Mexico', 'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 
                          'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 
                          'Haiti', 'Honduras', 'Jamaica', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 
                          'Saint Vincent and the Grenadines', 'Trinidad and Tobago']

south_america_standard = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 
                          'Suriname', 'Uruguay', 'Venezuela']

asia_standard = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 
                 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 
                 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 
                 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 
                 'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 
                 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen']

europe_standard = ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 
                   'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 
                   'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Latvia', 'Liechtenstein', 'Lithuania', 
                   'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 
                   'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 
                   'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City']

oceania_standard = ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 'Vanuatu', 'Samoa', 
                    'Kiribati', 'Tonga', 'Marshall Islands', 'Palau', 'Nauru', 'Tuvalu', 'Micronesia (country)']


# In[84]:


def GetContinent(country):
    if country in asia_standard:
        return "Asia"
    elif country in europe_standard:
        return "Europe"
    elif country in africa_standard:
        return "Africa"
    elif country in north_america_standard:
        return "North America"
    elif country in south_america_standard:
        return "South America"
    elif country in oceania_standard:
        return "Oceania"
    else:
        return "other"


# In[85]:


world_food_prod['Continent'] = world_food_prod['Entity'].apply(lambda x: GetContinent(x))


# In[86]:


world_food_prod.head(3)


# In[87]:


world_food_prod[world_food_prod['Continent'] == 'other']['Entity'].unique()


# ##### Dropping the "other" classification in continent

# In[88]:


world_food_prod = world_food_prod.drop(world_food_prod[world_food_prod['Continent'] == 'other'].index)


# In[89]:


world_food_prod['Continent'].unique()


# ##### Categorizing agricultural products into primary food classes

# ##### Primary Food Classes
# 
# 1. Cereals:
# 
#     * Maize
#     * Rice 
#     * Wheat
#     * Rye 
# 2. Tubers and Root Crops:
# 
#     * Yams
#     * Sweet Potatoes
#     * Potatoes
# 3. Vegetables:
#    
#    * Tomatoes
#    * Peas, Dry 
# 4. Fruits:
# 
#     * Oranges
#     * Bananas
#     * Grapes
#     * Apples
#     * Avocados
# 5. Oilseeds:
# 
#     * Sunflower
#     * Soybeans
#     * Palm Oil
# 6. Beverages:
# 
#     * Tea
#     * Coffee, Green
#     * Cocoa Beans 
# 7. Sugar and Sweeteners:
# 
#     * Sugar Cane
# 8. Livestock:
# 
#     * Meat, Chicken

# In[90]:


world_food_prod['Cereals'] = world_food_prod.iloc[:, [2, 3, 5, 12]].sum(axis=1)
world_food_prod['Roots and Tubers'] = world_food_prod.iloc[:, [4, 8, 13]].sum(axis=1)
world_food_prod['Vegetables'] = world_food_prod.iloc[:, [6, 15]].sum(axis=1)
world_food_prod['Fruits'] = world_food_prod.iloc[:, [14, 17, 21, 22, 23]].sum(axis=1)
world_food_prod['Oil seeds'] = world_food_prod.iloc[:, [9, 11, 16]].sum(axis=1)
world_food_prod['Beverages'] = world_food_prod.iloc[:, [7, 18, 19]].sum(axis=1)
world_food_prod['Sugar and Sweeteners'] = world_food_prod.iloc[:, 10]
world_food_prod['Livestock'] = world_food_prod.iloc[:, 20]


# In[91]:


# to view all columns
pd.set_option('display.max_columns', None)


# In[92]:


world_food_prod.head(3)


# ##### Creating a total production column

# In[93]:


world_food_prod['Total Production'] = world_food_prod.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                                                               21, 22, 23]].sum(axis=1)


# ##### Deleting extra spaces between column names

# In[94]:


world_food_prod.columns


# In[95]:


world_food_prod.columns = world_food_prod.columns.str.replace('  ', ' ')


# In[96]:


world_food_prod.columns


# #### Country Classification by Income (World Bank)

# In[97]:


high_income_countries = ['Andorra', 'Antigua and Barbuda', 'Australia', 'Austria', 'Bahamas', 'Belgium', 'Brunei Darussalam', 'Canada', 'Denmark', 'Finland', 
                         'France', 'Germany', 'Hong Kong SAR, China', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 
                         'Kuwait', 'Luxembourg', 'Macao SAR, China', 'Malta', 'Netherlands', 'New Zealand', 'Norway', 
                         'Oman', 'Qatar', 'San Marino', 'Saudi Arabia', 'Singapore', 'Slovenia', 'South Korea', 'Spain', 
                         'Sweden', 'Switzerland', 'United Arab Emirates', 'United Kingdom', 'United States', 'Barbados', 
                         'Czechia', 'Estonia', 'Greece', 'Hungary', 'Latvia', 'Lithuania', 'Romania', 'Saint Kitts and Nevis', 'Seychelles', 'Trinidad and Tobago', 'Poland', 'Panama', 'Portugal', 'Slovakia', 
                         'Taiwan', 'Guyana']

upper_middle_income_countries = ['Albania', 'Argentina', 'Belarus', 'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 
                                 'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba', 'Dominica', 'Dominican Republic', 
                                 'Ecuador', 'Equatorial Guinea', 'Fiji', 'Gabon', 'Grenada', 'Iraq', 'Jamaica',  
                                 'Kazakhstan', 'Libya', 'Malaysia', 'Maldives', 'Mauritius', 'Mexico', 'Montenegro', 'Paraguay', 'Peru', 'Russia', 'Saint Lucia', 
                                 'Saint Vincent and the Grenadines', 'Serbia', 'South Africa', 'Suriname', 
                                 'Thailand', 'Turkey', 'Uruguay', 'Venezuela', 'Vietnam', 'Azerbaijan', 
                                 'Belize', 'Botswana', 'Namibia', 'North Macedonia', 'Georgia', 'Kosovo', 'Moldova', 'Marshall Islands']

lower_middle_income_countries = ['Afghanistan', 'Angola', 'Armenia', 'Algeria', 'Bangladesh', 'Benin', 'Bhutan', 'Bolivia', 'Cape Verde', 'Cambodia', 
                                 'Cameroon', 'Comoros', 'Congo, Rep.', "Cote d'Ivoire", 'Djibouti', 'El Salvador', 'Eswatini', 
                                 'Eswatini', 'Ghana', 'Guatemala', 'Lebanon', 'Egypt', 'Guinea', 
                                 'Honduras', 'Haiti', 'India', 'Indonesia', 'Iran', 'Jordan', 'Kenya', 'Kiribati', 'Kyrgyz Republic', 'Lao PDR', 
                                 'Lesotho', 'Micronesia', 'Mongolia', 'Morocco', 'Myanmar', 'Nepal', 
                                 'Nicaragua', 'Nigeria', 'Pakistan', 'Papua New Guinea', 'Philippines', 'Samoa', 'São Tomé and Príncipe', 
                                 'Senegal', 'Solomon Islands', 'Sri Lanka', 'Timor-Leste', 'Tanzania', 'Tonga', 'Tunisia', 'Ukraine', 
                                 'Uzbekistan', 'Vanuatu', 'Zambia', 'Zimbabwe', 'Congo', 'Kyrgyzstan', 'Laos', 'Mauritania', 
                                 'Micronesia (country)', 'Sao Tome and Principe', 'Tajikistan', 'Turkmenistan']

low_income_countries = ['Burkina Faso', 'Burundi', 'Central African Republic', 'Chad', 'Democratic Republic of the Congo', 
                        'Eritrea', 'Ethiopia', 'Gambia', 'Guinea-Bissau', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mozambique', 
                        'Niger', 'Rwanda', 'Sierra Leone', 'Somalia', 'South Sudan', 'Sudan', 'Syria', 'Togo', 'Uganda', 'Yemen', 
                        'Democratic Republic of Congo', 'North Korea']


# In[98]:


def GetClassification(country):
    if country in high_income_countries:
        return "High Income Countries"
    elif country in upper_middle_income_countries:
        return "Upper Middle Income Countries"
    elif country in lower_middle_income_countries:
        return "Lower Middle Income Countries"
    elif country in low_income_countries:
        return "Low Income Countries"
    else:
        return "other"


# In[99]:


world_food_prod['Income Classification'] = world_food_prod['Entity'].apply(lambda x: GetClassification(x))


# #### Rename and shorten column headers

# In[100]:


column_rename = {'Entity' : 'Country', 'Maize Production (tonnes)' : 'Maize',
       'Rice Production ( tonnes)' : 'Rice', 'Yams Production (tonnes)' : 'Yam',
       'Wheat Production (tonnes)' : 'Wheat', 'Tomatoes Production (tonnes)' : 'Tomatoes',
       'Tea Production ( tonnes )' : 'Tea', 'Sweet potatoes Production (tonnes)' : 'Sweet Potatoes',
       'Sunflower seed Production (tonnes)' : 'Sunflower Seed', 'Sugar cane Production (tonnes)' : 'Sugar Cane',
       'Soybeans Production (tonnes)' : 'Soybeans', 'Rye Production (tonnes)' : 'Rye',
       'Potatoes Production (tonnes)' : 'Potatoes', 'Oranges Production (tonnes)' : 'Oranges',
       'Peas, dry Production ( tonnes)' : 'Peas, dry', 'Palm oil Production (tonnes)' : 'Palm Oil',
       'Grapes Production (tonnes)' : 'Grapes', 'Coffee, green Production ( tonnes)' : 'Coffee, green',
       'Cocoa beans Production (tonnes)' : 'Cocoa beans', 'Meat, chicken Production (tonnes)' : 'Meat, chicken',
       'Bananas Production ( tonnes)' : 'Bananas', 'Avocados Production (tonnes)' : 'Avocado',
       'Apples Production (tonnes)' : 'Apples'}


# In[101]:


world_food_prod.rename(columns=column_rename, inplace=True)


# #### Reordering columns

# In[102]:


world_food_prod = world_food_prod.reindex(columns=['Continent', 'Country', 'Income Classification', 'Decade', 'Year', 'Maize', 'Rice', 'Yam', 'Wheat', 'Tomatoes', 'Tea',
       'Sweet Potatoes', 'Sunflower Seed', 'Sugar Cane', 'Soybeans', 'Rye',
       'Potatoes', 'Oranges', 'Peas, dry', 'Palm Oil', 'Grapes',
       'Coffee, green', 'Cocoa beans', 'Meat, chicken', 'Bananas', 'Avocado',
       'Apples', 'Cereals', 'Roots and Tubers', 'Vegetables',
       'Fruits', 'Oil seeds', 'Beverages', 'Sugar and Sweeteners', 'Livestock',
       'Total Production'])
world_food_prod.head(1)


# #### Production Trends Over Time
# This section explores the historical trends in global food production over time. By analyzing production volumes across different years, we aim to identify patterns, fluctuations, and long-term trends that have shaped food production dynamics.

# In[103]:


decade_total_prod = world_food_prod.groupby('Decade')[['Total Production']].sum()
decade_total_prod = decade_total_prod.reset_index()


# In[104]:


# Abbreviate number function
def abbreviate_number(number):
    if number >= 1e9:
        return f'{number/1e9:.1f}B'
    elif number >= 1e6:
        return f'{number/1e6:.1f}M'
    else:
        return f'{number}'  

fig_trend = px.line(decade_total_prod, x="Decade", y="Total Production", title='<b>World Food Production</b>',
                    markers=True, text=decade_total_prod['Total Production'].apply(abbreviate_number))

fig_trend.update_traces(line=dict(color='green'), textposition='top center')
fig_trend.update_layout(xaxis_title='', yaxis_title='', xaxis=dict(showgrid=False), template=template_style)
fig_trend.show()


# - From 1960 to 2010, world food production exhibited consistent growth, with an increase from 155.2 billion tonnes to 270.0 billion tonnes. 
# 
# - However, post-2010, a sharp decline occurred, plummeting to 55.2 billion units by 2020.
# 
# - The peak in 2010 marks an inflection point where growth reversed, necessitating further investigation into underlying factors.

# #### Explore the causes of the drop in production post 2010's
# - In examining the factors contributing to the notable decline in production post-2010s, we start by querying the dataset for the past decade to pinpoint the inflection point and investigate potential causative factors.

# In[105]:


yearly_food_prod = world_food_prod.groupby('Year').sum(numeric_only=True)
yearly_food_prod.reset_index(inplace=True)
last_10_years = yearly_food_prod.iloc[-10:]

fig1 = px.line(last_10_years, x="Year", y="Total Production", title='<b>World Food Production (2012 - 2021)</b>',
                    markers=True, text=last_10_years['Total Production'].apply(abbreviate_number))

fig1.update_traces(line=dict(color='green'), textposition='top center')
fig1.update_layout(xaxis_title='', yaxis_title='', xaxis=dict(showgrid=False), template=template_style)
fig1.show()


# 1. **Annual Production Trends insights**:
# - In 2019, the world food production volume was 27.9 billion tonnes.
# - In 2020, there was a slight increase, reaching 28 billion tonnes.
# - However, in 2021, production dropped to 27.2 billion tonnes, a decrease of 800 million tonnes from 2020.
# 
# 2. **Explanation for the Sharp Decline (2010s to 2020s)**:
# 
# The drastic drop from 270 billion tonnes in the 2010s to 55 billion tonnes in the 2020s can be attributed to the following:
# 
# - **Limited Data Points**: The dataset only covers two years (2020 and 2021) within the 2020 decade, whereas a standard decade comprises ten years (e.g., 2020 - 2029).
# - **Sampling Bias**: With only two data points, the overall trend may not accurately represent the entire decade. It lacks the granularity needed to capture long-term patterns.

# ##### World Food Production Trend (1961 - 2019)¶
# To mitigate the sampling bias evident in the data, we generate a new dataframe that omits the 2020s decade. This revised dataset will be utilized specifically for decade-by-decade comparisons or analyses.

# In[106]:


world_food_prod2 = world_food_prod.drop(world_food_prod[world_food_prod['Year'] > 2019].index)


# In[107]:


decade_total_prod = world_food_prod2.groupby('Decade')[['Total Production']].sum()
decade_total_prod.reset_index(inplace=True)

fig_trend1 = px.line(decade_total_prod, x="Decade", y="Total Production", title='<b>World Food Production</b>',
                    markers=True, text=decade_total_prod['Total Production'].apply(abbreviate_number))

fig_trend1.update_traces(line=dict(color='green'), textposition='top center')

fig_trend1.update_layout(xaxis_title='', yaxis_title='', xaxis=dict(showgrid=False), template=template_style)

fig_trend1.show()


# - The overall trend shows a consistent increase in global food production over time.
# - Starting around 155 billion tonnes in the 1960s, production steadily rises to approximately 270 billion tonnes by the 2010s.
# 

# #### Product Ranking by Volume
# In this section, we rank agricultural products based on their production volumes. By identifying the top-producing crops, we gain insights into the relative importance of different food commodities in global agricultural output.

# In[108]:


food_crops = world_food_prod.columns[5:27]
total_crop_production = world_food_prod[food_crops].sum()

sorted_food_crops = total_crop_production.sort_values(ascending=False).index
sorted_total_production = total_crop_production[sorted_food_crops]

primary_food_crops = world_food_prod.columns[27:35]
primary_crop_production = world_food_prod[primary_food_crops].sum()

sorted_primary_crops = primary_crop_production.sort_values(ascending=False).index
sorted_total_primary_production = primary_crop_production[sorted_primary_crops]

# abbreviate function(html)
def abbreviate_number_bold(number):
    if number >= 1e9:
        return f'<b>{number/1e9:.1f}B</b>'
    elif number >= 1e6:
        return f'<b>{number/1e6:.1f}M</b>'
    else:
        return f'<b>{number}</b>'

# Bar chart
fig2 = go.Figure(data=go.Bar(x=sorted_food_crops, y=sorted_total_production, text=sorted_total_production
                            .apply(abbreviate_number_bold), textposition='outside',textfont=dict(size=10)))
fig2.update_layout(title='<b>Total Production by Food Crops</b>', template=template_style)
fig2.show()


# In[109]:


fig_primary = go.Figure(data=go.Bar(x=sorted_primary_crops, y=sorted_total_primary_production, 
                                    text=sorted_total_primary_production.apply(abbreviate_number_bold), textposition='outside', 
                                    textfont=dict(size=11)))
fig_primary.update_layout(title='<b>Total Production by Primary Food Crops</b>', template=template_style)
fig_primary.show()


# - Sugar Cane leads as the most produced crop, with an impressive volume of about 472.4 billion tonnes.
# 
# - Other significant agricultural products on the list of top producers (with volumes exceeding 100 billion tonnes) include wheat, rice, and potatoes.
# 
# - Least produced crops, with volumes below 10 billion tonnes, encompass meat, palm oil, peas, coffee, cocoa, tea, and avocado.
# 
# - Avocado holds the lowest production record, with just 1.2 billion tonnes over six decades.
# 
# - In the ranking by primary food categories, sugars and sweeteners emerge as the top performer, with an impressive production volume exceeding 472 billion tonnes, closely trailed by cereals boasting 420 billion tonnes.
# 
# - Despite avocado recording the lowest production among the products ranked, the fruit category secures the fourth position with 94.8 billion tonnes. This can be attributed to the higher production volumes of other fruit crops such as grapes, apples, bananas, and oranges, each exceeding 15.9 billion tonnes.
# 
# - The beverage category demonstrates the lowest production volume, totaling only 6.1 billion tonnes. This outcome aligns with the bottom five productions in the product ranking chart, primarily comprising coffee, cocoa, and tea.
# 

# #### Production Trend of Top 5 Crops
# Here, we delve deeper into the production trends of the top five agricultural crops. By examining how the production of these key crops has evolved over time, we uncover valuable insights into the changing dynamics of global food supply.

# In[110]:


top_5_crops = ['Sugar Cane', 'Wheat', 'Rice', 'Potatoes', 'Soybeans']
top_5_crops_df = world_food_prod2.groupby(['Decade'])[top_5_crops].sum()

fig_top5 = go.Figure()
for crop in top_5_crops:
    fig_top5.add_trace(go.Scatter(x=top_5_crops_df.index, y=top_5_crops_df[crop], mode='lines+markers+text', name=crop, 
                                  text=[abbreviate_number(val) for val in top_5_crops_df[crop]], textposition='top center', 
                                  textfont_size=9))

fig_top5.update_layout(title='<b>Production Trend of Top 5 Crops</b>', xaxis=dict(tickvals=top_5_crops_df.index, 
                                                                                  ticktext=top_5_crops_df.index, 
                                                                                  showgrid=False), template=template_style)
fig_top5.show()


# - Sugar Cane exhibits a significant upward trajectory in production volume. Rising from a base of 56.8 billion tonnes in 1960 to a peak of 94.2 billion tonnes in 2010.
# 
# - Wheat, Rice and Potatoes show moderate growth over the years. However, they all experience a  minimal drop (11% - 20%) in the 1980's which was quickly corrected in the 1990's with over 40% increment in production
# 
# - Soybeans depict a fluctuating trend with no substantial increase in production volume.
# 
# - While Sugar Cane has experienced exponential growth, other crops have witnessed only marginal increases or have remained relatively stable.

# #### Food Production Distribution by Continent
# This section investigates the distribution of food production across different continents. By analyzing production volumes by continent, we gain a comprehensive understanding of regional contributions to global food output.

# In[111]:


continent_prod1 = world_food_prod.groupby('Continent').sum(numeric_only=True)
continent_prod1.reset_index(inplace=True)
# Pie chart
fig_pie1 = px.pie(continent_prod1, values='Total Production', names='Continent',
                 title='<b>Food Production Distribution by Continent</b>', 
                  hover_name='Continent', labels={'Income Classification': 'Income Classification'})
fig_pie1.show()


# From the depicted figure illustrating the production distribution across continents, the following observations are noted:
# 
# - Africa holds the largest production share, closely followed by Asia, trailing by less than 7%.
# - Subsequently, North America, Europe, South America, and Oceania follow in respective order.
# - Oceania exhibits the smallest production share, amounting to 1.13%, which is 28% less than Africa's share.

# #### Production Trend by Continent
# Here, we explore the production trends within individual continents. By examining how food production has evolved over time in different regions, we identify regional patterns and dynamics that shape the global food landscape.

# In[112]:


decade_continent_prod = world_food_prod2.groupby(['Decade', 'Continent']).sum(numeric_only=True)
decade_continent_prod.reset_index(inplace=True)


# In[113]:


continent_list = ['Asia', 'Europe', 'Africa', 'North America', 'South America','Oceania']

traces = []
for continent in continent_list:
    continent_data = decade_continent_prod[decade_continent_prod['Continent'] == continent]
    trace = go.Scatter(
        x=continent_data['Decade'],
        y=continent_data['Total Production'],
        mode='lines+markers+text',
        name=continent, text=[abbreviate_number(val) for val in continent_data["Total Production"]], textposition='top center', 
        textfont_size=9)
    traces.append(trace)

layout = go.Layout(
    title='<b>Production Trend by Continent</b>',
    xaxis=dict(showgrid=False), template='plotly_white')

fig3 = go.Figure(data=traces, layout=layout)
fig3.show()


# - Africa has witnessed a remarkable upward trajectory in production, possibly attributed to advancements in technology, ample availability of arable land, and supportive policies.
# 
# - Asia's production has also risen, albeit at a slower rate than Africa. Following closely is Europe, which demonstrated consistent production growth until the 1990s, after which it experienced marginal declines of around 9% in subsequent decades.
# 
# - North America and South America have both encountered fluctuations in production levels, with South America experiencing the most significant decline (approximately 26%) post-1990s.
# 
# - Oceania's production trend appears relatively stagnant, indicating stable yet stagnant market conditions. Potential factors contributing to this include limited investment, infrastructure constraints, and regional economic challenges.

# #### Production Trend by Primary Food Classification
# In this section, we analyze the production trends of primary food classifications such as cereals, fruits, vegetables, and more. By examining production trends within each category, we gain insights into the dynamics of specific food groups.

# In[114]:


sorted_primary_foods = ['Sugar and Sweeteners', 'Cereals', 'Roots and Tubers', 'Fruits', 'Oil seeds', 'Vegetables', 
                        'Livestock', 'Beverages']
decade_df = world_food_prod2.groupby('Decade')[sorted_primary_foods].sum(numeric_only=True)
decade_df.reset_index(inplace=True)


# In[115]:


fig4 = go.Figure()

for col in decade_df.columns[1:]:
    fig4.add_trace(go.Bar(
        x=decade_df['Decade'],
        y=decade_df[col],
        name=col,
        text=decade_df[col].apply(abbreviate_number_bold),  
        textposition='outside',  
        hoverinfo='none',  
        textfont=dict(size=14)))

fig4.update_layout(
    title='<b>Production Trend by Food Classification</b>',
    barmode='group',  
    template='plotly_white')

layout = go.Layout(
    legend=dict(
        orientation="h", 
        yanchor="top",    
        y=1.1))

fig4.update_layout(layout)
fig4.show()


# - The data illustrates a persistent uptrend in production volumes across primary food categories.
# 
# - It also demonstrates a consistent ordinal arrangement of these categories within each decade under examination.
# 
# - The hierarchical order is as follows: Sugars and sweeteners, Cereals, Roots & Tubers, Fruits, Oil seeds, Vegetables, Livestock, and Beverages.
# 
# - There were marginal shifts in the 1990s and 2000s, with oil seeds and fruits interchanging positions.

# #### Top Producers by Primary Food Classification
# This section identifies the top producers within each primary food classification. By highlighting the leading producers in various food categories, we discern patterns of dominance and specialization across different regions.

# In[116]:


tempo1 = world_food_prod2.iloc[:, [0, 1, 27, 28, 29, 30, 31, 32, 33, 34]]
tempo_df1 = tempo1.groupby(["Continent", 'Country']).sum(numeric_only=True)
tempo_df1.reset_index(inplace=True)

melted_df1 = tempo_df1.melt(id_vars=['Continent', 'Country'], var_name='Food_Class', value_name='Production_Volume')
melted_df1['Production_Volume'] = pd.to_numeric(melted_df1['Production_Volume'], errors='coerce')
top_producers1 = melted_df1.groupby('Food_Class').apply(lambda x: x.nlargest(5, 'Production_Volume'))


# In[117]:


continent_colors = {
    'Asia': '#FF7F50',  
    'Europe': '#008080', 
    'Africa': '#FFD700', 
    'Oceania': '#E6E6FA',
    'South America': '#708090', 
    'North America': '#556B2F' 
}
food_classes = top_producers1['Food_Class'].unique()
num_rows = len(food_classes) // 3 + len(food_classes) % 3 
fig5 = make_subplots(rows=num_rows, cols=3, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.2, 
                    subplot_titles=("<b>Beverages</b>", "<b>Cereals</b>", "<b>Fruits</b>", "<b>Livestock</b>", "<b>Oil Seeds</b>", 
                                    "<b>Roots and Tubers</b>", "<b>Sugars and Sweeteners</b>", "<b>Vegetables</b>" ))

for i, food_class in enumerate(food_classes, start=1):
    row_num = (i - 1) // 3 + 1 
    col_num = (i - 1) % 3 + 1 
    food_class_data = top_producers1.loc[food_class]
    food_class_data1 = food_class_data.sort_values(by='Production_Volume', ascending=True)

    for index, row in food_class_data1.iterrows():
        continent = row['Continent']  
        
        color = continent_colors.get(continent, '#808080') 
        
        abbreviated_volume = abbreviate_number(row['Production_Volume'])
        
        trace = go.Bar(y=[row['Country']], x=[row['Production_Volume']],
                       orientation='h', marker_color=color, showlegend=False,
                      text=[abbreviated_volume], textposition='inside') 
        
        fig5.add_trace(trace, row=row_num, col=col_num)
        fig5.update_layout(title=f'<b>Top 5 Producing Countries by Food Class</b>')

for continent, color in continent_colors.items():
    fig5.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color=color),
                             showlegend=True, name=continent),
                  row=1, col=3)

fig5.update_layout(height=250 * num_rows, width=1000, template="plotly_white")
fig5.show()


# **Beverages**:
#   - Asia leads global beverage production, with Maldives as the top producer, while Africa also holds a significant share.
#   
# **Cereals**:
#   - North America, represented by Saint Lucia, United States, and Armenia, commands a strong presence in cereal production, while other continents have smaller shares.
#   
# **Fruits**:
#   - Although Argentina leads in fruit production with 4.2 billion tonnes (South America), Africa, represented by Sao Tome and Principe, Chad, and Morocco, dominates with a combined volume of 9.5 billion tonnes.
#   
# **Livestock**:
#   - Europe, particularly Italy, takes the lead in livestock production, with contributions from Africa and Asia.
#   
# **Oil Seeds**:
#   - Africa, led by Namibia, dominates oilseed production.
#   
# **Roots and Tubers**:
#   - Africa, notably Egypt, and North America play significant roles in roots and tubers production.
#   
# **Sugars and Sweeteners**:
#   - South America, especially Peru and Argentina, leads in sugars and sweeteners production.
#   
# **Vegetables**:
#   - North America, specifically the United States, Guatemala, and Dominican Republic, dominates global vegetable production.
# 
# In summary, Africa, North America, and Asia emerge as dominant continents across various food classes, with other continents contributing to specific categories. No country within the Oceania continent ranks in the top five for each food class.
# 

# #### Top Producers by Food Crops
# Here, we identify the top producers for individual food crops. By examining production volumes and leading producers for each crop, we gain insights into regional strengths and specialization in agricultural production.

# In[118]:


cont_count_prod = world_food_prod2.groupby(['Continent', 'Country']).sum(numeric_only=True)
cont_count_prod.reset_index(inplace=True)
cont_count_prod.head(3)
exclude_columns = [0, 1, 2, 3, 26, 27, 28, 29, 30, 31, 32, 33, 34]
food_produce_columns = cont_count_prod.columns.difference(cont_count_prod.columns[exclude_columns])

max_productions = {}
for column in food_produce_columns:
    max_production_continent = cont_count_prod.loc[cont_count_prod[column].idxmax()]['Continent']
    max_production_country = cont_count_prod.loc[cont_count_prod[column].idxmax()]['Country']
    max_production_volume = cont_count_prod[column].max()
    max_productions[column] = {'Continent':max_production_continent, 'Country': max_production_country, 'Production': max_production_volume}

max_productions_df = pd.DataFrame(max_productions).T
max_productions_df2 = max_productions_df.sort_values(by='Production', ascending=False)


# In[125]:


legend_items = []

fig6 = go.Figure()

for index, row in max_productions_df2.iterrows():
    color = continent_colors.get(row['Continent'], 'gray')
    abbreviated_volume = abbreviate_number(row['Production'])
    
    fig6.add_trace(go.Bar(
        x=[index], 
        y=[row['Production']],
        showlegend=False,
        text=[f"<b>{row['Country']}</b> ({abbreviated_volume})"], 
        name=row['Continent'],
        marker=dict(color=color), 
        textposition='outside', 
        textangle=-90, 
    ))
    if row['Continent'] not in legend_items:
        legend_items.append(row['Continent'])
        fig6.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=row['Continent']))
        
fig6.update_layout(
    title="<b>Highest Producing Country Per Food Crop</b>",
    xaxis_title=None,
    yaxis_title=None,
    legend_title="Continent",
    template="plotly_white",
    yaxis=dict(range=[0, max_productions_df2['Production'].max() * 1.4]),
    height=600,  # Specify the height of the plot
    width=800 
)

fig6.update_layout(uniformtext_minsize=6, uniformtext_mode='show')
fig6.show()


# In the figure above, designed to illustrate the country with the highest production volume per food crop, color-coded by continent, the following observations are made:
# 
# - Peru (representing South America), emerges as the top producer of sugarcane, which also constitutes the highest production volume among all food crops.
# - Africa, represented by South Africa, Namibia, Angola, Sao Tome and Principe, Morocco, Chad, Cape Verde, and Libya, dominates the list with the highest production volumes for rice, soybeans, sweet potatoes, oranges, apples, grapes, sunflower seeds, avocado, and tea, with Chad appearing twice for grapes and tea.
# -  Italy (represnting Europe) has the most appearance as highest producer for Yam, Meat and Cocoa beans, while United States (North America) follows with Maize and Tomatoes.
# - No country in Oceania is featured on this list.

# #### Max Production Years Per Crop
# In this section, we determine the years with the highest production for each food crop. By identifying peak production years, we uncover key periods of abundance and productivity for various agricultural commodities.

# In[120]:


dec_year_prod = world_food_prod.groupby(['Decade', 'Year']).sum(numeric_only=True)
dec_year_prod.reset_index(inplace=True)

max_production_years = {}
for crop in food_produce_columns:
    max_decade = dec_year_prod.loc[dec_year_prod[crop].idxmax(), 'Decade']
    max_year = dec_year_prod.loc[dec_year_prod[crop].idxmax(), 'Year']
    max_production = dec_year_prod.loc[dec_year_prod[crop].idxmax(), crop]
    max_production_years[crop] = {'Max_Decade': max_decade, 'Max_Year': max_year, 'Max_Production': max_production}

max_production_df = pd.DataFrame.from_dict(max_production_years, orient='index')
max_production_df1 = max_production_df.sort_values(by='Max_Production', ascending=False)


# In[124]:


decade_colors = {
    1960: '#1f77b4', 
    1970: '#9467bd',  
    1980: '#2ca02c',  
    1990: '#d62728', 
    2000: '#ff7f0e',  
    2010: '#17becf', 
    2020: '#bcbd22'   
}
legend_items = []

fig7 = go.Figure()
for index, row in max_production_df1.iterrows():
    color = decade_colors.get(row['Max_Decade'], 'gray')
    abbreviated_volume = abbreviate_number(row['Max_Production'])
    
    fig7.add_trace(go.Bar(
        x=[index], 
        y=[row['Max_Production']],
        showlegend=False,
        text=[f"<b>{int(row['Max_Year'])}</b> ({abbreviated_volume})"], 
        name=row['Max_Decade'],
        marker=dict(color=color), 
        textposition='outside', 
        textangle=-90, 
    ))
    if row['Max_Decade'] not in legend_items:
        legend_items.append(int(row['Max_Decade']))
        fig7.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=int(row['Max_Decade'])))
        
fig7.update_layout(
    title="<b>Highest Producing Year Per Food Crop</b>",
    xaxis_title=None,
    yaxis_title=None,
    legend_title="Decade",
    template="plotly_white",
    yaxis=dict(range=[0, max_production_df1['Max_Production'].max() * 1.4]),
    height=600,  # Specify the height of the plot
    width=800 
)

fig7.update_layout(uniformtext_minsize=6, uniformtext_mode='show')
fig7.show()


# In the provided figure, delineating the year of peak production for each crop, color-coded by their respective decades, the following trends are discerned:
# 
# - The year 2011 emerges as the pinnacle for sugarcane production, equally representing the highest production volume among all food crops.
# - The 2010s decade stands out as the period with the most frequent occurrences of record-high production for 8 out of the 22 observed crops, closely trailed by the 1990s and 2020s with 6 and 4 crops, respectively.
# - Notably, the year 2021 recurs most frequently as the peak production year for potatoes, maize, tomatoes, and sunflower seeds.

# #### Production Distribution by Income Classification
# This section examines the distribution of food production based on income classification. By analyzing production volumes across different income groups, we gain insights into the relationship between economic development and food production dynamics.

# In[122]:


fig_box = px.box(world_food_prod2, x='Income Classification', y='Total Production',
                 title='<b>Food Production Distribution by Income Classification</b>',
                 template=template_style, 
                 labels={'Income Classification': 'Income Classification', 'Total Production': 'Total Production'})
fig_box.update_layout(xaxis_title="", yaxis_title="")
fig_box.show()


# In the provided box plot, the following observations are made:
# 
# - The majority of global food production is attributed to upper-middle-income countries, with lower-middle-income countries closely following suit.
# - Low-income countries contribute the smallest share to world food production.

# ## Summary
# This project delves into the intricate dynamics of global food production, aiming to elucidate key trends, top producers, and regional contributions over the decades. Through comprehensive analysis of production trends, product rankings, and continent-wise distribution, the project sheds light on the evolving landscape of agricultural output.
# 
# **Key Findings**:
# 
# - Steady Growth in Production: Over the past six decades, global food production has witnessed a consistent upward trajectory, reflecting advancements in technology, agricultural practices, and evolving market demands.
# 
# - Diverse Crop Rankings: Sugar cane emerges as the leading crop in terms of production volume, underscoring its significance in the agricultural landscape. However, rankings vary across different crops, with staples like wheat, rice, and potatoes also holding significant shares.
# 
# - Continental Contributions: Africa and Asia emerge as the primary contributors to global food production, driven by factors such as arable land availability, technological advancements, and supportive policies. Other continents, including Europe, North America, and South America, also play crucial roles in shaping production trends.
# 
# - Food Classification Analysis: A detailed analysis of primary food classifications reveals distinct trends and top producers across categories such as cereals, fruits, and beverages, as well as a hierarchical order with sugars and sweeteners at the top and beverages at the bottom. Notable insights include the dominance of specific continents in particular food classes and the varying production volumes across sectors, with Africa, North America, and Asia emerging as most dominant continents.
# 
# - Region-wise Analysis: Examination of production distribution across income classifications highlights the significant contributions of upper-middle-income countries, closely followed by lower-middle-income countries. Low-income countries, although contributing the smallest share, play a crucial role in shaping global food production dynamics.
# 
# #### Implications and Future Directions:
# 
# Understanding the underlying trends and drivers of global food production is essential for policymakers, agricultural stakeholders, and international organizations to formulate effective strategies for sustainable agriculture and food security.
# 
# Future research endeavors could explore the impact of emerging technologies, climate change, and shifting consumer preferences on food production dynamics. Additionally, comparative analyses across regions and countries could offer deeper insights into disparities and opportunities in the global agricultural landscape.
# 
# By elucidating the intricacies of global food production dynamics, this project aims to contribute to broader discussions on sustainable development, food security, and the future of agriculture in an ever-changing world.
# 
# 

# In[129]:


markdown1 = '# Exploring Global Food Production Dynamics'
author = 'Author: Victory Elekwa'
markdown2 = """
### Introduction

In an era marked by unprecedented global challenges and the need for sustainable development, understanding and optimizing food production processes have become paramount. As we navigate complexities such as population growth, climate change, and resource scarcity, the role of data-driven insights in shaping agricultural strategies cannot be overstated.

This report explores an in-depth analysis of world food production trends, leveraging advanced data analytics techniques to uncover key patterns and insights. By examining production volumes, regional disparities, and emerging trends across various food categories, we aim to provide actionable insights for policymakers, agricultural stakeholders, and researchers alike.

Through a comprehensive exploration of global food production data spanning multiple decades, this report endeavors to shed light on the dynamics shaping our food systems. By harnessing the power of data analytics, we seek to empower decision-makers with the knowledge and tools necessary to foster a more sustainable and resilient food future.

Join us as we embark on a journey through the intricate web of food production, uncovering hidden trends, identifying opportunities for improvement, and paving the way towards a more nourished and prosperous world.
"""
markdown3 = """
#### Production Trends Over Time

This section explores the historical trends in global food production over time. By analyzing production volumes across different years, we aim to identify patterns, fluctuations, and long-term trends that have shaped food production dynamics.
"""
markdown4 = """
- From 1960 to 2010, world food production exhibited consistent growth, with an increase from 155.2 billion tonnes to 270.0 billion tonnes. 

- However, post-2010, a sharp decline occurred, plummeting to 55.2 billion units by 2020.

- The peak in 2010 marks an inflection point where growth reversed, necessitating further investigation into underlying factors.
"""
markdown5 = """
#### Explore the causes of the drop in production post 2010's

- In examining the factors contributing to the notable decline in production post-2010s, we start by querying the dataset for the past decade to pinpoint the inflection point and investigate potential causative factors.
"""
markdown6 = """
1. **Annual Production Trends insights**:
- In 2019, the world food production volume was 27.9 billion tonnes.
- In 2020, there was a slight increase, reaching 28 billion tonnes.
- However, in 2021, production dropped to 27.2 billion tonnes, a decrease of 800 million tonnes from 2020.

2. **Explanation for the Sharp Decline (2010s to 2020s)**:

The drastic drop from 270 billion tonnes in the 2010s to 55 billion tonnes in the 2020s can be attributed to the following:

- **Limited Data Points**: The dataset only covers two years (2020 and 2021) within the 2020 decade, whereas a standard decade comprises ten years (e.g., 2020 - 2029).
- **Sampling Bias**: With only two data points, the overall trend may not accurately represent the entire decade. It lacks the granularity needed to capture long-term patterns.
"""
markdown7 = """
##### World Food Production Trend (1961 - 2019)

To mitigate the sampling bias evident in the data, we generate a new dataframe that omits the 2020s decade. This revised dataset will be utilized specifically for decade-by-decade comparisons or analyses.
"""
markdown8 = """
- The overall trend shows a consistent increase in global food production over time.
- Starting around 155 billion tonnes in the 1960s, production steadily rises to approximately 270 billion tonnes by the 2010s.
"""
markdown9 = """
#### Product Ranking by Volume
In this section, we rank agricultural products based on their production volumes. By identifying the top-producing crops, we gain insights into the relative importance of different food commodities in global agricultural output.
"""
markdown10 = """
- Sugar Cane leads as the most produced crop, with an impressive volume of about 472.4 billion tonnes.

- Other significant agricultural products on the list of top producers (with volumes exceeding 100 billion tonnes) include wheat, rice, and potatoes.

- Least produced crops, with volumes below 10 billion tonnes, encompass meat, palm oil, peas, coffee, cocoa, tea, and avocado.

- Avocado holds the lowest production record, with just 1.2 billion tonnes over six decades.

- In the ranking by primary food categories, sugars and sweeteners emerge as the top performer, with an impressive production volume exceeding 472 billion tonnes, closely trailed by cereals boasting 420 billion tonnes.

- Despite avocado recording the lowest production among the products ranked, the fruit category secures the fourth position with 94.8 billion tonnes. This can be attributed to the higher production volumes of other fruit crops such as grapes, apples, bananas, and oranges, each exceeding 15.9 billion tonnes.

- The beverage category demonstrates the lowest production volume, totaling only 6.1 billion tonnes. This outcome aligns with the bottom five productions in the product ranking chart, primarily comprising coffee, cocoa, and tea.
"""
markdown11 = """
#### Production Trend of Top 5 Crops

Here, we delve deeper into the production trends of the top five agricultural crops. By examining how the production of these key crops has evolved over time, we uncover valuable insights into the changing dynamics of global food supply.
"""
markdown12 = """
- Sugar Cane exhibits a significant upward trajectory in production volume. Rising from a base of 56.8 billion tonnes in 1960 to a peak of 94.2 billion tonnes in 2010.

- Wheat, Rice and Potatoes show moderate growth over the years. However, they all experience a  minimal drop (11% - 20%) in the 1980's which was quickly corrected in the 1990's with over 40% increment in production

- Soybeans depict a fluctuating trend with no substantial increase in production volume.

- While Sugar Cane has experienced exponential growth, other crops have witnessed only marginal increases or have remained relatively stable.
"""
markdown13 = """
#### Food Production Distribution by Continent

This section investigates the distribution of food production across different continents. By analyzing production volumes by continent, we gain a comprehensive understanding of regional contributions to global food output.
"""

markdown14 = """
From the depicted figure illustrating the production distribution across continents, the following observations are noted:

- Africa holds the largest production share, closely followed by Asia, trailing by less than 7%.
- Subsequently, North America, Europe, South America, and Oceania follow in respective order.
- Oceania exhibits the smallest production share, amounting to 1.13%, which is 28% less than Africa's share.
"""
markdown15 = """
#### Production Trend by Continent

Here, we explore the production trends within individual continents. By examining how food production has evolved over time in different regions, we identify regional patterns and dynamics that shape the global food landscape.
"""

markdown16 = """
- Africa has witnessed a remarkable upward trajectory in production, possibly attributed to advancements in technology, ample availability of arable land, and supportive policies.

- Asia's production has also risen, albeit at a slower rate than Africa. Following closely is Europe, which demonstrated consistent production growth until the 1990s, after which it experienced marginal declines of around 9% in subsequent decades.

- North America and South America have both encountered fluctuations in production levels, with South America experiencing the most significant decline (approximately 26%) post-1990s.

- Oceania's production trend appears relatively stagnant, indicating stable yet stagnant market conditions. Potential factors contributing to this include limited investment, infrastructure constraints, and regional economic challenges.
"""
markdown17 = """
#### Production Trend by Primary Food Classification

In this section, we analyze the production trends of primary food classifications such as cereals, fruits, vegetables, and more. By examining production trends within each category, we gain insights into the dynamics of specific food groups.
"""
markdown18 = """
- The data illustrates a persistent uptrend in production volumes across primary food categories.

- It also demonstrates a consistent ordinal arrangement of these categories within each decade under examination.

- The hierarchical order is as follows: Sugars and sweeteners, Cereals, Roots & Tubers, Fruits, Oil seeds, Vegetables, Livestock, and Beverages.

- There were marginal shifts in the 1990s and 2000s, with oil seeds and fruits interchanging positions.
"""
markdown19 = """
#### Top Producers by Primary Food Classification

This section identifies the top producers within each primary food classification. By highlighting the leading producers in various food categories, we discern patterns of dominance and specialization across different regions.
"""

markdown20 = """
**Beverages**:
  - Asia leads global beverage production, with Maldives as the top producer, while Africa also holds a significant share.
  
**Cereals**:
  - North America, represented by Saint Lucia, United States, and Armenia, commands a strong presence in cereal production, while other continents have smaller shares.
  
**Fruits**:
  - Although Argentina leads in fruit production with 4.2 billion tonnes (South America), Africa, represented by Sao Tome and Principe, Chad, and Morocco, dominates with a combined volume of 9.5 billion tonnes.
  
**Livestock**:
  - Europe, particularly Italy, takes the lead in livestock production, with contributions from Africa and Asia.
  
**Oil Seeds**:
  - Africa, led by Namibia, dominates oilseed production.
  
**Roots and Tubers**:
  - Africa, notably Egypt, and North America play significant roles in roots and tubers production.
  
**Sugars and Sweeteners**:
  - South America, especially Peru and Argentina, leads in sugars and sweeteners production.
  
**Vegetables**:
  - North America, specifically the United States, Guatemala, and Dominican Republic, dominates global vegetable production.

In summary, Africa, North America, and Asia emerge as dominant continents across various food classes, with other continents contributing to specific categories. No country within the Oceania continent ranks in the top five for each food class.
"""
markdown21 = """
#### Top Producers by Food Crops

Here, we identify the top producers for individual food crops. By examining production volumes and leading producers for each crop, we gain insights into regional strengths and specialization in agricultural production.
"""

markdown22 = """
In the figure above, designed to illustrate the country with the highest production volume per food crop, color-coded by continent, the following observations are made:

- Peru (representing South America), emerges as the top producer of sugarcane, which also constitutes the highest production volume among all food crops.
- Africa, represented by South Africa, Namibia, Angola, Sao Tome and Principe, Morocco, Chad, Cape Verde, and Libya, dominates the list with the highest production volumes for rice, soybeans, sweet potatoes, oranges, apples, grapes, sunflower seeds, avocado, and tea, with Chad appearing twice for grapes and tea.
-  Italy (represnting Europe) has the most appearance as highest producer for Yam, Meat and Cocoa beans, while United States (North America) follows with Maize and Tomatoes.
- No country in Oceania is featured on this list.
"""
markdown23 = """
#### Max Production Years Per Crop

In this section, we determine the years with the highest production for each food crop. By identifying peak production years, we uncover key periods of abundance and productivity for various agricultural commodities.
"""
markdown24 = """
In the provided figure, delineating the year of peak production for each crop, color-coded by their respective decades, the following trends are discerned:

- The year 2011 emerges as the pinnacle for sugarcane production, equally representing the highest production volume among all food crops.
- The 2010s decade stands out as the period with the most frequent occurrences of record-high production for 8 out of the 22 observed crops, closely trailed by the 1990s and 2020s with 6 and 4 crops, respectively.
- Notably, the year 2021 recurs most frequently as the peak production year for potatoes, maize, tomatoes, and sunflower seeds.
"""
markdown25 = """
#### Production Distribution by Income Classification

This section examines the distribution of food production based on income classification. By analyzing production volumes across different income groups, we gain insights into the relationship between economic development and food production dynamics.
"""
markdown26 = """
In the provided box plot, the following observations are made:

- The majority of global food production is attributed to upper-middle-income countries, with lower-middle-income countries closely following suit.
- Low-income countries contribute the smallest share to world food production.
"""
markdown27 = """
## Summary

This project delves into the intricate dynamics of global food production, aiming to elucidate key trends, top producers, and regional contributions over the decades. Through comprehensive analysis of production trends, product rankings, and continent-wise distribution, the project sheds light on the evolving landscape of agricultural output.

**Key Findings**:

- Steady Growth in Production: Over the past six decades, global food production has witnessed a consistent upward trajectory, reflecting advancements in technology, agricultural practices, and evolving market demands.

- Diverse Crop Rankings: Sugar cane emerges as the leading crop in terms of production volume, underscoring its significance in the agricultural landscape. However, rankings vary across different crops, with staples like wheat, rice, and potatoes also holding significant shares.

- Continental Contributions: Africa and Asia emerge as the primary contributors to global food production, driven by factors such as arable land availability, technological advancements, and supportive policies. Other continents, including Europe, North America, and South America, also play crucial roles in shaping production trends.

- Food Classification Analysis: A detailed analysis of primary food classifications reveals distinct trends and top producers across categories such as cereals, fruits, and beverages, as well as a hierarchical order with sugars and sweeteners at the top and beverages at the bottom. Notable insights include the dominance of specific continents in particular food classes and the varying production volumes across sectors, with Africa, North America, and Asia emerging as most dominant continents.

- Region-wise Analysis: Examination of production distribution across income classifications highlights the significant contributions of upper-middle-income countries, closely followed by lower-middle-income countries. Low-income countries, although contributing the smallest share, play a crucial role in shaping global food production dynamics.

#### Implications and Future Directions:

Understanding the underlying trends and drivers of global food production is essential for policymakers, agricultural stakeholders, and international organizations to formulate effective strategies for sustainable agriculture and food security.

Future research endeavors could explore the impact of emerging technologies, climate change, and shifting consumer preferences on food production dynamics. Additionally, comparative analyses across regions and countries could offer deeper insights into disparities and opportunities in the global agricultural landscape.

By elucidating the intricacies of global food production dynamics, this project aims to contribute to broader discussions on sustainable development, food security, and the future of agriculture in an ever-changing world.

"""


# In[130]:


report = dp.Report(
    dp.Text(markdown1),
    dp.Text(author),
    dp.Text(markdown2),
    dp.Text(markdown3),
    dp.Plot(fig_trend),
    dp.Text(markdown4),
    dp.Text(markdown5),
    dp.Plot(fig1),
    dp.Text(markdown6),
    dp.Text(markdown7),
    dp.Plot(fig_trend1),
    dp.Text(markdown8),
    dp.Text(markdown9),
    dp.Plot(fig2),
    dp.Plot(fig_primary),
    dp.Text(markdown10),
    dp.Text(markdown11),
    dp.Plot(fig_top5),
    dp.Text(markdown12),
    dp.Text(markdown13),
    dp.Plot(fig_pie1),
    dp.Text(markdown14),
    dp.Text(markdown15),
    dp.Plot(fig3),
    dp.Text(markdown16),
    dp.Text(markdown17),
    dp.Plot(fig4),
    dp.Text(markdown18),
    dp.Text(markdown19),
    dp.Plot(fig5),
    dp.Text(markdown20),
    dp.Text(markdown21),
    dp.Plot(fig6),
    dp.Text(markdown22),
    dp.Text(markdown23),
    dp.Plot(fig7),
    dp.Text(markdown24),
    dp.Text(markdown25),
    dp.Plot(fig_box),
    dp.Text(markdown26),
    dp.Text(markdown27),
)

# Save the report
report.save(path='report2.html', open=True)

