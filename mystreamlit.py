import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load data with error handling
try:
    df = pd.read_csv('cancer.csv')
except FileNotFoundError:
    st.error("Error: File not found. Please make sure the CSV file 'cancer.csv' exists.")

# Add caching mechanism for data loading
@st.cache
def load_data():
    df = pd.read_csv('cancer.csv')
    return df
    
# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Dataset 🫁",
    layout="centered",
    page_icon=" 🫁"
)

# Convert categorical to numeric values
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Define a consistent color palette
color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']

# Set page layout to wide for better display
st.set_page_config(layout="wide")

# App title and description
st.title('Lung Cancer Data Analysis')
st.write("""
This app provides insights into lung cancer patient data, showcasing several visualizations to understand 
the relationships and distributions within the data.
""")

# Sidebar for selecting visualization
st.sidebar.title("Select Visualization")
visualization = st.sidebar.selectbox("Choose a visualization:", 
                                     ["Data Table", "Lung Cancer Distribution", "Smoking Status", 
                                      "Age vs. Chronic Disease",
                                      "Age Distribution", "Smoking vs. Lung Cancer", 
                                      "Gender Distribution of Lung Cancer", "Gender and Smoking Status",
                                      "Categorical Variables Heatmap",
                                      "Count of Lung Cancer Cases by Multiple Factors", 
                                      "Interactive Filters", "Pair Plots"])

# Display the selected visualization
if visualization == "Data Table":
    st.write("""
    By: Rex Ponce
    """)
    st.write("""
    Github: https://github.com/itsmerex/mystreamlitFinal
    """)
    st.write("""
    *Data Source*: [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/lung-cancer-detection)  
    Note: '1' indicates 'No' and '2' indicates 'Yes' for categorical values except for Gender that '1' indicates 'Male' and '0' indicates 'Female' and lung cancer that '1' indicates 'Yes' and '0' indicates 'No' .
    """)
    st.dataframe(df)
   
elif visualization == "Lung Cancer Distribution":
    st.header('Lung Cancer Distribution')
    lung_cancer_counts = df['LUNGCANCER'].value_counts()
    fig1 = px.pie(names=lung_cancer_counts.index, values=lung_cancer_counts.values, title='Lung Cancer Distribution',
                  color_discrete_sequence=color_palette)
    st.plotly_chart(fig1)

elif visualization == "Smoking Status":
    st.header('Smoking Status')
    smoking_counts = df['SMOKING'].value_counts()
    fig2 = px.bar(x=smoking_counts.index, y=smoking_counts.values, labels={'x': 'Smoking Status', 'y': 'Count'}, 
                  title='Smoking Status Distribution', color=smoking_counts.index, color_discrete_sequence=color_palette)
    st.plotly_chart(fig2)

elif visualization == "Age vs. Chronic Disease":
    st.header('Age vs. Chronic Disease')
    fig3 = px.scatter(df, x='AGE', y='CHRONIC DISEASE', color='LUNGCANCER', title='Age vs. Chronic Disease',
                      color_discrete_sequence=color_palette)
    st.plotly_chart(fig3)

elif visualization == "Age Distribution":
    st.header('Age Distribution')
    fig4 = px.histogram(df, x='AGE', nbins=20, title='Age Distribution', color_discrete_sequence=color_palette)
    st.plotly_chart(fig4)

elif visualization == "Smoking vs. Lung Cancer":
    st.header('Smoking vs. Lung Cancer')
    smoking_lung_cancer = df.groupby(['SMOKING', 'LUNGCANCER']).size().reset_index(name='counts')
    fig5 = px.bar(smoking_lung_cancer, x='SMOKING', y='counts', color='LUNGCANCER', barmode='group', 
                  title='Smoking vs. Lung Cancer', color_discrete_sequence=color_palette)
    st.plotly_chart(fig5)

elif visualization == "Gender Distribution of Lung Cancer":
    st.header('Gender Distribution of Lung Cancer')
    gender_lung_cancer = df.groupby(['GENDER', 'LUNGCANCER']).size().reset_index(name='counts')
    fig6 = px.bar(gender_lung_cancer, x='GENDER', y='counts', color='LUNGCANCER', barmode='group', 
                  title='Gender Distribution of Lung Cancer', color_discrete_sequence=color_palette)
    st.plotly_chart(fig6)

elif visualization == "Gender and Smoking Status":
    st.header('Gender and Smoking Status')
    gender_smoking = df.groupby(['GENDER', 'SMOKING']).size().reset_index(name='counts')
    fig7 = px.bar(gender_smoking, x='GENDER', y='counts', color='SMOKING', barmode='group', 
                  title='Gender and Smoking Status', color_discrete_sequence=color_palette)
    st.plotly_chart(fig7)

elif visualization == "Categorical Variables Heatmap":
    st.header('Heatmap of Categorical Variables')
    st.header('Correlation Heatmap')
    st.write(""" The correlation heatmap visualizes the correlation between different features in the dataset. 
             The values in the heatmap range from 0 to 1, where values of 0.5 and above indicate a strong correlation between the features.""")
    plt.figure(figsize=(12, 10))
    categorical_cols = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                        'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                        'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNGCANCER']
    sns.heatmap(pd.get_dummies(df[categorical_cols]).corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

elif visualization == "Count of Lung Cancer Cases by Multiple Factors":
    st.header('Count of Lung Cancer Cases by Multiple Factors')
    factor_counts = df.groupby(['GENDER', 'SMOKING', 'LUNGCANCER']).size().reset_index(name='counts')
    fig9 = px.bar(factor_counts, x='GENDER', y='counts', color='LUNGCANCER', barmode='group', facet_col='SMOKING',
                  title='Count of Lung Cancer Cases by Gender and Smoking Status', color_discrete_sequence=color_palette)
    st.plotly_chart(fig9)

elif visualization == "Interactive Filters":
    st.header('Interactive Filters')
    gender_filter = st.sidebar.multiselect("Select Gender:", options=df['GENDER'].unique(), default=df['GENDER'].unique())
    smoking_filter = st.sidebar.multiselect("Select Smoking Status:", options=df['SMOKING'].unique(), default=df['SMOKING'].unique())
    filtered_data = df[(df['GENDER'].isin(gender_filter)) & (df['SMOKING'].isin(smoking_filter))]
    st.dataframe(filtered_data)
    st.write(f"Filtered data contains {filtered_data.shape[0]} rows.")

elif visualization == "Pair Plots":
    st.header('Pair Plots')
    sns.pairplot(df, hue='LUNGCANCER', palette=['#3498db', '#e74c3c'])
    st.pyplot(plt)
    
# Add interactive elements - Slider for age filtering
age_range = st.sidebar.slider("Select Age Range", min_value=df['AGE'].min(), max_value=df['AGE'].max(),
                              value=(df['AGE'].min(), df['AGE'].max()))
filtered_data = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]

    
# Narrative and insights
st.sidebar.title("About the Data")
st.sidebar.write("""
The data represents various attributes associated with lung cancer patients, including demographic 
information, health conditions, and habits. The visualizations help in understanding the distributions and 
correlations among these attributes, potentially aiding in identifying key factors contributing to lung cancer.
""")

st.sidebar.write("""
- **Pie Chart**: Shows the distribution of lung cancer cases.
- **Bar Chart**: Highlights the smoking status of the patients.
- **Scatter Plot**: Demonstrates the relationship between age and chronic disease status.
- **Heatmap**: Displays the correlation between different attributes in the dataset.
- **Gender Distribution**: Compares the distribution of lung cancer cases between males and females.
- **Gender and Smoking Status**: Analyzes the smoking status among males and females.
""")

# Run the app with the command `streamlit run mystreamlit.py`
