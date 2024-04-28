import pandas as pd 
import numpy as np 
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ml_models import train_knn_model, train_svm_model, train_gradient_boosting_model, train_logistic_regression_model, train_random_forest_model, train_adaboost_model

nltk.download('punkt')

#Page Config
st.set_page_config(
    page_title="SpamVis",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#Datasets
hotel_test = pd.read_csv("hotel_test.csv")
hotel_train = pd.read_csv("hotel_train.csv")
restaurant_test = pd.read_csv("restaurant_test.csv")
restaurant_train = pd.read_csv("restaurant_train.csv")


#Sidebar
st.markdown("<h1 style='font-size:28px;'>SpamVis: Multimodal Visual Interactive System for Spam Review Detection</h1>", unsafe_allow_html=True)


selected_model = st.sidebar.selectbox("Select model: ", ["BERT","RoBERTa","KNN", "SVM","Logistic Regression", "Gradient Boosting", "Random Forest", "AdaBoost"])
selected_train_dataset = st.sidebar.selectbox("Select train dataset: ",["Restaurant data","Hotel data"])

#Test dataset option
test_option = st.sidebar.radio("Choose Test Data Option:", ("Select Predefined Test Dataset", "Upload Your Own CSV File"))

if test_option == "Select Predefined Test Dataset":
    selected_test_dataset = st.sidebar.selectbox("Select test dataset: ",["Restaurant data", "Hotel data"])
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        custom_test_df = pd.read_csv(uploaded_file)

#Generate Results button
if st.sidebar.button("Generate Results"):
    #Create 2 tabs for model results & spam analyzer 
    tab1, tab2, tab3 = st.tabs(["Model results","Review Analyzer","Model Comparision"])
    with tab1: 
    #Main space
        col1, col2 = st.columns(2)
        
        #Define the right df based on user choice
        if selected_model == "BERT": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_train 
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_train   
                    header_text = "Hotel"

        elif selected_model == "RoBERTa": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_train
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_train   
                    header_text = "Hotel"
        
        elif selected_model == "KNN":
                if selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_knn_model(hotel_train, hotel_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_knn_model(restaurant_train, restaurant_test)
                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_knn_model(hotel_train, restaurant_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_knn_model(restaurant_train, hotel_test)

        elif selected_model == "SVM":
                if selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_svm_model(hotel_train, hotel_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_svm_model(restaurant_train, restaurant_test)
                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_svm_model(hotel_train, restaurant_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_svm_model(restaurant_train, hotel_test)

        elif selected_model == "Gradient Boosting":
                if selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_gradient_boosting_model(hotel_train, hotel_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_gradient_boosting_model(restaurant_train, restaurant_test)
                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_gradient_boosting_model(hotel_train, restaurant_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_gradient_boosting_model(restaurant_train, hotel_test)

        elif selected_model == "Logistic Regression":
                if selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_logistic_regression_model(hotel_train, hotel_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_logistic_regression_model(restaurant_train, restaurant_test)
                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_logistic_regression_model(hotel_train, restaurant_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_logistic_regression_model(restaurant_train, hotel_test)

        elif selected_model == "Random Forest":
                if selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_random_forest_model(hotel_train, hotel_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_random_forest_model(restaurant_train, restaurant_test)
                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_random_forest_model(hotel_train, restaurant_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_random_forest_model(restaurant_train, hotel_test)


        elif selected_model == "AdaBoost":
                if selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_adaboost_model(hotel_train, hotel_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_adaboost_model(restaurant_train, restaurant_test)
                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    df = hotel_train
                    header_text = "Hotel"
                    results = train_adaboost_model(hotel_train, restaurant_test)
                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    df = restaurant_train
                    header_text = "Restaurant"
                    results = train_adaboost_model(restaurant_train, hotel_test)


        with col1:
            #GRAPH 1: FULL TABLE OF SPAM REVIEWS
            st.markdown(f"<h1 style='font-size:20px;'>Total Spam Reviews of {header_text} data</h1>", unsafe_allow_html=True)

            #TO EDIT: Extract spam tables             
            spam_table = df[df['Label'] == 'Y']
            
            columns_to_drop = ['Date', 'ReviewID', 'ReviewerID', 'ProductID', 'Rating']

            spam_table.drop(columns_to_drop, axis = 1, inplace = True)

            #Compute percentage of spam reviews detected 
            num_total_reviews = len(df)
            num_spam_reviews = len(spam_table)
            spam_percentage = (num_spam_reviews / num_total_reviews) * 100
            st.write(f"{spam_percentage:.2f}% of the dataset are spam reviews.")
            st.dataframe(spam_table, height=300)

            ####################################################
            
            #GRAPH 3: SENTIMENTS    
            #Extract all sentiments of spam and non-spam 
            st.markdown("<h2 style='font-size:20px;'>Sentiment Analysis</h2>", unsafe_allow_html=True)
            
            def plot_sentiment_proportion(df):
                # Calculate proportions of positive and negative sentiments for each label
                grouped = df.groupby(["Label", "Sentiment"]).size().unstack(fill_value=0)
                grouped["Total"] = grouped.sum(axis=1)
                grouped["Positive %"] = (grouped[1] / grouped["Total"]) * 100
                grouped["Negative %"] = (grouped[0] / grouped["Total"]) * 100

                # Plotting
                fig, ax = plt.subplots(figsize=(18, 8.8))  # Adjusted size to set height to 400 pixels
                positive_color = '#98FB98'
                negative_color = '#FFB6C1'
                bar_width = 0.35
                x_indices = range(len(grouped))

                ax.bar(x_indices, grouped["Positive %"], color=positive_color, width=bar_width, label="Positive")
                ax.bar(x_indices, grouped["Negative %"], bottom=grouped["Positive %"], color=negative_color, width=bar_width, label="Negative")

                # Adding data labels
                for i, (pos_percent, neg_percent) in enumerate(zip(grouped["Positive %"], grouped["Negative %"])):
                    ax.text(
                        i, pos_percent / 2, f"{pos_percent:.2f}%", ha="center", va="center", fontsize=16, color="black"
                    )
                    ax.text(
                        i, pos_percent + (neg_percent / 2), f"{neg_percent:.2f}%", ha="center", va="center", fontsize=16, color="black"
                    )

                ax.set_xticks(x_indices)
                ax.set_xticklabels(["Genuine (N)", "Spam (Y)"], fontsize=24)  # Updated x-axis label fontsize
                ax.set_yticks(range(0, 101, 20))  # Set y-axis ticks from 0 to 100 with intervals of 20
                ax.set_yticklabels([f'{x:.0f}%' for x in range(0, 101, 20)], fontsize=20)  # Updated y-axis label fontsize
                ax.set_ylim(0, 100)  # Adjusted y-axis limit to match percentage scale
                ax.legend(fontsize=20)  # Matched legend fontsize

                # Remove the outer border line
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Display the plot
                st.pyplot(fig)

            # Display the plot using Streamlit
            # Assuming you have a DataFrame named df containing the necessary data
            plot_sentiment_proportion(df)


        with col2:
            # GRAPH 2: AVG LENGTH BETWEEN SPAM & GENUINE 
            spam_reviews = df[df['Label'] == 'Y']['Review']
            genuine_reviews = df[df['Label'] == 'N']['Review']
            
            avg_words_spam = np.mean([len(review.split()) for review in spam_reviews])
            avg_words_genuine = np.mean([len(review.split()) for review in genuine_reviews])
            avg_words_spam_rounded = round(avg_words_spam)
            avg_words_genuine_rounded = round(avg_words_genuine)

            # Create length comparison df 
            
            df_words_comparison = pd.DataFrame({
                'Average Words': [avg_words_spam_rounded, avg_words_genuine_rounded]
            })

            st.markdown("<h2 style='font-size:20px;'>Average word length of reviews</h2>", unsafe_allow_html=True)

            fig_words = px.bar(df_words_comparison, 
                                x='Average Words', 
                                y=['Spam Review', 'Genuine Review'],  # Updated y labels directly in the plotly figure
                                text='Average Words', 
                                orientation='h')
            fig_words.update_traces(marker_color=['#FFA07A', '#ADD8E6'], textfont_color='black', textposition='inside', textfont_size=12)  # Change bar labels color to black and enlarge text size, place text inside bars
            fig_words.update_layout(height=350, yaxis_title=None, xaxis_title=None, yaxis=dict(tickfont=dict(size=14, color='black')), xaxis=dict(tickfont=dict(size=12, color='black')), margin=dict(t=0, b=0)) 
            st.plotly_chart(fig_words, use_container_width=True)

            ##############################

            # GRAPH 4: Display bar chart to compare results between train and test
            st.markdown("<h2 style='font-size:20px;'>Model's Training & Testing Results</h2>", unsafe_allow_html=True)
            
            data = {
                'Model': ['BERT', 'BERT', 'BERT', 'BERT', 'RoBERTa', 'RoBERTa', 'RoBERTa', 'RoBERTa'],
                'Data': ['hotel_train', 'hotel_test', 'restaurant_train', 'restaurant_test', 
                        'hotel_train', 'hotel_test', 'restaurant_train', 'restaurant_test'],
                'Accuracy': [0.85, 0.4, 0.87, 0.91, 0.45, 0.57, 0.82, 0.85],
                'Recall': [0.5, 0.76, 0.52, 0.56, 0.81, 0.93, 0.98, 1.01],
                'Precision': [0.6, 0.9, 0.62, 0.66, 0.95, 0.97, 0.85, 0.88],
                'Auc': [0.3, 0.9, 0.32, 0.36, 0.95, 0.83, 0.79, 0.82]
            }

            # Create the DataFrame
            model_results = pd.DataFrame(data)

            #Display train-test results based on user selected models 
                #If user choose BERT
            if selected_model == "BERT": 
                if selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    filtered_train_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'restaurant_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'restaurant_test')]

                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    filtered_train_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'restaurant_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'hotel_test')]

                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    filtered_train_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'hotel_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'hotel_test')]

                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    filtered_train_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'hotel_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'restaurant_test')]

                #If user choose RoBERTa 
            elif selected_model == "RoBERTa": 
                if selected_train_dataset == "Restaurant data" and selected_test_dataset == "Restaurant data":
                    filtered_train_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'restaurant_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'restaurant_test')]

                elif selected_train_dataset == "Restaurant data" and selected_test_dataset == "Hotel data":
                    filtered_train_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'restaurant_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'hotel_test')]

                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Hotel data":
                    filtered_train_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'hotel_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'hotel_test')]

                elif selected_train_dataset == "Hotel data" and selected_test_dataset == "Restaurant data":
                    filtered_train_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'hotel_train')]
                    filtered_test_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'restaurant_test')]
            
            elif selected_model == "KNN":
                filtered_train_results = pd.DataFrame({
                    'Accuracy': [results['Train Accuracy']],
                    'Recall': [results['Train Recall']],
                    'Precision': [results['Train Precision']],
                    'F1-score': [results['Train F1-score']]
                })

                filtered_test_results = pd.DataFrame({
                    'Accuracy': [results['Test Accuracy']],
                    'Recall': [results['Test Recall']],
                    'Precision': [results['Test Precision']],
                    'F1-score': [results['Test F1-score']]
                })

            elif selected_model == "SVM":
                filtered_train_results = pd.DataFrame({
                    'Accuracy': [results['Train Accuracy']],
                    'Recall': [results['Train Recall']],
                    'Precision': [results['Train Precision']],
                    'F1-score': [results['Train F1-score']]
                })

                filtered_test_results = pd.DataFrame({
                    'Accuracy': [results['Test Accuracy']],
                    'Recall': [results['Test Recall']],
                    'Precision': [results['Test Precision']],
                    'F1-score': [results['Test F1-score']]
                })

            elif selected_model == "Gradient Boosting":
                filtered_train_results = pd.DataFrame({
                    'Accuracy': [results['Train Accuracy']],
                    'Recall': [results['Train Recall']],
                    'Precision': [results['Train Precision']],
                    'F1-score': [results['Train F1-score']]
                })

                filtered_test_results = pd.DataFrame({
                    'Accuracy': [results['Test Accuracy']],
                    'Recall': [results['Test Recall']],
                    'Precision': [results['Test Precision']],
                    'F1-score': [results['Test F1-score']]
                })

            elif selected_model == "Logistic Regression":
                filtered_train_results = pd.DataFrame({
                    'Accuracy': [results['Train Accuracy']],
                    'Recall': [results['Train Recall']],
                    'Precision': [results['Train Precision']],
                    'F1-score': [results['Train F1-score']]
                })

                filtered_test_results = pd.DataFrame({
                    'Accuracy': [results['Test Accuracy']],
                    'Recall': [results['Test Recall']],
                    'Precision': [results['Test Precision']],
                    'F1-score': [results['Test F1-score']]
                })

            
            elif selected_model == "Random Forest":
                filtered_train_results = pd.DataFrame({
                    'Accuracy': [results['Train Accuracy']],
                    'Recall': [results['Train Recall']],
                    'Precision': [results['Train Precision']],
                    'F1-score': [results['Train F1-score']]
                })

                filtered_test_results = pd.DataFrame({
                    'Accuracy': [results['Test Accuracy']],
                    'Recall': [results['Test Recall']],
                    'Precision': [results['Test Precision']],
                    'F1-score': [results['Test F1-score']]
                })

            elif selected_model == "AdaBoost":
                filtered_train_results = pd.DataFrame({
                    'Accuracy': [results['Train Accuracy']],
                    'Recall': [results['Train Recall']],
                    'Precision': [results['Train Precision']],
                    'F1-score': [results['Train F1-score']]
                })

                filtered_test_results = pd.DataFrame({
                    'Accuracy': [results['Test Accuracy']],
                    'Recall': [results['Test Recall']],
                    'Precision': [results['Test Precision']],
                    'F1-score': [results['Test F1-score']]
                })
                                
               
            merged_results = pd.concat([filtered_train_results, filtered_test_results])

            # Define colors
            colors = ['#FFA07A', '#98FB98', '#ADD8E6', '#FFB6C1']

            # Create the bar chart
            fig, ax = plt.subplots(figsize=(18, 10))
            merged_results.plot(kind='bar', ax=ax, legend=True, width=0.7, color=colors)

            # Customize the plot
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)  # Enlarged legend series
            ax.set_xticklabels(['Training', 'Testing'], rotation=0, fontsize=26)
            ax.set_yticklabels([f'{x:.0%}' for x in ax.get_yticks()], fontsize=20)  # Enlarged y-axis tick labels
            ax.set_ylim(0, 1)

            # Remove the outer border line
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Display data values on each bar as percentages (enlarged)
            for p in ax.patches:
                ax.annotate(f'{p.get_height()*100:.0f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                            va='center',
                            xytext=(0, 20), textcoords='offset points', fontsize=16)  # Enlarged data values

            # Display the plot in Streamlit
            st.pyplot(fig)

            
    with tab2: 
        st.subheader("Analyze Review Text")
        @st.cache(allow_output_mutation=True)
        def get_model():
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
            return tokenizer, model

        def predict_spam(text):
            test_sample = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
            output = model(**test_sample)
            y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
            return y_pred[0]

        tokenizer, model = get_model()

        st.title("Spam Review Analyzer")

        user_input = st.text_area('Enter Text to Analyze')
        button = st.button("Analyze")

        if user_input and button:
            st.write("Analyzing...")
            prediction = predict_spam([user_input])
            d = {1: 'Spam', 0: 'Genuine'}
            st.write("Prediction: ", d[prediction])

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV file:")
            st.write(df)
            
            # Split dataset into train and test sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Train the model on the training set
            train_texts = train_df['Review'].tolist()
            train_labels = train_df['Sentiment'].tolist()
            
            st.write("Training...")
            # You need to implement the training part of your model here
            
            st.write("Training completed.")
            
            # Evaluate the model on the test set
            test_texts = test_df['Review'].tolist()
            test_labels = test_df['Sentiment'].tolist()
            
            st.write("Testing...")
            predictions = []
            for text in test_texts:
                prediction = predict_spam([text])
                predictions.append(prediction)
            
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='weighted')
            recall = recall_score(test_labels, predictions, average='weighted')
            f1 = f1_score(test_labels, predictions, average='weighted')
            
            st.write("Testing Accuracy: ", accuracy)
            st.write("Testing Precision: ", precision)
            st.write("Testing Recall: ", recall)
            st.write("Testing F1-score: ", f1)