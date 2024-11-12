import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY'],
)


#define a function to load the trained models
def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model("xgb_model.pkl")
randomforest_model = load_model("rf_model.pkl")
decisiontree_model = load_model("dt_model.pkl")
kneightbors_model = load_model("k_neighbors_model.pkl")
nb_model = load_model("nb_model.pkl")
SVM_model = load_model("svm_model.pkl")
voting_clf_model = load_model("voting_clf.pkl")
xgboostFeaturEngineering_model = load_model(
    "xgboost_feature_engineering_model.pkl")
xgboostSmote_model = load_model("xgboost_smote_model.pkl")
gb_model = load_model("gb_model.pkl")
sc_model = load_model("sc_model.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_of_products, has_credit_card, is_active_member,
                  estimated_salary):

  input_dict = {
      "CreditScore": credit_score,
      "Age": age,
      "Tenure": tenure,
      "Balance": balance,
      "NumOfProducts": num_of_products,
      "HasCrCard": int(has_credit_card),
      "IsActiveMember": int(is_active_member),
      "EstimatedSalary": estimated_salary,
      "Geography_France": 1 if location == "France" else 0,
      "Geography_Germany": 1 if location == "Germany" else 0,
      "Geography_Spain": 1 if location == "Spain" else 0,
      "Gender_Male": 1 if gender == "Male" else 0,
      "Gender_Female": 1 if gender == "Female" else 0
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def make_predictions(input_df, input_dict):

  probabilites = {
      "xgboost": xgboost_model.predict_proba(input_df)[0][1],
      "randomforest": randomforest_model.predict_proba(input_df)[0][1],
      "kneightbors": kneightbors_model.predict_proba(input_df)[0][1],
      # "decisiontree":
      # decisiontree_model.predict_proba(input_df)[0][1],
      # "nb":
      # nb_model.predict_proba(input_df)[0][1],
      # # "SVC":
      # # SVM_model.predict_proba(input_df)[0][1],
      # "voting_clf":
      # voting_clf_model.predict_proba(input_df)[0][1],
      # "xgboostFeaturEngineering":
      # xgboostFeaturEngineering_model.predict_proba(input_df)[0][1],
      # "xgboostSmote":
      # xgboostSmote_model.predict_proba(input_df)[0][1]
  }

  average_probability = np.mean(list(probabilites.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(average_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        f"The customer has a {average_probability:.2%} probability of churning."
    )

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilites)
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown("### Model Probabilties")
  for model, prob in probabilites.items():
    st.write(f"{model} : {prob}")
  st.write(f"Average_Probability: {average_probability}")

  return average_probability


def explain_prediction(probability, input_dict, surname):
  prompt = f""" You are an expert data scientist at HS bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100,1)}% probability of churning based on the information provided below.

    Here's the customer's infromation:
    {input_dict}



    Here are the machine learning model's top 10 most important features fro predicting churn:

        Feature	          | Importance
       ----------------	  | ---------------
        NumOfProducts	  | 0.323888
        IsActiveMember	  | 0.164146
        Age	              | 0.109550
        Geography_Germany |	0.091373
        Balance           |	0.052786
        Geography_France  |	0.046463
        Gender_Female	  | 0.045283
        Geography_Spain	  | 0.036855
        CreditScore	      | 0.035005
        EstimatedSalary	  | 0.032655
        HasCrCard	      | 0.031940
        Tenure	          | 0.030054
        Gender_Male	      | 0.000000

    {pd.set_option('display.max_columns',None)}

    Here are the summary statistics for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are the summary statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

    -If the customer has over a 40% risk of churning,generate a three senetnce detailed explanation of why they are at risk of churning.

    -If the customer has less than a 40% risk of churning,generate a three senetnce detailed explanation of why they might not be at risk of churning.

    -your explanation should be based on the customer's infromation, the summary statiustics of churned and non-churned customers and feature importances provided.

    Important information including any headings in the explanation should be decorated in bold .

    -Don't mention the probability of churning or machine learning model or say anything like "Based on the machine learning model's prediction and top 10 most important features",just explain the prediction.
    """
  print("Explanation Prompt:", prompt)

  raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }])
  return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
  prompt = f""" You are a manager at HS bank. You are responsible for ensuring customer's stay with the bank and are incentivized with varios offers.

    Your noticed a customer named {surname} has a {round(probability * 100,1)}% probability of churning.

    Here's the customer's infromation:
    {input_dict}



    Here is some explanation as to why the customer might be at risk of churning:

        {explanation}

    Generate an email to the customer based on the their information,asking them to stay if they are at risk of churning or offering them incentives so that they become more loyal to the bank.

    Important information in the email should be decorated in bold .

    Make sure to list out a set of incentives for the respective customer to stay with the bank based on their information in bullet point format.

    Each bullet point need to be seperated .

    Don't ever mention the probability of churning or machine learning models to the customer.
    """

  raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }])

  print("\n\nEmail Prompt:", prompt)

  return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")
df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']}-{row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split("-")[0])

  print("seleted customer id", selected_customer_id)

  selected_customer_surname = selected_customer_option.split("-")[1]

  print("seleted customer surname", selected_customer_surname)

  selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

  print("selected customer", selected_customer)

  col1, col2 = st.columns(2)
  with col1:
    credit_score = st.number_input("Credit Score",
                                   min_value=300,
                                   max_value=850,
                                   value=int(selected_customer["CreditScore"]))

    location = st.selectbox("location", ["Spain", "France", "Germany"],
                            index=["Spain", "France", "Germany"
                                   ].index(selected_customer["Geography"]))

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer["Gender"] == "Male" else 1)

    age = st.number_input("Age",
                          min_value=18,
                          max_value=100,
                          value=int(selected_customer["Age"]))

    tenure = st.number_input("Tenure (years)",
                             min_value=0,
                             max_value=50,
                             value=int(selected_customer["Tenure"]))
    with col2:

      balance = st.number_input("Balance",
                                min_value=0.0,
                                value=float(selected_customer["Balance"]))

      num_of_products = st.number_input(
          "Number Of Products",
          min_value=1,
          max_value=10,
          value=int(selected_customer["NumOfProducts"]))

      has_credit_card = st.checkbox("Has Credit Card",
                                    value=bool(selected_customer["HasCrCard"]))
      is_active_member = st.checkbox("Is Active Member",
                                     value=bool(
                                         selected_customer["IsActiveMember"]))

      estimated_salary = st.number_input(
          "Estimated_Salary",
          min_value=0.0,
          value=float(selected_customer["EstimatedSalary"]))
  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, num_of_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)
  average_probability = make_predictions(input_df, input_dict)

  explanation = explain_prediction(average_probability, input_dict,
                                   selected_customer['Surname'])

  st.markdown("---------")
  st.subheader("Explanation of Prediction:")
  st.markdown(explanation)
  email = generate_email(average_probability, input_dict, explanation,
                         selected_customer['Surname'])

  st.markdown("---------")
  st.subheader("Personalized Email:")
  st.markdown(email)
