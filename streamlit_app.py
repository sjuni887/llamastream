import streamlit as st
import pickle
import pandas as pd
import replicate
import os

# App title
st.set_page_config(page_title="Logistic Regression & Chatbot UI")

# Create tabs
tabs = st.sidebar.radio("Navigation", ("Logistic Regression Model", "Chatbot"))

# Define encoding mappings
Anemia_category_mapping = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
GradeofKidneydisease_mapping = {"g1": 1, "G2": 2, "G3a": 3.1, "G3b": 3.2, "G4": 4, "G5": 5}
SurgRiskCategory_mapping = {"Low": 1, "Moderate": 2, "High": 3}
ASAcategorybinned_mapping = {"I": 1, "II": 2, "III": 3, "IV-VI": 4}
RaceCategory_mapping = {"Chinese": 1, "Others": 2, "Indian": 3, "Malay": 4}
GENDER_mapping = {'MALE': 1, 'FEMALE': 0}
AnaestypeCategory_mapping = {'GA': 0, 'RA': 1}
PriorityCategory_mapping = {'Elective': 0, 'Emergency': 1}
RDW15_7_mapping = {'<= 15.7': 0, '>15.7': 1}

# Tab: Logistic Regression Model
if tabs == "Logistic Regression Model":
    st.title("Logistic Regression Model")

    # Load the saved logistic regression model
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Define function to preprocess input features
    def preprocess_features(features):
        features['Anemia category'] = Anemia_category_mapping.get(features['Anemia category'].lower(), 0)
        features['GradeofKidneydisease'] = GradeofKidneydisease_mapping.get(features['GradeofKidneydisease'].lower(), 1)
        features['SurgRiskCategory'] = SurgRiskCategory_mapping.get(features['SurgRiskCategory'].lower(), 1)
        features['ASAcategorybinned'] = ASAcategorybinned_mapping.get(features['ASAcategorybinned'].lower(), 1)
        features['GENDER'] = GENDER_mapping.get(features['GENDER'].upper(), 0)
        features['AnaestypeCategory'] = AnaestypeCategory_mapping.get(features['AnaestypeCategory'].upper(), 0)
        features['PriorityCategory'] = PriorityCategory_mapping.get(features['PriorityCategory'].upper(), 0)
        features['RDW15.7'] = RDW15_7_mapping.get(features['RDW15.7'].lower(), 0)
        features['RaceCategory'] = RaceCategory_mapping.get(features['RaceCategory'].lower(), 2)  # Default to 'Others' if not found
        return features

    # Define function to make predictions
    def predict_icu(input_features):
        input_features_processed = preprocess_features(input_features)
        input_features_processed = pd.DataFrame(input_features_processed, index=[0])
        prediction = model.predict(input_features_processed)
        probability = model.predict_proba(input_features_processed)[0][1]
        return prediction, probability

    # Input fields for each feature
    st.subheader("Enter Patient Details:")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=0, max_value=120, step=1, value=40)
        gender = st.selectbox("Gender", ("Male", "Female"))
        rcri_score = st.slider("RCRI score", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
        anemia_category = st.selectbox("Anemia Category", ("None", "Mild", "Moderate", "Severe"))
        preop_egfr_mdrd = st.slider("PreopEGFRMDRD", min_value=0.0, max_value=200.0, step=0.1, value=100.0)
    with col2:
        grade_of_kidney_disease = st.selectbox("Grade of Kidney Disease", ("G1", "G2", "G3a", "G3b", "G4", "G5"))
        anaestype_category = st.selectbox("Anesthesia Type Category", ("GA", "RA"))
        priority_category = st.selectbox("Priority Category", ("Elective", "Emergency"))
        surg_risk_category = st.selectbox("Surgical Risk Category", ("Low", "Moderate", "High"))
        race_category = st.selectbox("Race Category", ("Chinese", "Others", "Indian", "Malay"))
        rdw15_7 = st.selectbox("RDW15.7", ("<= 15.7", ">15.7"))
        asa_category_binned = st.selectbox("ASA Category Binned", ("I", "II", "III", "IV-VI"))

    # Prepare input features dictionary
    input_features = {
        'AGE': age,
        'GENDER': gender,
        'RCRI score': rcri_score,
        'Anemia category': anemia_category,
        'PreopEGFRMDRD': preop_egfr_mdrd,
        'GradeofKidneydisease': grade_of_kidney_disease,
        'AnaestypeCategory': anaestype_category,
        'PriorityCategory': priority_category,
        'SurgRiskCategory': surg_risk_category,
        'RaceCategory': race_category,
        'RDW15.7': rdw15_7,
        'ASAcategorybinned': asa_category_binned
    }

    # Prediction button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction, probability = predict_icu(input_features)
            st.write(f"**Prediction:** {prediction}", font_size=25)
            st.write(f"**Probability:** {probability:.2f}%", font_size=20)
            
# Tab: Chatbot
elif tabs == "Chatbot":
    st.title("🦙💬 Llama 2 Chatbot")
    # Your chatbot UI code here

    # Replicate Credentials
    with st.sidebar:
        st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
        if 'REPLICATE_API_TOKEN' in st.secrets:
            st.success('API key already provided!', icon='✅')
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ['REPLICATE_API_TOKEN'] = replicate_api

        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
        if selected_model == 'Llama2-7B':
            llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
        elif selected_model == 'Llama2-13B':
            llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
        st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
    def generate_llama2_response(prompt_input):
        string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
        output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                               input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                      "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
        return output

    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
