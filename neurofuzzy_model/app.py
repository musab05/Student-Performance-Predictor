import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    try:
        from tensorflow.keras.losses import mean_squared_error
        mse = mean_squared_error
    except:
        try:
            mse = tf.keras.losses.MSE
        except:
            try:
                mse = tf.keras.losses.mse
            except:
                mse = 'mse'
                
    try:
        from tensorflow.keras.metrics import mean_absolute_error
        mae = mean_absolute_error
    except:
        try:
            mae = tf.keras.metrics.MAE
        except:
            try:
                mae = tf.keras.metrics.mae
            except:
                mae = 'mae'
except:
    st.error("TensorFlow not available. Will use mock prediction model.")
    tf = None

EXPECTED_COLUMNS = [
    'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion',
    'engagement_level', 'Hours_Studied', 'Attendance', 'Parental_Involvement',
    'Access_to_Resources', 'Motivation_Level', 'Exam_Score', 'absence_days',
    'extracurricular_activities', 'weekly_self_study_hours'
]

def load_models():
    """Load the trained hybrid model and scaler with fallback options"""
    models = {
        'hybrid': None,
        'scaler': None
    }
    
    # Load scaler
    if os.path.exists('./scaler.pkl'):
        try:
            with open('./scaler.pkl', 'rb') as f:
                models['scaler'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")
    
    if tf is not None and os.path.exists('./hybrid_model.h5'):
        custom_objects_options = [
            {'mse': mse, 'mae': mae},
            {'mean_squared_error': mse, 'mean_absolute_error': mae},
            {'MSE': mse, 'MAE': mae}
        ]
        
        for custom_objects in custom_objects_options:
            try:
                models['hybrid'] = load_model('./hybrid_model.h5', custom_objects=custom_objects)
                break
            except Exception as e:
                continue
    
    if models['hybrid'] is None:
        models['hybrid'] = create_mock_hybrid_model()
        st.warning("Using mock hybrid model.")
        
    if models['scaler'] is None:
        models['scaler'] = create_mock_scaler()
        st.warning("Using mock scaler.")
    
    return models

def create_mock_hybrid_model():
    """Create a simple mock hybrid model for fallback"""
    if tf is not None:
        # Input layer
        inputs = Input(shape=(len(EXPECTED_COLUMNS),))
        
        # Neural network part
        x = Dense(32, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        
        # Output layer
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(0.001), loss=mse, metrics=[mae])
        return model
    else:
        # Return a simple function if TensorFlow is not available
        def mock_predict(input_data):
            # Simple rule-based prediction
            study_hours = input_data.get('Hours_Studied', 0)
            attendance = input_data.get('Attendance', 0)
            engagement = input_data.get('engagement_level', 0)
            
            base_score = 50 + (study_hours * 0.5) + (attendance * 0.3) + (engagement * 5)
            return min(max(base_score, 0), 100)
        
        return mock_predict

def create_mock_scaler():
    """Create a mock scaler that does simple normalization"""
    class MockScaler:
        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                X = X.values
            return X / 100
    
    return MockScaler()

def prepare_input_data(input_data):
    """Prepare input data with all expected features in correct order"""
    # Create DataFrame with all expected columns
    input_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    
    # Fill in provided values
    for col in EXPECTED_COLUMNS:
        if col in input_data:
            input_df[col] = [input_data[col]]
        else:
            # Provide default values for missing columns
            if col == 'Exam_Score':
                input_df[col] = [75]  # Default exam score
            elif col == 'engagement_level':
                input_df[col] = [1]  # Medium engagement
            else:
                input_df[col] = [0]  # Default to 0
    
    input_df = input_df[EXPECTED_COLUMNS]
    
    return input_df

def normalize_input(scaler, input_data):
    """Normalize input data with fallback options"""
    try:
        input_df = prepare_input_data(input_data)
        
        if scaler is not None:
            try:
                return scaler.transform(input_df)
            except Exception as e:
                st.warning(f"Scaler transformation failed: {e}. Using simple normalization.")
                return simple_normalization(input_df)
        else:
            return simple_normalization(input_df)
    except Exception as e:
        st.error(f"Error normalizing input: {e}")
        return np.array([list(input_data.values())])

def simple_normalization(input_df):
    """Simple normalization fallback"""
    normalized_df = input_df.copy()
    for col in normalized_df.columns:
        if col in ['raisedhands', 'VisITedResources', 'AnnouncementsView', 
                 'Discussion', 'Attendance', 'Exam_Score']:
            normalized_df[col] = normalized_df[col] / 100
        elif col in ['Hours_Studied', 'weekly_self_study_hours']:
            normalized_df[col] = normalized_df[col] / 50
        elif col in ['absence_days']:
            normalized_df[col] = normalized_df[col] / 30
        elif col in ['engagement_level', 'Parental_Involvement', 
                   'Access_to_Resources', 'Motivation_Level']:
            normalized_df[col] = normalized_df[col] / 2
        elif col in ['extracurricular_activities']:
            normalized_df[col] = normalized_df[col] 
    return normalized_df.values

def predict_with_hybrid(model, scaler, input_data):
    """Make performance prediction with the hybrid model"""
    try:
        normalized_input = normalize_input(scaler, input_data)
        
        if callable(model) and not isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
            return model(input_data)
        else:
            prediction = model.predict(normalized_input)
            
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    return prediction[0][0]
                else:
                    return prediction[0]
            else:
                return prediction
    except Exception as e:
        st.error(f"Error during prediction with hybrid model: {e}")
        study_hours = input_data.get('Hours_Studied', 0)
        attendance = input_data.get('Attendance', 0)
        engagement = input_data.get('engagement_level', 0)
        
        base_score = 50 + (study_hours * 0.5) + (attendance * 0.3) + (engagement * 5)
        return min(max(base_score, 0), 100)

def get_engagement_remarks(input_data):
    """Generate remarks based on student engagement metrics"""
    remarks = []
    
    # Engagement level remarks
    engagement_level = input_data.get('engagement_level', 1)
    if engagement_level == 0:
        remarks.append("⚠️ Low engagement level detected. Consider strategies to increase student participation.")
    elif engagement_level == 2:
        remarks.append("✅ High engagement level observed. This is positively impacting performance.")
    else:
        remarks.append("🔄 Medium engagement level. There's room for improvement to boost performance.")
    
    raised_hands = input_data.get('raisedhands', 0)
    discussion = input_data.get('Discussion', 0)
    
    if raised_hands < 30 and discussion < 30:
        remarks.append("📉 Low class participation (raised hands and discussions). Encouraging more interaction could help.")
    elif raised_hands >= 70 and discussion >= 70:
        remarks.append("📈 Excellent class participation! The student is actively engaging in discussions.")
    
    visited_resources = input_data.get('VisITedResources', 0)
    announcements_view = input_data.get('AnnouncementsView', 0)
    
    if visited_resources < 30 and announcements_view < 30:
        remarks.append("🔍 Low resource utilization. Student may benefit from guidance on using available materials.")
    elif visited_resources >= 70 and announcements_view >= 70:
        remarks.append("📚 Strong resource utilization. Student is making good use of available materials.")
    
    return remarks

def get_additional_factors_remarks(input_data):
    """Generate remarks based on additional factors"""
    remarks = []
    
    hours_studied = input_data.get('Hours_Studied', 0)
    weekly_self_study = input_data.get('weekly_self_study_hours', 0)
    
    if hours_studied < 15 or weekly_self_study < 10:
        remarks.append("⏳ Study time may be insufficient. Consider increasing study hours gradually.")
    elif hours_studied >= 30 and weekly_self_study >= 20:
        remarks.append("⏱️ Substantial study time invested. Ensure proper balance with rest and activities.")
    
    attendance = input_data.get('Attendance', 0)
    absence_days = input_data.get('absence_days', 0)
    
    if attendance < 70:
        remarks.append("🚨 Low attendance rate. Regular class attendance is crucial for better performance.")
    elif attendance >= 90:
        remarks.append("🏆 Excellent attendance! This consistency contributes to learning.")
    
    if absence_days > 10:
        remarks.append(f"⚠️ High absence days ({absence_days}). Frequent absences may be affecting learning continuity.")
    
    extracurricular = input_data.get('extracurricular_activities', 0)
    if extracurricular == 0:
        remarks.append("🎭 No extracurricular activities reported. Balanced involvement can enhance overall development.")
    else:
        remarks.append("🤹 Participates in extracurricular activities. This balanced approach supports holistic growth.")
    
    parental_involvement = input_data.get('Parental_Involvement', 1)
    if parental_involvement == 0:
        remarks.append("👪 Low parental involvement. Increased support from home could benefit the student.")
    elif parental_involvement == 2:
        remarks.append("👪 High parental involvement. This strong support system is valuable for the student.")
    
    return remarks

def get_prediction_remarks(prediction, input_data):
    """Generate remarks based on prediction results"""
    remarks = []
    
    if prediction >= 85:
        remarks.append("🌟 Exceptional performance! Maintain these effective learning strategies.")
        if input_data.get('engagement_level', 1) < 2:
            remarks.append("💡 With even higher engagement, you might reach even greater heights!")
    elif prediction >= 70:
        remarks.append("👍 Solid performance. Focus on identified areas for further improvement.")
        if input_data.get('Hours_Studied', 0) < 25:
            remarks.append("📖 Consider increasing study hours slightly for better results.")
    elif prediction >= 60:
        remarks.append("🔄 Satisfactory but needs improvement. Focus on key areas like engagement and study habits.")
        if input_data.get('Attendance', 0) < 80:
            remarks.append("⏰ Improving attendance could help boost your performance.")
    else:
        remarks.append("🚧 Performance needs significant improvement. Focus on foundational areas first.")
        if input_data.get('Motivation_Level', 1) == 0:
            remarks.append("💪 Addressing motivation could be the first step to improvement.")
    
    # Specific suggestions based on input data
    if input_data.get('Access_to_Resources', 1) == 0:
        remarks.append("💻 Limited access to resources may be hindering performance. Explore available support options.")
    
    if input_data.get('Motivation_Level', 1) == 0:
        remarks.append("🔋 Low motivation detected. Identifying personal learning goals might help.")
    
    return remarks

def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="wide")
    
    st.title("Student Performance Predictor")
    st.write("""
    This application predicts student performance based on various factors using a hybrid neuro-fuzzy model.
    Fill in the form below to get a prediction.
    """)
    
    models = load_models()
    
    with st.form("prediction_form"):
        st.subheader("Student Engagement Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            raised_hands = st.slider("Raised Hands", 0, 100, 30)
            visited_resources = st.slider("Visited Resources", 0, 100, 25)
            announcements_view = st.slider("Announcements View", 0, 100, 15)
            discussion = st.slider("Discussion Participation", 0, 100, 25)
            engagement_level = st.selectbox("Engagement Level", options=[0, 1, 2], 
                                           format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x])
        
        with col2:
            hours_studied = st.slider("Hours Studied", 0, 50, 20)
            attendance = st.slider("Attendance (%)", 0, 100, 85)
            exam_score = st.slider("Previous Exam Score", 0, 100, 75)
            parental_involvement = st.selectbox("Parental Involvement", options=[0, 1, 2], 
                                               format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x])
            access_to_resources = st.selectbox("Access to Resources", options=[0, 1, 2], 
                                              format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x])
            motivation_level = st.selectbox("Motivation Level", options=[0, 1, 2], 
                                          format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x])
        
        st.subheader("Additional Factors")
        col3, col4 = st.columns(2)
        
        with col3:
            absence_days = st.slider("Absence Days", 0, 30, 5)
            extracurricular = st.selectbox("Participates in Extracurricular Activities", 
                                          options=[0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
            weekly_self_study = st.slider("Weekly Self-Study Hours", 0, 50, 15)
        
        submitted = st.form_submit_button("Predict Performance")
        
        if submitted:
            input_data = {
                'raisedhands': raised_hands,
                'VisITedResources': visited_resources,
                'AnnouncementsView': announcements_view,
                'Discussion': discussion,
                'engagement_level': engagement_level,
                'Hours_Studied': hours_studied,
                'Attendance': attendance,
                'Exam_Score': exam_score,
                'Parental_Involvement': parental_involvement,
                'Access_to_Resources': access_to_resources,
                'Motivation_Level': motivation_level,
                'absence_days': absence_days,
                'extracurricular_activities': extracurricular,
                'weekly_self_study_hours': weekly_self_study
            }
            
            with st.spinner("Predicting performance using hybrid neuro-fuzzy model..."):
                prediction = predict_with_hybrid(
                    models['hybrid'], 
                    models['scaler'], 
                    input_data
                )
            
            st.success(f"Predicted Performance Score: {prediction:.2f}/100")
            
            if prediction >= 85:
                performance_level = "Excellent"
                color = "green"
            elif prediction >= 70:
                performance_level = "Good"
                color = "blue"
            elif prediction >= 60:
                performance_level = "Satisfactory"
                color = "orange"
            else:
                performance_level = "Needs Improvement"
                color = "red"
            
            st.markdown(f"<h3 style='color: {color}'>Performance Level: {performance_level}</h3>", unsafe_allow_html=True)
            
            with st.expander("Engagement Analysis"):
                engagement_remarks = get_engagement_remarks(input_data)
                for remark in engagement_remarks:
                    st.write(f"- {remark}")
            
            with st.expander("Additional Factors Analysis"):
                additional_remarks = get_additional_factors_remarks(input_data)
                for remark in additional_remarks:
                    st.write(f"- {remark}")
            
            with st.expander("Performance Recommendations"):
                prediction_remarks = get_prediction_remarks(prediction, input_data)
                for remark in prediction_remarks:
                    st.write(f"- {remark}")

if __name__ == "__main__":
    main()
