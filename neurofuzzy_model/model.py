import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pickle

def load_and_preprocess():
    df1 = pd.read_csv('../data/xAPI-Edu-Data.csv')
    df2 = pd.read_csv('../data/StudentPerformanceFactors.csv')
    df3 = pd.read_csv('../data/student-scores.csv')
    
    
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()

    le = LabelEncoder()
    categorical_cols = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 
                       'SectionID', 'Topic', 'Semester', 'Relation', 
                       'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 
                       'StudentAbsenceDays', 'Class']
    for col in categorical_cols:
        df1[col] = le.fit_transform(df1[col])

    categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 
                       'Extracurricular_Activities', 'Motivation_Level', 
                       'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                       'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                       'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    for col in categorical_cols:
        df2[col] = le.fit_transform(df2[col])

    weights = {
        'math_score': 0.2,
        'history_score': 0.1,
        'physics_score': 0.15,
        'chemistry_score': 0.15,
        'biology_score': 0.15,
        'english_score': 0.15,
        'geography_score': 0.1
    }
    df3['overall_score'] = sum(df3[col] * weight for col, weight in weights.items())

    categorical_cols = ['gender', 'part_time_job', 'extracurricular_activities', 'career_aspiration']
    for col in categorical_cols:
        df3[col] = le.fit_transform(df3[col])

    df1_features = df1[['raisedhands', 'VisITedResources', 'AnnouncementsView', 
                       'Discussion', 'Class']].rename(columns={'Class': 'engagement_level'})
    df2_features = df2[['Hours_Studied', 'Attendance', 'Parental_Involvement', 
                       'Access_to_Resources', 'Motivation_Level', 'Exam_Score']]
    df3_features = df3[['absence_days', 'extracurricular_activities', 
                       'weekly_self_study_hours', 'overall_score']]

    combined_df = pd.concat([df1_features, df2_features, df3_features], axis=1)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(combined_df.drop('overall_score', axis=1))
    scaled_df = pd.DataFrame(scaled_features, columns=combined_df.columns[:-1])
    scaled_df['overall_score'] = combined_df['overall_score']

    return scaled_df, scaler

def create_fuzzy_system():
    study_hours = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'study_hours')
    attendance = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'attendance')
    engagement = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'engagement')
    performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance')

    study_hours['low'] = fuzz.trimf(study_hours.universe, [0, 0, 0.5])
    study_hours['medium'] = fuzz.trimf(study_hours.universe, [0, 0.5, 1])
    study_hours['high'] = fuzz.trimf(study_hours.universe, [0.5, 1, 1])

    attendance['low'] = fuzz.trimf(attendance.universe, [0, 0, 0.5])
    attendance['medium'] = fuzz.trimf(attendance.universe, [0, 0.5, 1])
    attendance['high'] = fuzz.trimf(attendance.universe, [0.5, 1, 1])

    engagement['low'] = fuzz.trimf(engagement.universe, [0, 0, 0.5])
    engagement['medium'] = fuzz.trimf(engagement.universe, [0, 0.5, 1])
    engagement['high'] = fuzz.trimf(engagement.universe, [0.5, 1, 1])

    performance['poor'] = fuzz.trimf(performance.universe, [0, 0, 50])
    performance['average'] = fuzz.trimf(performance.universe, [0, 50, 100])
    performance['good'] = fuzz.trimf(performance.universe, [50, 100, 100])

    rules = [
        ctrl.Rule(study_hours['low'] & attendance['low'] & engagement['low'], performance['poor']),
        ctrl.Rule(study_hours['low'] & attendance['low'] & engagement['medium'], performance['poor']),
        ctrl.Rule(study_hours['low'] & attendance['low'] & engagement['high'], performance['average']),
        ctrl.Rule(study_hours['low'] & attendance['medium'] & engagement['low'], performance['poor']),
        ctrl.Rule(study_hours['low'] & attendance['medium'] & engagement['medium'], performance['average']),
        ctrl.Rule(study_hours['low'] & attendance['medium'] & engagement['high'], performance['average']),
        ctrl.Rule(study_hours['low'] & attendance['high'] & engagement['low'], performance['average']),
        ctrl.Rule(study_hours['low'] & attendance['high'] & engagement['medium'], performance['average']),
        ctrl.Rule(study_hours['low'] & attendance['high'] & engagement['high'], performance['good']),
        ctrl.Rule(study_hours['medium'] & attendance['low'] & engagement['low'], performance['poor']),
        ctrl.Rule(study_hours['medium'] & attendance['low'] & engagement['medium'], performance['average']),
        ctrl.Rule(study_hours['medium'] & attendance['low'] & engagement['high'], performance['average']),
        ctrl.Rule(study_hours['medium'] & attendance['medium'] & engagement['low'], performance['average']),
        ctrl.Rule(study_hours['medium'] & attendance['medium'] & engagement['medium'], performance['average']),
        ctrl.Rule(study_hours['medium'] & attendance['medium'] & engagement['high'], performance['good']),
        ctrl.Rule(study_hours['medium'] & attendance['high'] & engagement['low'], performance['average']),
        ctrl.Rule(study_hours['medium'] & attendance['high'] & engagement['medium'], performance['good']),
        ctrl.Rule(study_hours['medium'] & attendance['high'] & engagement['high'], performance['good']),
        ctrl.Rule(study_hours['high'] & attendance['low'] & engagement['low'], performance['average']),
        ctrl.Rule(study_hours['high'] & attendance['low'] & engagement['medium'], performance['average']),
        ctrl.Rule(study_hours['high'] & attendance['low'] & engagement['high'], performance['good']),
        ctrl.Rule(study_hours['high'] & attendance['medium'] & engagement['low'], performance['average']),
        ctrl.Rule(study_hours['high'] & attendance['medium'] & engagement['medium'], performance['good']),
        ctrl.Rule(study_hours['high'] & attendance['medium'] & engagement['high'], performance['good']),
        ctrl.Rule(study_hours['high'] & attendance['high'] & engagement['low'], performance['good']),
        ctrl.Rule(study_hours['high'] & attendance['high'] & engagement['medium'], performance['good']),
        ctrl.Rule(study_hours['high'] & attendance['high'] & engagement['high'], performance['good'])
    ]

    performance_ctrl = ctrl.ControlSystem(rules)
    fuzzy_model = ctrl.ControlSystemSimulation(performance_ctrl)
    
    return fuzzy_model

def create_hybrid_model(input_shape):
    inputs = Input(shape=(input_shape,))
    
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    
    fuzzy_branch = Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
    
    nn_branch = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    combined = Concatenate()([fuzzy_branch, nn_branch])
    
    output = Dense(1, activation='linear')(combined)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

def main():
    scaled_df, scaler = load_and_preprocess()
    
    X = scaled_df.drop('overall_score', axis=1)
    y = scaled_df['overall_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    train_df = pd.concat([X_train, y_train], axis=1).dropna()
    test_df = pd.concat([X_test, y_test], axis=1).dropna()

    X_train = train_df.drop('overall_score', axis=1)
    y_train = train_df['overall_score']
    X_test = test_df.drop('overall_score', axis=1)
    y_test = test_df['overall_score']
    
    fuzzy_model = create_fuzzy_system()
    with open('fuzzy_model.pkl', 'wb') as f:
        pickle.dump(fuzzy_model, f)
    
    model = create_hybrid_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)
    save_model(model, 'hybrid_model.h5')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Models and scaler saved successfully!")

if __name__ == "__main__":
    main()