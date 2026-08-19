import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import io

# -----------------------------------------------------------------------------
# SECTION 1: ANALYTICS ENGINE (Backend Logic)
# -----------------------------------------------------------------------------

class CourseAnalytics:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.is_trained = False

    def preprocess_data(self, df):
        """
        Prepares data for analysis and modeling.
        Expected columns: 'date', 'course_name', 'registered_count'
        """
        data = df.copy()
        
        # Convert date column
        data['date'] = pd.to_datetime(data['date'])
        
        # Extract time features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['quarter'] = data['date'].dt.quarter
        
        # Encode categorical data (Course Names)
        le = LabelEncoder()
        # Handle cases where new courses might appear in future data that weren't in training
        # For this simple version, we fit on the provided data. 
        # In production, you would save/load the encoder state.
        try:
            data['course_encoded'] = le.fit_transform(data['course_name'])
            self.label_encoders['course'] = le
        except Exception as e:
            st.error(f"Error encoding course names: {e}")
            raise e
        
        return data

    def get_performance_overview(self, df):
        """
        Generates aggregated statistics for management reporting.
        Returns: Dict of dataframes for reporting.
        """
        data = self.preprocess_data(df)
        
        # 1. Course Performance (Total & Average)
        course_perf = data.groupby('course_name')['registered_count'].agg(
            ['sum', 'mean', 'std', 'count']
        ).rename(columns={
            'sum': 'Total Registrations',
            'mean': 'Avg Registrations',
            'std': 'Std Dev',
            'count': 'Data Points'
        }).sort_values('Total Registrations', ascending=False)

        # 2. Seasonal/Monthly Trends
        monthly_trends = data.groupby('month')['registered_count'].sum()
        monthly_trends.index = [f"Month {i}" for i in monthly_trends.index]
        
        # 3. Quarterly Performance
        quarterly_perf = data.groupby('quarter')['registered_count'].sum()
        quarterly_perf.index = [f"Q{i}" for i in quarterly_perf.index]

        # 4. Course x Quarter Matrix (Heatmap data)
        course_quarter_matrix = pd.pivot_table(
            data, 
            values='registered_count', 
            index='course_name', 
            columns='quarter', 
            aggfunc='sum', 
            fill_value=0
        )

        return {
            'course_performance': course_perf,
            'monthly_trends': monthly_trends,
            'quarterly_performance': quarterly_perf,
            'course_quarter_matrix': course_quarter_matrix,
            'processed_data': data
        }

    def train_prediction_model(self, df):
        """
        Trains a Random Forest model to predict future registrations.
        Features: Month, Quarter, Course Encoding
        Target: Registered Count
        """
        data = self.preprocess_data(df)
        
        features = ['month', 'quarter', 'course_encoded']
        X = data[features]
        y = data['registered_count']

        # Train Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training error for reporting
        predictions = self.model.predict(X)
        mae = mean_absolute_error(y, predictions)
        
        return mae

    def predict_future(self, future_dates_config):
        """
        Generates predictions based on user configuration.
        future_dates_config: List of dicts with 'course_name', 'year', 'month'
        """
        if not self.is_trained:
            raise Exception("Model not trained. Please train the model first.")
        
        predictions = []
        for item in future_dates_config:
            # Encode course
            try:
                # Check if the course exists in our trained encoder
                if 'course' not in self.label_encoders:
                    raise Exception("Encoder not initialized.")
                
                # Handle unknown courses gracefully
                try:
                    course_code = self.label_encoders['course'].transform([item['course_name']])[0]
                except ValueError:
                    st.warning(f"Course '{item['course_name']}' was not in training data. Skipping prediction.")
                    continue
                
                input_data = pd.DataFrame({
                    'month': [item['month']],
                    'quarter': [pd.Timestamp(f"{item['year']}-{item['month']}-01").quarter],
                    'course_encoded': [course_code]
                })
                
                pred = self.model.predict(input_data)[0]
                predictions.append({
                    'course': item['course_name'],
                    'year': item['year'],
                    'month': item['month'],
                    'predicted_registrations': max(0, pred) # No negative registrations
                })
            except Exception as e:
                st.error(f"Prediction error for {item['course_name']}: {e}")
                continue
            
        if not predictions:
            return pd.DataFrame()
            
        return pd.DataFrame(predictions)

# -----------------------------------------------------------------------------
# SECTION 2: STREAMLIT INTERFACE (Frontend)
# -----------------------------------------------------------------------------

# Page Configuration
st.set_page_config(
    page_title="Course Registration Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("🎓 Course Registration & Performance Intelligence")
st.markdown("""
    **Management Dashboard**: Upload historical registration data to analyze course performance, 
    identify seasonal trends, and predict future enrollment numbers.
    
    *Note: All data processing happens locally in this session. No data is stored on external servers.*
""")

# Initialize Session State
if 'analytics' not in st.session_state:
    st.session_state.analytics = CourseAnalytics()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar for Data Upload
with st.sidebar:
    st.header("Data Configuration")
    st.info("Upload a CSV with columns: `date`, `course_name`, `registered_count`")
    
    # Provide a sample download button for testing
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=24, freq='M').repeat(3),
        'course_name': ['Math 101', 'Math 101', 'Math 101', 'CS 101', 'CS 101', 'CS 101'] * 24,
        'registered_count': np.random.randint(50, 500, size=72)
    })
    # Correcting sample data generation for simplicity
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='M')
    courses = ['Math 101', 'CS 101', 'Physics 202', 'History 303']
    sample_rows = []
    for date in dates:
        for course in courses:
            # Add some seasonality and noise
            month = date.month
            base = 100 if 'Math' in course else 50
            seasonal = 50 * np.sin(2 * np.pi * month / 12)
            count = int(base + seasonal + np.random.randint(-20, 50))
            sample_rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'course_name': course,
                'registered_count': max(0, count)
            })
    sample_df = pd.DataFrame(sample_rows)
    
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Data Template",
        data=csv,
        file_name='sample_registration_data.csv',
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Validate columns
            required_columns = ['date', 'course_name', 'registered_count']
            if not all(col in df.columns for col in required_columns):
                st.error(f"Missing columns. Required: {required_columns}")
                st.stop()
            
            st.session_state.data_loaded = True
            st.success("Data loaded successfully!")
            st.info(f"Records found: {len(df)}")
            
            # Show sample
            st.write("### Data Preview")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file to begin analysis.")
        st.stop()

if not st.session_state.data_loaded:
    st.stop()

# Main Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📈 Performance Overview", "🔮 Predictive Analytics", "📉 Detailed Reports"])

with tab1:
    st.header("Historical Performance Analysis")
    
    if st.button("Generate Performance Report", type="primary"):
        try:
            # Get analytics
            results = st.session_state.analytics.get_performance_overview(df)
            
            # 1. Top/Bottom Courses
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏆 Top Performing Courses (Total)")
                top_courses = results['course_performance'].head(5)
                if not top_courses.empty:
                    st.bar_chart(top_courses['Total Registrations'], color="#28a745")
                else:
                    st.write("No data available.")
            
            with col2:
                st.subheader("📉 Underperforming Courses (Total)")
                bottom_courses = results['course_performance'].tail(5)
                if not bottom_courses.empty:
                    st.bar_chart(bottom_courses['Total Registrations'], color="#dc3545")
                else:
                    st.write("No data available.")

            # 2. Seasonal Trends
            st.subheader("📅 Seasonal & Monthly Trends")
            fig_monthly = px.line(
                x=results['monthly_trends'].index, 
                y=results['monthly_trends'].values,
                labels={'x': 'Month', 'y': 'Total Registrations'},
                title="Registration Volume by Month (All Courses)"
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            st.warning("Insight: Use the 'Detailed Reports' tab to see which specific courses drive these seasonal spikes.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.error(f"Details: {str(e)}")

with tab2:
    st.header("AI Prediction Engine")
    st.markdown("Train a model on the last 3 years of data to forecast future registration numbers.")
    
    if st.button("Train Prediction Model", type="primary"):
        with st.spinner("Training Random Forest Regressor... This may take a moment."):
            try:
                mae = st.session_state.analytics.train_prediction_model(df)
                st.session_state.model_trained = True
                st.success(f"Model trained successfully! (Mean Absolute Error: {mae:.2f})")
                st.info("Model is ready for predictions in the section below.")
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.error(f"Details: {str(e)}")
    
    if st.session_state.model_trained:
        st.divider()
        st.subheader("Generate Future Forecast")
        
        # Simple interface to generate future scenarios
        st.write("Enter parameters to simulate future performance:")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            courses = df['course_name'].unique()
            selected_course = st.selectbox("Select Course", courses)
        with col_b:
            year = st.number_input("Year", min_value=2024, max_value=2030, value=2025)
        with col_c:
            month = st.slider("Month", 1, 12, 1)
            
        if st.button("Predict Registration"):
            try:
                future_config = [
                    {'course_name': selected_course, 'year': year, 'month': month}
                ]
                pred_df = st.session_state.analytics.predict_future(future_config)
                
                if not pred_df.empty:
                    st.metric(
                        label=f"Predicted Registrations for {selected_course}",
                        value=f"{pred_df['predicted_registrations'].iloc[0]:.0f}"
                    )
                    st.caption("Note: Predictions are based on historical patterns. Actual results may vary due to market changes.")
                else:
                    st.warning("Could not generate prediction. Ensure the course exists in training data.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.error(f"Details: {str(e)}")
    else:
        st.info("Please train the model first to enable predictions.")

with tab3:
    st.header("Detailed Management Reports")
    
    if st.button("Generate Detailed Matrices"):
        try:
            results = st.session_state.analytics.get_performance_overview(df)
            
            st.subheader("Course Performance by Quarter")
            st.write("Heatmap showing registration intensity. Darker colors indicate higher volume.")
            
            # Heatmap
            fig_heatmap = px.imshow(
                results['course_quarter_matrix'],
                labels={'x': 'Quarter', 'y': 'Course', 'color': 'Registrations'},
                color_continuous_scale="Reds",
                title="Registration Volume: Course vs Quarter"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.subheader("Raw Data Tables")
            st.write("### Course Performance Summary")
            st.dataframe(results['course_performance'])
            
            st.write("### Quarterly Aggregates")
            st.dataframe(results['quarterly_performance'])
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.error(f"Details: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Generated by Course Intelligence System | Powered by Python & Streamlit")
