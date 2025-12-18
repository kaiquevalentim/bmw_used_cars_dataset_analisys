import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score
from openai import OpenAI
import os
from dotenv import load_dotenv
import joblib

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="BMW used cars analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load data
@st.cache_data
def load_data():
    """Load the BMW dataset"""
    try:
        data = pd.read_csv('../data/bmw.csv')
        return data
    except FileNotFoundError:
        st.error("Error: Could not find '../data/bmw.csv'. Please ensure the file exists.")
        return None

# Load trained model
@st.cache_resource
def load_model():
    """Load the trained Random Forest model package"""
    try:
        model_package = joblib.load('../models/bmw_price_model.pkl')
        return model_package['model'], model_package['encoders'], model_package['features']
    except FileNotFoundError:
        st.error("Error: Could not find '../models/bmw_price_model.pkl'. Please ensure the model file exists.")
        return None, None, None

# Calculate test metrics
@st.cache_data
def calculate_metrics(_model, data, _encoders, feature_names):
    """Calculate model performance metrics using the same preprocessing as training"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Create a copy to avoid modifying original data
    dataset = data.copy()
    
    # Apply Label Encoding to categorical columns (recreate the encoding process)
    # Since encoders dict is empty, we need to encode the same way as training
    for categorical_column in dataset.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        dataset[categorical_column] = le.fit_transform(dataset[categorical_column].astype(str))
    
    # Now split - after encoding, just like in training
    X = dataset.drop(columns=['price'])
    y = dataset['price']
    
    # Split data with same random state as training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ensure features are in correct order
    X_test = X_test[feature_names]
    
    # Predictions
    y_pred = _model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return metrics

class SpecialistRAGSystem:
    """Multi-agent RAG system with specialized models for different tasks"""
    
    def __init__(self, data, model, encoders, feature_names):
        self.data = data
        self.model = model
        self.encoders = encoders
        self.feature_names = feature_names
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def _call_llm(self, system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Helper method to call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ============ SPECIALIST 1: INTENT CLASSIFIER ============
    def classify_intent(self, user_query: str) -> str:
        """Specialist RAG that classifies user intent"""
        system_prompt = """You are an intent classification specialist for a BMW car pricing system.

Your job is to classify the user's intent into ONE of these categories:
- "prediction": User wants to predict/estimate the price of a specific car
- "query": User wants to retrieve/see data from the dataset (show cars, list examples, find entries)
- "both": User wants both prediction AND data retrieval
- "general": User is asking general questions about the system, data statistics, or having a conversation

Examples:
- "How much is a 2020 BMW X3 worth?" -> prediction
- "Show me all 2019 BMW cars" -> query
- "What's the price of a 2020 X3 and show me similar cars in the dataset" -> both
- "What's the average BMW price?" -> general
- "Tell me about your model" -> general

Respond with ONLY one word: prediction, query, both, or general"""

        response = self._call_llm(system_prompt, user_query, temperature=0.1, max_tokens=10)
        return response.lower().strip()
    
    # ============ SPECIALIST 2: FEATURE EXTRACTOR ============
    def extract_features(self, user_query: str) -> dict:
        """Specialist RAG that extracts car features from natural language"""
        system_prompt = """You are a feature extraction specialist for BMW cars.

Your job is to extract car features from the user's query and return them as a JSON object.

Available features:
- model: BMW model (e.g., "X1", "X3", "3 Series", "5 Series")
- year: Manufacturing year (e.g., 2020, 2019)
- mileage: Mileage in kilometers (e.g., 50000)
- transmission: "Manual", "Automatic", or "Semi-Auto"
- fuelType: "Petrol", "Diesel", "Hybrid", "Electric", or "Other"
- tax: Annual tax amount
- mpg: Miles per gallon
- engineSize: Engine size in liters (e.g., 2.0, 3.0)

Rules:
1. Extract ONLY features that are explicitly mentioned
2. Use null for features that are not mentioned
3. Return ONLY valid JSON, no explanations
4. Be flexible with different ways users describe features

Example input: "How much is a 2020 X3 with automatic transmission worth?"
Example output: {"model": "X3", "year": 2020, "transmission": "Automatic", "mileage": null, "fuelType": null, "tax": null, "mpg": null, "engineSize": null}

Now extract features from the user's query and respond with ONLY the JSON object."""

        response = self._call_llm(system_prompt, user_query, temperature=0.2, max_tokens=200)
        
        # Clean response
        response = response.replace("```json", "").replace("```", "").strip()
        
        try:
            import json
            features = json.loads(response)
            return features
        except:
            return {}
    
    # ============ SPECIALIST 3: FEATURE FILLER ============
    def fill_missing_features(self, extracted_features: dict, user_query: str) -> dict:
        """Specialist RAG that intelligently fills missing features using data statistics"""
        
        # Get data context
        data_summary = {
            'year_range': f"{self.data['year'].min()}-{self.data['year'].max()}",
            'avg_year': int(self.data['year'].mean()),
            'most_common_year': int(self.data['year'].mode()[0]),
            'avg_mileage': int(self.data['mileage'].mean()),
            'avg_tax': float(self.data['tax'].mean()),
            'avg_mpg': float(self.data['mpg'].mean()),
            'avg_engine': float(self.data['engineSize'].mean()),
            'most_common_transmission': self.data['transmission'].mode()[0],
            'most_common_fuel': self.data['fuelType'].mode()[0],
            'most_common_model': self.data['model'].mode()[0]
        }
        
        system_prompt = f"""You are a missing feature filler specialist for BMW car price predictions.

Your job is to fill missing features using intelligent defaults based on the dataset statistics and context.

Dataset Statistics:
- Year range: {data_summary['year_range']}
- Average year: {data_summary['avg_year']}
- Most common year: {data_summary['most_common_year']}
- Average mileage: {data_summary['avg_mileage']:,} km
- Average tax: ${data_summary['avg_tax']:.0f}
- Average MPG: {data_summary['avg_mpg']:.1f}
- Average engine size: {data_summary['avg_engine']:.1f}L
- Most common transmission: {data_summary['most_common_transmission']}
- Most common fuel type: {data_summary['most_common_fuel']}
- Most common model: {data_summary['most_common_model']}

Given features: {extracted_features}
Original user query: "{user_query}"

Instructions:
1. For each missing feature (null values), decide on a reasonable default
2. Consider the user's query context and provided features
3. Use statistical averages for numeric features
4. Use most common values for categorical features
5. If year is known, adjust mileage accordingly (newer cars = less mileage)
6. Return a JSON object with ALL features filled (no null values)
7. Include a "reasoning" field explaining your choices

Example output:
{{
    "model": "X3",
    "year": 2020,
    "mileage": 45000,
    "transmission": "Automatic",
    "fuelType": "Diesel",
    "tax": 145,
    "mpg": 50.2,
    "engineSize": 2.0,
    "reasoning": "Used provided model, year, and transmission. Filled mileage with lower than average due to newer year (2020). Used most common fuel type (Diesel) and average values for tax, mpg, and engine size."
}}

Now fill the missing features and respond with ONLY the JSON object."""

        response = self._call_llm(system_prompt, str(extracted_features), temperature=0.3, max_tokens=400)
        
        # Clean and parse
        response = response.replace("```json", "").replace("```", "").strip()
        
        try:
            import json
            filled = json.loads(response)
            return filled
        except:
            # Fallback to simple filling
            return self._simple_fill(extracted_features)
    
    def _simple_fill(self, features: dict) -> dict:
        """Fallback simple feature filling"""
        filled = features.copy()
        
        for feature in self.feature_names:
            if feature not in filled or filled[feature] is None:
                if feature == 'year':
                    filled[feature] = int(self.data['year'].mode()[0])
                elif feature == 'mileage':
                    filled[feature] = int(self.data['mileage'].mean())
                elif feature in ['tax', 'mpg', 'engineSize']:
                    filled[feature] = float(self.data[feature].mean())
                elif feature in ['model', 'transmission', 'fuelType']:
                    filled[feature] = self.data[feature].mode()[0]
        
        filled['reasoning'] = "Used statistical defaults (means and modes)"
        return filled
    
    # ============ SPECIALIST 4: PRICE PREDICTOR ============
    def predict_price(self, filled_features: dict) -> dict:
        """Specialist that uses the Random Forest model to predict price"""
        from sklearn.preprocessing import LabelEncoder
        
        # Remove reasoning if present
        features_for_model = {k: v for k, v in filled_features.items() if k != 'reasoning'}
        
        # Create dataframe
        feature_df = pd.DataFrame([features_for_model])
        
        # Encode categorical variables
        for col in ['model', 'transmission', 'fuelType']:
            if col in feature_df.columns:
                le = LabelEncoder()
                le.fit(self.data[col].astype(str))
                try:
                    feature_df[col] = le.transform(feature_df[col].astype(str))
                except:
                    feature_df[col] = le.transform([self.data[col].mode()[0]])[0]
        
        # Ensure correct column order
        feature_df = feature_df[self.feature_names]
        
        # Predict
        prediction = self.model.predict(feature_df)[0]
        
        # Get confidence interval using tree predictions
        tree_predictions = [tree.predict(feature_df)[0] for tree in self.model.estimators_]
        confidence_low = np.percentile(tree_predictions, 10)
        confidence_high = np.percentile(tree_predictions, 90)
        
        return {
            'predicted_price': float(prediction),
            'confidence_range': (float(confidence_low), float(confidence_high)),
            'features_used': features_for_model
        }
    
    # ============ SPECIALIST 5: DATA QUERIER ============
    def query_dataset(self, user_query: str) -> str:
        """Specialist RAG that queries the dataset using pandas"""
        
        # Get sample of data for context
        sample_columns = list(self.data.columns)
        sample_data = self.data.head(3).to_dict('records')
        
        system_prompt = f"""You are a data query specialist for a BMW car dataset.

Dataset info:
- Total records: {len(self.data)}
- Columns: {sample_columns}
- Sample data: {sample_data}

Your job is to understand what data the user wants and generate a pandas query strategy.

Respond with a JSON object containing:
1. "filters": dict of column names and values to filter (e.g., {{"year": 2020, "model": "X3"}})
2. "sort_by": column name to sort by (or null)
3. "limit": maximum number of results to show (default 10)
4. "aggregation": type of aggregation if needed ("count", "average", "min", "max", or null for raw data)

Example:
User: "Show me 2020 BMW X3 cars"
Response: {{"filters": {{"year": 2020, "model": "X3"}}, "sort_by": "price", "limit": 10, "aggregation": null}}

User: "What's the average price of diesel cars?"
Response: {{"filters": {{"fuelType": "Diesel"}}, "sort_by": null, "limit": null, "aggregation": "average"}}

Now analyze the user's query and respond with ONLY the JSON object."""

        response = self._call_llm(system_prompt, user_query, temperature=0.2, max_tokens=200)
        
        # Clean and parse
        response = response.replace("```json", "").replace("```", "").strip()
        
        try:
            import json
            query_spec = json.loads(response)
            return self._execute_query(query_spec)
        except:
            return "I couldn't understand your query. Please try rephrasing."
    
    def _execute_query(self, query_spec: dict) -> str:
        """Execute the query specification on the dataset"""
        filtered_data = self.data.copy()
        
        # Apply filters
        filters = query_spec.get('filters', {})
        for col, value in filters.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]
        
        # Check if aggregation needed
        aggregation = query_spec.get('aggregation')
        if aggregation:
            if aggregation == 'average':
                result = filtered_data['price'].mean()
                filter_desc = ', '.join([f"{k}={v}" for k, v in filters.items()]) if filters else "all cars"
                return f"**Average price for {filter_desc}**: ${result:,.2f}\n\nBased on {len(filtered_data)} cars in the dataset."
            elif aggregation == 'count':
                return f"**Count**: {len(filtered_data)} cars found"
            elif aggregation == 'min':
                result = filtered_data['price'].min()
                return f"**Minimum price**: ${result:,.2f}"
            elif aggregation == 'max':
                result = filtered_data['price'].max()
                return f"**Maximum price**: ${result:,.2f}"
        
        # Regular data retrieval
        if len(filtered_data) == 0:
            filter_str = ', '.join([f"{k}={v}" for k, v in filters.items()])
            return f"No cars found matching: {filter_str}"
        
        # Sort if specified
        sort_by = query_spec.get('sort_by')
        if sort_by and sort_by in filtered_data.columns:
            filtered_data = filtered_data.sort_values(sort_by, ascending=False)
        
        # Limit results
        limit = query_spec.get('limit', 10)
        sample_data = filtered_data.head(limit)
        
        # Format response
        response = f"**Found {len(filtered_data)} cars**"
        if filters:
            response += f" matching: {', '.join([f'{k}={v}' for k, v in filters.items()])}"
        response += f"\n\n**Showing first {min(limit, len(filtered_data))} results:**\n\n"
        
        for idx, (_, row) in enumerate(sample_data.iterrows(), 1):
            response += f"**Car #{idx}:**\n"
            response += f"- Model: {row['model']} ({row['year']})\n"
            response += f"- Price: ${row['price']:,.0f}\n"
            response += f"- Mileage: {row['mileage']:,} km\n"
            response += f"- Transmission: {row['transmission']} | Fuel: {row['fuelType']}\n"
            response += f"- Engine: {row['engineSize']}L | MPG: {row['mpg']} | Tax: ${row['tax']}\n\n"
        
        if len(filtered_data) > limit:
            response += f"*({len(filtered_data) - limit} more cars match your criteria)*"
        
        return response
    
    # ============ ORCHESTRATOR ============
    def orchestrate(self, user_query: str, chat_history: list = None) -> str:
        """Main orchestrator that coordinates all specialist models"""
        
        if not self.client:
            return "Error: OpenAI API key not found. Please add OPENAI_API_KEY to your .env file."
        
        if chat_history is None:
            chat_history = []
        
        # Step 1: Classify intent
        intent = self.classify_intent(user_query)
        
        # Step 2: Route to appropriate specialists based on intent
        
        if intent == "prediction":
            # Prediction workflow
            features = self.extract_features(user_query)
            
            if not features or all(v is None for v in features.values()):
                return """I'd be happy to predict a car price! However, I need some information about the car.

Please provide details such as:
- **Model** (e.g., X1, X3, 3 Series)
- **Year** (manufacturing year)
- **Mileage** (in km)
- **Transmission** (Automatic, Manual, Semi-Auto)
- **Fuel Type** (Petrol, Diesel, Hybrid, Electric)

Example: "How much is a 2020 X3 with 50,000 km worth?" """
            
            # Fill missing features
            filled = self.fill_missing_features(features, user_query)
            
            # Make prediction
            prediction_result = self.predict_price(filled)
            
            # Format response
            response = f"**Estimated Price**: ${prediction_result['predicted_price']:,.2f}\n\n"
            response += f"**Confidence Range**: ${prediction_result['confidence_range'][0]:,.0f} - ${prediction_result['confidence_range'][1]:,.0f}\n\n"
            
            response += "**Features Used:**\n"
            for k, v in prediction_result['features_used'].items():
                response += f"- {k}: {v}\n"
            
            if 'reasoning' in filled:
                response += f"\n**Note**: {filled['reasoning']}\n"
            
            response += f"\n*Prediction based on Random Forest model (R²=0.944) trained on {len(self.data)} BMW cars*"
            
            return response
        
        elif intent == "query":
            # Data query workflow
            return self.query_dataset(user_query)
        
        elif intent == "both":
            # Combined workflow: prediction + query
            response = "## Combined Analysis\n\n"
            
            # First do prediction
            response += "### Price Prediction\n"
            features = self.extract_features(user_query)
            if features and any(v is not None for v in features.values()):
                filled = self.fill_missing_features(features, user_query)
                prediction_result = self.predict_price(filled)
                response += f"**Estimated Price**: ${prediction_result['predicted_price']:,.2f}\n\n"
            
            # Then do query
            response += "### Similar Cars in Dataset\n"
            response += self.query_dataset(user_query)
            
            return response
        
        else:  # general
            # General conversation with context
            context = self._generate_context()
            
            system_prompt = f"""You are an AI assistant specialized in BMW car pricing analysis.

Dataset Information:
- Total cars: {len(self.data)}
- Price range: ${self.data['price'].min():,.0f} - ${self.data['price'].max():,.0f}
- Average price: ${self.data['price'].mean():,.0f}
- Model performance: R² = 0.944, MAE = $2,695

{context}

You can help users with:
1. Price predictions for specific cars
2. Querying the dataset to find cars
3. General insights about BMW pricing
4. Understanding model features and their importance

Be helpful, concise, and data-driven in your responses."""

            # Build messages with history
            messages = [{"role": "system", "content": system_prompt}]
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_query})
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"
    
    def _generate_context(self) -> str:
        """Generate contextual information about the dataset"""
        context_parts = []
        
        # Feature importance
        top_features = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        context_parts.append(f"Most important features: {', '.join([f[0] for f in top_features])}")
        
        # Popular models
        top_models = self.data['model'].value_counts().head(3)
        context_parts.append(f"Most common models: {', '.join(top_models.index)}")
        
        return " | ".join(context_parts)
    
    def query(self, user_query: str, chat_history: list = None) -> str:
        """Main entry point - delegates to orchestrator"""
        return self.orchestrate(user_query, chat_history)

# Load data and model
data = load_data()
model, encoders, feature_names = load_model()

if data is None or model is None:
    st.stop()

# Calculate metrics
metrics = calculate_metrics(model, data, encoders, feature_names)

# Initialize RAG
rag_system = SpecialistRAGSystem(data, model, encoders, feature_names)

# Header
st.title("BMW used cars analysis")
st.markdown("### Machine Learning & Agentic AI Showcase")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "RAG Chatbot"])

# TAB 1: DASHBOARD
with tab1:
    st.header("BMW Car Price Prediction Dashboard")
    
    # Model Performance Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model R² Score", f"{metrics['r2']:.3f}", help="Coefficient of determination")
    with col2:
        st.metric("Mean Absolute Error", f"${metrics['mae']:,.0f}", help="Average prediction error")
    with col3:
        st.metric("Dataset Size", len(data), help="Number of BMW cars")
    
    st.markdown("---")
    
    # Two column layout for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Feature Importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # 2. Price Distribution
        st.subheader("Price Distribution")
        fig_dist = px.histogram(
            data,
            x='price',
            nbins=50,
            title="Distribution of BMW Car Prices",
            labels={'price': 'Price ($)', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.add_vline(
            x=data['price'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${data['price'].mean():,.0f}"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # 3. Actual vs Predicted
        st.subheader("Model Predictions")
        pred_df = pd.DataFrame({
            'Actual': metrics['y_test'],
            'Predicted': metrics['y_pred']
        })
        
        fig_scatter = px.scatter(
            pred_df,
            x='Actual',
            y='Predicted',
            title="Actual vs Predicted Prices",
            labels={'Actual': 'Actual Price ($)', 'Predicted': 'Predicted Price ($)'},
            opacity=0.6
        )
        # Add perfect prediction line
        min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
        max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 4. Price by categorical features
        st.subheader("Price Analysis by Category")
        
        # Choose which categorical features to display
        if 'transmission' in data.columns and 'fuelType' in data.columns:
            avg_by_category = data.groupby(['transmission', 'fuelType'])['price'].mean().reset_index()
            
            fig_category = px.bar(
                avg_by_category,
                x='transmission',
                y='price',
                color='fuelType',
                title="Average Price by Transmission and Fuel Type",
                labels={'price': 'Average Price ($)', 'transmission': 'Transmission'},
                barmode='group'
            )
            st.plotly_chart(fig_category, use_container_width=True)
        elif 'model' in data.columns:
            # Show top models by average price
            top_models = data.groupby('model')['price'].mean().sort_values(ascending=False).head(10)
            fig_models = px.bar(
                x=top_models.values,
                y=top_models.index,
                orientation='h',
                title="Top 10 Models by Average Price",
                labels={'x': 'Average Price ($)', 'y': 'Model'}
            )
            st.plotly_chart(fig_models, use_container_width=True)
    
    # Full width visualizations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 5. Price Trends by Year
        st.subheader("Price Trends Over Years")
        if 'year' in data.columns:
            year_stats = data.groupby('year').agg({
                'price': ['mean', 'min', 'max']
            }).reset_index()
            year_stats.columns = ['year', 'mean', 'min', 'max']
            
            fig_year = go.Figure()
            fig_year.add_trace(go.Scatter(
                x=year_stats['year'],
                y=year_stats['mean'],
                mode='lines+markers',
                name='Average Price',
                line=dict(color='blue', width=3)
            ))
            fig_year.add_trace(go.Scatter(
                x=year_stats['year'],
                y=year_stats['max'],
                mode='lines',
                name='Max Price',
                line=dict(color='lightblue', dash='dash')
            ))
            fig_year.add_trace(go.Scatter(
                x=year_stats['year'],
                y=year_stats['min'],
                mode='lines',
                name='Min Price',
                line=dict(color='lightblue', dash='dash')
            ))
            fig_year.update_layout(
                title="BMW Prices by Manufacturing Year",
                xaxis_title="Year",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("Year column not found in dataset")
    
    with col2:
        # 6. Mileage vs Price
        st.subheader("Mileage Impact on Price")
        if 'mileage' in data.columns:
            sample_data = data.sample(min(200, len(data)))
            
            color_col = 'fuelType' if 'fuelType' in data.columns else None
            
            fig_mileage = px.scatter(
                sample_data,
                x='mileage',
                y='price',
                color=color_col,
                title="Price vs Mileage",
                labels={'mileage': 'Mileage (km)', 'price': 'Price ($)'},
                opacity=0.7,
                trendline="lowess"
            )
            st.plotly_chart(fig_mileage, use_container_width=True)
        else:
            st.info("Mileage column not found in dataset")

# TAB 2: RAG CHATBOT
with tab2:
    st.header("RAG-Powered Car Pricing Chatbot")
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about BMW pricing..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_system.query(prompt, st.session_state.chat_history)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()