# 🚗 BMW Used Car Price Analysis & Prediction

A comprehensive machine learning project that combines **Random Forest regression** with an **intelligent multi-agent RAG (Retrieval-Augmented Generation) system** for BMW car price prediction and analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Specialist RAG Architecture](#specialist-rag-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project analyzes BMW used car prices using machine learning and provides an interactive interface for price predictions and data exploration. The system features a **novel multi-specialist RAG architecture** that uses multiple specialized language models working together to handle different aspects of user queries.

### Key Highlights:

- **Random Forest Model** with 94.4% R² score
- **Multi-Agent RAG System** with 6 specialized AI agents
- **Interactive Dashboard** with visualizations and insights
- **Natural Language Interface** for predictions and queries

---

## ✨ Features

### 1. Machine Learning Model
- **Random Forest Regressor** trained on BMW car dataset
- Handles multiple features: year, mileage, transmission, fuel type, engine size, etc.
- High accuracy: R² = 0.944, MAE = $2,695

### 2. Interactive Dashboard
- Feature importance visualization
- Price distribution and trends
- Actual vs. predicted price comparison
- Category-based price analysis
- Mileage impact visualization

### 3. Intelligent RAG Chatbot
- Natural language price predictions
- Dataset querying in plain English
- Conversational interface with memory
- Automatic feature extraction and filling

---

## 🧠 Specialist RAG Architecture

The chatbot uses a **multi-agent architecture** where specialized language models handle different tasks. This approach is more robust and explainable than traditional keyword-based systems.

### Architecture Overview

```
User Query → Orchestrator → Specialist Models → Response
                  ↓
        Intent Classifier
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
   Prediction Path      Query Path
        ↓                   ↓
   [Extract] → [Fill] → [Predict]  [Query Dataset]
```

### The 6 Specialist Models

#### 1. **Intent Classifier** 🎯
**Role**: Determines what the user wants

**Input**: User query  
**Output**: Classification into:
- `prediction` - User wants price estimation
- `query` - User wants to see dataset entries
- `both` - User wants prediction AND similar cars
- `general` - General conversation about the system

**Example**:
```
Input: "How much is a 2020 BMW X3 worth?"
Output: "prediction"

Input: "Show me all 2019 diesel cars"
Output: "query"
```

**Why LLM?**: Understanding user intent from natural language is complex. An LLM can handle variations, ambiguity, and context better than rule-based systems.

---

#### 2. **Feature Extractor** 🔍
**Role**: Extracts car features from natural language

**Input**: User query  
**Output**: JSON object with extracted features
```json
{
  "model": "X3",
  "year": 2020,
  "mileage": 50000,
  "transmission": "Automatic",
  "fuelType": null,
  "tax": null,
  "mpg": null,
  "engineSize": null
}
```

**Example**:
```
Input: "What's a 2020 X3 with 50k km and automatic transmission worth?"
Output: {
  "model": "X3",
  "year": 2020,
  "mileage": 50000,
  "transmission": "Automatic",
  ...
}
```

**Why LLM?**: Users describe cars in many different ways. An LLM can understand variations like "50k km", "50,000 kilometers", "fifty thousand km" and map them correctly.

---

#### 3. **Feature Filler** 🔧
**Role**: Intelligently fills missing features using dataset statistics and reasoning

**Input**: 
- Partially filled features
- Dataset statistics (averages, modes)
- Original user query for context

**Output**: Complete feature set with reasoning
```json
{
  "model": "X3",
  "year": 2020,
  "mileage": 45000,
  "transmission": "Automatic",
  "fuelType": "Diesel",
  "tax": 145,
  "mpg": 50.2,
  "engineSize": 2.0,
  "reasoning": "Used provided model, year, transmission. Filled mileage with lower than average due to newer year (2020). Used most common fuel type and statistical averages for other features."
}
```

**Why LLM?**: Simply using mean/mode for missing values ignores context. An LLM can make intelligent decisions based on relationships (e.g., newer cars typically have lower mileage, diesel engines usually have better MPG).

---

#### 4. **Price Predictor** 💰
**Role**: Uses the trained Random Forest model to predict prices

**Input**: Complete feature set  
**Output**: Price prediction with confidence interval
```json
{
  "predicted_price": 28500.00,
  "confidence_range": [25000, 32000],
  "features_used": {...}
}
```

**Process**:
1. Encodes categorical features (Label Encoding)
2. Orders features correctly for the model
3. Gets prediction from Random Forest
4. Calculates confidence interval using individual tree predictions

**Why Specialist?**: This specialist handles all ML-specific operations (encoding, feature ordering, prediction, confidence calculation) keeping the ML logic separate from the LLM logic.

---

#### 5. **Data Querier** 📊
**Role**: Translates natural language into pandas operations and retrieves data

**Input**: User query  
**Output**: Query specification
```json
{
  "filters": {"year": 2020, "model": "X3"},
  "sort_by": "price",
  "limit": 10,
  "aggregation": null
}
```

**Example**:
```
Input: "Show me 2020 X3 cars"
Output: Executes pandas query and returns formatted results

Input: "What's the average price of diesel cars?"
Output: Calculates and returns average with statistics
```

**Why LLM?**: Users ask for data in many ways. An LLM can understand "show me", "find", "list", "get" all mean data retrieval, and can extract complex filter conditions from natural language.

---

#### 6. **Orchestrator** 🎼
**Role**: Coordinates all specialists and manages workflow

**Responsibilities**:
- Calls Intent Classifier first
- Routes to appropriate specialists based on intent
- Manages multi-step workflows (prediction path, query path, or both)
- Handles general conversation with context
- Formats final responses

**Workflow Examples**:

**Prediction Flow**:
```
Query → Classifier → "prediction" → Extractor → Filler → Predictor → Response
```

**Query Flow**:
```
Query → Classifier → "query" → Querier → Response
```

**Combined Flow**:
```
Query → Classifier → "both" → [Extractor → Filler → Predictor] + [Querier] → Combined Response
```

**Why Orchestrator?**: Having a central coordinator ensures the right specialists are called in the right order, prevents conflicts, and allows for complex multi-step reasoning.

---

### 🎓 Why This Architecture?

#### Traditional Approach (Keyword-Based):
```python
if 'price' in query and 'worth' in query:
    # Predict
elif 'show' in query or 'list' in query:
    # Query dataset
```

**Problems**:
- ❌ Brittle - breaks with variations
- ❌ Hard to maintain - lots of if-else
- ❌ No reasoning - just pattern matching
- ❌ Can't handle ambiguity

#### Specialist RAG Approach:
```python
intent = intent_classifier.classify(query)  # LLM decides
if intent == 'prediction':
    features = extractor.extract(query)    # LLM extracts
    filled = filler.fill(features)         # LLM reasons
    result = predictor.predict(filled)     # ML predicts
```

**Advantages**:
- ✅ Flexible - handles natural language variations
- ✅ Explainable - each specialist explains its reasoning
- ✅ Maintainable - clear separation of concerns
- ✅ Extensible - easy to add new specialists
- ✅ Robust - LLMs handle edge cases naturally

---

### 🔄 Example: Complete Interaction Flow

**User**: "How much would a 2020 X3 with automatic transmission be worth?"

**Step 1 - Intent Classifier**:
```
Input: "How much would a 2020 X3 with automatic transmission be worth?"
Output: "prediction"
Reasoning: User is asking for price estimation
```

**Step 2 - Feature Extractor**:
```
Input: Same query
Output: {
  "model": "X3",
  "year": 2020,
  "transmission": "Automatic",
  "mileage": null,
  "fuelType": null,
  ...
}
Reasoning: Extracted explicitly mentioned features
```

**Step 3 - Feature Filler**:
```
Input: Partial features + dataset stats
Output: {
  "model": "X3",
  "year": 2020,
  "transmission": "Automatic",
  "mileage": 45000,  # Filled: lower than avg for 2020
  "fuelType": "Diesel",  # Filled: most common
  ...
}
Reasoning: "2020 is relatively new, so used lower mileage. 
            Diesel is most common fuel type in dataset."
```

**Step 4 - Price Predictor**:
```
Input: Complete features
Output: {
  "predicted_price": 28500.00,
  "confidence_range": [25000, 32000]
}
Process: Encoded features → RF model → Prediction
```

**Step 5 - Orchestrator**:
```
Formats final response with price, confidence interval,
features used, and reasoning explanation
```

**Final Response to User**:
```
## 🎯 Price Prediction

Estimated Price: $28,500.00
Confidence Range: $25,000 - $32,000

Features Used:
- model: X3
- year: 2020
- transmission: Automatic
- mileage: 45,000 km (filled)
- fuelType: Diesel (filled)
...

Note: Used lower mileage than average since 2020 is relatively new.
Most common fuel type (Diesel) was used.

*Prediction based on Random Forest model (R²=0.944) trained on 4,843 BMW cars*
```

---

## 📁 Project Structure

```
bmw-car-analysis/
│
├── data/
│   └── bmw.csv                  
│
├── models/
│   └── bmw_price_model.pkl      
|   └── random_forest.ipynb
│
├── app/
│   └── app.py                   # Streamlit application
│
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for RAG chatbot)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/bmw-car-analysis.git
cd bmw-car-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
```

5. **Train the model (optional)**
```bash
# Open and run the Jupyter notebook
jupyter notebook notebooks/model_training.ipynb
```

---

## 💻 Usage

### Running the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Dashboard Tab**: Explore visualizations and model performance metrics
2. **RAG Chatbot Tab**: Interact with the AI assistant

### Example Queries

**Price Predictions**:
- "How much is a 2020 BMW X3 worth?"
- "What would a 2019 5 Series with 60,000 km cost?"
- "Predict the price of an automatic diesel X5 from 2021"

**Data Queries**:
- "Show me all 2020 BMW cars"
- "Find diesel cars under $30,000"
- "What's the average price of X3 models?"

**Combined**:
- "What's a 2020 X3 worth and show me similar cars in the dataset"

**General**:
- "What features are most important for pricing?"
- "Tell me about your model's performance"
- "What's the price range in your dataset?"

---

## 📊 Model Performance

### Training Results

- **Algorithm**: Random Forest Regressor
- **Features**: 8 (year, mileage, transmission, fuelType, tax, mpg, engineSize, model)
- **Training Set**: 80% (3,874 cars)
- **Test Set**: 20% (969 cars)

### Metrics

| Metric | Score |
|--------|-------|
| **R² Score** | 0.944 |
| **Mean Absolute Error (MAE)** | $2,695 |
| **Root Mean Squared Error (RMSE)** | $3,842 |

### Feature Importance

The model identifies these as the most important features for price prediction:

1. **Year** - Manufacturing year (most important)
2. **Mileage** - Total kilometers driven
3. **Model** - BMW model type
4. **Engine Size** - Engine displacement
5. **MPG** - Fuel efficiency

---

## 🔧 Technical Details

### Technologies Used

**Machine Learning**:
- scikit-learn (Random Forest, preprocessing)
- pandas & NumPy (data manipulation)
- joblib (model serialization)

**Web Application**:
- Streamlit (interactive UI)
- Plotly (visualizations)

**AI/NLP**:
- OpenAI GPT-3.5-turbo (RAG specialists)
- LangChain concepts (multi-agent system)

### Model Training Process

1. **Data Loading & Exploration**
   - Load BMW dataset
   - Analyze distributions and correlations
   - Identify outliers and missing values

2. **Data Preprocessing**
   - Handle missing values
   - Label encode categorical features
   - Train-test split (80/20)

3. **Model Training**
   - Random Forest with 100 estimators
   - Hyperparameter tuning
   - Cross-validation

4. **Model Evaluation**
   - Calculate performance metrics
   - Analyze feature importance
   - Visualize predictions vs. actuals

5. **Model Deployment**
   - Save model with joblib
   - Include encoders and feature names
   - Integrate with Streamlit app

### RAG System Components

**Prompt Engineering**:
- Each specialist has a carefully crafted system prompt
- Prompts include examples, constraints, and output formats
- Temperature tuned per specialist (lower for classification, higher for reasoning)

**Error Handling**:
- Fallback mechanisms for each specialist
- JSON parsing with error recovery
- Graceful degradation if OpenAI API fails

**Context Management**:
- Chat history maintained in session state
- Recent context window (last 10 messages)
- Dataset statistics cached for efficiency

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Try ensemble methods (XGBoost, LightGBM)
- [ ] Add more features (color, condition, location)
- [ ] Implement time-series analysis for price trends
- [ ] Create separate models for different BMW series

### RAG System Enhancements
- [ ] Add a "Recommendation Specialist" for car suggestions
- [ ] Implement "Comparison Specialist" for side-by-side analysis
- [ ] Add memory persistence across sessions
- [ ] Create a "Market Analysis Specialist" for trends

### Application Features
- [ ] User authentication and saved preferences
- [ ] Export predictions to PDF/Excel
- [ ] Price alerts for specific models
- [ ] Integration with real-time car listing APIs
- [ ] Mobile-responsive design improvements

### Architecture
- [ ] Add function calling for more reliable structured outputs
- [ ] Implement Claude API as alternative to GPT-3.5
- [ ] Add caching layer for repeated queries
- [ ] Create API endpoints (FastAPI/Flask)

--

## 🙏 Acknowledgments

- BMW dataset from [Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)
- Streamlit for the amazing framework
- OpenAI for GPT-3.5-turbo API
- scikit-learn community for excellent ML tools