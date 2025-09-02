# UNICEF Agent Benchmark System Documentation

## Overview

The UNICEF Agent Benchmark System is a comprehensive testing framework designed to evaluate the performance of an AI agent across multiple data sources and domains. The system assesses the agent's ability to provide accurate responses to questions about climate risks, development indicators, and technical documentation through a multi-server architecture.

## System Architecture

The benchmark system operates on a **multi-server architecture** that simulates the production environment:

### 1. **MCP Servers (Model Context Protocol)**

The system starts three specialized servers that provide different data capabilities:

- **GEE MCP Server** (`unicef-gee-mcp`): Provides access to Google Earth Engine data for geospatial analysis and climate hazard information
- **RAG MCP Server** (`unicef-rag-mcp`): Handles Retrieval-Augmented Generation for technical documentation queries
- **Datawarehouse MCP Server** (`unicef-datawarehouse-mcp`): Provides access to UNICEF's structured development indicators

### 2. **Agent Under Test**

The UNICEF Agent connects to all three servers and processes user questions using the available tools and data sources.

### 3. **Evaluation System**

The benchmark runner evaluates responses using both automated scoring and LLM-based evaluation metrics.

## Question Categories

The benchmark includes three main categories of questions, each designed to test different aspects of the system:

### 1. **Technical Documentation Questions** (`technical_doc.py`)

- **Purpose**: Tests the agent's ability to retrieve and accurately convey information from technical documentation
- **Data Source**: RAG MCP Server
- **Question Type**: Textual responses
- **Example**: _"What is the primary aim of the Children's Climate Risk Index (CCRI)?"_
- **Expected Answer**: _"The CCRI aims to rank countries where vulnerable children are also exposed to a wide range of climate and environmental hazards..."_

### 2. **Google Earth Engine Questions** (`gee.py`)

- **Purpose**: Tests geospatial analysis capabilities and climate hazard exposure calculations
- **Data Source**: GEE MCP Server
- **Question Type**: Numerical responses
- **Focus Areas**:
  - Single hazard exposure (e.g., agricultural drought, river floods)
  - Multi-hazard analysis (intersection and union operations)
  - Country-specific child population exposure data
- **Example**: _"How many children were exposed to agricultural drought in Angola?"_
- **Expected Answer**: `4,734,925` (numerical value)

### 3. **Datawarehouse Questions** (`datawarehouse.py`)

- **Purpose**: Tests access to structured UNICEF development indicators
- **Data Source**: Datawarehouse MCP Server
- **Question Type**: Numerical responses
- **Focus Areas**: Health indicators, vaccination rates, birth registration data
- **Example**: _"What's the percentage of births without a birth weight registered in Nigeria?"_
- **Expected Answer**: `77` (percentage)

## Evaluation Methodology

### 1. **Response Types & Scoring**

#### **Numerical Responses**

- **Extraction**: Uses an LLM with a structured prompt to extract numerical values from agent responses
- **Scoring**: Binary correctness with 1% tolerance for expected values
- **Metric**: `answer_correctness` (correct/incorrect)

#### **Textual Responses**

- **Evaluation**: Uses LLM-based scoring across three dimensions
- **Metrics**:
  - **Faithfulness** (1-5): How accurately the response reflects the ground truth without introducing outside information
  - **Completeness** (1-5): How well the response addresses all aspects of the question
  - **Conciseness** (1-5): How well the response provides relevant information without excessive details

### 2. **Evaluation Infrastructure**

#### **Langfuse Integration**

All benchmark runs are tracked using Langfuse for:

- Trace-level tracking of each question-answer pair
- Score storage and historical analysis
- Session-based organization of benchmark runs

#### **Results Storage**

- **TSV Files**: Local storage of detailed results
  - Numerical results: `benchmark/results/numerical/results_{TIMESTAMP}.tsv`
  - Textual results: `benchmark/results/textual/results_{TIMESTAMP}.tsv`
- **Langfuse Scores**: Cloud-based storage for historical analysis

## Running the Benchmark

### 1. **Basic Usage**

```bash
# Run with default 10 parallel workers
./run_benchmark.sh

# Run with custom number of workers
./run_benchmark.sh -n 5
./run_benchmark.sh --workers 20
```

For the script to work, you need to have the following repositories cloned in the same directory:

- unicef-agent
- unicef-gee-mcp
- unicef-rag-mcp
- unicef-datawarehouse-mcp

### 2. **Execution Flow**

The `run_benchmark.sh` script orchestrates the entire benchmark process:

1. **Server Startup**: Launches all three MCP servers in background processes
2. **Environment Setup**: Configures Python path and removes previous session IDs
3. **Test Execution**: Runs `pytest` with parallel worker processes
4. **Cleanup**: Automatically terminates all servers on completion

### 3. **Parallel Execution**

- Uses `pytest-xdist` for parallel test execution
- Configurable number of workers (default: 10)
- Shared session ID across all parallel processes
- Prevents server overload while maximizing throughput

## Key Implementation Details

### 1. **Session Management**

```python
# Shared session ID file for parallel processes
SESSION_FILE = RESULTS_PATH / ".session_id"
```

- Ensures all parallel workers use the same session ID
- Creates consistent grouping in Langfuse traces
- Enables proper benchmark run identification

### 2. **Question Assembly**

```python
# Questions are assembled from multiple modules
benchmark_questions = [
    *technical_doc_questions,
    *gee_questions,
    *warehouse_questions,
]
```

- Modular question organization by domain
- Consistent data structure using Pydantic models

### 3. **Evaluation Prompts**

The system uses prompts for evaluation:

- **Number Extraction**: Specialized prompt for extracting numerical values from free-form responses
- **Textual Scoring**: Evaluation prompt that compares responses against ground truth answers

## Historical Analysis

The benchmark system includes a Jupyter notebook (`historic.ipynb`) for analyzing historical performance:

- **Data Retrieval**: Fetches scores from Langfuse for specified time periods
- **Visualization**:
  - Categorical metrics (bar charts for correctness)
  - Numeric metrics (line plots with standard deviation bands)
- **Trend Analysis**: Performance evolution over time
- **Metric Breakdown**: Separate analysis for each evaluation dimension

### **Usage**

```python
# Configure time range
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Analyze trends
plot_categorical_metrics(scores_df, ["answer_correctness"])
plot_numeric_metrics(scores_df, ["completeness", "faithfulness", "conciseness"])
```

## File Structure

```
benchmark/
├── run_benchmark.py      # Main test runner
├── test_data.py         # Question assembly and evaluation functions
├── schemas.py           # Data models
├── historic.ipynb       # Historical analysis notebook
├── questions/
│   ├── __init__.py     # Question module imports
│   ├── technical_doc.py # RAG/documentation questions
│   ├── gee.py          # Geospatial analysis questions
│   └── datawarehouse.py # Development indicator questions
└── results/
    ├── numerical/      # Numerical test results
    └── textual/       # Textual evaluation results
```

## Adding New Questions

Add new questions to the appropriate module in `benchmark/questions/`.

All questions and within a json object, so you can add new questions by just changing the json object.

## Troubleshooting

### **Common Issues**

1. **Server Startup Failures**

   - Check if required MCP server repositories are available
   - Verify server dependencies are installed
   - Ensure ports are not already in use

2. **Question Evaluation Errors**

   - Verify LLM model configuration in `config.llm.model`
   - Check prompt templates in `agent/prompts.yaml`
   - Ensure Langfuse credentials are properly configured
