# Data_Cleaning_Assistant 🧹

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square)](https://www.python.org/downloads/release/python-390/)

## Description

Data Cleaning Assistant is a powerful tool designed to simplify the data cleaning process. It leverages AI to automatically identify data quality issues, explain them in plain English, and generate ready-to-run cleaning code. Upload your CSV file, and the assistant will help you detect missing values, identify data type issues, find duplicates and outliers, and spot inconsistent categories. It uses the Groq API to analyze data and suggest cleaning steps. It is built using Streamlit, Pandas, NumPy, Langchain, and Groq.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How to use](#how-to-use)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Important Links](#important-links)
- [Footer](#footer)

## Features ✨

- **Automated Data Quality Analysis**: Automatically detects missing values, data type issues, duplicates, outliers, and inconsistent categories in CSV files.
- **AI-Powered Recommendations**: Generates cleaning recommendations and Python code using the Groq API.
- **Interactive User Interface**: Provides a user-friendly interface built with Streamlit for easy file uploading and data preview.
- **Data Privacy**: Processes data locally and sends it only to the Groq API for analysis, ensuring no data is stored on external servers.
- **Customizable Settings**: Allows users to input their Groq API key for personalized data analysis.
- **Comprehensive Reporting**: Generates detailed inspection reports with organized sections for missing values, outliers, and inconsistent categories.

## Tech Stack 💻

- Python
- Streamlit
- Pandas
- NumPy
- Langchain
- Langchain-Groq
- Pytest
- Pytest-mock
- Python-dotenv

## Installation ⚙️

To install and run the Data Cleaning Assistant, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Surrmay/Data_Cleaning_Assistant.git
    cd Data_Cleaning_Assistant
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your Groq API key:**

    -   You can set the `GROQ_API_KEY` environment variable or enter it in the sidebar of the Streamlit application.

    ```bash
    # Option 1: Set environment variable
    export GROQ_API_KEY="YOUR_GROQ_API_KEY"
    
    # Option 2: Enter in the Streamlit app sidebar
    ```

## Usage 🚀

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Upload your CSV file:**

    -   Click on the "Browse files" button to upload your CSV file.
    -   Ensure that your CSV file is properly formatted.

3.  **View Data Quality Analysis:**

    -   The application will automatically analyze the data and display any quality issues.

4.  **Generate Cleaning Recommendations:**

    -   Click the "✨ Generate Cleaning Recommendations" button to get AI-generated cleaning advice.

5.  **Download Recommendations:**

    -   Click the "📥 Download Recommendations" button to download the cleaning recommendations as a text file.

## How to use

This project helps to clean data using AI. You just need to add your CSV file and Groq API key to get started. Here's a breakdown of how the Data Cleaning Assistant can be used in real-world scenarios:

1.  **Data Preprocessing for Machine Learning**: Clean and prepare datasets before training machine learning models to improve model accuracy and performance.
2.  **Business Intelligence**: Ensure data quality for accurate and reliable business insights and reporting.
3.  **Data Migration**: Clean and validate data before migrating it to a new system or database to prevent data loss and inconsistencies.
4.  **Research**: Clean and standardize research data to ensure accurate and reproducible results.

To use this project, you need to run the `app.py` file using Streamlit. Once the application is up and running, you can upload your CSV file and follow the prompts to analyze and clean your data.

## Project Structure 📂

```text
Data_Cleaning_Assistant/
├── app.py              # Main Streamlit application
├── data_cleaner.py     # Comprehensive data cleaning module
├── inspector.py        # Data inspection engine
├── qa_service.py       # LLM integration layer
├── utils.py            # Helper functions
├── requirements.txt    # Project dependencies
├── setup.py            # Project setup script
├── README.md           # Project documentation
├── .env.example        # Example environment variables
├── data/
│   └── sample.csv      # Sample CSV file
└── tests/
    └── test_inspector.py # Test cases for data inspection
```

## Contributing 🤝

Contributions are welcome! Please follow these steps to contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License 📜

This project has no specified license. Please check with the repository owner for more details.

## Important Links 🔗

-   **GitHub Repository**: [Data_Cleaning_Assistant](https://github.com/Surrmay/Data_Cleaning_Assistant)

## Footer 📝


<div align="center">
		 Data_Cleaning_Assistant, <a href="https://github.com/Surrmay/Data_Cleaning_Assistant">https://github.com/Surrmay/Data_Cleaning_Assistant</a>, by <a href="https://github.com/Surrmay">@Surrmay</a>, Feel free to fork, like, star, and raise issues.
</div>
