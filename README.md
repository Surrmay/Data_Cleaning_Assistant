# 🧹 AI Data Cleaning Assistant

An intelligent tool that automatically detects data quality issues in CSV files and generates Python code to fix them. Built for data analysts and scientists who spend too much time on data cleaning.

## 🎯 Problem Statement

Data analysts spend approximately 80% of their time cleaning and preparing data before they can perform actual analysis. Common issues include missing values, inconsistent data types, duplicates, outliers, and formatting inconsistencies. This manual process is tedious, error-prone, and repetitive.

## 💡 Solution

This AI-powered assistant automates the data quality assessment process. Simply upload a CSV file, and the system will automatically detect issues using statistical methods and Pandas analysis, then use a Large Language Model to explain the problems in plain English and generate ready-to-run cleaning code that you can execute immediately.

## 🏗️ Architecture

The application follows a modular architecture with clear separation of concerns:

**Data Flow**: CSV Upload → Data Inspection → Issue Detection → LLM Analysis → Code Generation → Recommendations

**Components**:
- **utils.py**: Handles file I/O operations, CSV loading with encoding detection, and data validation to ensure the DataFrame is suitable for processing
- **inspector.py**: Core analysis engine that detects missing values, duplicates, outliers using IQR method, inconsistent categorical values, and computes summary statistics
- **qa_service.py**: LangChain integration layer that formats inspection findings and sends them to Groq API, then receives human-readable explanations and executable cleaning code
- **app.py**: Streamlit interface orchestrating the entire workflow from upload to recommendations

## 🚀 Features

The assistant performs comprehensive data quality checks including detecting missing values with percentage calculations, identifying duplicate rows, finding statistical outliers using the Interquartile Range method, spotting inconsistent categorical values like different capitalizations or formatting, analyzing data types for potential conversion issues, and generating summary statistics for numeric columns.

The AI-powered recommendations provide executive summaries of critical issues, detailed explanations of why each problem matters, and production-ready Pandas code with comments that handles edge cases properly.

## 🛠️ Technology Stack

The project uses Streamlit for the web interface, Pandas and NumPy for data analysis, LangChain for LLM orchestration, Groq API with Llama 3.1 70B model for code generation, and Python for all backend logic. This combination provides a powerful yet efficient solution.

## 📦 Installation

Follow these steps to set up the project locally.

### Prerequisites

You'll need Python 3.8 or higher installed on your system, a Groq API key which you can obtain for free from console.groq.com, and Git for cloning the repository.

### Setup Steps

First, clone the repository using `git clone https://github.com/yourusername/data-cleaning-assistant.git` and navigate into it with `cd data-cleaning-assistant`.

Create a virtual environment by running `python -m venv venv` and activate it. On Windows use `venv\Scripts\activate`, on Mac or Linux use `source venv/bin/activate`.

Install the required packages with `pip install -r requirements.txt`.

Create a file named `.env` in the project root and add your Groq API key like this: `GROQ_API_KEY=your_api_key_here`. You can also use the `.env.example` file as a template.

## 🎮 Usage

To run the application locally, simply execute `streamlit run app.py` from your terminal. Your browser should automatically open to `http://localhost:8501`. If it doesn't, navigate there manually.

Once the app is running, click the "Browse files" button to upload your CSV file. The system will validate the file and show you a preview of the first few rows. The app will then automatically analyze your data and display the findings, showing metrics like row count, missing data columns, and duplicate rows. You'll see warnings for each detected issue with detailed breakdowns in expandable sections.

Click the "Generate Cleaning Recommendations" button to send the analysis to the AI. The system will return a comprehensive report with an executive summary, detailed explanations of each issue, and ready-to-run Pandas code with comments. You can download both the recommendations and the generated cleaning code using the provided download buttons.

## 🧪 Testing

The project includes unit tests for the inspection module. To run all tests, use `pytest tests/` from the project root. For verbose output, add the `-v` flag: `pytest -v tests/`. To test a specific file, run `pytest tests/test_inspector.py`.

## 🌐 Deployment

### Deploying to Streamlit Cloud

Streamlit Cloud offers free hosting for Streamlit apps with straightforward deployment. Push your code to GitHub, then go to share.streamlit.io and sign in with your GitHub account. Click "New app" and select your repository. Choose the branch, typically main or master, and set the main file path to `app.py`.

Under "Advanced settings" click "Secrets" and add your Groq API key in TOML format: `GROQ_API_KEY = "your_key_here"`. Click "Deploy" and wait a few minutes for the app to build. You'll receive a public URL like `your-app.streamlit.app` that you can share with others.

### Deploying to Hugging Face Spaces

Hugging Face Spaces is another excellent option for free hosting. Create an account on huggingface.co if you don't have one. Click "Create new Space" and select "Streamlit" as the SDK. Name your Space something like "data-cleaning-assistant". Under "Settings" navigate to "Repository secrets" and add `GROQ_API_KEY` with your API key value.

Clone the Space repository to your local machine using the provided Git URL. Copy all your project files into this cloned directory, including app.py, inspector.py, qa_service.py, utils.py, requirements.txt, and the data folder. Commit and push your changes: `git add .` then `git commit -m "Initial deployment"` and `git push`. The Space will automatically build and deploy your app, giving you a public URL like `huggingface.co/spaces/username/data-cleaning-assistant`.

## 📁 Project Structure

Understanding the file organization helps you navigate and modify the project. The root directory contains app.py as the main Streamlit application, inspector.py for data quality analysis logic, qa_service.py for LLM integration with Groq, and utils.py with helper functions for file operations. You'll also find requirements.txt listing all Python dependencies, README.md with this documentation, and .env.example as a template for environment variables.

The data directory contains sample.csv as an example CSV file for testing. The tests directory includes test_inspector.py with unit tests for the inspection module.

## 🔒 Privacy & Security

The application is designed with privacy in mind. Your uploaded CSV data is processed entirely in memory and is never stored on disk unless you explicitly choose to download cleaned results. Data sent to the Groq API includes only a small sample of your dataset, typically the first few rows plus aggregated statistics, not the entire file. No personal data or API keys are logged or stored by the application.

For additional security when deploying, always use environment variables or secure secret management for API keys, never commit .env files to version control, and consider adding rate limiting if deploying publicly. You might also want to implement file size restrictions to prevent abuse.

## 🤝 Contributing

Contributions are welcome and appreciated. If you'd like to contribute, fork the repository, create a feature branch with `git checkout -b feature/your-feature-name`, make your changes with clear commit messages, write or update tests as needed, and submit a pull request with a clear description of your changes.

Please ensure your code follows PEP 8 style guidelines, includes docstrings for all functions and classes, has appropriate error handling, and includes unit tests for new functionality.

## 🐛 Troubleshooting

If you encounter issues, here are some common solutions. For "API Key not found" errors, verify that your .env file contains the correct key and that you've restarted the Streamlit app after creating or modifying the .env file.

If you see "Module not found" errors, ensure all dependencies are installed by running `pip install -r requirements.txt` and verify you're using the correct virtual environment. For CSV loading errors, check that your file is properly formatted as CSV and try specifying a different encoding if the file uses non-UTF-8 characters.

If the LLM generates unexpected output, the model might occasionally produce inconsistent results, so try regenerating the recommendations. You can also adjust the temperature parameter in qa_service.py for more deterministic outputs.

## 📄 License

This project is open source and available under the MIT License. You're free to use, modify, and distribute this software as you see fit.

## 🙏 Acknowledgments

This project was built using several excellent open source tools. Streamlit provides the beautiful and easy-to-use web framework. LangChain offers powerful LLM orchestration capabilities. Groq delivers fast and efficient LLM inference. Pandas and NumPy enable robust data analysis. The open source community continues to make projects like this possible through their contributions and support.

## 📞 Contact & Support

If you have questions, suggestions, or run into issues, please open an issue on GitHub or reach out through the repository's discussion section. For general questions about data cleaning best practices or AI integration, feel free to start a discussion in the repository.

---

**Happy Data Cleaning!** 🎉
