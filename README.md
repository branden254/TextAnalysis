# Text Analysis and Natural Language Processing (NLP) Project

This project focuses on analyzing text data using various Natural Language Processing (NLP) techniques. It involves entity recognition, topic modeling, text summarization, classification, and trend detection. The project is organized into distinct folders for better structure and reusability.

## Project Structure

### Folders:
- **`data/`**: Contains input data files.
  - Example: `text_data.xlsx`.
- **`src/`**: Contains the main Python script and other source code files.
  - Example: `text_analysis.py` (entry point for the project).
- **`output/`**: Stores visualization results and analysis reports.
  - Example: `entity_distribution.png`, `topic_modeling.png`.
- **`requirements.txt`**: Specifies the necessary dependencies for the project.

### Key Functionalities:
1. **Entity Recognition**:
   - Identifies named entities (e.g., people, organizations, locations) in the text data.
   - Visualizes the distribution of entities.

2. **Topic Modeling**:
   - Uses Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) to identify topics in the text.
   - Generates word clouds and bar charts for better understanding of topics.

3. **Text Summarization**:
   - Summarizes text using transformer models (e.g., DistilBART) or fallback extractive methods.

4. **Text Classification**:
   - Classifies documents into predefined categories using unsupervised machine learning techniques.

5. **Trend Analysis**:
   - Detects emerging trends in the data by analyzing keywords and category evolution over time.

6. **Visualization**:
   - Generates various plots and word clouds to visualize the results effectively.

## Prerequisites

### Installation:
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### Additional Resources:
- SpaCy's pre-trained model (`en_core_web_sm`) is required for Named Entity Recognition (NER). Install it using:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Place the input data file (e.g., `text_data.xlsx`) in the `data/` folder.
2. Run the main script:
   ```bash
   python src/text_analysis.py
   ```
3. The script performs the following steps:
   - Loads the data.
   - Processes the text for entity recognition, topic modeling, summarization, classification, and trend analysis.
   - Saves the outputs (e.g., images, summaries) in the `output/` folder.

## Outputs
The project generates the following outputs:
- Entity distribution and details (`entity_distribution.png`, `entity_details.png`).
- Topic modeling visualizations (`topic_modeling.png`, `topic_wordclouds.png`).
- Category distribution and classification results (`category_distribution.png`, `document_categories.png`).
- Emerging trend analysis (`category_trends.png`, `emerging_trends.png`).

## Future Enhancements
- Incorporate more advanced models for summarization and classification.
- Add support for real-time data analysis.
- Develop a web interface for easier user interaction.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- [NLTK](https://www.nltk.org/) for natural language processing utilities.
- [SpaCy](https://spacy.io/) for advanced NLP tools.
- [Scikit-learn](https://scikit-learn.org/) for machine learning models.
- [HuggingFace Transformers](https://huggingface.co/) for pre-trained summarization models.