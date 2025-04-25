# LangGraph Q&A with Neo4j Graph Database

![LangGraph + Neo4j](https://raw.githubusercontent.com/langchain-ai/langchain/main/docs/static/img/langchain_langgraph.png)

A showcase implementation demonstrating how to build an intelligent Q&A system using LangGraph and Neo4j Graph Database. This project combines the power of Large Language Models (LLMs) with graph database capabilities to answer questions about movie data.

## 🌟 Features

- **Intelligent Movie Database Q&A**: Ask natural language questions about movies, actors, directors, and genres
- **Graph-based Knowledge Representation**: Leverages Neo4j's graph capabilities for complex relationship queries
- **Multi-stage Processing Pipeline**: Uses LangGraph for dynamic orchestration of the Q&A process
- **Query Validation and Self-correction**: Intelligent validation and correction of generated Cypher queries
- **Semantic Similarity**: Uses example-based learning for improved query generation
- **Smart Guardrails**: Ensures the system only answers movie-related questions

## 📋 Requirements

- Python 3.8+
- Neo4j Database (accessible via connection string)
- OpenAI API Key
- LangSmith API Key (for tracing)

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/langgraph_qa_with_graph_db_showscase.git
   cd langgraph_qa_with_graph_db_showscase
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # Create a .env file with the following variables
   OPENAI_API_KEY=your_openai_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=GRAPH-QA
   NEO4J_URI=neo4j://localhost:7687  # Adjust as needed
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

## 🚀 Usage

Run the application:

```bash
python graph_qa.py
```

### Example Questions

You can ask questions like:
- "Which actors played in the movie Casino?"
- "How many movies has Tom Hanks acted in?"
- "List all the genres of the movie Schindler's List"
- "Which actors have worked in movies from both the comedy and action genres?"

## 🧠 How It Works

This system follows a multi-stage process to answer questions:

1. **Guardrails**: Determines if the question is movie-related
2. **Cypher Generation**: Transforms the natural language question into a Cypher query
3. **Validation**: Checks the Cypher query for errors
4. **Correction**: Fixes any identified errors in the query
5. **Execution**: Runs the query against the Neo4j database
6. **Answer Generation**: Creates a natural language answer based on database results

### Architecture Diagram

```
Input Question → Guardrails → Generate Cypher → Validate → Correct → Execute → Generate Answer
                     ↓             ↑               ↓         ↑
                    End         Correction ←── Validation
```

## 💾 Database Schema

The movie database includes:
- **Movie nodes**: With properties like title, released date, and IMDB rating
- **Person nodes**: Representing actors and directors
- **Genre nodes**: Different movie genres
- **Relationships**: ACTED_IN, DIRECTED, IN_GENRE

### Graph Visualization

![visualisation](https://github.com/user-attachments/assets/034dc4ea-28f1-400d-b6b8-2c6afa60d0c3)
![visualisation(1)](https://github.com/user-attachments/assets/cd10c772-3e27-43c8-a29a-a6aa246442d9)


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check issues page.

## 🔮 Future Work

- Support for more complex queries
- Integration with additional data sources
- User interface for easier interaction
- Support for additional languages
