import os
import logging
from typing import Annotated, List, Literal, Optional
from operator import add
from neo4j.exceptions import CypherSyntaxError
import neo4j

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_qa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not os.environ.get("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"

if not os.environ.get("LANGSMITH_PROJECT"):
    os.environ["LANGSMITH_PROJECT"] = "GRAPH-QA"

required_env_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
missing_vars = [var for var in required_env_vars if var not in os.environ]

if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    exit(1)

llm = ChatOpenAI(model="gpt-4", temperature=0)

class InputState(TypedDict):
    question: str

class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]

class OutputState(TypedDict):
    answer: str
    steps: List[str]
    cypher_statement: str

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=True
)

def setup_movie_database(neo_4j_graph: Neo4jGraph) -> None:
    """
    Initialize the Neo4j database with movie data and schema using the tutorial approach.
    
    Args:
        neo_4j_graph: Neo4jGraph instance to initialize
        
    This function:
    1. Loads movie data from a CSV file
    2. Creates nodes for movies, persons, and genres
    3. Establishes relationships between nodes
    4. Uses enhanced schema for better query generation
    """
    try:
        try:
            query_result = neo_4j_graph.query("MATCH (m:Movie) RETURN count(m) as count")
            if query_result and query_result[0]["count"] > 0:
                logger.info("Movie database already exists and has data")
                return
        except neo4j.exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j service is unavailable: {str(e)}")
            raise
        except neo4j.exceptions.AuthError as e:
            logger.error(f"Authentication error with Neo4j: {str(e)}")
            raise
        except neo4j.exceptions.ClientError as e:
            logger.error(f"Client error with Neo4j: {str(e)}")
            raise
        
        # Clear existing data
        logger.info("Clearing existing data...")
        neo_4j_graph.query("MATCH (n) DETACH DELETE n")
        
        # Load movie data from CSV
        logger.info("Loading movie data from CSV...")
        movies_query = """
        LOAD CSV WITH HEADERS FROM 
        'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
        AS row
        MERGE (m:Movie {id:row.movieId})
        SET m.released = date(row.released),
            m.title = row.title,
            m.imdbRating = toFloat(row.imdbRating)
        FOREACH (director in split(row.director, '|') | 
            MERGE (p:Person {name:trim(director)})
            MERGE (p)-[:DIRECTED]->(m))
        FOREACH (actor in split(row.actors, '|') | 
            MERGE (p:Person {name:trim(actor)})
            MERGE (p)-[:ACTED_IN]->(m))
        FOREACH (genre in split(row.genres, '|') | 
            MERGE (g:Genre {name:trim(genre)})
            MERGE (m)-[:IN_GENRE]->(g))
        """
        
        neo_4j_graph.query(movies_query)

        query_result = neo_4j_graph.query("MATCH (m:Movie) RETURN count(m) as count")
        if not query_result or query_result[0]["count"] == 0:
            raise Exception("Failed to create movie data")

        logger.info("Refreshing schema...")
        neo_4j_graph.refresh_schema()
            
        logger.info("Successfully initialized movie database with data from CSV")
    except neo4j.exceptions.ServiceUnavailable as e:
        logger.error(f"Neo4j service is unavailable: {str(e)}")
        raise
    except neo4j.exceptions.AuthError as e:
        logger.error(f"Authentication error with Neo4j: {str(e)}")
        raise
    except neo4j.exceptions.ClientError as e:
        logger.error(f"Client error with Neo4j: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize movie database: {str(e)}")
        raise

setup_movie_database(graph)

examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: 'Schindler's List'})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    }
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(), Neo4jVector, k=5, input_keys=["question"]
)

corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph.structured_schema.get("relationships")
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)

NO_RESULTS = "I couldn't find any relevant information in the database"

guardrails_system = """
As an intelligent assistant, your primary objective is to decide whether a given question is related to movies or not. 
If the question is related to movies, output "movie". Otherwise, output "end".
To make this decision, assess the content of the question and determine if it refers to any movie, actor, director, film industry, 
or related topics. Provide only the specified output: "movie" or "end".
"""

guardrails_prompt = ChatPromptTemplate.from_messages([
    ("system", guardrails_system),
    ("human", "{question}"),
])

class GuardrailsOutput(BaseModel):
    decision: Literal["movie", "end"] = Field(
        description="Decision on whether the question is related to movies"
    )

guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput, method="function_calling")

def guardrails(state: InputState) -> OverallState:
    """
    Decides if the question is related to movies or not.
    
    Args:
        state: InputState containing the user's question
        
    Returns:
        OverallState with all required fields initialized
    """
    guardrails_output = guardrails_chain.invoke({"question": state.get("question")})
    database_records: List[dict] = []
    if guardrails_output.decision == "end":
        database_records = [{"message": "This question is not about movies or their cast. Therefore I cannot answer this question."}]
    return {
        "question": state.get("question"),
        "next_action": guardrails_output.decision,
        "cypher_statement": "",
        "cypher_errors": [],
        "database_records": database_records,
        "steps": ["guardrail"],
    }

text2cypher_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "Given an input question, convert it to a Cypher query. No pre-amble."
            "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
        ),
    ),
    (
        "human",
        (
            """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
            Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
            Here is the schema information
            {schema}
            Below are a number of examples of questions and their corresponding Cypher queries.
            {fewshot_examples}
            User input: {question}
            Cypher query:"""
        ),
    ),
])

text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

def generate_cypher(state: OverallState) -> OverallState:
    """
    Generates a cypher statement based on the provided schema and user input.
    
    Args:
        state: OverallState containing the current state
        
    Returns:
        OverallState with all required fields, including the generated Cypher statement
    """
    nl = "\n"
    few_shot_examples = (nl * 2).join(
        [
            f"Question: {el['question']}{nl}Cypher:{el['query']}"
            for el in example_selector.select_examples(
                {"question": state.get("question")}
            )
        ]
    )
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("question"),
            "fewshot_examples": few_shot_examples,
            "schema": graph.schema,
        }
    )

    return {
        "question": state.get("question", ""),
        "next_action": "validate_cypher",
        "cypher_statement": generated_cypher,
        "cypher_errors": [],
        "database_records": [],
        "steps": state.get("steps", []) + ["generate_cypher"],
    }

validate_cypher_system = """
You are a Cypher expert reviewing a statement written by a junior developer.
"""

validate_cypher_user = """You must check the following:
* Are there any syntax errors in the Cypher statement?
* Are there any missing or undefined variables in the Cypher statement?
* Are any node labels missing from the schema?
* Are any relationship types missing from the schema?
* Are any of the properties not included in the schema?
* Does the Cypher statement include enough information to answer the question?

Examples of good errors:
* Label (:Foo) does not exist, did you mean (:Bar)?
* Property bar does not exist for label Foo, did you mean baz?
* Relationship FOO does not exist, did you mean FOO_BAR?

Schema:
{schema}

The question is:
{question}

The Cypher statement is:
{cypher}

Make sure you don't make any mistakes!"""

validate_cypher_prompt = ChatPromptTemplate.from_messages([
    ("system", validate_cypher_system),
    ("human", validate_cypher_user),
])

class Property(BaseModel):
    """Represents a filter condition based on a specific node property in a graph in a Cypher statement."""
    node_label: str = Field(description="The label of the node to which this property belongs.")
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(description="The value that the property is being matched against.")

class ValidateCypherOutput(BaseModel):
    """Represents the validation result of a Cypher query's output."""
    errors: Optional[List[str]] = Field(
        description="A list of syntax or semantically errors in the Cypher statement."
    )
    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )

validate_cypher_chain = validate_cypher_prompt | llm.with_structured_output(ValidateCypherOutput, method="function_calling")

def validate_cypher(state: OverallState) -> OverallState:
    """
    Validates the Cypher statements and maps any property values to the database.
    """
    errors = []
    mapping_errors = []
    try:
        graph.query(f"EXPLAIN {state.get('cypher_statement')}")
    except CypherSyntaxError as e:
        errors.append(e.message)
    corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the graph schema")
    if not corrected_cypher == state.get("cypher_statement"):
        print("Relationship direction was corrected")
    llm_output = validate_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "schema": graph.schema,
            "cypher": state.get("cypher_statement"),
        }
    )
    if llm_output.errors:
        errors.extend(llm_output.errors)
    if llm_output.filters:
        for llm_output_filter in llm_output.filters:
            if llm_output_filter.node_label not in graph.structured_schema["node_props"]:
                errors.append(f"Node label '{llm_output_filter.node_label}' does not exist in the database")
                continue

            if (
                not [
                    prop
                    for prop in graph.structured_schema["node_props"][llm_output_filter.node_label]
                    if prop["property"] == llm_output_filter.property_key][0]["type"] == "STRING"
            ):
                continue
            mapping = graph.query(
                f"MATCH (n:{llm_output_filter.node_label}) WHERE toLower(n.`{llm_output_filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                {"value": llm_output_filter.property_value},
            )
            if not mapping:
                print(
                    f"Missing value mapping for {llm_output_filter.node_label} on property {llm_output_filter.property_key} with value {llm_output_filter.property_value}"
                )
                mapping_errors.append(
                    f"Missing value mapping for {llm_output_filter.node_label} on property {llm_output_filter.property_key} with value {llm_output_filter.property_value}"
                )
    if mapping_errors:
        next_action = "end"
    elif errors:
        next_action = "correct_cypher"
    else:
        next_action = "execute_cypher"

    return {
        "question": state.get("question", ""),
        "next_action": next_action,
        "cypher_statement": corrected_cypher,
        "cypher_errors": errors,
        "database_records": [],
        "steps": state.get("steps", []) + ["validate_cypher"],
    }

correct_cypher_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a Cypher expert reviewing a statement written by a junior developer. "
            "You need to correct the Cypher statement based on the provided errors. No pre-amble."
            "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
        ),
    ),
    (
        "human",
        (
            """Check for invalid syntax or semantics and return a corrected Cypher statement.
            Schema:
            {schema}
            Note: Do not include any explanations or apologies in your responses.
            Do not wrap the response in any backticks or anything else.
            Respond with a Cypher statement only!
            Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
            The question is:
            {question}
            The Cypher statement is:
            {cypher}
            The errors are:
            {errors}
            Corrected Cypher statement: """
        ),
    ),
])

correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()

def correct_cypher(state: OverallState) -> OverallState:
    """
    Correct the Cypher statement based on the provided errors.
    """
    corrected_cypher = correct_cypher_chain.invoke({
        "question": state.get("question"),
        "errors": state.get("cypher_errors"),
        "cypher": state.get("cypher_statement"),
        "schema": graph.schema,
    })

    return {
        "question": state.get("question", ""),
        "next_action": "validate_cypher",
        "cypher_statement": corrected_cypher,
        "cypher_errors": [],
        "database_records": [],
        "steps": state.get("steps", []) + ["correct_cypher"],
    }

def execute_cypher(state: OverallState) -> OverallState:
    """
    Executes the given Cypher statement.
    """
    records = graph.query(state.get("cypher_statement"))
    return {
        "question": state.get("question", ""),
        "next_action": "end",
        "cypher_statement": state.get("cypher_statement", ""),
        "cypher_errors": [],
        "database_records": records if records else [{"message": NO_RESULTS}],
        "steps": state.get("steps", []) + ["execute_cypher"],
    }

generate_final_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant",
    ),
    (
        "human",
        (
            """Use the following results retrieved from a database to provide
            a succinct, definitive answer to the user's question.
            Respond as if you are answering the question directly.
            Results: {results}
            Question: {question}"""
        ),
    ),
])

generate_final_chain = generate_final_prompt | llm | StrOutputParser()

def generate_final_answer(state: OverallState) -> OutputState:
    """
    Generates the final answer based on the database results.
    """
    final_answer = generate_final_chain.invoke({
        "question": state.get("question"),
        "results": state.get("database_records"),
    })
    return {
        "answer": final_answer,
        "steps": state.get("steps", []),
        "cypher_statement": state.get("cypher_statement", ""),
    }

def guardrails_condition(state: OverallState) -> Literal["generate_cypher", "generate_final_answer"]:
    """
    Determines the next node based on the guardrails decision.
    
    Args:
        state: The current state containing the next_action field
        
    Returns:
        The name of the next node to execute
    """
    next_action = state.get("next_action", "")
    if next_action == "end":
        return "generate_final_answer"
    elif next_action == "movie":
        return "generate_cypher"
    else:
        return "generate_final_answer"

def validate_cypher_condition(state: OverallState) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
    """
    Determines the next node based on the Cypher validation results.
    
    Args:
        state: The current state containing the next_action field
        
    Returns:
        The name of the next node to execute
    """
    next_action = state.get("next_action", "")
    if next_action == "end":
        return "generate_final_answer"
    elif next_action == "correct_cypher":
        return "correct_cypher"
    elif next_action == "execute_cypher":
        return "execute_cypher"
    else:
        return "generate_final_answer"

langgraph = StateGraph(OverallState, input=InputState, output=OutputState)

langgraph.add_node("guardrails", guardrails)
langgraph.add_node("generate_cypher", generate_cypher)
langgraph.add_node("validate_cypher", validate_cypher)
langgraph.add_node("correct_cypher", correct_cypher)
langgraph.add_node("execute_cypher", execute_cypher)
langgraph.add_node("generate_final_answer", generate_final_answer)

langgraph.add_edge(START, "guardrails")
langgraph.add_conditional_edges("guardrails", guardrails_condition)
langgraph.add_edge("generate_cypher", "validate_cypher")
langgraph.add_conditional_edges("validate_cypher", validate_cypher_condition)
langgraph.add_edge("execute_cypher", "generate_final_answer")
langgraph.add_edge("correct_cypher", "validate_cypher")
langgraph.add_edge("generate_final_answer", END)

app = langgraph.compile()

if __name__ == "__main__":
    result = app.invoke({"question": "What was the cast of the Casino?"})
    print("\nMovie question result:")
    print(f"Answer: {result['answer']}")
    print(f"Steps: {result['steps']}")
    print(f"Cypher: {result['cypher_statement']}")

    result = app.invoke({"question": "What's the weather in Spain?"})
    print("\nNon-movie question result:")
    print(f"Answer: {result['answer']}")
    print(f"Steps: {result['steps']}")
    print(f"Cypher: {result['cypher_statement']}") 