from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatMistralAI(model="mistral-medium-latest", temperature=0.3)



summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at summarizing spoken content. "
        "You write clear, structured, and insightful summaries. "
        "Always respond in {output_language}.",
    ),
    (
        "human",
        """Summarize the transcript below.

Transcript:
{transcript}

Structure your summary as:
- **Overview** (2-3 sentences, what this is about)
- **Key Points** (bullet points, minimum 3)
- **Conclusion** (1-2 sentences wrapping up)

Be concise but thorough. Preserve important names, numbers, and facts.""",
    ),
])

summary_chain = summary_prompt | llm | StrOutputParser()




topics_prompt = ChatPromptTemplate.from_messages([
    ("system", "You extract key topics and named entities from transcripts. Be brief and precise."),
    (
        "human",
        """From the transcript below, extract:
1. Main topics discussed (max 5, one line each)
2. Named entities (people, places, organizations, products)

Transcript:
{transcript}

Respond in JSON format:
{{
  "topics": ["...", "..."],
  "entities": ["...", "..."]
}}""",
    ),
])

topics_chain = topics_prompt | llm | StrOutputParser()


def summarize(transcript: str, output_language: str = "English") -> dict:
    """
    Run the full summarization pipeline on a transcript.

    Returns:
        {
            "summary": markdown string,
            "topics": list of topics,
            "entities": list of named entities
        }
    """
    import json, re

    print("[summarizer] Generating summary...")
    summary = summary_chain.invoke({
        "transcript": transcript,
        "output_language": output_language,
    })

    print("[summarizer] Extracting topics and entities...")
    topics_raw = topics_chain.invoke({"transcript": transcript[:4000]})  

   
    try:
       
        clean = re.sub(r"```json|```", "", topics_raw).strip()
        parsed = json.loads(clean)
        topics = parsed.get("topics", [])
        entities = parsed.get("entities", [])
    except Exception:
        topics = []
        entities = []

    print("[summarizer] Done.")

    return {
        "summary": summary,
        "topics": topics,
        "entities": entities,
    }