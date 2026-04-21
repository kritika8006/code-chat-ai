const { ChatOllama, OllamaEmbeddings } = require("@langchain/ollama");
const { MemoryVectorStore } = require("@langchain/community/vectorstores/memory");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { startIngestion } = require("./ingest");

async function runAI() {
    try {
        const docs = await startIngestion("./");


        console.log("🤖 Initializing AI Brain...");
        const embeddings = new OllamaEmbeddings({ 
            model: "nomic-embed-text",
            baseUrl: "http://localhost:11434"
        });
        
        const model = new ChatOllama({ 
            model: "gemma3:4b", 
            baseUrl: "http://localhost:11434"
        });

        console.log("📚 Indexing chunks into memory...");
        const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

        const question = "How does the ingestion process handle different file types like .js or .cpp?";
        console.log(`🤔 Question: ${question}`);

        const relevantDocs = await vectorStore.similaritySearch(question, 1);
        const context = relevantDocs[0].pageContent;

        const template = `You are a Senior Developer. Answer the question based on this code context:
        {context}
        
        Question: {question}
        
        Answer:`;

        const prompt = PromptTemplate.fromTemplate(template);
        const chain = prompt.pipe(model).pipe(new StringOutputParser());

        const response = await chain.invoke({ context, question });
        console.log(`\n🚀 AI ANSWER:\n${response}`);

    } catch (err) {
        console.error("❌ Pipeline Error:", err.message);
    }
}

runAI();
