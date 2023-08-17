from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
        

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = "your_hugging_face_auth_token"
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    cache_dir="/mnt/s.shrikant.bhurke/s.shrikant.bhurke/model"
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth,
    cache_dir="/mnt/s.shrikant.bhurke/s.shrikant.bhurke/model"
)

# enable evaluation mode to allow model inference
model.eval()

#print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1000,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse.")
# print(res[0]["generated_text"])

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

# checking again that everything is working fine
#llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse.")

from langchain.document_loaders import TextLoader

loader = TextLoader("./fielname.txt")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vectorstore
vectorstore = FAISS.from_documents(all_splits, embeddings)
#vectorstore = FAISS.from_documents(documents, embeddings)
template = '''Context: {context}

Based on Context that is the chunks of the document provide me the complete text under section given for following question
Question: {question}

The answer should be from context only do not use general knowledge to answer the query'''

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)
# qa_chain = RetrievalQA.from_chain_type(
    # llm,
    # retriever=vectorstore.as_retriever(),
    # return_source_documents=True
# )

question = "Give the section text for 'use' from the given document.Do not modify or summarize the text"
result = qa_chain({"query": question})
print("query",question)
print("-------------result",result)

from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

chat_history = []

query = "Can you give me the address of the leased premises from the document?"
result = chain({"question": query, "chat_history": chat_history})

print("query",query)
print("-------------result",result['answer'])

query = "Give me currency type used in the document?"
result = chain({"question": query, "chat_history": chat_history})

print("query",query)
print("-------------result",result['answer'])


query = "give the county name from the document?"
result = chain({"question": query, "chat_history": chat_history})

print("query",query)
print("-------------result",result['answer'])

query = "Could you please extract the content located under the section labeled 'use' in the given document?"
result = chain({"question": query, "chat_history": chat_history})
print("query",query)
print("-------------result",result['answer'])
