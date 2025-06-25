from llama_cpp import llama, Llama, llama_cpp
import os;


def chat_with_history():
    chat_history = "User: What is the capital of France?\nBot: The capital of France is Paris."
    response = llm(
        chat_history + "\nUser: Tell me more about Paris.",
        max_tokens=100,
        temperature=0.7,
    )
    print(response['choices'][0]['text'])


def summarize():
    text_to_summarize = """
    Artificial intelligence is a rapidly evolving field with applications in various domains 
    such as healthcare, finance, and transportation. Machine learning, a subset of AI, enables 
    computers to learn from data and make predictions.
    """
    response = llm(
        f"Summarize the following text:\n{text_to_summarize}",
        max_tokens=50,
        temperature=0.5,
    )
    print(response['choices'][0]['text'])


def prompt_engineer():
    prompt = """
    You are an expert language model trained to assist with coding questions. 
    Explain Python decorators in simple terms.
    """
    response = llm(prompt, max_tokens=100, temperature=0.5)
    print(response['choices'][0]['text'])


def check_gpu():
    print("Checking GPU")
    llm=Llama(model_path, n_gpu_layers=1, verbose=True)
    print("GPU support: ", llm.ctx)
    print("GPU support: ", llm.ctx.gpu)

if __name__=="__main__":
    # chat_with_history()
    # summarize()
    # prompt_engineer()
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    CONTEXT_SIZE = 512
    ngl = 20
    nThread=6
    llm = Llama(model_path, n_ctx=CONTEXT_SIZE, n_gpu_layers=ngl, n_threads=nThread)
    # check_gpu()