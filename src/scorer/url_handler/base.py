'''
Helper functions for classifying URLs from the CLI
'''

def classify_url(url: str) -> str:
    '''
    Classifies urls into either "code", "dataset", "model", or "unknown"
    Assumes that code URLs are from GitHub, and that dataset and model URLs are from Hugging Face
    '''
    if "github.com" in url:
        return "code"
    elif "huggingface.co/datasets" in url:
        return "dataset"
    elif url.startswith("https://huggingface.co/"):
        return "model"
    else:
        return "unknown"
