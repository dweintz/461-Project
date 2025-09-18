'''
Helper function for splicing url
'''

def get_repo_id(url: str) -> str:
    parts = url.split("/")

    # Splice URL to get the repo id
    if type == "model":
        try:
            index = parts.index("huggingface.co")
            repo_id = f"{parts[index + 1]}/{parts[index + 2]}"
        except (ValueError, IndexError):
            print("Error getting repo id")
            return None
    elif type == "dataset":
        pass
    elif type == "code":
        pass

    return repo_id