"""
Helper function for splicing url
"""


def get_repo_id(url: str, url_type: str) -> str:
    parts = url.split("/")

    # Splice URL to get the repo id
    try:
        if url_type == "model":
            # Ex: https://huggingface.co/google/gemma-3-270m/tree/main
            index = parts.index("huggingface.co")
            repo_id = f"{parts[index + 1]}/{parts[index + 2]}"
        elif url_type == "dataset":
            # Ex: https://huggingface.co/datasets/xlangai/AgentNet
            index = parts.index("huggingface.co")
            if parts[index + 1] == "datasets":
                if len(parts) > index + 3:
                    repo_id = f"{parts[index + 2]}/{parts[index + 3]}"
                else:
                    repo_id = parts[index + 2]
        elif url_type == "code":
            # Ex: https://github.com/SkyworkAI/Matrix-Game
            index = parts.index("github.com")
            repo_id = f"{parts[index + 1]}/{parts[index + 2]}"
    except (ValueError, IndexError):
        print("Error getting repo id")
        return None

    return repo_id
