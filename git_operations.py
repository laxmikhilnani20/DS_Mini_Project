#!/usr/bin/env python3
import subprocess
import os

def git_pull():
    try:
        # Change to the repository directory
        repo_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(repo_path)
        
        # Check if it's a git repository
        if not os.path.exists('.git'):
            print("Error: This is not a git repository.")
            return False
        
        # Fetch the latest changes from the remote
        print("Fetching latest changes from remote...")
        subprocess.run(['git', 'fetch'], check=True)
        
        # Pull changes from the main branch
        print("Pulling changes from main branch...")
        subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
        
        print("Successfully updated to the latest version from main branch.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    git_pull()
