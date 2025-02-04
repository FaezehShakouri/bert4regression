import datetime
import logging
import os
import subprocess
from typing import Dict, List

class GitLogAnalyzer:
    def __init__(self, git_remote_url: str):
        self.git_remote_url = git_remote_url
        
        # Create temp directory for repo
        self.repo_name = os.path.splitext(os.path.basename(git_remote_url))[0]
        self.repo_path = os.path.join("data/repos", self.repo_name)
        os.makedirs(self.repo_path, exist_ok=True)

        # Clone repo if it doesn't exist
        if not os.path.exists(os.path.join(self.repo_path, '.git')):
            subprocess.run(['git', 'clone', git_remote_url, self.repo_path], check=True)
        
        # Setup logging with file handler
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_git_log(self) -> str:
        """Fetch git log with detailed information"""
        self.logger.info(f"Fetching git log from {self.repo_path}")
        cmd = [
            'git',
            '-C', self.repo_path,
            'log',
            '--pretty=format:%h|%an|%ad|%s',
            '--date=iso'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def split_log_by_periods(self, log_text: str, weeks_per_period: int) -> Dict[str, str]:
        """Split git log into periods of specified number of weeks
        
        Args:
            log_text: The git log text to split
            weeks_per_period: Number of weeks per period
        """
        self.logger.info(f"Splitting git log into {weeks_per_period}-week periods")
        commits = log_text.split('\n')
        periods = {}

        for commit in commits:
            if not commit.strip():
                continue

            try:
                _, _, date_str, _ = commit.split('|', 3)
                date = datetime.datetime.strptime(date_str.split()[0], '%Y-%m-%d')
                
                # Get week number (1-52) and create period key
                week_num = date.isocalendar()[1]
                period_num = (week_num - 1) // weeks_per_period + 1
                period_key = f"{date.year}-W{period_num}"

                if period_key not in periods:
                    periods[period_key] = []
                periods[period_key].append(commit)
            except Exception as e:
                print(f"Error processing commit: {commit}")
                continue

        self.logger.info(f"Found {len(periods)} distinct periods")
        return periods

    def split_log_by_max_token_size(self, log_text: str, max_token_size: int) -> List[str]:
        """Split git log into chunks of specified maximum token size, ensuring commits aren't split"""
        self.logger.info(f"Splitting git log into chunks of {max_token_size} tokens")
        return [log_text[i:i+max_token_size] for i in range(0, len(log_text), max_token_size)]

    def analyze_log_chunk_promt(self, chunk: str, promt_context: str) -> str:
        """Generate a prompt for the LLM to analyze a chunk of git commit logs"""
        self.logger.info(f"Analyzing chunk for period ({len(chunk)} characters)")
        
        prompt = f"""You are an expert Git commit log analyzer with deep experience in software development patterns and project management.
        
        Task: Analyze the provided git commit logs in the context of the project and generate a comprehensive yet concise, short, without any markdown formatting and to the point summary.

        Required Analysis:
        1. Key Milestones & Features:
            - Identify major project milestones and significant features
            - Focus on user-facing changes and architectural improvements
            - Highlight any breaking changes or important refactoring

        2. Development Patterns:
            - Identify recurring themes in development
            - Note any shifts in development focus
            - Observe code quality and testing patterns
            - Detect potential technical debt indicators

        Remember to:
            - Be specific
            - Focus on substantial changes over minor updates
            - Consider the project context in your analysis
            - Maintain objectivity in your observations
        Project Context:
        {promt_context}

        Format your response strictly as follows:

        [description]
        A short description of the project

        [milestones]
        • Major milestone or feature
            - Supporting details or sub-features
        • Next milestones...

        [patterns]
        • Pattern observation
            - Supporting evidence
            - Impact assessment
        • Next patterns...

        Git Commit Logs to Analyze:
        {chunk}

        Remember to:
        - Be specific
        - Focus on substantial changes over minor updates
        - Consider the project context in your analysis
        - Maintain objectivity in your observations
        """

        return prompt