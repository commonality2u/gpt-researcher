import os
import io
import argparse
from typing import List, Tuple

def read_file(file_path: str) -> Tuple[str, bool]:
    """
    Reads the content of a file if it exists.

    Args:
        file_path (str): The absolute path to the file.

    Returns:
        Tuple[str, bool]: A tuple containing the file content or an error message,
                          and a boolean indicating success.
    """
    if not os.path.isfile(file_path):
        return f"**Error:** `{file_path}` does not exist.\n", False
    try:
        with io.open(file_path, 'r', encoding='utf-8') as file:
            return file.read(), True
    except Exception as e:
        return f"**Error reading {file_path}:** {str(e)}\n", False

def combine_files(file_paths: List[str], include_errors: bool = False) -> Tuple[str, List[str]]:
    """
    Combines the content of multiple files into a single string.

    Args:
        file_paths (List[str]): List of absolute file paths to combine.
        include_errors (bool): Whether to include error messages for missing files.

    Returns:
        Tuple[str, List[str]]: Combined content and a list of missing files.
    """
    combined_content = ""
    missing_files = []

    for file_path in file_paths:
        combined_content += f"\n\n# {file_path}\n\n"
        content, success = read_file(file_path)
        if not success:
            missing_files.append(file_path)
            if include_errors:
                combined_content += content
            else:
                combined_content += f"**{file_path}** is missing and was not included.\n"
        else:
            combined_content += content

    return combined_content, missing_files

def write_combined_file(combined_content: str, output_file: str) -> bool:
    """
    Writes the combined content to the specified output file.

    Args:
        combined_content (str): The content to write.
        output_file (str): The absolute path to the output file.

    Returns:
        bool: True if writing was successful, False otherwise.
    """
    try:
        with io.open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        print(f"✅ Combined files have been written to `{output_file}`.")
        return True
    except Exception as e:
        print(f"❌ Failed to write to `{output_file}`: {str(e)}")
        return False

def log_missing_files(missing_files: List[str], log_file: str = "missing_files.log"):
    """
    Logs the list of missing files to a separate log file.

    Args:
        missing_files (List[str]): List of missing file paths.
        log_file (str): Path to the log file.
    """
    if not missing_files:
        print("🎉 All files were successfully combined. No missing files.")
        return

    try:
        with io.open(log_file, 'w', encoding='utf-8') as f:
            f.write("### Missing Files Report\n\n")
            for file in missing_files:
                f.write(f"- {file}\n")
        print(f"📝 Missing files have been logged to `{log_file}`.")
    except Exception as e:
        print(f"❌ Failed to write missing files log: {str(e)}")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Combine multiple files into a single Markdown file.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="combined_files.md",
        help="Output Markdown file name with absolute path."
    )
    parser.add_argument(
        "-e", "--include-errors",
        action="store_true",
        help="Include error messages for missing files in the combined output."
    )
    parser.add_argument(
        "-l", "--log-file",
        type=str,
        default="missing_files.log",
        help="Log file to record missing files with absolute path."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Updated list of absolute file paths to combine
    files_to_combine = [
        r"D:\Documents\Github\gpt-researcher-wes\gpt_researcher\retrievers\__init__.py",
        r"D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\agent\master.py",
        r"D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\actions\__init__.py",
        r"D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\actions\query_processing.py",
        r"D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\actions\report_generation.py",
        r"D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\agent\writer.py",
        r"D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\prompts.py",
        r"D:\Documents\Github\gpt-researcher-wes\backend\server\server.py",
        r"D:\Documents\Github\gpt-researcher-wes\main.py",
        r"D:\Documents\Github\gpt-researcher-wes\requirements.txt",
        r"D:\Documents\Github\gpt-researcher-wes\README.md",
        r"D:\Documents\Github\gpt-researcher-wes\frontend\scripts.js",
        r"D:\Documents\Github\gpt-researcher-wes\tests\documents-report-source.py",
        r"D:\Documents\Github\gpt-researcher-wes\docs\docs\examples\hybrid_research.md",
        r"D:\Documents\Github\gpt-researcher-wes\docs\blog\2023-09-22-gpt-researcher\index.md",
        r"D:\Documents\Github\gpt-researcher-wes\docs\blog\2024-09-7-hybrid-research\index.md",
        r"D:\Documents\Github\gpt-researcher-wes\multi_agents\README.md",
        r"D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\gptr\config.md",
        r"D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\gptr\example.md",
        r"D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\getting-started\how-to-choose.md",
        r"D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\getting-started\introduction.md"
    ]

    # Combine the files
    combined_content, missing_files = combine_files(files_to_combine, include_errors=args.include_errors)

    # Write the combined content to a new file
    output_file = os.path.abspath(args.output)
    if write_combined_file(combined_content, output_file):
        # Log missing files separately
        log_missing_files(missing_files, log_file=os.path.abspath(args.log_file))

    # Summary Report
    total_files = len(files_to_combine)
    successfully_combined = total_files - len(missing_files)
    print("\n### 📝 Summary Report")
    print(f"- **Total Files Attempted:** {total_files}")
    print(f"- **Successfully Combined:** {successfully_combined}")
    print(f"- **Missing Files:** {len(missing_files)}")

    if missing_files:
        print("\n#### ❌ Missing Files:")
        for file in missing_files:
            print(f"  - {file}")

if __name__ == "__main__":
    main()
