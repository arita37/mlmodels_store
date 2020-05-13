# -*- coding: utf-8 -*-
# Created: 06/05/2020
# Description: A script to parse errors from a logfile and store the errors into a new file

import glob
import os
import re
import sys
import datetime as dt
import argparse
import json



def cli_load_arguments(config_file=None):
    """
        Load CLI input
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--github_user",
                        default="arita37",
                        help="GitHub Username")
    args = parser.parse_args()
    return args


def create_error_list(latest_file):
    """
        Return List of Errors
    """
    traceback_found = False  # Flag for Traceback Line
    error_re = re.compile('[a-zA-z]+Error: .+')  # Regex to match error lines
    output = []

    with open(latest_file, 'r') as fp:
        lines = fp.readlines()  # read the lines in the file
        for idx, line in enumerate(lines):  # loop over lines
            error_found = False
            if 'TAG ::' in line:
                json_acceptable_string = line.split(
                    "::")[1].strip().replace("'", "\"")
                tag_dict = json.loads(json_acceptable_string)
            if 'Traceback' in line:
                traceback_found = True
                error_found = False
                line = str(idx + 1) + '..' + line
            if error_re.match(line) is not None:
                traceback_found = False
                error_found = True
            if traceback_found or error_found:
                output.append(line)

    return output, tag_dict


def create_error_file(
        output,
        tag_dict,
        output_file_dir,
        output_file_name,
        latest_file_link):
    """
        Create Error Files
    """
    error_cnt = 0

    # Create the new dir
    if not os.path.isdir(output_file_dir):
        os.mkdir(output_file_dir)

    output_file_path = output_file_dir + output_file_name

    # Create Empty File if no error is found
    if len(output) == 0:
        output = ["\n\n### No Error"]

    with open(output_file_path, 'w') as f:  # Write to o/p file
        f.write(f"## Original File URL: {latest_file_link}")
        for idx, line in enumerate(output):
            if 'Traceback' in line:
                error_cnt += 1
                line_of_traceback = line.split('..')[0]
                # Write error number and error file's location
                f.write(
                    f"\n\n\n### Error {error_cnt}, [Traceback at line {line_of_traceback}]({latest_file_link}#L{line_of_traceback})")
                
            ii = ""
            if ", line" in line2 "
               ii = line2.split(",")[1].replace("line", " ").strip()    

            if '/home/runner/work/mlmodels/mlmodels/mlmodels/' in line:
                line = line.replace(
                    '/home/runner/work/mlmodels/mlmodels',
                    f"{tag_dict['github_repo_url']}")
            f.write(f"<br />{line}")

        print(f"Sucessfully created the error file {output_file_name}")


def log_info_repo(arg=None):
    """
       Grab Github Variables
       https://help.github.com/en/actions/configuring-and-managing-workflows/using-environment-variables
       log_info_repo(arg=None)
    """
    repo = os_bash("echo $GITHUB_REPOSITORY")
    sha = os_bash("echo $GITHUB_SHA")
    workflow = os_bash("echo $GITHUB_WORKFLOW")
    branch = os_bash("echo $GITHUB_REF")

    repo = repo.replace("\n", "").replace("\r", "").strip()
    workflow = workflow.replace("\n", "").replace("\r", "").strip()
    sha = sha.replace("\n", "").replace("\r", "").strip()
    branch = branch.replace(
        "\n",
        "").replace(
        "\r",
        "").strip().replace(
            "refs/heads/",
        "")

    github_repo_url = f"https://github.com/{repo}/tree/{sha}"
    url_branch_file = f"https://github.com/{repo}/blob/{branch}/"

    url_branch_file2 = f"https://github.com/{repo}/tree/{branch}/"

    return_dict = {
        "github_repo_url": github_repo_url,
        "url_branch_file": url_branch_file,
        "repo": repo,
        "branch": branch,
        "sha": sha,
        "workflow": workflow
    }

    print(return_dict)
    return return_dict


def main():
    """
    Main Execution
    """
    args = cli_load_arguments()
    USERNAME = args.github_user  # GitHub Username

    # List of all log folders
    log_folders = [
        'log_benchmark',
        'log_dataloader',
        'log_import',
        'log_json',
        'log_jupyter',
        'log_pullrequest',
        'log_test_cli',
        'log_testall']

    # Main Script
    for log_folder in log_folders:
        dir = os.path.dirname(__file__)  # File Path
        file_path = os.path.join(dir, log_folder + '/*.py')
        list_of_files = glob.glob(file_path)
        latest_file_path = max(list_of_files, key=os.path.getctime)
        latest_file_name = latest_file_path.split('/')[-1]
        latest_file_link = "https://github.com/{}/mlmodels_store/blob/master/{}".format(
            USERNAME, latest_file_path.split('/')[-2] + '/' + latest_file_name)  # Original file url

        output, tag_dict = create_error_list(latest_file_path)

        # In the format
        # /error_list/execution_date/list_log_folder_executionDate.md
        execution_date = str(dt.datetime.now().date()).replace('-', '')
        output_file_dir = os.path.join(dir,
                                       "error_list/" +
                                       execution_date +
                                       '/')

        output_file_name = "list" + \
            "_{}_".format(log_folder) + execution_date + '.md'
        create_error_file(
            output,
            tag_dict,
            output_file_dir,
            output_file_name,
            latest_file_link)


if __name__ == "__main__":
    main()
