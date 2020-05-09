# -*- coding: utf-8 -*-

# Author: Suyog Dahal
# Created: 06/05/2020
# Description: A script to create new issues in a GitHub repository using
# GitHub API

import os
import requests
import json
import glob
import datetime as dt
import argparse


def cli_load_arguments(config_file=None):
    """
        Load CLI input
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("github_user", help="GitHub Username")
    parser.add_argument("github_token", help="GitHub Token")
    args = parser.parse_args()
    return args


def create_issue(
        REPO_OWNER,
        REPO_NAME,
        headers,
        title,
        body=None,
        labels=None):
    '''
    Function to create an issue
    Parameters: https://developer.github.com/v3/issues/#create-an-issue
    '''

    # Create url
    url = 'https://api.github.com/repos/{}/{}/import/issues'.format(
        REPO_OWNER, REPO_NAME)

    # Create data
    data = {"issue": {"title": title,
                      "body": body,
                      "labels": labels}
            }

    # Constructing JSON payload from data
    payload = json.dumps(data)

    # Request to create new issue
    response = requests.request("POST", url, data=payload, headers=headers)
    if response.status_code == 202:
        print('Successfully created the issue: {}'.format(title))
    else:
        print('Could not create the issue: {}'.format(title))
        print('Response: {}'.format(response.content))


def main():
    """
    Main Execution
    """
    args = cli_load_arguments()
    USERNAME = args.github_user  # GitHub Username
    TOKEN = args.github_token  # GitHub Token

    # The repository to add this issue to
    REPO_OWNER = USERNAME
    REPO_NAME = 'mlmodels_store'

    # Header for Authentication
    headers = {
        "Authorization": "token %s" % TOKEN,
        "Accept": "application/vnd.github.golden-comet-preview+json"
    }

    # Get the files in the error_list folder
    execution_date = str(dt.datetime.now().date()).replace('-', '')
    list_of_files = glob.glob('error_list/' + execution_date + '/*.md')

    # Error Tile and labels
    title = f"Error List {execution_date}"
    lables = ["Errors"]

    # Error Body
    body = ''
    for file in list_of_files:
        body += (
            f"https://github.com/{USERNAME}/mlmodels_store/blob/master/{file}" +
            ' \n')
    create_issue(REPO_OWNER, REPO_NAME, headers, title, body, lables)


if __name__ == "__main__":
    main()
