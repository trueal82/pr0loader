import os
import json
import time
import logging
from typing import Any, List, Tuple

import requests
import dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
import pymongo.errors

# Configure logging with level INFO and a specific format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)

# Global configuration variables
REQUIRED_CONFIG_KEYS = ['PP', 'CONSENT', 'MONGODB_STRING', 'FILESYSTEM_PREFIX']
CONTENT_FLAGS = 15  # Adjust as needed
HTTP_MAX_TRIES = 100
HTTP_TIMEOUT = 30
BASE_API_URL = "https://pr0gramm.com/api"
REMOTE_MEDIA_PREFIX = "https://img.pr0gramm.com"


def get_mongo_collection(collection_name: str, config: dict) -> Collection:
    """
    Connect to the MongoDB database and return the specified collection.

    Args:
        collection_name (str): The name of the collection to retrieve.
        config (dict): Configuration dictionary containing the MongoDB connection string.

    Returns:
        Collection: The MongoDB collection object.
    """
    connection_string = config['MONGODB_STRING']
    client = MongoClient(connection_string)
    database = client["pr0loader"]
    return database[collection_name]


def validate_config(config: dict) -> bool:
    """
    Check if the required configuration keys are present in the config dictionary.

    Args:
        config (dict): Configuration dictionary to validate.

    Returns:
        bool: True if all required keys are present, False otherwise.
    """
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {', '.join(missing_keys)}")
        return False
    return True


def setup_http_session(config: dict) -> requests.Session:
    """
    Set up the HTTP session with the necessary cookies.

    Args:
        config (dict): Configuration dictionary containing cookie values.

    Returns:
        requests.Session: Configured HTTP session with cookies set.
    """
    session = requests.Session()
    cookies = {
        'me': config['ME'],
        'pp': config['PP']
    }
    session.cookies.update(cookies)
    return session


def fetch_json(url: str, session: requests.Session) -> dict:
    """
    Fetch JSON data from a given URL using the provided session, with retries.

    Args:
        url (str): The URL to fetch data from.
        session (requests.Session): The HTTP session to use for the request.

    Returns:
        dict: The JSON data fetched from the URL.

    Raises:
        Exception: If the data cannot be fetched after max retries.
    """
    tries = 1
    while tries <= HTTP_MAX_TRIES:
        try:
            logging.info(f"Attempt {tries}/{HTTP_MAX_TRIES} - Fetching data from {url}")
            response = session.get(url, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else None
            if status_code == 429:
                # Handle rate limiting
                retry_after = int(http_err.response.headers.get('Retry-After', 60))
                logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            elif status_code and 400 <= status_code < 500:
                logging.error(f"Client error {status_code}: {http_err.response.reason}")
                break  # Do not retry on client errors
            else:
                logging.error(f"HTTP error occurred: {http_err}")
                time.sleep(2 ** tries)
        except requests.RequestException as req_err:
            logging.error(f"Network error occurred: {req_err}")
            time.sleep(2 ** tries)
        tries += 1
    logging.error(f"Failed to fetch data from {url} after {HTTP_MAX_TRIES} attempts")
    raise Exception(f"Failed to fetch data from {url}")


def get_db_item_id(sort_order: List[Tuple[str, int]], collection: Collection) -> int:
    """
    Retrieve the item ID from the database based on the specified sort order.

    Args:
        sort_order (List[Tuple[str, int]]): The sort order for the query.
        collection (Collection): The MongoDB collection to query.

    Returns:
        int: The item ID from the database, or -1 if not found.
    """
    try:
        document = collection.find_one({}, projection={'id': 1}, sort=sort_order)
        if document:
            return document['id']
        else:
            return -1
    except pymongo.errors.PyMongoError as db_err:
        logging.error(f"Database error: {db_err}")
        return -1


def get_min_db_id(collection: Collection) -> int:
    """
    Retrieve the minimum item ID from the database.

    Args:
        collection (Collection): The MongoDB collection to query.

    Returns:
        int: The minimum item ID, or -1 if not found.
    """
    return get_db_item_id([('id', 1)], collection)


def get_max_db_id(collection: Collection) -> int:
    """
    Retrieve the maximum item ID from the database.

    Args:
        collection (Collection): The MongoDB collection to query.

    Returns:
        int: The maximum item ID, or -1 if not found.
    """
    return get_db_item_id([('id', -1)], collection)


def fetch_highest_remote_id(session: requests.Session) -> int:
    """
    Fetch the highest item ID available from the remote API.

    Args:
        session (requests.Session): The HTTP session to use for the request.

    Returns:
        int: The highest remote item ID.

    Raises:
        Exception: If no items are found in the remote data.
    """
    logging.info("Fetching the highest remote item ID")
    url = f"{BASE_API_URL}/items/get?flags={CONTENT_FLAGS}"
    data = fetch_json(url, session)
    items = data.get('items', [])
    if items:
        return items[0]['id']
    else:
        logging.error("No items found in remote data.")
        raise Exception("Failed to fetch highest remote item ID")


def determine_id_range(collection: Collection, session: requests.Session, full_update: bool, start_from: int) -> Tuple[int, int]:
    """
    Determine the range of item IDs to process based on local and remote data.

    Args:
        collection (Collection): The MongoDB collection to query.
        session (requests.Session): The HTTP session to use for API requests.
        full_update (bool): Whether to perform a full update.

    Returns:
        Tuple[int, int]: A tuple containing the start ID and end ID.
    """
    highest_remote_id = fetch_highest_remote_id(session)

    if start_from:
        logging.info(f"START_FROM is set. Start preset to {start_from}")
        return start_from, 1
    elif full_update:
        # Process all items from highest remote ID down to ID 1
        logging.info("FULL_UPDATE is enabled. Processing all items.")
        return highest_remote_id, 1
    else:
        # Proceed with the existing logic
        min_db_id = get_min_db_id(collection)
        max_db_id = get_max_db_id(collection)
        logging.info(f"Local DB min ID: {min_db_id}, max ID: {max_db_id}")
        logging.info(f"Highest remote ID: {highest_remote_id}")

        if max_db_id == -1 or min_db_id == -1:
            # Case a: No local data, start from highest remote ID to 1
            logging.info("No local data found. Starting from the highest remote ID.")
            return highest_remote_id, 1
        elif min_db_id != 1:
            # Case b: Local data exists, but min ID is not 1
            logging.info("Local data found, but min ID is not 1. Continuing from min DB ID.")
            return min_db_id, 1
        else:
            # Case c: Local data exists, min ID is 1
            logging.info("Local data complete from ID 1. Starting from highest remote ID to max DB ID.")
            return highest_remote_id, max_db_id


def fetch_item_info(item_id: int, session: requests.Session) -> dict:
    """
    Fetch detailed information for a specific item from the API.

    Args:
        item_id (int): The ID of the item to fetch information for.
        session (requests.Session): The HTTP session to use for the request.

    Returns:
        dict: The JSON data containing item information.
    """
    url = f"{BASE_API_URL}/items/info?itemId={item_id}&flags={CONTENT_FLAGS}"
    return fetch_json(url, session)


def process_items_metadata(items_data: dict, collection: Collection, session: requests.Session):
    """
    Process the metadata of items and insert or update them in the database.

    Args:
        items_data (dict): The JSON data containing items.
        collection (Collection): The MongoDB collection to insert or update data.
        session (requests.Session): The HTTP session to use for additional API requests.
    """
    items = items_data.get('items', [])
    for item in items:
        try:
            # Fetch additional info for the item
            detailed_info = fetch_item_info(item['id'], session)
            if 'comments' in detailed_info:
                item['comments'] = detailed_info['comments']
            if 'tags' in detailed_info:
                item['tags'] = detailed_info['tags']
            
            # Define the filter and update for upsert operation
            filter_query = {'id': item['id']}
            update_data = {'$set': item}
            
            # Insert or update the item in the database
            result = collection.update_one(filter_query, update_data, upsert=True)
            
            if result.matched_count > 0:
                logging.info(f"Updated item {item['id']} in the database")
            elif result.upserted_id is not None:
                logging.info(f"Inserted new item {item['id']} into the database")
            else:
                logging.info(f"Item {item['id']} was already up-to-date in the database")
        except pymongo.errors.PyMongoError as db_err:
            logging.error(f"Failed to insert or update item {item['id']} in the database: {db_err}")
        except Exception as e:
            logging.error(f"Error processing item {item['id']}: {e}")


def get_filesystem_prefix(config: dict) -> str:
    """
    Get the filesystem prefix from the configuration, ensuring it ends with a slash.

    Args:
        config (dict): Configuration dictionary containing the filesystem prefix.

    Returns:
        str: The filesystem prefix with a trailing slash.
    """
    fs_prefix = str(config['FILESYSTEM_PREFIX'])
    if not fs_prefix.endswith("/"):
        fs_prefix += "/"
    return fs_prefix


def download_media_files(items_data: dict, config: dict, session: requests.Session):
    """
    Download media files for the items in the provided data.

    Args:
        items_data (dict): The JSON data containing items.
        config (dict): Configuration dictionary for filesystem settings.
        session (requests.Session): The HTTP session to use for media requests.
    """
    fs_prefix = get_filesystem_prefix(config)
    items = items_data.get('items', [])
    for item in items:
        media_filename = item['image']
        local_file_path = os.path.join(fs_prefix, media_filename)
        media_url = f"{REMOTE_MEDIA_PREFIX}/{media_filename}"

        logging.info(f"Preparing to download media: {media_url}")

        # Skip download if file already exists
        if os.path.exists(local_file_path):
            logging.info(f"File {local_file_path} already exists. Skipping download.")
            continue

        tries = 1
        while tries <= HTTP_MAX_TRIES:
            try:
                logging.info(f"Attempt {tries}/{HTTP_MAX_TRIES} - Downloading media from {media_url}")
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                response = session.get(media_url, timeout=HTTP_TIMEOUT, stream=True)
                response.raise_for_status()
                size = 0
                with open(local_file_path, 'wb') as file_handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file_handle.write(chunk)
                            size += len(chunk)
                logging.info(f"Downloaded {size} bytes for {media_filename}")
                break  # Download successful, exit retry loop
            except requests.HTTPError as http_err:
                status_code = http_err.response.status_code if http_err.response is not None else None
                logging.error(f"Caught HTTPError with status: {status_code}")
                if status_code == 429:
                    retry_after = int(http_err.response.headers.get('Retry-After', 60))
                    logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                elif status_code and 400 <= status_code < 500:
                    logging.error(f"Client error {status_code}: {http_err.response.reason}")
                    break  # Do not retry on client errors
                else:
                    logging.error(f"HTTP error occurred: {http_err}")
                    time.sleep(2 ** tries)
            except requests.RequestException as req_err:
                logging.error(f"Network error occurred: {req_err}")
                time.sleep(2 ** tries)
            except Exception as e:
                logging.exception(f"Unexpected error occurred: {e}")
                raise
            tries += 1
        else:
            logging.error(f"Failed to download media from {media_url} after {HTTP_MAX_TRIES} attempts")


def get_next_item_id(items_data: dict) -> int:
    """
    Get the next item ID from the items data for pagination.

    Args:
        items_data (dict): The JSON data containing items.

    Returns:
        int: The next item ID, or None if not available.
    """
    items = items_data.get('items', [])
    if items:
        return items[-1]['id']
    else:
        return None


def main():
    """
    Main function to orchestrate the data fetching and processing.
    """
    logging.info("Starting the data loader script")

    # Load configuration from environment and .env file
    config = {
        **dotenv.dotenv_values(".env"),
        **os.environ
    }

    # Validate configuration
    if not validate_config(config):
        logging.error(
            f"The following configuration keys are required: {', '.join(REQUIRED_CONFIG_KEYS)}"
        )
        exit(1)

    # Parse FULL_UPDATE parameter from configuration
    full_update_str = config.get('FULL_UPDATE', 'False').lower()
    full_update = full_update_str in ('true', '1', 'yes')

    # Parse START_FROM parameter from configuration
    start_from = int(config.get('START_FROM', 1))

    if full_update:
        logging.info("FULL_UPDATE is enabled in configuration")

    # Set up the HTTP session
    session = setup_http_session(config)

    # Get the MongoDB collection
    mongo_collection = get_mongo_collection("pr0items", config)

    # Ensure an index on 'id' field for efficient upsert operations
    mongo_collection.create_index('id', unique=True)

    # Determine the range of item IDs to process
    start_id, end_id = determine_id_range(mongo_collection, session, full_update, start_from)
    current_id = start_id
    logging.info(f"Starting to process items from ID {current_id} down to {end_id}")

    # Main processing loop
    try:
        while True:
            logging.info(f"Fetching items starting with ID {current_id}")
            url = f"{BASE_API_URL}/items/get?older={current_id}&flags={CONTENT_FLAGS}"
            items_data = fetch_json(url, session)

            # Check if any items were returned
            if not items_data.get('items'):
                logging.info("No more items to process.")
                break

            # Process item metadata and download media files
            process_items_metadata(items_data, mongo_collection, session)
            download_media_files(items_data, config, session)

            # Get the next item ID for pagination
            next_id = get_next_item_id(items_data)
            if next_id is None or next_id >= current_id or next_id <= end_id:
                logging.info("No more items to process or reached the end ID.")
                break
            else:
                current_id = next_id

            # Sleep to respect API rate limits
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

    logging.info("Data loader script has completed")


if __name__ == "__main__":
    main()