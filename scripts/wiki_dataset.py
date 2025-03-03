import requests
from bs4 import BeautifulSoup
import pandas as pd
import uuid
import re
from datasets import Dataset


def extract_day_from_link(item, month):
    """
    Extract day from a link in the item.

    Args:
        item (BeautifulSoup): The list item element
        month (str): The current month

    Returns:
        str or None: The day as a zero-padded string, or None if not found
    """
    month_day_pattern = (
        r"^(January|February|March|April|May|June|"
        + r"July|August|September|October|November|December) \d+$"
    )
    date_link = item.find("a", title=re.compile(month_day_pattern))

    if date_link:
        date_parts = date_link.get("title").split()
        if len(date_parts) == 2 and date_parts[0] == month:
            try:
                return date_parts[1].zfill(2)
            except ValueError:
                pass

    return None


def extract_day_from_text(item_text, month):
    """
    Extract day from the item text using regex patterns.

    Args:
        item_text (str): The text content of the item
        month (str): The current month

    Returns:
        str or None: The day as a zero-padded string, or None if not found
    """
    # Try to match patterns like "January 2" or "2 January"
    date_pattern = r"(?:" + month + r" (\d+)|(\d+) " + month + r")"
    date_match = re.search(date_pattern, item_text)
    if date_match:
        day_group = date_match.group(1) if date_match.group(1) else date_match.group(2)
        return day_group.zfill(2)

    # If still not found, try to match just a number at the beginning
    date_match = re.match(r"^(\d+)", item_text)
    if date_match:
        return date_match.group(1).zfill(2)

    return None


def split_into_events(text):
    """
    Split a text into multiple events based on newlines.

    Args:
        text (str): The text to split

    Returns:
        list: List of event descriptions
    """
    # Split the text by newlines
    raw_events = text.split("\n")

    # Clean up each event by removing citation markers and trimming whitespace
    events = []
    for event in raw_events:
        # Skip empty lines
        if not event.strip():
            continue

        # Remove citation markers like [25]
        clean_event = re.sub(r"\[\d+\]", "", event).strip()

        # Only add if there's meaningful content
        if clean_event and len(clean_event) > 5:
            events.append(clean_event)

    return events


def create_event_dict(day, month_num, year, description, url):
    """
    Create an event dictionary with all required fields.

    Args:
        day (str): The day of the event
        month_num (str): The month number as a string
        year (str): The year of the event
        description (str): The event description
        url (str): The source URL

    Returns:
        dict: The event dictionary
    """
    date_str = f"{year}-{month_num}-{day}"

    # Clean up the description - remove leading dash and spaces
    description = description.lstrip("– ").strip()

    # Remove year prefix if the first word is a year
    words = description.split()
    if words and words[0].isdigit() and len(words[0]) == 4:
        cleaned_description = " ".join(words[1:])
    else:
        cleaned_description = description

    # Try to extract location
    location = extract_location(cleaned_description)

    return {
        "id": str(uuid.uuid4()),
        "date": date_str,
        "description": cleaned_description,
        "location": location,
    }


def process_month_section(section, year, url):
    """
    Process a month section from a Wikipedia page.

    Args:
        section (bs4.element.Tag): The section element
        year (str): The year
        url (str): The URL of the page

    Returns:
        list: List of events extracted from the section
    """
    events = []
    section_title = section.get_text().strip()
    print(f"Processing section: {section_title[:50]}...")

    # Try to extract month from section title
    month_pattern = (
        r"(January|February|March|April|May|June|July|August|"
        + r"September|October|November|December)"
    )
    month_match = re.search(month_pattern, section_title)

    if not month_match:
        print(f"Could not find month in section title: {section_title}")
        return events

    month = month_match.group(1)
    print(f"Found month: {month}")

    month_num = {
        "January": "01",
        "February": "02",
        "March": "03",
        "April": "04",
        "May": "05",
        "June": "06",
        "July": "07",
        "August": "08",
        "September": "09",
        "October": "10",
        "November": "11",
        "December": "12",
    }.get(month)

    # Get the next ul element which contains the events
    event_list = section.find_next("ul")
    if not event_list:
        print(f"No event list found for {month}")
        # Try to find the ul within the section
        event_list = section.find("ul")
        if not event_list:
            return events

    print(
        f"Found event list with {len(event_list.find_all('li', recursive=False))} items"
    )

    for item in event_list.find_all("li", recursive=False):
        item_text = item.get_text().strip()
        print(f"Processing item: {item_text[:50]}...")

        # Try different methods to extract the day
        day = extract_day_from_link(item, month)
        if not day:
            day = extract_day_from_text(item_text, month)

        # If we found a day, create events
        if day:
            print(f"Found day: {day}")
            # Remove date prefix (like "January 1" or "January 1–29") if present
            # First remove the month name
            month_pattern = r"^" + month + r"\s*"
            cleaned_text = re.sub(month_pattern, "", item_text)

            # Then remove any day numbers and ranges at the beginning
            day_range_pattern = r"^\d+(?:[–\-]\d+)?\s*[–\-]?\s*"
            cleaned_text = re.sub(day_range_pattern, "", cleaned_text)

            # Split the cleaned text into multiple events if possible
            event_descriptions = split_into_events(cleaned_text)

            for description in event_descriptions:
                # Clean any remaining day range artifacts
                clean_desc = re.sub(
                    r"^\d+[–\-]\d+\s*[–\-]?\s*", "", description.strip()
                )
                event = create_event_dict(day, month_num, year, clean_desc, url)
                events.append(event)

            print(
                f"Extracted {len(event_descriptions)} events for {month} {day}, {year}"
            )
        else:
            print(f"Could not extract date from: {item_text[:100]}...")

    return events


def scrape_wikipedia_page(url):
    """
    Scrape events from a Wikipedia page.

    Args:
        url (str): URL of the Wikipedia page

    Returns:
        list: List of events
    """
    print(f"Scraping {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the year from the URL
    year = url.split("/")[-1]

    # Find all month sections by looking for headings
    month_sections = []
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Look for headings that contain month names
    for heading in soup.find_all(["h2", "h3"]):
        heading_text = heading.get_text().strip()
        for month in month_names:
            if month in heading_text:
                print(f"Found section for {month} in heading: {heading_text}")
                month_sections.append(heading)
                break

    events = []
    print(f"Processing {len(month_sections)} month sections")

    for section in month_sections:
        section_events = process_month_section(section, year, url)
        events.extend(section_events)
        print(f"Processed section, found {len(section_events)} events")

    print(f"Total events found: {len(events)}")
    return events


def extract_location(text):
    """
    Extract location from event description if possible.
    This is a simple implementation and might not be accurate for all cases.

    Args:
        text (str): Event description

    Returns:
        str: Location or 'Unknown'
    """
    # List of common countries and regions
    countries = [
        "Afghanistan",
        "Albania",
        "Algeria",
        "Argentina",
        "Australia",
        "Austria",
        "Bangladesh",
        "Belgium",
        "Brazil",
        "Canada",
        "China",
        "Colombia",
        "Denmark",
        "Egypt",
        "Ethiopia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "India",
        "Indonesia",
        "Iran",
        "Iraq",
        "Ireland",
        "Israel",
        "Italy",
        "Japan",
        "Kenya",
        "Malaysia",
        "Mexico",
        "Netherlands",
        "New Zealand",
        "Nigeria",
        "Norway",
        "Pakistan",
        "Philippines",
        "Poland",
        "Portugal",
        "Russia",
        "Saudi Arabia",
        "South Africa",
        "South Korea",
        "Spain",
        "Sweden",
        "Switzerland",
        "Taiwan",
        "Thailand",
        "Turkey",
        "Ukraine",
        "United Arab Emirates",
        "United Kingdom",
        "United States",
        "Venezuela",
        "Vietnam",
    ]

    for country in countries:
        if country in text or f"{country}n" in text:
            return country

    return "Unknown"


def create_wikipedia_dataset():
    """
    Create a dataset from Wikipedia pages for 2024 and 2025.

    Returns:
        Dataset: Hugging Face dataset
    """
    # URLs of the Wikipedia pages
    urls = ["https://en.wikipedia.org/wiki/2024", "https://en.wikipedia.org/wiki/2025"]

    # Scrape events from each page
    all_events = []
    for url in urls:
        events = scrape_wikipedia_page(url)
        all_events.extend(events)

    # Convert to DataFrame
    df = pd.DataFrame(all_events)

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    return dataset


def upload_to_huggingface(dataset, repo_id):
    """
    Upload the dataset to Hugging Face.

    Args:
        dataset (Dataset): Dataset to upload
        repo_id (str): Repository ID on Hugging Face
    """
    dataset.push_to_hub(repo_id)
    print(f"Dataset uploaded to {repo_id}")


def delete_from_huggingface(repo_id, token=None):
    """
    Delete a dataset from Hugging Face.

    Args:
        repo_id (str): Repository ID on Hugging Face
        token (str, optional): Hugging Face API token. Defaults to None.

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        from huggingface_hub import delete_repo

        # Delete the repository
        delete_repo(repo_id=repo_id, token=token)
        print(f"Dataset {repo_id} successfully deleted from Hugging Face")
        return True
    except Exception as e:
        print(f"Error deleting dataset {repo_id}: {str(e)}")
        return False
