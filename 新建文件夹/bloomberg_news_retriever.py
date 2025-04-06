#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bloomberg News Retriever
This script connects to Bloomberg and retrieves news stories and headlines.
"""

import blpapi
import datetime
import logging
import argparse
import os
import pandas as pd
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bloomberg API constants
NEWS_API_SVC = "//blp/newsapi"
NEWS_SEARCH_SVC = "//blp/newssearch"
REFDATA_SVC = "//blp/refdata"
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
RESPONSE_ERROR = blpapi.Name("ResponseError")


class BloombergNewsRetriever:
    """Class to retrieve news from Bloomberg"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Bloomberg News Retriever

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.news_api_service = None
        self.news_search_service = None
        self.refdata_service = None

    def start_session(self) -> bool:
        """Start a Bloomberg API session

        Returns:
            bool: True if session started successfully, False otherwise
        """
        logger.info("Starting Bloomberg API session...")
        
        # Initialize session options
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)
        
        # Create a session
        self.session = blpapi.Session(session_options)
        
        # Start the session
        if not self.session.start():
            logger.error("Failed to start session.")
            return False
        
        logger.info("Session started successfully.")
        
        # Open the news API service
        if not self.session.openService(NEWS_API_SVC):
            logger.error("Failed to open news API service.")
            return False
        
        self.news_api_service = self.session.getService(NEWS_API_SVC)
        logger.info("News API service opened successfully.")
        
        # Open the news search service
        if not self.session.openService(NEWS_SEARCH_SVC):
            logger.error("Failed to open news search service.")
            return False
        
        self.news_search_service = self.session.getService(NEWS_SEARCH_SVC)
        logger.info("News search service opened successfully.")
        
        # Open the reference data service
        if not self.session.openService(REFDATA_SVC):
            logger.error("Failed to open reference data service.")
            return False
        
        self.refdata_service = self.session.getService(REFDATA_SVC)
        logger.info("Reference data service opened successfully.")
        
        return True

    def stop_session(self) -> None:
        """Stop the Bloomberg API session"""
        if self.session:
            self.session.stop()
            logger.info("Session stopped.")

    def get_news_by_topic(self, topic: str, max_headlines: int = 10) -> List[Dict]:
        """Get news headlines by topic

        Args:
            topic: News topic (e.g., "TOP", "FIRST", "MARKET_STORIES")
            max_headlines: Maximum number of headlines to retrieve

        Returns:
            List[Dict]: List of news headlines
        """
        logger.info(f"Retrieving news for topic: {topic}")
        
        # Create a request for news by topic
        request = self.news_api_service.createRequest("NewsByTopicRequest")
        request.set("topic", topic)
        request.set("maxHeadlines", max_headlines)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        headlines = []
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("headlines"):
                    headlines_element = msg.getElement("headlines")
                    
                    for i in range(headlines_element.numValues()):
                        headline = headlines_element.getValue(i)
                        
                        headline_data = {
                            "time": headline.getElementAsDatetime("time"),
                            "title": headline.getElementAsString("title"),
                            "source": headline.getElementAsString("source") if headline.hasElement("source") else "",
                            "id": headline.getElementAsString("id")
                        }
                        
                        headlines.append(headline_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        logger.info(f"Retrieved {len(headlines)} headlines for topic: {topic}")
        return headlines

    def get_news_by_security(self, security: str, max_headlines: int = 10) -> List[Dict]:
        """Get news headlines for a specific security

        Args:
            security: Bloomberg security identifier
            max_headlines: Maximum number of headlines to retrieve

        Returns:
            List[Dict]: List of news headlines
        """
        logger.info(f"Retrieving news for security: {security}")
        
        # Create a request for news by security
        request = self.news_api_service.createRequest("NewsBySecurityRequest")
        request.set("security", security)
        request.set("maxHeadlines", max_headlines)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        headlines = []
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("headlines"):
                    headlines_element = msg.getElement("headlines")
                    
                    for i in range(headlines_element.numValues()):
                        headline = headlines_element.getValue(i)
                        
                        headline_data = {
                            "time": headline.getElementAsDatetime("time"),
                            "title": headline.getElementAsString("title"),
                            "source": headline.getElementAsString("source") if headline.hasElement("source") else "",
                            "id": headline.getElementAsString("id")
                        }
                        
                        headlines.append(headline_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        logger.info(f"Retrieved {len(headlines)} headlines for security: {security}")
        return headlines

    def search_news(self, query: str, max_results: int = 10, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None) -> List[Dict]:
        """Search for news using a query

        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            start_date: Start date for search (default: 7 days ago)
            end_date: End date for search (default: now)

        Returns:
            List[Dict]: List of news headlines
        """
        logger.info(f"Searching news with query: {query}")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.datetime.now()
        if not start_date:
            start_date = end_date - datetime.timedelta(days=7)
        
        # Create a request for news search
        request = self.news_search_service.createRequest("NewsSearchRequest")
        request.set("query", query)
        request.set("maxResults", max_results)
        request.set("startDateTime", start_date)
        request.set("endDateTime", end_date)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        results = []
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("results"):
                    results_element = msg.getElement("results")
                    
                    for i in range(results_element.numValues()):
                        result = results_element.getValue(i)
                        
                        result_data = {
                            "time": result.getElementAsDatetime("time"),
                            "title": result.getElementAsString("title"),
                            "source": result.getElementAsString("source") if result.hasElement("source") else "",
                            "id": result.getElementAsString("id")
                        }
                        
                        results.append(result_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        logger.info(f"Retrieved {len(results)} results for query: {query}")
        return results

    def get_news_story(self, story_id: str) -> str:
        """Get the full text of a news story

        Args:
            story_id: News story ID

        Returns:
            str: Full text of the news story
        """
        logger.info(f"Retrieving news story: {story_id}")
        
        # Create a request for news story
        request = self.news_api_service.createRequest("NewsStoryRequest")
        request.set("id", story_id)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        story_text = ""
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("story"):
                    story_element = msg.getElement("story")
                    story_text = story_element.getElementAsString("body")
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return story_text

    def save_headlines_to_csv(self, headlines: List[Dict], file_path: str) -> None:
        """Save news headlines to a CSV file

        Args:
            headlines: List of news headlines
            file_path: Path to save the CSV file
        """
        df = pd.DataFrame(headlines)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(headlines)} headlines to {file_path}")

    def save_story_to_file(self, story_id: str, file_path: str) -> None:
        """Save a news story to a text file

        Args:
            story_id: News story ID
            file_path: Path to save the text file
        """
        story_text = self.get_news_story(story_id)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(story_text)
        
        logger.info(f"Saved news story to {file_path}")


def main():
    """Main function to run the Bloomberg News Retriever"""
    parser = argparse.ArgumentParser(description='Retrieve news from Bloomberg')
    parser.add_argument('--topic', help='News topic (e.g., TOP, FIRST, MARKET_STORIES)')
    parser.add_argument('--security', help='Bloomberg security identifier')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--max', type=int, default=10, help='Maximum number of results')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back for search')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--story', help='News story ID to retrieve')
    
    args = parser.parse_args()
    
    # Initialize the Bloomberg News Retriever
    retriever = BloombergNewsRetriever()
    
    try:
        # Start the Bloomberg session
        if not retriever.start_session():
            logger.error("Failed to initialize Bloomberg session. Exiting.")
            return
        
        # Create output directory if it doesn't exist
        output_dir = "bloomberg_news"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process based on arguments
        if args.story:
            # Retrieve a specific news story
            output_file = args.output or os.path.join(output_dir, f"story_{args.story}.txt")
            retriever.save_story_to_file(args.story, output_file)
            print(f"News story saved to {output_file}")
        
        elif args.topic:
            # Retrieve news by topic
            headlines = retriever.get_news_by_topic(args.topic, args.max)
            
            # Display headlines
            print(f"\nLatest {len(headlines)} headlines for topic '{args.topic}':")
            for i, headline in enumerate(headlines, 1):
                print(f"{i}. [{headline['time']}] {headline['title']} (Source: {headline['source']})")
                print(f"   ID: {headline['id']}")
            
            # Save to CSV if output is specified
            if args.output:
                retriever.save_headlines_to_csv(headlines, args.output)
                print(f"Headlines saved to {args.output}")
            else:
                output_file = os.path.join(output_dir, f"topic_{args.topic}.csv")
                retriever.save_headlines_to_csv(headlines, output_file)
                print(f"Headlines saved to {output_file}")
        
        elif args.security:
            # Retrieve news by security
            headlines = retriever.get_news_by_security(args.security, args.max)
            
            # Display headlines
            print(f"\nLatest {len(headlines)} headlines for security '{args.security}':")
            for i, headline in enumerate(headlines, 1):
                print(f"{i}. [{headline['time']}] {headline['title']} (Source: {headline['source']})")
                print(f"   ID: {headline['id']}")
            
            # Save to CSV if output is specified
            if args.output:
                retriever.save_headlines_to_csv(headlines, args.output)
                print(f"Headlines saved to {args.output}")
            else:
                output_file = os.path.join(output_dir, f"security_{args.security.replace(' ', '_')}.csv")
                retriever.save_headlines_to_csv(headlines, output_file)
                print(f"Headlines saved to {output_file}")
        
        elif args.query:
            # Search for news
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=args.days)
            
            results = retriever.search_news(args.query, args.max, start_date, end_date)
            
            # Display results
            print(f"\nTop {len(results)} results for query '{args.query}' in the last {args.days} days:")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['time']}] {result['title']} (Source: {result['source']})")
                print(f"   ID: {result['id']}")
            
            # Save to CSV if output is specified
            if args.output:
                retriever.save_headlines_to_csv(results, args.output)
                print(f"Results saved to {args.output}")
            else:
                output_file = os.path.join(output_dir, f"search_{args.query.replace(' ', '_')}.csv")
                retriever.save_headlines_to_csv(results, output_file)
                print(f"Results saved to {output_file}")
        
        else:
            # No specific action, retrieve top news
            headlines = retriever.get_news_by_topic("TOP", args.max)
            
            # Display headlines
            print(f"\nLatest {len(headlines)} top headlines:")
            for i, headline in enumerate(headlines, 1):
                print(f"{i}. [{headline['time']}] {headline['title']} (Source: {headline['source']})")
                print(f"   ID: {headline['id']}")
            
            # Save to CSV
            output_file = os.path.join(output_dir, "top_news.csv")
            retriever.save_headlines_to_csv(headlines, output_file)
            print(f"Headlines saved to {output_file}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Stop the Bloomberg session
        retriever.stop_session()


if __name__ == "__main__":
    main()
