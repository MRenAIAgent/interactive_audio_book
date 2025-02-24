        class SerpySearchActionFactory(ActionFactory):
            def create_actions(self) -> List[Action]:
                return [
                    Action(
                        name="search",
                        description="Search for information in the story",
                        parameters=[
                            ActionParameter(
                                name="query",
                                description="The search query",
                                type="string"
                            )
                        ],
                        implementation=self.search
                    )
                ]
            
            def search(self, query: str) -> str:
                # Here you would implement the actual search logic
                # For now returning a placeholder response
                
                # Initialize search parameters
                search_params = {
                    "q": query,
                    "api_key": os.getenv("SERPAPI_API_KEY"),
                    "engine": "google"
                }
                
                try:
                    # Perform the search
                    search = GoogleSearch(search_params)
                    results = search.get_dict()
                    
                    # Extract organic results
                    if "organic_results" in results:
                        top_result = results["organic_results"][0]
                        return f"Found: {top_result['title']}\n{top_result['snippet']}"
                    else:
                        return "No results found"
                        
                except Exception as e:
                    print(f"Search error: {e}")
                    return "Sorry, there was an error performing the search"
