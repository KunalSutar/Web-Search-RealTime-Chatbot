Hi Kunal! I've tackled a persistent issue in building AI models by creating a Web Search Chatbot that provides real-time answers using the Retrieval-Augmented Generation (RAG) system. I've named the model #"Furflexcity"#, inspired by its ability to deliver immediate and relevant information, akin to the concept of perplexity's next evolutionary step (alright, maybe not quite, but let's pretend it is for a moment).

Also the model includes clickable links at the end of each response, allowing users to explore the sources used for processing.To ensure the data is always current, I've integrated a Serper Google Search API. This allows the chatbot to fetch the latest top links from Google directly in response to queries. This approach overcomes the common challenge of using outdated data in AI model fine-tuning.

For the model itself, I've implemented Llama-2.7B-Chat, leveraging weaviate as a vector store to optimize data retrieval speed. This combination ensures the chatbot delivers timely and relevant information, making it a robust solution for accessing up-to-date data in AI applications.

You can find the test images in a folder that's quite intriguing; I encourage you to take a look! 
