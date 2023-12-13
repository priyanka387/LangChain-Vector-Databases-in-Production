#**Introduction**

In the digital era, the abundance of information can be overwhelming, and we often find ourselves scrambling to consume as much content as possible within our limited time. YouTube is a treasure trove of knowledge and entertainment, but it can be challenging to sift through long videos to extract the key takeaways. So we build a powerful AI that efficiently summarize YouTube videos using two cutting-edge tools: Whisper and LangChain.

##**Project Architecture**

![Image Alt Text](img/youtube_summarizer_archtecture.jpg)

##**Workflow**:

- 1.Download the YouTube audio file.
- 2. Transcribe the audio using Whisper.
- 3.Summarize the transcribed text using LangChain with three different approaches: stuff, refine, and map_reduce.
- 4.Adding multiple URLs to DeepLake database, and retrieving information. 

##**Installations**:

Remember to install the required packages with the following command: pip install langchain==0.0.208 deeplake openai==0.27.8 tiktoken. Additionally, install also the yt_dlp and openai-whisper packages

```
!pip install -q yt_dlp
!pip install -q git+https://github.com/openai/whisper.git

```

